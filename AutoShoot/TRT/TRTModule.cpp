//
// Created by yuesang on 23-1-13.
//
#include "TRTModule.h"
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include<vector>
#include<string.h>
#include <fmt/color.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#define IMAGE_SHOW

static inline int argmax(const float *ptr, int len) {
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

template<class F, class T, class ...Ts>
T reduce(F &&func, T x, Ts... xs) {
    if constexpr (sizeof...(Ts) > 0){
        return func(x, reduce(std::forward<F>(func), xs...));
    } else {
        return x;
    }
}

template<class T, class ...Ts>
T reduce_max(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::max(a, b);}, x, xs...);
}

template<class T, class ...Ts>
T reduce_min(T x, Ts... xs) {
    return reduce([](auto &&a, auto &&b){return std::min(a, b);}, x, xs...);
}

static inline float iou(const float pts1[8], const float pts2[8]) {
    cv::Rect2f bbox1, bbox2;
    bbox1.x = reduce_min(pts1[0], pts1[2], pts1[4], pts1[6]);
    bbox1.y = reduce_min(pts1[1], pts1[3], pts1[5], pts1[7]);
    bbox1.width = reduce_max(pts1[0], pts1[2], pts1[4], pts1[6]) - bbox1.x;
    bbox1.height = reduce_max(pts1[1], pts1[3], pts1[5], pts1[7]) - bbox1.y;

    bbox2.x = reduce_min(pts2[0], pts2[2], pts2[4], pts2[6]);
    bbox2.y = reduce_min(pts2[1], pts2[3], pts2[5], pts2[7]);
    bbox2.width = reduce_max(pts2[0], pts2[2], pts2[4], pts2[6]) - bbox2.x;
    bbox2.height = reduce_max(pts2[1], pts2[3], pts2[5], pts2[7]) - bbox2.y;

    cv::Rect And = bbox1 | bbox2;
    cv::Rect U = bbox1 & bbox2;

    return U.area()*1.0 / And.area();
}

TRTModule::TRTModule(const std::string &onnx_file) {
    Init(onnx_file);
    assert(Engine->getNbBindings() == 2);
    assert((input_Index = Engine->getBindingIndex("input")) == 0);
    assert((output_Index = Engine->getBindingIndex("output-topk")) == 1);
    assert(Engine->getBindingDataType(input_Index) == nvinfer1::DataType::kFLOAT);
    assert(Engine->getBindingDataType(output_Index) == nvinfer1::DataType::kFLOAT);


    inputdims = Engine->getBindingDimensions(input_Index);
    std::cout << "[INFO]: input dims " << inputdims.d[0] << " " << inputdims.d[1] << " " << inputdims.d[2] << " " << inputdims.d[3] << std::endl;
    outpusdims = Engine->getBindingDimensions(output_Index);
    std::cout << "[INFO]: output dims "<< outpusdims.d[0] << " " << outpusdims.d[1] << " " << outpusdims.d[2] << std::endl;

    cudaMalloc(&device_buffer[input_Index], inputdims.d[0]*inputdims.d[1]*inputdims.d[2]*inputdims.d[3] * sizeof(float));
    cudaMalloc(&device_buffer[output_Index], outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2] * sizeof(float));


    cudaStreamCreate(&stream);
    output_buffer = new float[outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2]];

    assert(output_buffer != nullptr);
}

TRTModule::~TRTModule() {
    delete[] output_buffer;
    cudaStreamDestroy(stream);
    cudaFree(device_buffer[input_Index]);
    cudaFree(device_buffer[output_Index]);
    Engine->destroy();

}

std::vector<bbox_t> TRTModule::operator()( cv::Mat &src,float conf_thres,float iou_thres,float kpts_conf,int nc,int kpts) const{
    auto start = std::chrono::system_clock::now();
    cv::Mat x=src.clone();
    if (src.cols != inputdims.d[1] || src.rows != inputdims.d[2]) {
        cv::resize(x, x, {inputdims.d[1], inputdims.d[2]});
    }
    cv::Mat visual_img=x.clone();
    x.convertTo(x, CV_32F);

    cudaMemcpyAsync(device_buffer[input_Index], x.data, inputdims.d[0]*inputdims.d[1]*inputdims.d[2]*inputdims.d[3] * sizeof(float), cudaMemcpyHostToDevice, stream);
    Context->enqueueV2(device_buffer,stream, nullptr);
    cudaMemcpyAsync(output_buffer, device_buffer[output_Index], outpusdims.d[0]*outpusdims.d[1]*outpusdims.d[2] * sizeof(float), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    std::vector<bbox_t> rst;
    rst.reserve(outpusdims.d[1]);
    std::vector<uint8_t> removed(outpusdims.d[1]);

    for (int i = 0; i < outpusdims.d[1]; i++) {
        auto *box_buffer = output_buffer + i * outpusdims.d[2];

        if(box_buffer[4]<0.5) break;
        if(removed[i]) continue;

//        for(int j=0;j<outpusdims.d[2];j++){
//            std::cout<<box_buffer[j]<<" ";
//        }
//        std::cout<<std::endl;

        bbox_t temp_box;

        //xywh
        for(int j=0;j<4;j++){
            temp_box.rect[j]=box_buffer[j];
        }
        //conf
        temp_box.conf=box_buffer[4];
        //label
        temp_box.class_id=argmax(box_buffer+5,nc);
        //Point
        for(int j=0;j<kpts*2;j++){
            temp_box.pts[j]=box_buffer[5+nc+j];
        }
        //Point_conf
        for(int j=0;j<kpts;j++){
            temp_box.kpts_conf[j]=box_buffer[5+nc+kpts*2+j];
        }
        rst.emplace_back(temp_box);
        //nms
        for(int j=i+1;j<outpusdims.d[1];j++){
            auto *box_buffer2 = output_buffer + j * outpusdims.d[2];  // 20->23
            if(box_buffer2[4]<conf_thres) break;
            if(removed[j]) continue;
            float temppoint[2*kpts];
            for(int k=0;k<8;k++){
                temppoint[k]=box_buffer2[5+nc+k];
            }
            if(iou(temp_box.pts,temppoint)>iou_thres) removed[j] = true;
        }
    }

#ifdef IMAGE_SHOW
    for(auto & i : rst){
        cv::Rect2f bbox1;

        bbox1.width = i.rect[2];
        bbox1.height = i.rect[3];
        bbox1.x = i.rect[0]-i.rect[2]/2;
        bbox1.y = i.rect[1]-i.rect[3]/2;

        cv::rectangle(visual_img,bbox1,cv::Scalar(0,255,255),2);
        for(size_t j=0;j<kpts;j++){
            auto Point_conf=i.kpts_conf[j];
            if(Point_conf>kpts_conf){
                cv::circle(visual_img,cv::Point(i.pts[j],i.pts[j+kpts]),3,cv::Scalar(255,0,255),-1);
            }
        }

        cv::putText(visual_img,std::to_string(i.class_id)+":"+std::to_string(i.conf),cv::Point(bbox1.x,bbox1.y),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2,2,false);
        auto end = std::chrono::system_clock::now();
        cv::putText(visual_img,"FPS:"+std::to_string(1000/(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())),cv::Point(0,30),cv::FONT_HERSHEY_SIMPLEX,1.5,cv::Scalar(0,0,255),2,2,false);
    }

    cv::imshow("aaa", visual_img);
    cv::waitKey(1);
#endif
    return rst;
}

void TRTModule::Init(const std::string &strModelName) {
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    Logger gLogger;
    if(!exists(strTrtName))
    {
        std::cout << "[INFO]：Loading onnx model..." <<std::endl;
        std::cout << "[INFO]: build engine from onnx" << std::endl;
        // Logger gLogger;
        IRuntime* m_CudaRuntime = createInferRuntime(gLogger);
        IBuilder* builder = createInferBuilder(gLogger);
        builder->setMaxBatchSize(1);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

        parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kINFO));

        auto yolov7_output = network->getOutput(0);

        auto slice_layer = network->addSlice(*yolov7_output, Dims3{0, 0, 4}, Dims3{1, 25200, 1}, Dims3{1, 1, 1});
        auto yolov7_conf = slice_layer->getOutput(0);

        auto shuffle_layer = network->addShuffle(*yolov7_conf);
        shuffle_layer->setReshapeDimensions(Dims2{1, 25200});
        yolov7_conf = shuffle_layer->getOutput(0);

        auto topk_layer = network->addTopK(*yolov7_conf, TopKOperation::kMAX, TOPK_NUM, 1 << 1);
        auto topk_idx = topk_layer->getOutput(1);

        auto gather_layer = network->addGather(*yolov7_output, *topk_idx, 1);
        gather_layer->setNbElementWiseDims(1);

        auto yolov7_output_topk = gather_layer->getOutput(0);
        yolov7_output_topk->setName("output-topk");

        network->getInput(0)->setName("input");
        network->markOutput(*yolov7_output_topk);

        network->unmarkOutput(*yolov7_output);

        IBuilderConfig* config = builder->createBuilderConfig();
        //TODO:有的人电脑可能需要添加下面这句不然会报错
        config->setMaxWorkspaceSize(1ULL<<30);

        //启用 FP16 精度推理
        if (builder->platformHasFastFp16()) {
            std::cout << "[INFO]: platform support fp16, enable fp16" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        } else {
            std::cout << "[INFO]: platform do not support fp16, enable fp32" << std::endl;
        }


        Engine = builder->buildEngineWithConfig(*network, *config);
        Context = Engine->createExecutionContext();

        IHostMemory *gieModelStream = Engine->serialize();
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(gieModelStream->size());
        memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
        serialize_output_stream.open(strTrtName);
        serialize_output_stream<<serialize_str;
        serialize_output_stream.close();

        size_t free, total;
        cuMemGetInfo(&free, &total);
        std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB" << std::endl;
        std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;

        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
    }
    else{
        std::cout << "[INFO]: build engine from cache" << std::endl;
        std::cout << "[INFO]：Loading trt model..." <<std::endl;

        IRuntime* runtime = createInferRuntime(gLogger);

        std::string cached_path = strTrtName;
        std::ifstream fin(cached_path);
        std::string cached_engine = "";
        while (fin.peek() != EOF){
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
        }
        fin.close();
        Engine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        Context = Engine->createExecutionContext();
        runtime->destroy();
    }

}

bool TRTModule::exists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}
