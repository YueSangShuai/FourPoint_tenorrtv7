#include <iostream>
#include"opencv2/opencv.hpp"
#include "AutoShoot/TRT/TRTModule.h"

int main(){

    cv::Mat temp=cv::imread("../test/test2.jpg");
    TRTModule model("../AutoShoot/model/merge.onnx");
//    auto temp1=model(temp);

    cv::Mat frame;
    cv::VideoCapture capture = cv::VideoCapture("../test/1.mp4");
    auto start = std::chrono::system_clock::now();
    int count=0;
    while(capture.read(frame)){
        model(frame);
        count++;
    }
    auto end = std::chrono::system_clock::now();
    std::cout<<"[INFO]: Average_time "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/count<<std::endl;
}
