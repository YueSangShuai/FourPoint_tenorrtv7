/***************************************************************
 * @file       Logger.h
 * @brief      tensorrt的调用必须需要的头文件，这部分可以参考sampleminst
 * @author     Yue
 * @version    V1.0
 * @date       2022.9.8
 **************************************************************/

#ifndef TEST_LOGGER_H
#define TEST_LOGGER_H

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include <iostream>
#include <cassert>



using namespace nvinfer1;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)
    {
    }

    void log(Severity severity, char const* msg) noexcept
    // void log(Severity severity, const char* msg) noexcept
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};


#endif //TEST_LOGGER_H
