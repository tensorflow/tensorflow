//
// Created by skama on 1/23/18.
//

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_

#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_

#include <NvInfer.h>
#include <thread>
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/TRTInt8Calibrator.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
namespace trt {

struct TRTCalibrationResource : public tensorflow::ResourceBase {
  TRTCalibrationResource()
      : calibrator(nullptr),
        builder(nullptr),
        network(nullptr),
        engine(nullptr),
        logger(nullptr),
        thr(nullptr) {}
  string DebugString() override {
    return "";
  }
  TRTInt8Calibrator* calibrator;
  nvinfer1::IBuilder* builder;
  nvinfer1::INetworkDefinition* network;
  nvinfer1::ICudaEngine* engine;
  tensorflow::tensorrt::Logger* logger;
  std::thread* thr;
};

struct TRTEngineResource : public tensorflow::ResourceBase {
  TRTEngineResource() : runtime(nullptr), ctx(nullptr){};
  nvinfer1::IRuntime* runtime;
  nvinfer1::IExecutionContext* ctx;
};

}  // namespace trt
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCEMGR_TRTRESOURCES_H_
