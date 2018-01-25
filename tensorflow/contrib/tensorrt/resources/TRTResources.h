//
// Created by skama on 1/23/18.
//

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_

#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_

#include <thread>
#include <NvInfer.h>
#include "tensorflow/contrib/tensorrt/resourcemgr/TRTInt8Calibrator.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
namespace trt {

struct TRTCalibrationResource : public tensorflow::ResourceBase {
  TRTCalibrationResource():calibrator(nullptr), builder(nullptr), thr(nullptr){};
  TRTInt8Calibrator* calibrator;
  nvinfer1::IBuilder* builder;
  std::thread *thr;
};

struct TRTEngineResource:public tensorflow::ResourceBase{
  TRTEngineResource():runtime(nullptr), ctx(nullptr){};
  nvinfer1::IRuntime *runtime;
  nvinfer1::IExecutionContext *ctx;
};

}
}
#endif // TENSORFLOW_CONTRIB_TENSORRT_RESOURCEMGR_TRTRESOURCES_H_
