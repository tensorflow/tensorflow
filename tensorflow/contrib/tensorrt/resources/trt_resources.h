/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTRESOURCES_H_

#include <list>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/core/framework/resource_mgr.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
class TRTCalibrationResource : public tensorflow::ResourceBase {
 public:
  TRTCalibrationResource()
      : calibrator_(nullptr),
        builder_(nullptr),
        network_(nullptr),
        engine_(nullptr),
        logger_(nullptr),
        thr_(nullptr) {}
  string DebugString() override {
    std::stringstream oss;
    oss << " Calibrator = " << std::hex << calibrator_ << std::dec << std::endl
        << " Builder    = " << std::hex << builder_ << std::dec << std::endl
        << " Network    = " << std::hex << network_ << std::dec << std::endl
        << " Engine     = " << std::hex << engine_ << std::dec << std::endl
        << " Logger     = " << std::hex << logger_ << std::dec << std::endl
        << " Thread     = " << std::hex << thr_ << std::dec << std::endl;
    return oss.str();
  }
  ~TRTCalibrationResource() {
    VLOG(0) << "Destroying Calibration Resource " << std::endl << DebugString();
  }
  TRTInt8Calibrator* calibrator_;
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;
  nvinfer1::ICudaEngine* engine_;
  tensorflow::tensorrt::Logger* logger_;
  // TODO(sami): Use threadpool threads!
  std::thread* thr_;
};

class TRTWeightStore : public tensorflow::ResourceBase {
 public:
  TRTWeightStore() {}
  std::list<std::vector<uint8_t>> store_;
  string DebugString() override {
    std::stringstream oss;
    size_t lenBytes = 0;
    for (const auto& v : store_) {
      lenBytes += v.size() * sizeof(uint8_t);
    }
    oss << " Number of entries     = " << store_.size() << std::endl
        << " Total number of bytes = "
        << store_.size() * sizeof(std::vector<uint8_t>) + lenBytes << std::endl;
    return oss.str();
  }
  virtual ~TRTWeightStore() { VLOG(1) << "Destroying store" << DebugString(); }
};

class TRTEngineResource : public tensorflow::ResourceBase {
 public:
  TRTEngineResource() : runtime_(nullptr), ctx_(nullptr){};
  string DebugString() override { return string(""); }
  nvinfer1::IRuntime* runtime_;
  nvinfer1::IExecutionContext* ctx_;
};

}  // namespace tensorrt
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCEMGR_TRTRESOURCES_H_
#endif
#endif
