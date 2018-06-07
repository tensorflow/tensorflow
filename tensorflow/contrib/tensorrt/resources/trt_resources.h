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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_RESOURCES_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_RESOURCES_H_

#include <list>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorflow/core/framework/resource_mgr.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

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

  ~TRTCalibrationResource() {
    VLOG(0) << "Destroying Calibration Resource " << std::endl << DebugString();
    builder_->destroy();
    builder_ = nullptr;
    network_->destroy();
    network_ = nullptr;
    engine_->destroy();
    engine_ = nullptr;
    delete thr_;
    thr_ = nullptr;
    delete logger_;
    logger_ = nullptr;
    delete calibrator_;
    calibrator_ = nullptr;
  }

  string DebugString() override {
    std::stringstream oss;
    oss << " Calibrator = " << std::hex << calibrator_ << std::dec << std::endl
        << " Builder    = " << std::hex << builder_ << std::dec << std::endl
        << " Network    = " << std::hex << network_ << std::dec << std::endl
        << " Engine     = " << std::hex << engine_ << std::dec << std::endl
        << " Logger     = " << std::hex << logger_ << std::dec << std::endl
        << " Allocator  = " << std::hex << allocator_.get() << std::dec
        << std::endl
        << " Thread     = " << std::hex << thr_ << std::dec << std::endl;
    return oss.str();
  }

  TRTInt8Calibrator* calibrator_;
  nvinfer1::IBuilder* builder_;
  nvinfer1::INetworkDefinition* network_;
  nvinfer1::ICudaEngine* engine_;
  std::shared_ptr<nvinfer1::IGpuAllocator> allocator_;
  tensorflow::tensorrt::Logger* logger_;
  // TODO(sami): Use threadpool threads!
  std::thread* thr_;
};

class TRTWeightStore : public tensorflow::ResourceBase {
 public:
  TRTWeightStore() {}

  virtual ~TRTWeightStore() { VLOG(1) << "Destroying store" << DebugString(); }

  string DebugString() override {
    std::stringstream oss;
    size_t len_bytes = 0;
    for (const auto& v : store_) {
      len_bytes += v.size() * sizeof(uint8_t);
    }
    oss << " Number of entries     = " << store_.size() << std::endl
        << " Total number of bytes = "
        << store_.size() * sizeof(std::vector<uint8_t>) + len_bytes
        << std::endl;
    return oss.str();
  }

  std::list<std::vector<uint8_t>> store_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif
#endif
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_RESOURCES_H_
