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
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/tensorrt/convert/utils.h"
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/contrib/tensorrt/resources/trt_allocator.h"
#include "tensorflow/contrib/tensorrt/resources/trt_int8_calibrator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "tensorrt/include/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

class SerializableResourceBase : public tensorflow::ResourceBase {
 public:
  virtual Status SerializeToString(string* serialized) = 0;
};

class TRTCalibrationResource : public SerializableResourceBase {
 public:
  ~TRTCalibrationResource() override;

  string DebugString() const override;

  Status SerializeToString(string* serialized) override;

  // Lookup table for temporary staging areas of input tensors for calibration.
  std::unordered_map<string, std::pair<void*, size_t>> device_buffers_;

  // Temporary staging areas for calibration inputs.
  std::vector<PersistentTensor> device_tensors_;

  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  TrtUniquePtrType<nvinfer1::IBuilder> builder_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<TRTBaseAllocator> allocator_;
  tensorflow::tensorrt::Logger logger_;
  // TODO(sami): Use threadpool threads!
  std::unique_ptr<std::thread> thr_;
};

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRT_RESOURCES_H_
