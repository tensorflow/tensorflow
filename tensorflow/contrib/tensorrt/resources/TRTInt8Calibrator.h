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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTINT8CALIBRATOR_H_
#define TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTINT8CALIBRATOR_H_

#include <atomic>
#include <string>
#include <unordered_map>
#include <utility>
#include "tensorflow/core/platform/mutex.h"
#include "tensorrt/include/NvInfer.h"
namespace tensorflow {
namespace trt {

struct TRTInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator {
 public:
  TRTInt8Calibrator(
      const std::unordered_map<string, std::pair<void*, size_t>>& dev_buffers,
      int batch_size, string engineName);
  int getBatchSize() const;
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
  bool setBatch(const std::unordered_map<string, void*>& data);
  void setDone() { done_ = true; }
  const void* readCalibrationCache(std::size_t& length) override;
  void writeCalibrationCache(const void* ptr, std::size_t length) override;
  ~TRTInt8Calibrator();

 private:
  int batch_size_;
  tensorflow::mutex cond_mtx_;
  tensorflow::condition_variable cond_;
  bool done_;
  const std::unordered_map<string, std::pair<void*, size_t>> dev_buffers_;
  std::atomic_bool calib_running_;
  string engine_name_;
};
}  // namespace trt
}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_RESOURCES_TRTINT8CALIBRATOR_H_
