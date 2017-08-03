/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

namespace test {
class Benchmark;
}
struct SessionOptions;

// This class is shared by ThreadPoolDevice and GPUDevice and
// initializes a shared Eigen compute device used by both.  This
// should eventually be removed once we refactor ThreadPoolDevice and
// GPUDevice into more 'process-wide' abstractions.
class LocalDevice : public Device {
 public:
  LocalDevice(const SessionOptions& options,
              const DeviceAttributes& attributes);
  ~LocalDevice() override;

 private:
  static bool use_global_threadpool_;

  static void set_use_global_threadpool(bool use_global_threadpool) {
    use_global_threadpool_ = use_global_threadpool;
  }

  struct EigenThreadPoolInfo;
  std::unique_ptr<EigenThreadPoolInfo> owned_tp_info_;

  friend class test::Benchmark;

  TF_DISALLOW_COPY_AND_ASSIGN(LocalDevice);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_
