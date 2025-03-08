// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ENVIRONMENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ENVIRONMENT_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"

namespace litert {
namespace internal {

// Inner singleton class that is for storing the MLD global environment.
// This class is used to store OpenCL, OpenGL environment objects.
class EnvironmentSingleton {
 public:
  EnvironmentSingleton(const EnvironmentSingleton&) = delete;
  EnvironmentSingleton& operator=(const EnvironmentSingleton&) = delete;
  ~EnvironmentSingleton() = default;
  litert::cl::ClDevice* getDevice() { return &device_; }
  litert::cl::ClContext* getContext() { return &context_; }
  litert::cl::ClCommandQueue* getCommandQueue() { return &command_queue_; }

  static EnvironmentSingleton& GetInstance() {
    if (instance_ == nullptr) {
      instance_ = new EnvironmentSingleton(nullptr);
    }
    return *instance_;
  }

  // Create the singleton instance with the given environment.
  // It will fail if the singleton instance already exists.
  static Expected<EnvironmentSingleton*> Create(
      LiteRtEnvironmentT* environment) {
    if (instance_ == nullptr) {
      instance_ = new EnvironmentSingleton(environment);
      LITERT_LOG(LITERT_INFO, "Created LiteRT EnvironmentSingleton.");
    } else {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "EnvironmentSingleton already exists");
    }
    return instance_;
  }

 private:
  // Load the OpenCL device, context and command queue from the environment if
  // available. Otherwise, create the default device, context and command queue.
  explicit EnvironmentSingleton(LiteRtEnvironmentT* environment);

  litert::cl::ClDevice device_;
  litert::cl::ClContext context_;
  litert::cl::ClCommandQueue command_queue_;
  static EnvironmentSingleton* instance_;
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_ENVIRONMENT_H_
