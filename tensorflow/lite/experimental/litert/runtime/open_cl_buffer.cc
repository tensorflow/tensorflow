// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/runtime/open_cl_buffer.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include <CL/cl.h>
#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_device.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace internal {
namespace {

// Inner singleton class that is for storing the MLD OpenCL environment.
// TODO(piyu): MLD CL environment will need to be per model configuration.
class EnvironmentSingleton {
 public:
  EnvironmentSingleton(const EnvironmentSingleton&) = delete;
  EnvironmentSingleton& operator=(const EnvironmentSingleton&) = delete;
  ~EnvironmentSingleton() = default;
  litert::cl::ClDevice* getDevice() { return &device_; }
  litert::cl::ClContext* getContext() { return &context_; }
  litert::cl::ClCommandQueue* getCommandQueue() { return &command_queue_; }

  static EnvironmentSingleton& GetInstance() {
    static EnvironmentSingleton* instance = new EnvironmentSingleton();
    return *instance;
  }

 private:
  // Load the OpenCL device, context and command queue from the environment if
  // available. Otherwise, create the default device, context and command queue.
  EnvironmentSingleton() {
    auto environment = litert::internal::Environment::Instance();
    cl_device_id device_id = nullptr;
    cl_platform_id platform_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue command_queue = nullptr;
    if (environment) {
      auto device_option =
          (*environment)->GetOption(kLiteRtEnvOptionTagOpenClDeviceId);
      if (device_option.has_value() &&
          device_option->type == kLiteRtAnyTypeInt) {
        device_id = reinterpret_cast<cl_device_id>(device_option->int_value);
      }
      auto platform_option =
          (*environment)->GetOption(kLiteRtEnvOptionTagOpenClPlatformId);
      if (platform_option.has_value() &&
          platform_option->type == kLiteRtAnyTypeInt) {
        platform_id =
            reinterpret_cast<cl_platform_id>(platform_option->int_value);
      }
      auto context_option =
          (*environment)->GetOption(kLiteRtEnvOptionTagOpenClContext);
      if (context_option.has_value() &&
          context_option->type == kLiteRtAnyTypeInt) {
        context = reinterpret_cast<cl_context>(platform_option->int_value);
      }
      auto command_queue_option =
          (*environment)->GetOption(kLiteRtEnvOptionTagOpenClCommandQueue);
      if (command_queue_option.has_value() &&
          command_queue_option->type == kLiteRtAnyTypeInt) {
        command_queue =
            reinterpret_cast<cl_command_queue>(platform_option->int_value);
      }
    }
    if (device_id && platform_id) {
      device_ = litert::cl::ClDevice(device_id, platform_id);
    } else {
      auto status = litert::cl::CreateDefaultGPUDevice(&device_);
      if (!status.ok()) {
        LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL device");
      }
    }
    if (context) {
      context_ = litert::cl::ClContext(context, /*has_ownership=*/false);
    } else {
      auto status = litert::cl::CreateClContext(device_, &context_);
      if (!status.ok()) {
        LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL contxt");
      }
    }
    if (command_queue) {
      command_queue_ =
          litert::cl::ClCommandQueue(command_queue, /*has_ownership=*/false);
    } else {
      auto status =
          litert::cl::CreateClCommandQueue(device_, context_, &command_queue_);
      if (!status.ok()) {
        LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL command queue");
      }
    }
  }
  litert::cl::ClDevice device_;
  litert::cl::ClContext context_;
  litert::cl::ClCommandQueue command_queue_;
};
}  // namespace

template Expected<float*> OpenClBuffer::Lock<float>();
template Expected<void> OpenClBuffer::Unlock<float>();

template <typename T>
Expected<T*> OpenClBuffer::Lock() {
  absl::MutexLock lock(&mutex_);
  // The buffer has not been locked, so we need to read from the OpenCL
  // buffer.
  if (data_ == nullptr) {
    litert::cl::ClCommandQueue* queue =
        EnvironmentSingleton::GetInstance().getCommandQueue();
    std::vector<T> result;
    auto status = buffer_.ReadData(queue, &result);
    if (!status.ok()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to read OpenCL buffer");
    }
    // Ensure the data is aligned.
    if (auto rc = ::posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                   size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
    // Copy the data from the OpenCL buffer to the aligned memory.
    // TODO(piyu): Consider adding support in MLD OpenCL buffer to directly
    // write to the aligned memory.
    std::copy(result.begin(), result.end(), static_cast<T*>(data_));
  }
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> OpenClBuffer::Unlock() {
  absl::MutexLock lock(&mutex_);
  litert::cl::ClCommandQueue* queue =
      EnvironmentSingleton::GetInstance().getCommandQueue();
  // The buffer has not been locked, so we don't need to write back.
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  size_t write_size = (size_ + sizeof(T) - 1) / sizeof(T);
  auto status = buffer_.WriteData(
      queue, absl::MakeSpan(static_cast<T*>(data_), write_size));

  if (status.ok()) {
    return Expected<void>();
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "The data failed to write to the OpenCL buffer when unlocked");
}

bool OpenClBuffer::IsSupported() {
  static bool is_supported = ::litert::cl::LoadOpenCL().ok();
  return is_supported;
}

Expected<OpenClBuffer> OpenClBuffer::Alloc(size_t bytes_size) {
  litert::cl::Buffer buffer;

  litert::cl::ClContext* cl_context =
      EnvironmentSingleton::GetInstance().getContext();
  auto result =
      litert::cl::CreateReadWriteBuffer(bytes_size, cl_context, &buffer);
  if (!result.ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create OpenCL buffer");
  }

  return Expected<OpenClBuffer>(std::move(buffer), bytes_size);
}
}  // namespace internal
}  // namespace litert
