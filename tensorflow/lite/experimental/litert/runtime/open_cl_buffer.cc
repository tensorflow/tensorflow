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

#include <stdlib.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/runtime/gpu_environment.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_command_queue.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/cl_context.h"
#include "tensorflow/lite/experimental/litert/runtime/opencl/opencl_wrapper.h"

namespace litert {
namespace internal {

template Expected<float*> OpenClBuffer::Lock<float>();
template Expected<char*> OpenClBuffer::Lock<char>();
template Expected<void> OpenClBuffer::Unlock<float>();
template Expected<void> OpenClBuffer::Unlock<char>();

template <typename T>
Expected<T*> OpenClBuffer::Lock() {
  absl::MutexLock lock(&mutex_);
  // The buffer has not been locked, so we need to read from the OpenCL
  // buffer.
  if (data_ == nullptr) {
    litert::cl::ClCommandQueue* queue =
        GpuEnvironmentSingleton::GetInstance().getCommandQueue();
    std::vector<T> result;
    auto status = buffer_.ReadData(queue, &result);
    if (!status.ok()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to read OpenCL buffer");
    }
    // Ensure the data is aligned.
    if (auto rc =
            posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT, size_);
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
      GpuEnvironmentSingleton::GetInstance().getCommandQueue();
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
  LITERT_RETURN_IF_ERROR(
      IsSupported(),
      Unexpected(kLiteRtStatusErrorRuntimeFailure, "OpenCL is not supported"));

  litert::cl::Buffer buffer;

  litert::cl::ClContext* cl_context =
      GpuEnvironmentSingleton::GetInstance().getContext();
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
