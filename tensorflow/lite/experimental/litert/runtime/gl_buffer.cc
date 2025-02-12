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

#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#if LITERT_HAS_OPENGL_SUPPORT

#include <GLES3/gl31.h>
#include <GLES3/gl32.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"

namespace litert {
namespace internal {

Expected<GlBuffer> GlBuffer::Alloc(size_t bytes_size) {
  tflite::gpu::gl::GlBuffer tflite_gl_buffer;

  if (!tflite::gpu::gl::CreateReadWriteShaderStorageBuffer<std::byte>(
           bytes_size, &tflite_gl_buffer)
           .ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to allocate GL buffer");
  };

  return GlBuffer(std::move(tflite_gl_buffer));
}

template Expected<float*> GlBuffer::Lock<float>();
template Expected<void> GlBuffer::Unlock<float>();

template <typename T>
Expected<T*> GlBuffer::Lock() {
  absl::MutexLock lock(&mutex_);
  if (data_ == nullptr) {
    // Ensure the data is aligned.
    if (auto rc = ::posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                   size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
    if (auto status = tflite_gl_buffer_.Read(
            absl::MakeSpan(static_cast<T*>(data_), size_ / sizeof(T)));
        !status.ok()) {
      return Unexpected(
          kLiteRtStatusErrorRuntimeFailure,
          absl::StrCat("Failed to read GL buffer: ", status.message()));
    }
  }
  return Expected<T*>(static_cast<T*>(data_));
}

template <typename T>
Expected<void> GlBuffer::Unlock() {
  absl::MutexLock lock(&mutex_);
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  if (auto status = tflite_gl_buffer_.Write(
          absl::MakeSpan(static_cast<const T*>(data_), size_ / sizeof(T)));
      !status.ok()) {
    return Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        absl::StrCat("Failed to write GL buffer: ", status.message()));
  }
  return Expected<void>();
}

}  // namespace internal
}  // namespace litert

#endif  // LITERT_HAS_OPENGL_SUPPORT
