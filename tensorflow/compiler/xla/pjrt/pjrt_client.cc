/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"

namespace xla {

PjRtBuffer::ExternalReference::~ExternalReference() = default;

StatusOr<std::uintptr_t> PjRtClient::UnsafeBufferPointer(PjRtBuffer* buffer) {
  if (buffer->on_device_shape().IsTuple()) {
    return Unimplemented(
        "unsafe_buffer_pointer is not implemented for tuple buffers.");
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      buffer->AcquireExternalReference());
  const void* ptr = external_reference_hold->OpaqueDeviceMemoryDataPointer();
  return absl::bit_cast<std::uintptr_t>(ptr);
}

MultiSliceConfig::~MultiSliceConfig() {}

std::string CompiledMemoryStats::DebugString() const {
  return absl::Substitute(
      "CompiledMemoryStats("
      "generated_code_size_in_bytes=$0, "
      "argument_size_in_bytes=$1, "
      "output_size_in_bytes=$2, "
      "alias_size_in_bytes=$3, "
      "temp_size_in_bytes=$4)",
      generated_code_size_in_bytes, argument_size_in_bytes,
      output_size_in_bytes, alias_size_in_bytes, temp_size_in_bytes);
}

Status CopyToDeviceStream::AddChunk(PjRtChunk chunk) {
  absl::MutexLock lock(&mu_);
  if (current_bytes_ >= total_bytes_) {
    return xla::Status(tensorflow::error::Code::FAILED_PRECONDITION,
                       "Stream is already complete");
  }
  current_bytes_ += chunk.size();
  if (current_bytes_ > total_bytes_) {
    return xla::Status(tensorflow::error::Code::FAILED_PRECONDITION,
                       absl::StrCat("Stream byte size mismatch: ",
                                    current_bytes_, " > ", total_bytes_));
  }

  buffered_chunks_.push_back(std::move(chunk));
  return ::tensorflow::OkStatus();
}

std::optional<PjRtChunk> CopyToDeviceStream::ConsumeNextChunk() {
  absl::MutexLock lock(&mu_);
  if (buffered_chunks_.empty() && current_bytes_ >= total_bytes_) {
    return std::nullopt;
  }
  mu_.Await(absl::Condition(
      +[](std::deque<PjRtChunk>* buffered_chunks) {
        return !buffered_chunks->empty();
      },
      &buffered_chunks_));
  PjRtChunk chunk = std::move(buffered_chunks_.front());
  buffered_chunks_.pop_front();
  return chunk;
}

}  // namespace xla
