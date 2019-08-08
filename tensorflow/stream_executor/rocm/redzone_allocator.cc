/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/redzone_allocator.h"

#include "absl/container/fixed_array.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/kernel_spec.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpToNearest(13, 8) => 16
template <typename T>
static T RoundUpToNearest(T value, T divisor) {
  return tensorflow::MathUtil::CeilOfRatio(value, divisor) * divisor;
}

using RedzoneCheckStatus = RedzoneAllocator::RedzoneCheckStatus;

RedzoneAllocator::RedzoneAllocator(Stream* stream,
                                   DeviceMemoryAllocator* memory_allocator,
                                   GpuAsmOpts gpu_compilation_opts, int64 memory_limit,
                                   int64 redzone_size, uint8 redzone_pattern)
    : device_ordinal_(stream->parent()->device_ordinal()),
      stream_(stream),
      memory_limit_(memory_limit),
      redzone_size_(RoundUpToNearest(
          redzone_size,
          static_cast<int64>(tensorflow::Allocator::kAllocatorAlignment))),
      redzone_pattern_(redzone_pattern),
      memory_allocator_(memory_allocator),
      gpu_compilation_opts_(gpu_compilation_opts) {}

port::StatusOr<DeviceMemory<uint8>> RedzoneAllocator::AllocateBytes(
    int64 byte_size) {
  return port::Status{port::error::UNIMPLEMENTED,
                      "Redzone allocator is not implemented in ROCm"};
}

port::StatusOr<RedzoneCheckStatus> RedzoneAllocator::CheckRedzones() const {
  return port::Status{port::error::UNIMPLEMENTED,
                      "Redzone allocator is not implemented in ROCm"};
}

std::string RedzoneCheckStatus::RedzoneFailureMsg() const {
  return absl::StrFormat("Redzone allocator is not implemented in ROCm");
}

}  // namespace stream_executor
