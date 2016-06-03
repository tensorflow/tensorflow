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

#include "tensorflow/stream_executor/scratch_allocator.h"

#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/stream.h"

namespace perftools {
namespace gputools {

ScratchAllocator::~ScratchAllocator() {}

OneTimeScratchAllocator::OneTimeScratchAllocator() {}
OneTimeScratchAllocator::~OneTimeScratchAllocator() {}

int64 OneTimeScratchAllocator::GetMemoryLimitInBytes(Stream* stream) {
  return -1;
}

port::StatusOr<DeviceMemory<uint8>> OneTimeScratchAllocator::AllocateBytes(
    Stream* stream, int64 byte_size) {
  CHECK(temporary_ == nullptr);
  SE_ASSIGN_OR_RETURN(temporary_,
                      stream->AllocateTemporaryArray<uint8>(byte_size));
  return temporary_->device_memory();
}

}  // namespace gputools
}  // namespace perftools
