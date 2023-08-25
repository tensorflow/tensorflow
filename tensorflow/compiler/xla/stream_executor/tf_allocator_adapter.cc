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

#include "tensorflow/compiler/xla/stream_executor/tf_allocator_adapter.h"

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/errors.h"

namespace stream_executor {

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator *wrapped, Stream *stream)
    : DeviceMemoryAllocator(stream->parent()->platform()),
      wrapped_(wrapped),
      stream_(stream) {}

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator *wrapped,
                                       Platform *platform)
    : DeviceMemoryAllocator(platform), wrapped_(wrapped), stream_(nullptr) {}

TfAllocatorAdapter::~TfAllocatorAdapter() {}

tsl::StatusOr<OwningDeviceMemory> TfAllocatorAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  CHECK_EQ(memory_space, 0);
  tsl::AllocationAttributes attrs;
  attrs.retry_on_failure = retry_on_failure;
  void *data = nullptr;
  if (size != 0) {
    data =
        wrapped_->AllocateRaw(tsl::Allocator::kAllocatorAlignment, size, attrs);
    if (data == nullptr) {
      return tsl::errors::ResourceExhausted(
          "Out of memory while trying to allocate ", size, " bytes.");
    }
  }
  return OwningDeviceMemory(DeviceMemoryBase(data, size), device_ordinal, this);
}

tsl::Status TfAllocatorAdapter::Deallocate(int device_ordinal,
                                           DeviceMemoryBase mem) {
  wrapped_->DeallocateRaw(mem.opaque());
  return ::tsl::OkStatus();
}

tsl::StatusOr<Stream *> TfAllocatorAdapter::GetStream(int device_ordinal) {
  CHECK_EQ(stream_->parent()->device_ordinal(), device_ordinal);
  return stream_;
}

}  // namespace stream_executor
