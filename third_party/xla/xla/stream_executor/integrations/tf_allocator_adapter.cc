/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/integrations/tf_allocator_adapter.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"

namespace stream_executor {

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator *wrapped, Stream *stream)
    : DeviceMemoryAllocator(stream->parent()->platform()),
      wrapped_(wrapped),
      stream_(stream) {}

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator *wrapped,
                                       Platform *platform)
    : DeviceMemoryAllocator(platform), wrapped_(wrapped), stream_(nullptr) {}

TfAllocatorAdapter::~TfAllocatorAdapter() {}

absl::StatusOr<OwningDeviceMemory> TfAllocatorAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  tsl::AllocationAttributes attrs;
  attrs.retry_on_failure = retry_on_failure;
  void *data = nullptr;
  if (size != 0) {
    data =
        wrapped_->AllocateRaw(tsl::Allocator::kAllocatorAlignment, size, attrs);
    if (data == nullptr) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "Out of memory while trying to allocate ", size, " bytes."));
    }
  }
  return OwningDeviceMemory(DeviceMemoryBase(data, size), device_ordinal, this);
}

absl::Status TfAllocatorAdapter::Deallocate(int device_ordinal,
                                            DeviceMemoryBase mem) {
  wrapped_->DeallocateRaw(mem.opaque());
  return absl::OkStatus();
}

absl::StatusOr<Stream *> TfAllocatorAdapter::GetStream(int device_ordinal) {
  CHECK_EQ(stream_->parent()->device_ordinal(), device_ordinal);
  return stream_;
}

absl::StatusOr<tsl::Allocator *> TfAllocatorAdapter::GetAllocator(
    int device_ordinal) {
  if (stream_ == nullptr) {
    return absl::UnavailableError("stream_ is null for TfAllocatorAdapter.");
  }
  if (stream_->parent()->device_ordinal() != device_ordinal) {
    return absl::InternalError(
        absl::StrCat("stream_->parent()->device_ordinal() ",
                     stream_->parent()->device_ordinal(),
                     " not equal to device_ordinal ", device_ordinal));
  }
  return wrapped_;
}

}  // namespace stream_executor
