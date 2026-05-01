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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/layout.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator* wrapped, Stream* stream,
                                       size_t min_alignment)
    : DeviceAddressAllocator(CHECK_NOTNULL(stream)->parent()->GetPlatform()),
      wrapped_(wrapped),
      stream_(stream),
      min_alignment_(min_alignment) {}

TfAllocatorAdapter::TfAllocatorAdapter(tsl::Allocator* wrapped,
                                       const Platform* platform,
                                       size_t min_alignment)
    : DeviceAddressAllocator(platform),
      wrapped_(wrapped),
      stream_(nullptr),
      min_alignment_(min_alignment) {}

TfAllocatorAdapter::~TfAllocatorAdapter() {}

absl::StatusOr<ScopedDeviceAddress<uint8_t>> TfAllocatorAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  tsl::AllocationAttributes attrs;
  attrs.retry_on_failure = retry_on_failure;
  void* data = nullptr;
  if (size != 0) {
    data = wrapped_->AllocateRaw(min_alignment_, size, attrs);
    if (data == nullptr) {
      return MemoryAllocationError(
          size, memory_space == xla::Layout::kHostMemorySpace);
    }
  }
  return ScopedDeviceAddress<uint8_t>(DeviceAddressBase(data, size),
                                      device_ordinal, this);
}

absl::Status TfAllocatorAdapter::Deallocate(int device_ordinal,
                                            DeviceAddressBase mem) {
  wrapped_->DeallocateRaw(mem.opaque());
  return absl::OkStatus();
}

absl::StatusOr<Stream*> TfAllocatorAdapter::GetStream(int device_ordinal) {
  CHECK(stream_ != nullptr) << "GetStream requires a non-null stream";
  CHECK_EQ(stream_->parent()->device_ordinal(), device_ordinal);
  return stream_;
}

absl::StatusOr<tsl::Allocator*> TfAllocatorAdapter::GetAllocator(
    int device_ordinal) {
  if (stream_ && stream_->parent()->device_ordinal() != device_ordinal) {
    return absl::InternalError(
        absl::StrCat("stream_->parent()->device_ordinal() ",
                     stream_->parent()->device_ordinal(),
                     " not equal to device_ordinal ", device_ordinal));
  }
  return wrapped_;
}

//===----------------------------------------------------------------------===//
// MultiDeviceAdapter
//===----------------------------------------------------------------------===//

static int GetDeviceOrdinal(const MultiDeviceAdapter::AllocatorInfo& info) {
  return info.device_ordinal.has_value()
             ? *info.device_ordinal
             : CHECK_NOTNULL(info.stream)->parent()->device_ordinal();
}

MultiDeviceAdapter::MultiDeviceAdapter(const Platform* platform,
                                       std::vector<AllocatorInfo> allocators)
    : DeviceAddressAllocator(platform) {
  // Sort allocators by device ordinal and memory space to get user-friendly
  // logging below. It doesn't change the runtime behavior.
  absl::c_sort(allocators, [](const AllocatorInfo& a, const AllocatorInfo& b) {
    return std::make_pair(a.memory_space, GetDeviceOrdinal(a)) <
           std::make_pair(b.memory_space, GetDeviceOrdinal(b));
  });

  for (AllocatorInfo& info : allocators) {
    std::vector<std::shared_ptr<TfAllocatorAdapter>>& per_device_allocators =
        memory_space_to_per_device_allocators_[info.memory_space];
    int device_ordinal = GetDeviceOrdinal(info);
    if (per_device_allocators.size() <= device_ordinal) {
      per_device_allocators.resize(device_ordinal + 1);
    }
    CHECK(!per_device_allocators[device_ordinal]);
    if (info.stream != nullptr) {
      per_device_allocators[device_ordinal] =
          std::make_shared<TfAllocatorAdapter>(info.allocator.get(),
                                               info.stream, info.min_alignment);
    } else {
      per_device_allocators[device_ordinal] =
          std::make_shared<TfAllocatorAdapter>(
              info.allocator.get(), info.platform, info.min_alignment);
    }
    VLOG(3) << absl::StrFormat(
        "MultiDeviceAdapter: device_ordinal=%d memory_space=%d "
        "min_alignment=%d",
        device_ordinal, info.memory_space, info.min_alignment);
    allocators_.push_back(std::move(info.allocator));
  }
}

absl::StatusOr<ScopedDeviceAddress<uint8_t>> MultiDeviceAdapter::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  auto it = memory_space_to_per_device_allocators_.find(memory_space);
  CHECK(it != memory_space_to_per_device_allocators_.end());
  CHECK_LT(device_ordinal, it->second.size());
  ASSIGN_OR_RETURN(auto result,
                   it->second[device_ordinal]->Allocate(
                       device_ordinal, size, retry_on_failure, memory_space));

  absl::MutexLock lock(mu_);
  buffer_memory_spaces_[{device_ordinal, result->opaque()}] = memory_space;
  return result;
}

absl::Status MultiDeviceAdapter::Deallocate(int device_ordinal,
                                            DeviceAddressBase mem) {
  if (mem.opaque() == nullptr) {
    return absl::OkStatus();
  }
  int64_t memory_space;
  {
    absl::MutexLock lock(mu_);
    auto it = buffer_memory_spaces_.find({device_ordinal, mem.opaque()});
    if (it == buffer_memory_spaces_.end()) {
      // There might be situation when device memory was allocated somewhere
      // outside of the current allocator. For backward compatibility in
      // this case we are falling back to the first allocator to deallocate
      // the memory.
      // See b/325527293 for more details.
      return memory_space_to_per_device_allocators_[0][device_ordinal]
          ->Deallocate(device_ordinal, mem);
    }
    memory_space = it->second;
    buffer_memory_spaces_.erase(it);
  }

  auto it = memory_space_to_per_device_allocators_.find(memory_space);
  CHECK(it != memory_space_to_per_device_allocators_.end());
  CHECK_LT(device_ordinal, it->second.size());
  return it->second[device_ordinal]->Deallocate(device_ordinal, mem);
}

absl::StatusOr<Stream*> MultiDeviceAdapter::GetStream(int device_ordinal) {
  return memory_space_to_per_device_allocators_[0][device_ordinal]->GetStream(
      device_ordinal);
}

absl::StatusOr<tsl::Allocator*> MultiDeviceAdapter::GetAllocator(
    int device_ordinal) {
  return memory_space_to_per_device_allocators_[0][device_ordinal]
      ->GetAllocator(device_ordinal);
}

//===----------------------------------------------------------------------===//
// Error helpers
//===----------------------------------------------------------------------===//

static constexpr absl::string_view kMemoryAllocationErrorPayloadKey =
    "tf-allocator-allocation-error";

absl::Status MemoryAllocationError(uint64_t size, bool is_host_mem) {
  constexpr absl::string_view kHostMemoryExplanation =
      " Please set the environment variable "
      "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB to allocate larger "
      "host memory than the default 64 GB.";

  absl::Status status = absl::ResourceExhaustedError(
      absl::StrCat("Out of ", (is_host_mem ? "host " : ""),
                   "memory while trying to allocate ",
                   tsl::strings::HumanReadableNumBytes(size), ".",
                   (is_host_mem ? kHostMemoryExplanation : "")));
  status.SetPayload(kMemoryAllocationErrorPayloadKey, absl::Cord());
  return status;
}

bool IsMemoryAllocationError(absl::Status status) {
  return status.GetPayload(kMemoryAllocationErrorPayloadKey).has_value();
}

}  // namespace stream_executor
