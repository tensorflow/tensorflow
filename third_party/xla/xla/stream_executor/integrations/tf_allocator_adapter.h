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

#ifndef XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_
#define XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/framework/allocator.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {

// Adapter class that wraps a Tensorflow allocator.
//
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class TfAllocatorAdapter : public DeviceMemoryAllocator {
 public:
  // stream: a Stream on which the allocator can only be used. If non-null, the
  // allocator can not be used on any other stream.
  TfAllocatorAdapter(tsl::Allocator *wrapped, Stream *stream);

  // Constructor for the cases where `stream` can not be provided.
  TfAllocatorAdapter(tsl::Allocator *wrapped, Platform *platform);

  ~TfAllocatorAdapter() override;

  absl::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  absl::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  absl::StatusOr<Stream *> GetStream(int device_ordinal) override;

  absl::StatusOr<tsl::Allocator *> GetAllocator(int device_ordinal);

 private:
  tsl::Allocator *wrapped_;
  Stream *stream_;
};

// Adapter class that wraps per-device TF allocators with corresponding streams
// as a TfAllocatorAdapter. Assumes that the Tensorflow allocator permits
// asynchronous deallocation; see comment on `AllowsAsynchronousDeallocation()`.
class MultiDeviceAdapter : public DeviceMemoryAllocator {
 public:
  struct AllocatorInfo {
    std::unique_ptr<tsl::Allocator> allocator;
    Stream *stream;
    int64_t memory_space;
    std::optional<int> device_ordinal = std::nullopt;

    AllocatorInfo(std::unique_ptr<tsl::Allocator> allocator, Stream *stream,
                  int64_t memory_space,
                  std::optional<int> device_ordinal = std::nullopt)
        : allocator(std::move(allocator)),
          stream(stream),
          memory_space(memory_space),
          device_ordinal(device_ordinal) {}
  };

  MultiDeviceAdapter(const Platform *platform,
                     std::vector<AllocatorInfo> tf_allocators)
      : DeviceMemoryAllocator(platform) {
    tf_allocators_.reserve(tf_allocators.size());
    for (AllocatorInfo &info : tf_allocators) {
      auto &per_device_allocators =
          memory_space_to_per_device_allocators_[info.memory_space];
      int device_ordinal = info.device_ordinal.has_value()
                               ? *info.device_ordinal
                               : info.stream->parent()->device_ordinal();
      if (per_device_allocators.size() <= device_ordinal) {
        per_device_allocators.resize(device_ordinal + 1);
      }
      CHECK(!per_device_allocators[device_ordinal]);
      per_device_allocators[device_ordinal] =
          std::make_unique<TfAllocatorAdapter>(info.allocator.get(),
                                               info.stream);
      tf_allocators_.push_back(std::move(info.allocator));
    }
  }

  absl::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override {
    // memory_space is used here to select allocator. This isn't a need to pass
    // it any lower to TfAllocatorAdapter.
    auto it = memory_space_to_per_device_allocators_.find(memory_space);
    CHECK(it != memory_space_to_per_device_allocators_.end());
    CHECK_LT(device_ordinal, it->second.size());
    TF_ASSIGN_OR_RETURN(
        auto result, it->second[device_ordinal]->Allocate(
                         device_ordinal, size, retry_on_failure, memory_space));

    absl::MutexLock lock(&mu_);
    buffer_memory_spaces_[{device_ordinal, result->opaque()}] = memory_space;
    return result;
  }

  absl::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override {
    if (mem.opaque() == nullptr) return absl::OkStatus();
    // Memory space is not passed to deallocate, look up in
    // buffer_memory_spaces_.
    int64_t memory_space;
    {
      absl::MutexLock lock(&mu_);
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

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  absl::StatusOr<Stream *> GetStream(int device_ordinal) override {
    // Both allocators should use the same stream, so just use 0.
    return memory_space_to_per_device_allocators_[0][device_ordinal]->GetStream(
        device_ordinal);
  }

  absl::StatusOr<tsl::Allocator *> GetAllocator(int device_ordinal) {
    // GetAllocator is used for memory stats. Currently we will only see stats
    // for main device memory allocator.
    return memory_space_to_per_device_allocators_[0][device_ordinal]
        ->GetAllocator(device_ordinal);
  }

 private:
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<TfAllocatorAdapter>>>
      memory_space_to_per_device_allocators_;
  // Map of device ordinal, buffer to which memory space it resides in.
  absl::Mutex mu_;
  absl::flat_hash_map<std::pair<int, void *>, int64_t> buffer_memory_spaces_
      ABSL_GUARDED_BY(mu_);
  // The wrapped TF allocators backing per_device_allocators_
  // (TfAllocatorAdapter does not take ownership of its underlying Allocator).
  std::vector<std::unique_ptr<tsl::Allocator>> tf_allocators_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_INTEGRATIONS_TF_ALLOCATOR_ADAPTER_H_
