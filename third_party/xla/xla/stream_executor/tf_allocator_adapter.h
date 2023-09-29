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

#ifndef XLA_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
#define XLA_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
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

  tsl::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                             bool retry_on_failure,
                                             int64_t memory_space) override;

  tsl::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  tsl::StatusOr<Stream *> GetStream(int device_ordinal) override;

  tsl::StatusOr<tsl::Allocator *> GetAllocator(int device_ordinal);

 private:
  tsl::Allocator *wrapped_;
  Stream *stream_;
};

// Adapter class that wraps per-device TF allocators with corresponding streams
// as a TfAllocatorAdapter. Assumes that the Tensorflow allocator permits
// asynchronous deallocation; see comment on `AllowsAsynchronousDeallocation()`.
class MultiDeviceAdapter : public DeviceMemoryAllocator {
 public:
  using AllocatorWithStream =
      std::pair<std::unique_ptr<tsl::Allocator>, Stream *>;
  using AllocatorWithLogicalIdAndStream =
      std::tuple<std::unique_ptr<tsl::Allocator>, int, Stream *>;

  MultiDeviceAdapter(const Platform *platform,
                     std::vector<AllocatorWithStream> tf_allocators)
      : DeviceMemoryAllocator(platform) {
    tf_allocators_.reserve(tf_allocators.size());
    for (AllocatorWithStream &p : tf_allocators) {
      int device_ordinal = p.second->parent()->device_ordinal();
      if (per_device_allocators_.size() <= device_ordinal) {
        per_device_allocators_.resize(device_ordinal + 1);
      }
      CHECK(!per_device_allocators_[device_ordinal]);
      per_device_allocators_[device_ordinal] =
          std::make_unique<TfAllocatorAdapter>(p.first.get(), p.second);
      tf_allocators_.push_back(std::move(p.first));
    }
  }

  MultiDeviceAdapter(const Platform *platform,
                     std::vector<AllocatorWithLogicalIdAndStream> tf_allocators)
      : DeviceMemoryAllocator(platform) {
    tf_allocators_.reserve(tf_allocators.size());
    for (AllocatorWithLogicalIdAndStream &t : tf_allocators) {
      const int device_ordinal = std::get<1>(t);
      Stream *stream = std::get<2>(t);
      if (per_device_allocators_.size() <= device_ordinal) {
        per_device_allocators_.resize(device_ordinal + 1);
      }
      CHECK(!per_device_allocators_[device_ordinal]);
      per_device_allocators_[device_ordinal] =
          std::make_unique<TfAllocatorAdapter>(std::get<0>(t).get(), stream);
      tf_allocators_.push_back(std::move(std::get<0>(t)));
    }
  }

  tsl::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64_t size,
                                             bool retry_on_failure,
                                             int64_t memory_space) override {
    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal]->Allocate(
        device_ordinal, size, retry_on_failure, memory_space);
  }

  tsl::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override {
    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal]->Deallocate(device_ordinal,
                                                              mem);
  }

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  tsl::StatusOr<Stream *> GetStream(int device_ordinal) override {
    return per_device_allocators_[device_ordinal]->GetStream(device_ordinal);
  }

  tsl::StatusOr<tsl::Allocator *> GetAllocator(int device_ordinal) {
    return per_device_allocators_[device_ordinal]->GetAllocator(device_ordinal);
  }

 private:
  std::vector<std::unique_ptr<TfAllocatorAdapter>> per_device_allocators_;
  // The wrapped TF allocators backing per_device_allocators_
  // (TfAllocatorAdapter does not take ownership of its underlying Allocator).
  std::vector<std::unique_ptr<tsl::Allocator>> tf_allocators_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
