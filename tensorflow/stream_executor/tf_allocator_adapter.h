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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

// Adapter class that wraps a Tensorflow allocator.
//
// Assumes that the Tensorflow allocator permits asynchronous deallocation:
// see comment on `AllowsAsynchronousDeallocation()`.
class TfAllocatorAdapter : public DeviceMemoryAllocator {
 public:
  // stream: a Stream on which the allocator can only be used. If non-null, the
  // allocator can not be used on any other stream.
  TfAllocatorAdapter(tensorflow::Allocator *wrapped, Stream *stream);

  // Constructor for the cases where `stream` can not be provided.
  TfAllocatorAdapter(tensorflow::Allocator *wrapped, Platform *platform);

  ~TfAllocatorAdapter() override;

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64 size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override;

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override;

  // The Tensorflow BFC allocator used on GPU allows host-side deallocation
  // before GPU execution takes place. Tensorflow uses the ordering of the main
  // compute stream to enforce a happens-before relationship between a memory
  // allocation and code that reuses the same memory. If Tensorflow adds
  // support for multiple GPU streams or allocators with different ordering
  // requirements, this code may need to change.
  // (This attribute has no effect on CPU.)
  bool AllowsAsynchronousDeallocation() const override { return true; }

  port::StatusOr<Stream *> GetStream(int device_ordinal) override;

 private:
  tensorflow::Allocator *wrapped_;
  Stream *stream_;
};

// Adapter class that wraps per-device TF allocators with corresponding streams
// as a TfAllocatorAdapter. Assumes that the Tensorflow allocator permits
// asynchronous deallocation; see comment on `AllowsAsynchronousDeallocation()`.
class MultiDeviceAdapter : public DeviceMemoryAllocator {
 public:
  using AllocatorWithStream =
      std::pair<std::unique_ptr<tensorflow::Allocator>, Stream *>;
  MultiDeviceAdapter(const Platform *platform,
                     std::vector<AllocatorWithStream> tf_allocators)
      : DeviceMemoryAllocator(platform) {
    tf_allocators_.reserve(tf_allocators.size());
    for (AllocatorWithStream &p : tf_allocators) {
      per_device_allocators_.emplace_back(p.first.get(), p.second);
      tf_allocators_.push_back(std::move(p.first));
    }
  }

  port::StatusOr<OwningDeviceMemory> Allocate(int device_ordinal, uint64 size,
                                              bool retry_on_failure,
                                              int64_t memory_space) override {
    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal].Allocate(
        device_ordinal, size, retry_on_failure, memory_space);
  }

  port::Status Deallocate(int device_ordinal, DeviceMemoryBase mem) override {
    CHECK_LT(device_ordinal, per_device_allocators_.size());
    return per_device_allocators_[device_ordinal].Deallocate(device_ordinal,
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

  port::StatusOr<Stream *> GetStream(int device_ordinal) override {
    return per_device_allocators_[device_ordinal].GetStream(device_ordinal);
  }

 private:
  std::vector<TfAllocatorAdapter> per_device_allocators_;
  // The wrapped TF allocators backing per_device_allocators_
  // (TfAllocatorAdapter does not take ownership of its underlying Allocator).
  std::vector<std::unique_ptr<tensorflow::Allocator>> tf_allocators_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_TF_ALLOCATOR_ADAPTER_H_
