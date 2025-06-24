/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_HOST_MEMORY_POOL_H_
#define XLA_BACKENDS_GPU_RUNTIME_HOST_MEMORY_POOL_H_

#include <cstdint>
#include <memory>
#include <queue>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// HostMemoryPool is a pool of paged-locked CUDA host memory values. Each value
// is a scalar value of a specific PrimitiveType. The pool has a fixed size of
// kNumElems values.
//
// HostMemoryPool is used by thunks which need host memory during execution, as
// allocating/deallocating host memory each execution is expensive. As a result,
// if there are more than kNumElems simultaneous executions of an XLA program,
// an error may be raised. kNumElems is high so this is unlikely to occur in
// practice.
class HostMemoryPool {
 public:
  static constexpr int64_t kNumElems = 128;

  static absl::StatusOr<std::unique_ptr<HostMemoryPool>> Create(
      se::StreamExecutor* executor, PrimitiveType type);

  HostMemoryPool(const HostMemoryPool&) = delete;
  HostMemoryPool& operator=(const HostMemoryPool&) = delete;
  HostMemoryPool(HostMemoryPool&&) = delete;
  HostMemoryPool& operator=(HostMemoryPool&&) = delete;

  // A handle to a value in the pool. The value is released in the handle
  // destructor.
  class Handle {
   public:
    // Gets a pointer to the value. T must be the corresponding C++ type to the
    // PrimitiveType passed to to HostMemoryPool::Create.
    template <typename T>
    T* get() const {
      CHECK_EQ(primitive_util::NativeToPrimitiveType<T>(), pool_->type_);
      return static_cast<T*>(ptr_);
    }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other);
    Handle& operator=(Handle&& other);

    ~Handle();

    friend class HostMemoryPool;

   private:
    Handle(HostMemoryPool* pool, void* ptr) : pool_(pool), ptr_(ptr) {}

    HostMemoryPool* pool_;
    void* ptr_;
  };

  // Acquire a handle to a value in the pool. Returns an error if kNumElems
  // handles are already currently acquired.
  absl::StatusOr<Handle> Acquire();

 private:
  HostMemoryPool(std::unique_ptr<se::MemoryAllocation> allocation,
                 PrimitiveType type);

  void Release(void* ptr) {
    absl::MutexLock lock(&mutex_);
    free_list_.push(ptr);
  }

  const std::unique_ptr<se::MemoryAllocation> allocation_;
  const PrimitiveType type_;
  absl::Mutex mutex_;
  std::queue<void*> free_list_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_HOST_MEMORY_POOL_H_
