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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_WORK_QUEUE_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_WORK_QUEUE_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <new>
#include <optional>

#include "absl/base/optimization.h"
#include "absl/container/fixed_array.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/logging.h"

namespace xla::cpu {

// A work queue that partitions `num_tasks` tasks into `num_partitions`
// partitions processed by parallel workers.
class WorkQueue {
  // Align all atomic counters to a cache line boundary to avoid false
  // sharing between multiple worker threads.
  static constexpr size_t kAtomicAlignment =
#if defined(__cpp_lib_hardware_interference_size)
      std::hardware_destructive_interference_size;
#else
      64;
#endif

 public:
  WorkQueue(size_t num_tasks, size_t num_partitions);

  // Returns the next task in the given partition. Returns std::nullopt
  // if the partition is complete.
  std::optional<size_t> Pop(size_t partition_index);

  size_t num_partitions() const { return partitions_.size(); }

  bool empty() const { return empty_.load(std::memory_order_relaxed); }

 private:
  friend class Worker;

  struct Partition {
    void Initialize(size_t begin, size_t end);

    // Tracks index of the next task in the assigned partition.
    alignas(kAtomicAlignment) std::atomic<size_t> index;
    size_t begin;
    size_t end;
  };

  absl::FixedArray<Partition, 32> partitions_;
  alignas(kAtomicAlignment) std::atomic<size_t> empty_;
};

// Worker processes tasks from the work queue starting from the assigned
// work partition. Once the assigned partition is complete it tries to pop
// the task from the next partition. Once the work queue is empty (the worker
// wraps around to the initial partition) it returns and empty task.
class Worker {
 public:
  Worker(size_t worker_index, WorkQueue* queue);

  std::optional<size_t> Pop();

 private:
  size_t worker_index_;
  size_t partition_index_;
  WorkQueue* queue_;
};

inline void WorkQueue::Partition::Initialize(size_t begin, size_t end) {
  index.store(begin, std::memory_order_relaxed);
  this->begin = begin;
  this->end = end;
}

inline WorkQueue::WorkQueue(size_t num_tasks, size_t num_partitions)
    : partitions_(num_partitions), empty_(num_tasks == 0) {
  size_t partition_size = tsl::MathUtil::CeilOfRatio(num_tasks, num_partitions);
  for (size_t i = 0, begin = 0, end = partition_size; i < num_partitions;
       ++i, begin = end, end = std::min(num_tasks, end + partition_size)) {
    partitions_[i].Initialize(begin, end);
  }
}

inline std::optional<size_t> WorkQueue::Pop(size_t partition_index) {
  DCHECK(partition_index < partitions_.size()) << "Invalid partition index";
  Partition& partition = partitions_.data()[partition_index];

  // Check if partition is already empty.
  if (size_t index = partition.index.load(std::memory_order_relaxed);
      index >= partition.end) {
    return std::nullopt;
  }

  // Try to acquire the next task in the partition.
  size_t index = partition.index.fetch_add(1, std::memory_order_relaxed);
  return index >= partition.end ? std::nullopt : std::make_optional(index);
}

inline Worker::Worker(size_t worker_index, WorkQueue* queue)
    : worker_index_(worker_index),
      partition_index_(worker_index),
      queue_(queue) {}

inline std::optional<size_t> Worker::Pop() {
  std::optional<size_t> task = queue_->Pop(partition_index_);
  if (task) return task;

  // If work queue is empty, we are not going to find any more tasks.
  if (queue_->empty()) return std::nullopt;

  while (!task.has_value()) {
    // Wrap around to the first partition.
    if (ABSL_PREDICT_FALSE(++partition_index_ >= queue_->num_partitions())) {
      partition_index_ = 0;
    }

    // We checked all partitions and got back to the partition we started from.
    if (ABSL_PREDICT_FALSE(partition_index_ == worker_index_)) {
      queue_->empty_.store(true, std::memory_order_relaxed);
      break;
    }

    task = queue_->Pop(partition_index_);
  }

  return task;
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_WORK_QUEUE_H_
