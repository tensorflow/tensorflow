/* Copyright 2022 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_CONCURRENCY_CONCURRENT_VECTOR_H_
#define TENSORFLOW_TSL_CONCURRENCY_CONCURRENT_VECTOR_H_

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace tsl {
namespace internal {

// A simple concurrent sequential container that allows concurrent reads and
// writes and is optimized for read access. It is designed for the usage pattern
// where objects are inserted once but are read many times. The key difference
// between this data structure and std::vector is that when we re-allocate the
// underlying buffer, we do not free the previous buffer. This allows us to
// implement read access with a single atomic load.
//
// Sample usage:
//
// ConcurrentVector<T> vec;
//
// On the writer side, concurrent writers are allowed;
//
// size_t index1 = vec.emplace_back(args);
// size_t index2 = vec.emplace_back(args);
//
// On the reader side, concurrent readers are allowed.
//
// auto& t1 = vec[index1];
// auto& t2 = vec[index1];
//
// Requirements:
//
// Type T needs to be copyable.

template <typename T>
class ConcurrentVector {
 public:
  // Initialize the vector with the given initial_capapcity
  explicit ConcurrentVector(size_t initial_capacity) : state_(0ull) {
    // ConcurrentVector does not support inserting more than 2^64 elements,
    // which should be more than enough for any reasonable use case.
    all_allocated_elements_.reserve(65);
    all_allocated_elements_.emplace_back();
    auto& v = all_allocated_elements_.back();
    v.reserve(std::max(static_cast<size_t>(1), initial_capacity));
  }

  T& operator[](size_t index) {
    auto state = State::Decode(state_.load(std::memory_order_acquire));
    assert(index < state.size);
    // .data() is a workaround for libc++ assertions in operator[], which will
    // cause data race when container is resized from another thread.
    return all_allocated_elements_.data()[state.last_allocated].data()[index];
  }

  const T& operator[](size_t index) const {
    auto state = State::Decode(state_.load(std::memory_order_acquire));
    assert(index < state.size);
    // .data() is a workaround for libc++ assertions in operator[], which will
    // cause data race when container is resized from another thread.
    return all_allocated_elements_.data()[state.last_allocated].data()[index];
  }

  absl::Span<const T> ToConstSpan() const {
    auto state = State::Decode(state_.load(std::memory_order_acquire));
    auto& storage = all_allocated_elements_[state.last_allocated];
    // .data() is a workaround for libc++ assertions in operator[], which will
    // cause data race when container is resized from another thread.
    return absl::MakeConstSpan(storage.data(), state.size);
  }

  // Return the number of elements currently valid in this vector.  The vector
  // only grows, so this is conservative w.r.t. the execution of the current
  // thread.
  size_t size() const {
    return State::Decode(state_.load(std::memory_order_relaxed)).size;
  }

  // Insert a new element at the end. If the current buffer is full, we allocate
  // a new buffer with twice as much capacity and copy the items in the
  // previous buffer over.
  //
  // Returns the index of the newly inserted item.
  template <typename... Args>
  size_t emplace_back(Args&&... args) {
    absl::MutexLock lock(&mutex_);

    auto& last = all_allocated_elements_.back();

    if (last.size() < last.capacity()) {
      // There is still room in the current vector without reallocation. Just
      // add the new element there.
      last.emplace_back(std::forward<Args>(args)...);

      // Increment the size of the concurrent vector.
      auto state = State::Decode(state_.load(std::memory_order_relaxed));
      state.size += 1;
      state_.store(state.Encode(), std::memory_order_release);

      return state.size - 1;  // return insertion index
    }
    // There is no more room in the current vector without reallocation.
    // Allocate a new vector with twice as much capacity, copy the elements
    // from the previous vector, and set elements_ to point to the data of the
    // new vector.
    auto& new_last = all_allocated_elements_.emplace_back();
    auto& prev = *(all_allocated_elements_.rbegin() + 1);
    new_last.reserve(prev.capacity() * 2);
    assert(prev.size() == prev.capacity());

    // Copy over the previous vector to the new vector.
    new_last.insert(new_last.begin(), prev.begin(), prev.end());
    new_last.emplace_back(std::forward<Args>(args)...);

    // Increment the size of the concurrent vector and index of the last
    // allocated vector.
    auto state = State::Decode(state_.load(std::memory_order_relaxed));
    state.last_allocated += 1;
    state.size += 1;
    state_.store(state.Encode(), std::memory_order_release);

    return state.size - 1;  // return insertion index
  }

 private:
  // Concurrent vector state layout:
  // - Low 32 bits encode the index of the last allocated vector.
  // - High 32 bits encode the size of the concurrent vector.
  static constexpr uint64_t kLastAllocatedMask = (1ull << 32) - 1;
  static constexpr uint64_t kSizeMask = ((1ull << 32) - 1) << 32;

  struct State {
    uint64_t last_allocated;  // index of last allocated vector
    uint64_t size;            // size of the concurrent vector

    static State Decode(uint64_t state) {
      uint64_t last_allocated = (state & kLastAllocatedMask);
      uint64_t size = (state & kSizeMask) >> 32;
      return {last_allocated, size};
    }

    uint64_t Encode() const { return (size << 32) | last_allocated; }
  };

  // Stores/loads to/from this atomic used to enforce happens-before
  // relationship between emplace_back and operator[].
  std::atomic<uint64_t> state_;

  absl::Mutex mutex_;
  std::vector<std::vector<T>> all_allocated_elements_;
};

}  // namespace internal
}  // namespace tsl

#endif  // TENSORFLOW_TSL_CONCURRENCY_CONCURRENT_VECTOR_H_
