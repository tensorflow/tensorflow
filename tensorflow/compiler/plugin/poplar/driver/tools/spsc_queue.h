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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_

#include <atomic>
#include <bitset>
#include <cassert>
#include <memory>
#include <vector>

namespace xla {
namespace poplarplugin {

namespace {
constexpr bool is_powerof2(std::size_t v) { return v && ((v & (v - 1)) == 0); }
}  // namespace

/**
 * Statically bounded single-producer/single-consumer lock-free queue.
 *
 * This can be used for buffered unidirectional communication between two
 * threads. https://en.wikipedia.org/wiki/Producer–consumer_problem
 *
 * Particular attention has be paid to keeping the Pop as cheap as possible.
 * This is achieved through the `post_apply` function, which can be used for
 * resource management purely on the "Push" thread.
 *
 * \tparam T The element type to store in the queue.
 * \tparam Capacity The capacity of the queue.
 */
template <typename T, std::size_t Capacity>
class SPSCQueue {
  static_assert(is_powerof2(Capacity),
                "SPSCQueue requires a power of 2 capacity");
  static_assert(Capacity > 8, "SPSCQueue requires a capacity greater than 8");

 public:
  /**
   * Construct the SPSCQueue.
   *
   * \param init The initial value to fill the queue.
   * \param post_apply The function that is called on each element after it has
   *        been popped.
   *
   * \note post_apply must be resistant to multiple applications on the same
   *       element.
   */
  explicit SPSCQueue(T init, std::function<void(T&)> post_apply)
      : size_(0),
        write_position_(0),
        read_position_(0),
        post_apply_(post_apply) {
    assert(post_apply);
    std::fill(buffer_.begin(), buffer_.end(), init);
  }

  ~SPSCQueue() {
    std::atomic_store(&size_, std::size_t{0});
    for (auto& elem : buffer_) {
      post_apply_(elem);
    }
  }

  /**
   * Push an element into the queue.
   *
   * \param item The element to push.
   *
   * \note This function won't block, but assumes there is space
   *       (i.e. IsFull() == false).
   */
  inline void Push(const T& item) {
    assert(!IsFull());

    post_apply_(buffer_[write_position_]);
    buffer_[write_position_] = item;
    write_position_ = (write_position_ + 1) % Capacity;

    std::atomic_fetch_add(&size_, std::size_t{1});
  }

  /**
   * Similar to push, except it will block until a slot is available.
   *
   * \param item The element to push.
   */
  inline void BlockPush(const T& item) {
    while (IsFull()) {
    }

    Push(item);
  }

  /**
   * Similar to push, except it will return whether the operation was
   * successful.
   *
   * \param item The element to push.
   *
   * \return true if the element was successfully pushed, otherwise false.
   */
  inline bool TryPush(const T& item) {
    if (IsFull()) {
      return false;
    }

    Push(item);
    return true;
  }

  /**
   * Pop an element from the queue.
   *
   * \param item The element to pop into.
   *
   * \note This function won't block, but assumes there is at least a single
   * element (i.e. IsEmpty() == false).
   */
  inline void Pop(T& item) {
    assert(!IsEmpty());

    item = buffer_[read_position_];
    read_position_ = (read_position_ + 1) % Capacity;

    std::atomic_fetch_sub(&size_, std::size_t{1});
  }

  /**
   * Similar to Pop, but will block until a slot is occupied.
   *
   * \param item The element to pop into.
   */
  inline void BlockPop(T& item) {
    while (IsEmpty()) {
    }

    Pop(item);
  }

  /**
   * Similar to Pop, except it will return whether the operation was
   * successful
   *
   * \param item The element to pop into.
   *
   * \return true if the element was successfully poped, otherwise false.
   */
  inline bool TryPop(T& item) {
    if (IsEmpty()) {
      return false;
    }

    Pop(item);
    return true;
  }

  /**
   * Test whether the queue is full.
   *
   * \return True if the queue is full, otherwise false.
   */
  inline bool IsFull() const {
    return std::atomic_load(&size_) >= Capacity - 8;
  }

  /**
   * Test whether the queue is empty.
   *
   * \return True if the queue is empty, otherwise false.
   */
  inline bool IsEmpty() const { return std::atomic_load(&size_) == 0; }

 private:
  std::array<T, Capacity> buffer_;

  alignas(64) std::atomic<std::size_t> size_;
  alignas(64) std::size_t write_position_;
  alignas(64) std::size_t read_position_;

  std::function<void(T&)> post_apply_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_SPSC_QUEUE_H_
