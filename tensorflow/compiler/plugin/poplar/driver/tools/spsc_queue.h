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
#include <memory>
#include <vector>

namespace xla {
namespace poplarplugin {

namespace {
constexpr bool is_powerof2(std::size_t v) { return v && ((v & (v - 1)) == 0); }
}  // namespace

template <typename T, std::size_t Capacity>
class SPSCQueue {
  static_assert(is_powerof2(Capacity),
                "SPSCQueue requires a power of 2 capacity");

 public:
  explicit SPSCQueue(T init, std::function<void(T&)> post_apply)
      : size_(0),
        write_position_(0),
        read_position_(0),
        post_apply_(post_apply) {
    std::fill(buffer_.begin(), buffer_.end(), init);
  }

  ~SPSCQueue() {
    std::atomic_store(&size_, std::size_t{0});
    for (auto& elem : buffer_) {
      post_apply_(elem);
    }
  }

  inline void Push(T& item) {
    // We make an assumption that we only ever push when the queue is not full
    // (i.e. IsFull() returned false).
    post_apply_(buffer_[write_position_]);
    buffer_[write_position_] = item;
    write_position_ = (write_position_ + 1) % Capacity;

    std::atomic_fetch_add(&size_, std::size_t{1});
  }

  inline void Pop(T& item) {
    while (std::atomic_load(&size_) == 0) {
    }

    item = buffer_[read_position_];
    read_position_ = (read_position_ + 1) % Capacity;

    std::atomic_fetch_sub(&size_, std::size_t{1});
  }

  inline bool IsFull() const {
    return std::atomic_load(&size_) >= Capacity - 8;
  }

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
