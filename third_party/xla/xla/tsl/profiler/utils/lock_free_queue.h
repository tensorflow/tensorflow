/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_TSL_PROFILER_UTILS_LOCK_FREE_QUEUE_H_
#define XLA_TSL_PROFILER_UTILS_LOCK_FREE_QUEUE_H_

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/utils/no_init.h"

namespace tsl {
namespace profiler {

namespace QueueBaseInternal {

// Internally implement the base block listed queue which will later extends to:
//    * A single-producer single-consumer queue -- LockFreeQueue
//    * A normal BlockedQueue which is not concerning the concurrency
//
// The queue base is a linked-list of blocks containing numbered slots, with
// start and end pointers:
//
//  [ slots........ | next-]--> [ slots......... | next ]
//  ^start_block_ ^start_         ^end_block_ ^end_
//
// start_ is the first occupied slot, end_ is the first unoccupied slot.
//
// Push writes at end_, and then advances it, allocating a block if needed.
// Pop takes ownership of the element at start_, if any.
// Clear removes all elements in the range [start_, end_).
//
// end_ will be atomic<size_t> when using under single-producer single-consumer
// case, or it could be size_t for a thread-compatible data structure. In the
// single-producer single-consumer scenario,
//   Push and Pop are lock free and each might be called from a single thread.
// Push is called by the producer thread. Pop is called by the consumer thread.
// Since Pop might race with Push, Pop only removes an element if Push finished
// before Pop was called. If Push is called while Pop is active, the new element
// remains in the queue.
//
// Beyond the QueueBase,
//   * LockFreeQueue's PopAll() will generate a BlockedQueue effeciently
//   * BlockedQueue support move constructor/assignment and iterators

template <typename T, size_t kBlockSize>
struct InternalBlock {
  // The number of slots in a block is chosen so the block fits in kBlockSize.
  static constexpr size_t kNumSlots =
      (kBlockSize -
       (sizeof(size_t /*start*/) + sizeof(InternalBlock* /*next*/))) /
      sizeof(NoInit<T>);

  size_t start;  // The number of the first slot.
  InternalBlock* next;
  NoInit<T> slots[kNumSlots];
};

// Wraps size_t or std::atomic<size_t> used as index to the queue.
// Index<true> wraps std::atomic<size_t>, Index<false> wraps size_t.
template <bool kIsAtomic>
struct Index;

template <>
struct Index<false> {
  size_t value;
  explicit Index(size_t pos = 0) : value(pos) {}
  size_t Get() const { return value; }
  void Set(size_t pos) { value = pos; }
};

template <>
struct Index<true> {
  std::atomic<size_t> value;
  explicit Index(size_t pos = 0) : value(pos) {}
  size_t Get() const { return value.load(std::memory_order_acquire); }
  void Set(size_t pos) { value.store(pos, std::memory_order_release); }
};

template <typename T, size_t kBlockSize, bool kAtomicEnd>
class BlockedQueueBase {
  using Block = InternalBlock<T, kBlockSize>;

 public:
  static constexpr size_t kNumSlotsPerBlockForTesting = Block::kNumSlots;

  BlockedQueueBase()
      : start_block_(new Block{/*start=*/0, /*next=*/nullptr}),
        start_(start_block_->start),
        end_block_(start_block_),
        end_(end_block_->start) {}

  // Memory should be deallocated and elements destroyed on destruction.
  // This doesn't require global lock as this discards all the stored elements
  // and we assume of destruction of this instance only after the last Push()
  // has been called.
  ~BlockedQueueBase() {
    Clear();
    DCHECK(Empty());
    delete end_block_;
  }

  // Adds a new element to the back of the queue. Fast and lock-free.
  void Push(T&& element) {
    size_t end = End();
    auto& slot = end_block_->slots[end++ - end_block_->start];
    slot.Emplace(std::move(element));
    if (TF_PREDICT_FALSE(end - end_block_->start == Block::kNumSlots)) {
      auto* new_block = new Block{/*start=*/end, /*next=*/nullptr};
      end_block_ = (end_block_->next = new_block);
    }
    SetEnd(end);  // Write index after contents.
  }

  // Removes all elements from the queue.
  void Clear() {
    size_t end = End();
    while (start_ != end) {
      PopImpl();
    }
  }

  // Removes one element off the front of the queue and returns it.
  std::optional<T> Pop() {
    std::optional<T> element;
    size_t end = End();
    if (start_ != end) {
      element = PopImpl();
    }
    return element;
  }

 protected:
  void SetEnd(size_t end) { end_.Set(end); }

  size_t End() const { return end_.Get(); }

  // Returns true if the queue is empty.
  bool Empty() const { return (start_ == End()); }

  // Removes one element off the front of the queue and returns it.
  // REQUIRES: The queue must not be empty.
  T PopImpl() {
    DCHECK(!Empty());
    // Move the next element into the output.
    auto& slot = start_block_->slots[start_++ - start_block_->start];
    T element = std::move(slot).Consume();
    // If we reach the end of a block, we own it and should delete it.
    // The next block is present: end_ always points to something.
    if (TF_PREDICT_FALSE(start_ - start_block_->start == Block::kNumSlots)) {
      auto* old_block = std::exchange(start_block_, start_block_->next);
      delete old_block;
      DCHECK_EQ(start_, start_block_->start);
    }
    return element;
  }

  Block* start_block_;     // Head: updated only by consumer thread.
  size_t start_;           // Non-atomic: read only by consumer thread.
  Block* end_block_;       // Tail: updated only by producer thread.
  Index<kAtomicEnd> end_;  // Maybe atomic: read also by consumer thread.
};

}  // namespace QueueBaseInternal

template <typename T, size_t kBlockSize>
class LockFreeQueue;

template <typename T, size_t kBlockSize = 1 << 16 /* 64 KiB */>
class BlockedQueue final
    : public QueueBaseInternal::BlockedQueueBase<T, kBlockSize, false> {
  using Block = QueueBaseInternal::InternalBlock<T, kBlockSize>;
  friend class LockFreeQueue<T, kBlockSize>;

 public:
  BlockedQueue() = default;

  BlockedQueue(BlockedQueue&& src) { *this = std::move(src); }

  BlockedQueue& operator=(BlockedQueue&& src) {
    this->Clear();
    std::swap(this->start_block_, src.start_block_);
    std::swap(this->start_, src.start_);
    std::swap(this->end_block_, src.end_block_);
    auto origin_end = this->End();
    this->SetEnd(src.End());
    src.SetEnd(origin_end);
    return *this;
  }

  class Iterator {
   public:
    bool operator==(const Iterator& another) const {
      return (index_ == another.index_) && (queue_ == another.queue_);
    }

    bool operator!=(const Iterator& another) const {
      return !(*this == another);
    }

    T& operator*() const {
      DCHECK(block_ != nullptr);
      DCHECK_GE(index_, block_->start);
      DCHECK_LT(index_, block_->start + Block::kNumSlots);
      DCHECK_LT(index_, queue_->End());
      return block_->slots[index_ - block_->start].value;
    }

    T* operator->() const { return &(this->operator*()); }

    Iterator& operator++() {
      DCHECK(queue_ != nullptr);
      DCHECK(block_ != nullptr);
      if (index_ < queue_->End()) {
        ++index_;
        auto next_block_start = block_->start + Block::kNumSlots;
        DCHECK_LE(index_, next_block_start);
        if (index_ == next_block_start) {
          block_ = block_->next;
          DCHECK_NE(block_, nullptr);
        }
      }
      return (*this);
    }

    Iterator operator++(int) {
      auto temp(*this);
      this->operator++();
      return temp;
    }

   private:
    friend class BlockedQueue;
    Iterator(BlockedQueue* queue, BlockedQueue::Block* block, size_t index)
        : queue_(queue), block_(block), index_(index) {};
    BlockedQueue* queue_ = nullptr;
    BlockedQueue::Block* block_ = nullptr;
    size_t index_ = 0;
  };

  Iterator begin() { return Iterator(this, this->start_block_, this->start_); }

  Iterator end() { return Iterator(this, this->end_block_, this->End()); }
};

template <typename T, size_t kBlockSize = 1 << 16 /* 64 KiB */>
class LockFreeQueue final
    : public QueueBaseInternal::BlockedQueueBase<T, kBlockSize, true> {
  using Block = QueueBaseInternal::InternalBlock<T, kBlockSize>;

 public:
  // Pop all events into an normal block storage queue, blocks are directly
  // moved into new queue except the last block. Those events
  // that are in the last block are in fact copied one by one.
  BlockedQueue<T, kBlockSize> PopAll() {
    BlockedQueue<T, kBlockSize> result;
    auto* empty_block = result.start_block_;
    result.start_block_ = result.end_block_ = nullptr;
    result.start_ = this->start_;
    // Use the end we see now, skip further growing if any in another thread
    size_t end = this->End();
    result.SetEnd(end);
    while (this->start_block_->start + Block::kNumSlots <= end) {
      auto* old_block =
          std::exchange(this->start_block_, this->start_block_->next);
      this->start_ = this->start_block_->start;
      old_block->next = nullptr;
      if (result.end_block_) {
        result.end_block_->next = old_block;
      } else {
        result.start_block_ = old_block;
      }
      result.end_block_ = old_block;
    }
    empty_block->start = this->start_block_->start;
    if (result.end_block_ == nullptr) {
      result.end_block_ = result.start_block_ = empty_block;
    } else {
      result.end_block_->next = empty_block;
      result.end_block_ = empty_block;
    }
    size_t bs = this->start_block_->start;
    for (size_t i = std::max(this->start_, bs); i < end; i++) {
      auto& src_slot = this->start_block_->slots[i - bs];
      auto& dst_slot = result.end_block_->slots[i - bs];
      dst_slot.Emplace(std::move(src_slot).Consume());
    }
    this->start_ = end;
    return result;
  }
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_LOCK_FREE_QUEUE_H_
