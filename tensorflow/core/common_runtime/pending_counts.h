#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>

#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// PendingCounts is an internal helper class to keep track of pending and
// dead counts for nodes, for use in the ExecutorState module.  It
// holds a map from Handles to various counts for that handle.  This
// information is needed per frame iteration. The amount of memory
// needed for an iteration is the same across all executions of the
// iteration. The memory amount and handles are precomputed at startup
// using a Layout object.
//
//    PendingCounts::Layout layout;
//    std::vector<PendingCounts::Handle> h(C);
//    for (int id = 0; id < C; id++) {
//      h[id] = r.AddHandle(max_pending[id], max_dead[id]);
//    }
//
// When we actually want to start an iteration we first create a
// PendingCounts object and then index into it using the precomputed
// handles:

//    PendingCounts counts(layout);
//    ...
//    counts.decrement_pending(h[id], 1);
class PendingCounts {
 public:
  // The state machine for a node's execution.
  enum NodeState {
    // The pending count for the node > 0.
    PENDING_NOTREADY,
    // The pending count for the node == 0, but the node has not
    // started executing.
    PENDING_READY,
    // The node has started executing.
    STARTED,
    // The node has finished executing.
    COMPLETED
  };

  // An opaque handle indicating where in the PendingCounts data structure
  // the appropriate count information can be found.
  class Handle;
  // Given a node that needs to represent counts no larger than the
  // specified "max_pending_count" and "max_dead_count", create a
  // handle that can be passed to various PendingCounts routines
  // to retrieve the count data for this node.
  class Layout {
   public:
    Handle CreateHandle(size_t max_pending_count, size_t max_dead_count);

   private:
    friend class PendingCounts;
    int next_offset_ = 0;  // Next byte offset to allocate
  };

  // Create a new PendingCounts object that can hold the state of
  // all the Handles allocated from "final_allocator".
  explicit PendingCounts(Layout layout)
      : num_bytes_(layout.next_offset_), bytes_(new char[num_bytes_]) {
    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
  }

  // Create a new PendingCounts object with the same layout and counts
  // as "other".
  explicit PendingCounts(const PendingCounts& other)
      : num_bytes_(other.num_bytes_), bytes_(new char[num_bytes_]) {
    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
    memcpy(bytes_, other.bytes_, other.num_bytes_);
  }

  ~PendingCounts() { delete[] bytes_; }

  void set_initial_count(Handle h, size_t pending_count) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      DCHECK_LE(pending_count, kMaxCountForPackedCounts);
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }

  NodeState node_state(Handle h) {
    if (h.is_large_) {
      return NodeStateForStruct(Large(h)->load(std::memory_order_relaxed));
    } else {
      return NodeStateForStruct(Packed(h)->load(std::memory_order_relaxed));
    }
  }
  void mark_started(Handle h) {
    DCHECK_EQ(pending(h), 0);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  void mark_completed(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  int pending(Handle h) {
    if (h.is_large_) {
      LargeCounts c = Large(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    } else {
      PackedCounts c = Packed(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    }
  }
  struct AdjustResult {
    int dead_count;
    int pending_count;

    AdjustResult(int dead_count, int pending_count)
        : dead_count(dead_count), pending_count(pending_count) {}
  };
  int decrement_pending(Handle h, int v) {
    DCHECK_GE(pending(h), v);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    }
  }

  // Mark a merge node as live
  // REQUIRES: Node corresponding to "h" is a merge node
  void mark_live(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.pending &= ~static_cast<int>(0x1);
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        c.pending &= 0x6;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }

  int dead_count(Handle h) {
    int r = h.is_large_ ? Large(h)->load(std::memory_order_relaxed).dead_count
                        : Packed(h)->load(std::memory_order_relaxed).dead_count;
    return r;
  }
  void increment_dead_count(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        DCHECK_LT(c.dead_count, kMaxCountForPackedCounts);
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }

  // Mark a merge node as live. Please note that the pending count it returns
  // is before the update.
  AdjustResult adjust_for_mark_live(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      auto ret_pending = 0;
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        ret_pending = c.pending;
        c.pending &= ~static_cast<int>(0x1);
        c_ptr->store(c, std::memory_order_relaxed);
      }
      return AdjustResult(c.dead_count, ret_pending);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      auto ret_pending = 0;
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        ret_pending = c.pending;
        c.pending &= 0x6;
        c_ptr->store(c, std::memory_order_relaxed);
      }
      return AdjustResult(c.dead_count, ret_pending);
    }
  }

  // The same as the above, but performs the operation atomically. This
  // is thread-safe to run concurrently with other threads.
  AdjustResult adjust_for_mark_live_atomic(Handle h) {
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto old_val = c_ptr->load(std::memory_order_relaxed);
      while (true) {
        auto new_val = old_val;
        auto ret_pending = 0;
        // Only do anything if the node hasn't already started executing.
        if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
          ret_pending = old_val.pending;
          new_val.pending &= ~static_cast<int>(0x1);
        }
        AdjustResult ret(old_val.dead_count, ret_pending);
        if (TF_PREDICT_TRUE(c_ptr->compare_exchange_weak(old_val, new_val)))
          return ret;
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto old_val = c_ptr->load(std::memory_order_relaxed);
      while (true) {
        auto new_val = old_val;
        auto ret_pending = 0;
        // Only do anything if the node hasn't already started executing.
        if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
          static_assert(7 == kMaxCountForPackedCounts,
                        "Live flag incorrect for max packed count");
          ret_pending = old_val.pending;
          new_val.pending &= 0x6;
        }
        AdjustResult ret(old_val.dead_count, ret_pending);
        if (TF_PREDICT_TRUE(c_ptr->compare_exchange_weak(old_val, new_val)))
          return ret;
      }
    }
  }

  // A streamlined routine that does several pieces of bookkeeping at
  // once.  Equivalent to:
  //    increment_dead_count(h);
  //    return {dead_count(h) pending(h)};
  AdjustResult adjust_for_increment_dead(Handle h) {
    if (h.is_large_) {
      return adjust_for_increment_dead_shared(Large(h));
    } else {
      return adjust_for_increment_dead_shared(Packed(h));
    }
  }

  // The same as the above, but performs the operation atomically. This
  // is thread-safe to run concurrently with other threads.
  AdjustResult adjust_for_increment_dead_atomic(Handle h) {
    if (h.is_large_) {
      return adjust_for_increment_dead_shared_atomic(Large(h));
    } else {
      return adjust_for_increment_dead_shared_atomic(Packed(h));
    }
  }

  // A streamlined routine that does several pieces of bookkeeping at
  // once.  Equivalent to:
  //    decrement_pending(h, decrement_pending);
  //    return {dead_count(h) pending(h)};
  AdjustResult adjust_for_decrement_pending(Handle h, int decrement_pending) {
    DCHECK_GE(pending(h), decrement_pending);
    if (h.is_large_) {
      return adjust_for_decrement_pending_shared(Large(h), decrement_pending);
    } else {
      return adjust_for_decrement_pending_shared(Packed(h), decrement_pending);
    }
  }

  // The same as the above, but performs the operation atomically. This
  // is thread-safe to run concurrently with other threads.
  AdjustResult adjust_for_decrement_pending_atomic(Handle h,
                                                   int decrement_pending) {
    DCHECK_GE(pending(h), decrement_pending);
    if (h.is_large_) {
      return adjust_for_decrement_pending_shared_atomic(Large(h),
                                                        decrement_pending);
    } else {
      return adjust_for_decrement_pending_shared_atomic(Packed(h),
                                                        decrement_pending);
    }
  }

  // A streamlined routine that does several pieces of bookkeeping at
  // once.  Equivalent to:
  //    if (increment_dead) increment_dead_count(h);
  //    decrement_pending(h, 1);
  //    return {dead_count(h), pending(h)};
  AdjustResult adjust_for_activation(Handle h, bool increment_dead) {
    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared(Packed(h), increment_dead);
    }
  }

  // The same as the above, but performs the operation atomically. This
  // is thread-safe to run concurrently with other threads.
  AdjustResult adjust_for_activation_atomic(Handle h, bool increment_dead) {
    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared_atomic(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared_atomic(Packed(h), increment_dead);
    }
  }

  class Handle {
   public:
    Handle() : byte_offset_(0), is_large_(0) {}

   private:
    friend class PendingCounts;
    int byte_offset_ : 31;  // Byte offset of the rep in PendingCounts object
    bool is_large_ : 1;  // If true, rep is LargeCounts; otherwise PackedCounts
  };

 private:
  template <typename T>
  inline AdjustResult adjust_for_increment_dead_shared(std::atomic<T>* c) {
    T val = c->load(std::memory_order_relaxed);
    auto ret_pending = 0;
    // Only do anything if the node hasn't already started executing.
    if (PENDING_NOTREADY == NodeStateForStruct(val)) {
      val.dead_count++;
      ret_pending = val.pending;
      c->store(val, std::memory_order_relaxed);
    }
    return AdjustResult(val.dead_count, ret_pending);
  }

  template <typename T>
  inline AdjustResult adjust_for_increment_dead_shared_atomic(
      std::atomic<T>* c) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      auto new_val = old_val;
      auto ret_pending = 0;
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateForStruct(new_val)) {
        ret_pending = new_val.pending;
        new_val.dead_count++;
      }
      AdjustResult ret(new_val.dead_count, ret_pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }

  template <typename T>
  inline AdjustResult adjust_for_decrement_pending_shared(
      std::atomic<T>* c, int decrement_pending) {
    T val = c->load(std::memory_order_relaxed);
    DCHECK_GE(val.pending, decrement_pending);
    val.pending -= decrement_pending;
    c->store(val, std::memory_order_relaxed);
    return AdjustResult(val.dead_count, val.pending);
  }

  template <typename T>
  inline AdjustResult adjust_for_decrement_pending_shared_atomic(
      std::atomic<T>* c, int decrement_pending) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      T new_val = old_val;
      DCHECK_GE(new_val.pending, decrement_pending);
      new_val.pending -= decrement_pending;
      AdjustResult ret(new_val.dead_count, new_val.pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }

  template <typename T>
  inline AdjustResult adjust_for_activation_shared(std::atomic<T>* c,
                                                   bool increment_dead) {
    T val = c->load(std::memory_order_relaxed);
    if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(val)) {
      val.dead_count++;
    }
    DCHECK_GE(val.pending, 1);
    val.pending--;
    c->store(val, std::memory_order_relaxed);
    return AdjustResult(val.dead_count, val.pending);
  }

  template <typename T>
  inline AdjustResult adjust_for_activation_shared_atomic(std::atomic<T>* c,
                                                          bool increment_dead) {
    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      T new_val = old_val;
      if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(new_val)) {
        new_val.dead_count++;
      }
      DCHECK_GE(new_val.pending, 1);
      new_val.pending--;
      AdjustResult ret(new_val.dead_count, new_val.pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }

  // We keep track of the pending count and dead input count for each
  // graph node.  The representation used here is designed to be cache
  // efficient for graphs with large numbers of nodes, where most
  // nodes have relatively small maximum pending counts (e.g. for one
  // LSTM model, 99% of 5000+ nodes had in-degrees of 3 or less).  We
  // use one byte to hold both the pending and dead count for a node
  // where these together can fit in one byte, and we use a hash table
  // to handle the rare node ids that need larger counts than this.
  // Each frame in this subgraph has its own PendingCounts.

  // We use 3 bits each for dead_count and pending.
  static constexpr int kMaxCountForPackedCounts = 7;

  // Most counts are small, so we pack a pending count and a dead
  // count into 3 bits each, use 1 bit to indicate that the node has
  // started computing.
  struct PackedCounts {
    uint8 pending : 3;
    uint8 dead_count : 3;
    uint8 has_started : 1;
  };

  // NOTE: alignas(8) is critical to implement efficient atomic<LargeCounts>
  // on MSVC.
  struct alignas(8) LargeCounts {
    uint32 pending;
    uint32 dead_count : 31;
    // NOTE(tlipcon): MSVC won't pack this struct into 8 bytes unless
    // all of the member types are uint32.
    uint32 has_started : 1;
  };

  template <typename T>
  NodeState NodeStateForStruct(const T& c) const {
    if (c.has_started) {
      return (c.pending == 0) ? STARTED : COMPLETED;
    } else {
      return (c.pending == 0) ? PENDING_READY : PENDING_NOTREADY;
    }
  }
  inline std::atomic<LargeCounts>* Large(Handle h) {
    DCHECK(h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(std::atomic<LargeCounts>), num_bytes_);
    DCHECK_EQ(h.byte_offset_ % alignof(std::atomic<LargeCounts>), 0);
    return reinterpret_cast<std::atomic<LargeCounts>*>(bytes_ + h.byte_offset_);
  }
  inline std::atomic<PackedCounts>* Packed(Handle h) {
    DCHECK(!h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(PackedCounts), num_bytes_);
    return reinterpret_cast<std::atomic<PackedCounts>*>(bytes_ +
                                                        h.byte_offset_);
  }

  const int num_bytes_;  // Just for bounds checking in debug mode
  char* bytes_;          // Array of num_bytes_ bytes

  void operator=(const PendingCounts&) = delete;
};

inline PendingCounts::Handle PendingCounts::Layout::CreateHandle(
    size_t max_pending_count, size_t max_dead_count) {
  Handle result;
  if ((max_pending_count > kMaxCountForPackedCounts) ||
      (max_dead_count > kMaxCountForPackedCounts)) {
    constexpr int B = sizeof(std::atomic<LargeCounts>);
    // Round byte offset to proper alignment
    static_assert(
        sizeof(std::atomic<LargeCounts>) >= alignof(std::atomic<LargeCounts>),
        "std::atomic<LargeCounts> must be packed");
    int64_t offset = ((static_cast<int64_t>(next_offset_) + B - 1) / B) * B;
    result.byte_offset_ = offset;
    result.is_large_ = true;
    next_offset_ = result.byte_offset_ + B;
  } else {
    result.byte_offset_ = next_offset_;
    result.is_large_ = false;
    static_assert(sizeof(std::atomic<PackedCounts>) == 1,
                  "std::atomic<PackedCounts> should be a single byte");
    next_offset_ += sizeof(std::atomic<PackedCounts>);
  }
  return result;
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
