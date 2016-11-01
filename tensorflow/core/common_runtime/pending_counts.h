#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_

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

#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// An internal helper class to keep track of pending and dead counts for nodes,
// for use in the ExecutorState module.
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

  explicit PendingCounts(int num_nodes)
      : num_nodes_(num_nodes), counts_(new PackedCounts[num_nodes]) {}

  ~PendingCounts() { delete[] counts_; }

  void set_initial_count(int id, int pending_count, int max_dead_count) {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    if ((pending_count > kMaxCountForPackedCounts) ||
        (max_dead_count > kMaxCountForPackedCounts)) {
      // A value for which we have to use the large representation
      DCHECK(overflow_.count(id) == 0);
      LargeCounts c;
      c.pending = pending_count;
      c.dead_count = 0;
      overflow_[id] = c;
      PackedCounts pc;
      pc.pending = 0;
      pc.dead_count = 0;
      pc.is_large = 1;
      counts_[id] = pc;
    } else {
      PackedCounts pc;
      pc.pending = pending_count;
      pc.dead_count = 0;
      pc.has_started = 0;
      pc.is_large = 0;
      counts_[id] = pc;
    }
  }

  inline int num_nodes() const { return num_nodes_; }
  NodeState node_state(int id) {
    if (IsLarge(id)) {
      return NodeStateLarge(id);
    } else {
      return NodeStatePacked(id);
    }
  }
  void mark_started(int id) {
    if (IsLarge(id)) {
      auto& pending = overflow_[id].pending;
      DCHECK_EQ(pending, 0);
      pending = -1;
    } else {
      DCHECK_EQ(counts_[id].pending, 0);
      DCHECK_EQ(counts_[id].has_started, 0);
      counts_[id].has_started = 1;
    }
  }
  void mark_completed(int id) {
    if (IsLarge(id)) {
      auto& pending = overflow_[id].pending;
      DCHECK_EQ(pending, -1);
      pending = -2;
    } else {
      DCHECK_EQ(counts_[id].pending, 0);
      DCHECK_EQ(counts_[id].has_started, 1);
      counts_[id].pending = 1;
    }
  }
  int pending(int id) {
    if (IsLarge(id)) {
      if (PENDING_NOTREADY == NodeStateLarge(id)) {
        return overflow_[id].pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    } else {
      if (PENDING_NOTREADY == NodeStatePacked(id)) {
        return counts_[id].pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    }
  }
  int decrement_pending(int id, int v) {
    DCHECK_GE(pending(id), v);
    if (IsLarge(id)) {
      int* p = &(overflow_[id].pending);
      (*p) -= v;
      return *p;
    } else {
      counts_[id].pending -= v;
      return counts_[id].pending;
    }
  }
  // Mark a merge node as live
  // REQUIRES: Node corresponding to "id" is a merge node
  void mark_live(int id) {
    if (IsLarge(id)) {
      int& count = overflow_[id].pending;
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateLarge(id)) {
        count &= ~static_cast<int>(0x1);
      }
    } else {
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStatePacked(id)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        counts_[id].pending &= 0x6;
      }
    }
  }

  int dead_count(int id) {
    int r = IsLarge(id) ? overflow_[id].dead_count : counts_[id].dead_count;
    return r;
  }
  void increment_dead_count(int id) {
    if (IsLarge(id)) {
      if (PENDING_NOTREADY == NodeStateLarge(id)) {
        overflow_[id].dead_count++;
      }
    } else {
      if (PENDING_NOTREADY == NodeStatePacked(id)) {
        DCHECK_LT(counts_[id].dead_count, kMaxCountForPackedCounts);
        counts_[id].dead_count++;
      }
    }
  }

  // Initialize the state from "b".
  // REQUIRES: "num_nodes_ == b.num_nodes_"
  void InitializeFrom(const PendingCounts& b) {
    DCHECK_EQ(num_nodes_, b.num_nodes_);
    for (int id = 0; id < num_nodes_; id++) {
      counts_[id] = b.counts_[id];
    }
    overflow_ = b.overflow_;
  }

 private:
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
  static const int kMaxCountForPackedCounts = 7;

  bool IsLarge(int id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    return counts_[id].is_large;
  }
  // Requires !IsLarge(id).
  NodeState NodeStatePacked(int id) const {
    if (counts_[id].has_started) {
      return (counts_[id].pending == 0) ? STARTED : COMPLETED;
    } else {
      return (counts_[id].pending == 0) ? PENDING_READY : PENDING_NOTREADY;
    }
  }
  // Requires IsLarge(id).
  NodeState NodeStateLarge(int id) {
    int pending = overflow_[id].pending;
    if (pending > 0) {
      return PENDING_NOTREADY;
    } else if (pending == 0) {
      return PENDING_READY;
    } else if (pending == -1) {
      return STARTED;
    } else {
      return COMPLETED;
    }
  }
  // Most counts are small, so we pack a pending count and a dead
  // count into 3 bits each, use 1 bit to indicate that the node has
  // started computing, and then use the final bit as a marker bit.
  // If "is_large" is true, then the true pending and dead_count for
  // that "id" are stored as full 32-bit counts in "overflow_", a hash
  // table indexed by id.
  struct PackedCounts {
    uint8 pending : 3;
    uint8 dead_count : 3;
    uint8 has_started : 1;
    uint8 is_large : 1;
  };

  struct LargeCounts {
    // A negative value for pending indicates that the node has
    // started executing.
    int pending = 0;
    int dead_count = 0;
  };

  const int num_nodes_;  // Just for bounds checking in debug mode
  PackedCounts* counts_;
  gtl::FlatMap<int, LargeCounts> overflow_;

  TF_DISALLOW_COPY_AND_ASSIGN(PendingCounts);
};

}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
