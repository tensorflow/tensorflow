#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_

/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <unordered_map>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// An internal helper class to keep track of pending and dead counts for nodes,
// for use in the ExecutorState module.
class PendingCounts {
 public:
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
      pc.is_large = 0;
      counts_[id] = pc;
    }
  }

  int pending(int id) {
    if (IsLarge(id)) {
      return overflow_[id].pending;
    } else {
      return counts_[id].pending;
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
      overflow_[id].pending |= 0x1;
    } else {
      counts_[id].pending |= 0x1;
    }
  }

  int dead_count(int id) {
    int r = IsLarge(id) ? overflow_[id].dead_count : counts_[id].dead_count;
    return r;
  }
  void increment_dead_count(int id) {
    if (IsLarge(id)) {
      overflow_[id].dead_count++;
    } else {
      DCHECK_LT(counts_[id].dead_count, kMaxCountForPackedCounts);
      counts_[id].dead_count++;
    }
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

  // TODO(yuanbyu): We current use O(# of nodes in partition) space
  // even for nested iterations where only a small fraction of the
  // nodes are involved.  This is not efficient if the subgraph for
  // the frame is only a small subset of the partition. We should make
  // the vector size to be only the size of the frame subgraph.

  const int kMaxCountForPackedCounts = 7;  // We use 3 bits for dead_count

  bool IsLarge(int id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    return counts_[id].is_large;
  }
  // Most counts are small, so we pack a pending count and a dead
  // count into 4 bits and 3 bits, respectively, and then use one bit
  // as a marker bit.  If "is_large" is true, then the true pending
  // and dead_count for that "id" are stored as full 32-bit counts in
  // "overflow_", a hash table indexed by id.
  struct PackedCounts {
    uint8 pending : 4;
    uint8 dead_count : 3;
    uint8 is_large : 1;
  };

  struct LargeCounts {
    int pending = 0;
    int dead_count = 0;
  };

  const int num_nodes_;  // Just for bounds checking in debug mode
  PackedCounts* counts_;
  std::unordered_map<int, LargeCounts> overflow_;

  TF_DISALLOW_COPY_AND_ASSIGN(PendingCounts);
};

}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
