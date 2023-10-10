/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_PRESIZED_CUCKOO_MAP_H_
#define TENSORFLOW_CORE_UTIL_PRESIZED_CUCKOO_MAP_H_

#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {

// Class for efficiently storing key->value mappings when the size is
// known in advance and the keys are pre-hashed into uint64s.
// Keys should have "good enough" randomness (be spread across the
// entire 64 bit space).
//
// Important:  Clients wishing to use deterministic keys must
// ensure that their keys fall in the range 0 .. (uint64max-1);
// the table uses 2^64-1 as the "not occupied" flag.
//
// Inserted keys must be unique, and there are no update
// or delete functions (until some subsequent use of this table
// requires them).
//
// Threads must synchronize their access to a PresizedCuckooMap.
//
// The cuckoo hash table is 4-way associative (each "bucket" has 4
// "slots" for key/value entries).  Uses breadth-first-search to find
// a good cuckoo path with less data movement (see
// http://www.cs.cmu.edu/~dga/papers/cuckoo-eurosys14.pdf )

namespace presized_cuckoo_map {
// Utility function to compute (x * y) >> 64, or "multiply high".
// On x86-64, this is a single instruction, but not all platforms
// support the __uint128_t type, so we provide a generic
// implementation as well.
inline uint64 multiply_high_u64(uint64 x, uint64 y) {
#if defined(__SIZEOF_INT128__)
  return (uint64)(((__uint128_t)x * (__uint128_t)y) >> 64);
#else
  // For platforms without int128 support, do it the long way.
  uint64 x_lo = x & 0xffffffff;
  uint64 x_hi = x >> 32;
  uint64 buckets_lo = y & 0xffffffff;
  uint64 buckets_hi = y >> 32;
  uint64 prod_hi = x_hi * buckets_hi;
  uint64 prod_lo = x_lo * buckets_lo;
  uint64 prod_mid1 = x_hi * buckets_lo;
  uint64 prod_mid2 = x_lo * buckets_hi;
  uint64 carry =
      ((prod_mid1 & 0xffffffff) + (prod_mid2 & 0xffffffff) + (prod_lo >> 32)) >>
      32;
  return prod_hi + (prod_mid1 >> 32) + (prod_mid2 >> 32) + carry;
#endif
}
}  // namespace presized_cuckoo_map

template <class value>
class PresizedCuckooMap {
 public:
  // The key type is fixed as a pre-hashed key for this specialized use.
  typedef uint64 key_type;

  explicit PresizedCuckooMap(uint64 num_entries) { Clear(num_entries); }

  void Clear(uint64 num_entries) {
    cpq_.reset(new CuckooPathQueue());
    double n(num_entries);
    n /= kLoadFactor;
    num_buckets_ = (static_cast<uint64>(n) / kSlotsPerBucket);
    // Very small cuckoo tables don't work, because the probability
    // of having same-bucket hashes is large.  We compromise for those
    // uses by having a larger static starting size.
    num_buckets_ += 32;
    Bucket empty_bucket;
    for (int i = 0; i < kSlotsPerBucket; i++) {
      empty_bucket.keys[i] = kUnusedSlot;
    }
    buckets_.clear();
    buckets_.resize(num_buckets_, empty_bucket);
  }

  // Returns false if k is already in table or if the table
  // is full; true otherwise.
  bool InsertUnique(const key_type k, const value& v) {
    uint64 tk = key_transform(k);
    uint64 b1 = fast_map_to_buckets(tk);
    uint64 b2 = fast_map_to_buckets(h2(tk));

    // Merged find and duplicate checking.
    uint64 target_bucket = 0;
    int target_slot = kNoSpace;

    for (auto bucket : {b1, b2}) {
      Bucket* bptr = &buckets_[bucket];
      for (int slot = 0; slot < kSlotsPerBucket; slot++) {
        if (bptr->keys[slot] == k) {  // Duplicates are not allowed.
          return false;
        } else if (target_slot == kNoSpace && bptr->keys[slot] == kUnusedSlot) {
          target_bucket = bucket;
          target_slot = slot;
        }
      }
    }

    if (target_slot != kNoSpace) {
      InsertInternal(tk, v, target_bucket, target_slot);
      return true;
    }

    return CuckooInsert(tk, v, b1, b2);
  }

  // Returns true if found.  Sets *out = value.
  bool Find(const key_type k, value* out) const {
    uint64 tk = key_transform(k);
    return FindInBucket(k, fast_map_to_buckets(tk), out) ||
           FindInBucket(k, fast_map_to_buckets(h2(tk)), out);
  }

  // Prefetch memory associated with the key k into cache levels specified by
  // hint.
  template <port::PrefetchHint hint = port::PREFETCH_HINT_T0>
  void PrefetchKey(const key_type k) const {
    const uint64 tk = key_transform(k);
    port::prefetch<hint>(&buckets_[fast_map_to_buckets(tk)].keys);
    port::prefetch<hint>(&buckets_[fast_map_to_buckets(h2(tk))].keys);
  }

  int64_t MemoryUsed() const {
    return sizeof(PresizedCuckooMap<value>) + sizeof(CuckooPathQueue);
  }

 private:
  static constexpr int kSlotsPerBucket = 4;

  // The load factor is chosen slightly conservatively for speed and
  // to avoid the need for a table rebuild on insertion failure.
  // 0.94 is achievable, but 0.85 is faster and keeps the code simple
  // at the cost of a small amount of memory.
  // NOTE:  0 < kLoadFactor <= 1.0
  static constexpr double kLoadFactor = 0.85;

  // Cuckoo insert:  The maximum number of entries to scan should be ~400
  // (Source:  Personal communication with Michael Mitzenmacher;  empirical
  // experiments validate.).  After trying 400 candidate locations, declare
  // the table full - it's probably full of unresolvable cycles.  Less than
  // 400 reduces max occupancy;  much more results in very poor performance
  // around the full point.  For (2,4) a max BFS path len of 5 results in ~682
  // nodes to visit, calculated below, and is a good value.

  static constexpr uint8 kMaxBFSPathLen = 5;

  // Constants for BFS cuckoo path search:
  // The visited list must be maintained for all but the last level of search
  // in order to trace back the path.  The BFS search has two roots
  // and each can go to a total depth (including the root) of 5.
  // The queue must be sized for 2 * \sum_{k=0...4}{kSlotsPerBucket^k} = 682.
  // The visited queue, however, does not need to hold the deepest level,
  // and so it is sized 2 * \sum{k=0...3}{kSlotsPerBucket^k} = 170
  static constexpr int kMaxQueueSize = 682;
  static constexpr int kVisitedListSize = 170;

  static constexpr int kNoSpace = -1;  // SpaceAvailable return
  static constexpr uint64 kUnusedSlot = ~(0ULL);

  // Buckets are organized with key_types clustered for access speed
  // and for compactness while remaining aligned.
  struct Bucket {
    key_type keys[kSlotsPerBucket];
    value values[kSlotsPerBucket];
  };

  // Insert uses the BFS optimization (search before moving) to reduce
  // the number of cache lines dirtied during search.

  struct CuckooPathEntry {
    uint64 bucket;
    int depth;
    int parent;       // To index in the visited array.
    int parent_slot;  // Which slot in our parent did we come from?  -1 == root.
  };

  // CuckooPathQueue is a trivial circular queue for path entries.
  // The caller is responsible for not inserting more than kMaxQueueSize
  // entries.  Each PresizedCuckooMap has one (heap-allocated) CuckooPathQueue
  // that it reuses across inserts.
  class CuckooPathQueue {
   public:
    CuckooPathQueue() : head_(0), tail_(0) {}

    void push_back(CuckooPathEntry e) {
      queue_[tail_] = e;
      tail_ = (tail_ + 1) % kMaxQueueSize;
    }

    CuckooPathEntry pop_front() {
      CuckooPathEntry& e = queue_[head_];
      head_ = (head_ + 1) % kMaxQueueSize;
      return e;
    }

    bool empty() const { return head_ == tail_; }

    bool full() const { return ((tail_ + 1) % kMaxQueueSize) == head_; }

    void reset() { head_ = tail_ = 0; }

   private:
    CuckooPathEntry queue_[kMaxQueueSize];
    int head_;
    int tail_;
  };

  typedef std::array<CuckooPathEntry, kMaxBFSPathLen> CuckooPath;

  // Callers are expected to have pre-hashed the keys into a uint64
  // and are expected to be able to handle (a very low rate) of
  // collisions, OR must ensure that their keys are always in
  // the range 0 - (uint64max - 1).  This transforms 'not found flag'
  // keys into something else.
  inline uint64 key_transform(const key_type k) const {
    return k + (k == kUnusedSlot);
  }

  // h2 performs a very quick mix of h to generate the second bucket hash.
  // Assumes there is plenty of remaining entropy in the initial h.
  inline uint64 h2(uint64 h) const {
    const uint64 m = 0xc6a4a7935bd1e995;
    return m * ((h >> 32) | (h << 32));
  }

  // alt_bucket identifies the "other" bucket for key k, where
  // other is "the one that isn't bucket b"
  inline uint64 alt_bucket(key_type k, uint64 b) const {
    if (fast_map_to_buckets(k) != b) {
      return fast_map_to_buckets(k);
    }
    return fast_map_to_buckets(h2(k));
  }

  inline void InsertInternal(key_type k, const value& v, uint64 b, int slot) {
    Bucket* bptr = &buckets_[b];
    bptr->keys[slot] = k;
    bptr->values[slot] = v;
  }

  // For the associative cuckoo table, check all of the slots in
  // the bucket to see if the key is present.
  bool FindInBucket(key_type k, uint64 b, value* out) const {
    const Bucket& bref = buckets_[b];
    for (int i = 0; i < kSlotsPerBucket; i++) {
      if (bref.keys[i] == k) {
        *out = bref.values[i];
        return true;
      }
    }
    return false;
  }

  //  returns either kNoSpace or the index of an
  //  available slot (0 <= slot < kSlotsPerBucket)
  inline int SpaceAvailable(uint64 bucket) const {
    const Bucket& bref = buckets_[bucket];
    for (int i = 0; i < kSlotsPerBucket; i++) {
      if (bref.keys[i] == kUnusedSlot) {
        return i;
      }
    }
    return kNoSpace;
  }

  inline void CopyItem(uint64 src_bucket, int src_slot, uint64 dst_bucket,
                       int dst_slot) {
    Bucket& src_ref = buckets_[src_bucket];
    Bucket& dst_ref = buckets_[dst_bucket];
    dst_ref.keys[dst_slot] = src_ref.keys[src_slot];
    dst_ref.values[dst_slot] = src_ref.values[src_slot];
  }

  bool CuckooInsert(key_type k, const value& v, uint64 b1, uint64 b2) {
    int visited_end = 0;
    cpq_->reset();

    cpq_->push_back({b1, 1, 0, 0});  // Note depth starts at 1.
    cpq_->push_back({b2, 1, 0, 0});

    while (!cpq_->empty()) {
      CuckooPathEntry e = cpq_->pop_front();
      int free_slot;
      free_slot = SpaceAvailable(e.bucket);
      if (free_slot != kNoSpace) {
        while (e.depth > 1) {
          // "copy" instead of "swap" because one entry is always zero.
          // After, write target key/value over top of last copied entry.
          CuckooPathEntry parent = visited_[e.parent];
          CopyItem(parent.bucket, e.parent_slot, e.bucket, free_slot);
          free_slot = e.parent_slot;
          e = parent;
        }
        InsertInternal(k, v, e.bucket, free_slot);
        return true;
      } else {
        if (e.depth < (kMaxBFSPathLen)) {
          auto parent_index = visited_end;
          visited_[visited_end] = e;
          visited_end++;
          // Don't always start with the same slot, to even out the path depth.
          int start_slot = (k + e.bucket) % kSlotsPerBucket;
          const Bucket& bref = buckets_[e.bucket];
          for (int i = 0; i < kSlotsPerBucket; i++) {
            int slot = (start_slot + i) % kSlotsPerBucket;
            uint64 next_bucket = alt_bucket(bref.keys[slot], e.bucket);
            // Optimization:  Avoid single-step cycles (from e, don't
            // add a child node that is actually e's parent).
            uint64 e_parent_bucket = visited_[e.parent].bucket;
            if (next_bucket != e_parent_bucket) {
              cpq_->push_back({next_bucket, e.depth + 1, parent_index, slot});
            }
          }
        }
      }
    }

    LOG(WARNING) << "Cuckoo path finding failed: Table too small?";
    return false;
  }

  inline uint64 fast_map_to_buckets(uint64 x) const {
    // Map x (uniform in 2^64) to the range [0, num_buckets_ -1]
    // using Lemire's alternative to modulo reduction:
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    // Instead of x % N, use (x * N) >> 64.
    return presized_cuckoo_map::multiply_high_u64(x, num_buckets_);
  }

  // Set upon initialization: num_entries / kLoadFactor / kSlotsPerBucket.
  uint64 num_buckets_;
  std::vector<Bucket> buckets_;

  std::unique_ptr<CuckooPathQueue> cpq_;
  CuckooPathEntry visited_[kVisitedListSize];

  PresizedCuckooMap(const PresizedCuckooMap&) = delete;
  void operator=(const PresizedCuckooMap&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PRESIZED_CUCKOO_MAP_H_
