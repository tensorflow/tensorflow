/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACKED_CACHE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACKED_CACHE_H_

#include <cstddef>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

#include "tensorflow/lite/experimental/ruy/allocator.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/time.h"

namespace ruy {

namespace detail {

// Tracks a set of blocks allocated from the underlying system allocator.
class SystemBlockAllocator {
 public:
  void *Alloc(std::ptrdiff_t num_bytes) {
    void *p = detail::SystemAlignedAlloc(num_bytes);
    blocks_.push_back(p);
    return p;
  }

  void Free(void *block) {
    for (auto it = blocks_.begin(); it != blocks_.end(); ++it) {
      if (*it == block) {
        detail::SystemAlignedFree(block);
        blocks_.erase(it);
        return;
      }
    }
    RUY_DCHECK(false);  // Trying to free pointer we did not allocate.
  }

  ~SystemBlockAllocator() {
    for (void *block : blocks_) {
      detail::SystemAlignedFree(block);
    }
  }

 private:
  std::vector<void *> blocks_;
};

}  // namespace detail

enum CachePolicy { kNoCache, kCacheLHSOnGemV };

// "Low effort" Least Recently Used Cache for Prepacked Matrices
// A cache mechanism for prepacked matrices that ejects oldest entries.
// The implementation is "low effort" in the following ways:
//  - we just linearly search for the oldest entry when doing an ejection
//  - the ejection policy is very simple: if the new size would be above the
// .  threshold, we will eject one entry when adding an entry. Therefore,
//    there are no guarantees on maximum cache size since one may
//    insert an item larger than the ejection threshold (it will be ejected on
//    the next insert, but inserts always succeed).
// Current use cases (RNNs with GEMV operations) indicate that ejection is rare
// and memory constraints are tight, so we devote no additional storage to the
// LRU mechanism and accept O(n) search to eject oldest entry. In practice,
// the number of total entries has not been shown to be large.
// This class is not thread safe. In Ruy, memory allocation for packed matrices
// is done in a single threaded context and the actual packing activity may
// be done in a multi-threaded context.
class PrepackedCache {
 public:
  static constexpr int kDefaultEjectionThresholdBytes = 1 << 28;

  using CacheKey = std::pair<void *, void *>;

  using MatrixWithTimeStamp = std::pair<PrepackedMatrix, TimePoint>;

  using CacheIterator = std::map<CacheKey, MatrixWithTimeStamp>::const_iterator;

  using AlignedAllocator = detail::AlignedAllocator;

  explicit PrepackedCache(
      int32_t ejection_threshold = kDefaultEjectionThresholdBytes)
      : ejection_threshold_(ejection_threshold), cache_size_(0) {}

  // Looks for an entry with `key`. If found, update its time stamp.
  CacheIterator FindAndUpdate(const CacheKey &key);

  // Returns end iterator for internal cache. The iterator type is appropriate
  // to use with `FindAndUpdate`.
  CacheIterator cend() const { return cache_.end(); }

  // Returns the total size (in bytes) of data held in this cache.
  int TotalSize() const { return cache_size_; }

  // All calls to get current TimePoints go through here.
  // TODO(b/145625614) Profile timestamps on relevant models to see if
  // this level of granularity is sufficient. CoarseNow is cheap so
  // it would be nice to keep it.
  TimePoint CacheNow() const { return CoarseNow(); }

  // Performs the memory allocation for the `data` and `sums` members of a
  // PrepackedMatrix.
  void AllocatePrepackedMatrix(PrepackedMatrix *pmatrix);

  // Adds the PrepackedMatrix to the cache, possibly ejecting other values.
  void Insert(const CacheKey &key, const PrepackedMatrix &matrix);

 private:
  void EjectOne();
  void DoInsert(const CacheKey &key, const PrepackedMatrix &matrix);
  detail::SystemBlockAllocator allocator_;
  std::map<CacheKey, MatrixWithTimeStamp> cache_;
  const int32_t ejection_threshold_;
  size_t cache_size_;
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_PREPACKED_CACHE_H_
