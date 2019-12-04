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

#include "tensorflow/lite/experimental/ruy/prepacked_cache.h"

#include "tensorflow/lite/experimental/ruy/matrix.h"

namespace ruy {

using CacheIterator = PrepackedCache::CacheIterator;

// Looks for an entry with `key`. If found, update its time stamp.
CacheIterator PrepackedCache::FindAndUpdate(const CacheKey &key) {
  auto itr = cache_.find(key);
  // If found, update with new access time for this entry.
  if (itr != cache_.end()) {
    const TimePoint time = CoarseNow();
    itr->second.second = time;
  }
  return itr;
}

void PrepackedCache::Insert(const CacheKey &key,
                            const PrepackedMatrix &matrix) {
  // Calculate size of this new item.
  const size_t size_bytes = matrix.data_size + matrix.sums_size;

  // If we are above the threshold of ejection, eject the LRU entry.
  while (!cache_.empty() &&
         ((TotalSize() + size_bytes) > ejection_threshold_)) {
    EjectOne();
  }
  DoInsert(key, matrix);
  cache_size_ += matrix.data_size + matrix.sums_size;
}

void PrepackedCache::EjectOne() {
  TimePoint oldest_time = CoarseNow();
  auto oldest = cache_.begin();
  for (auto itr = cache_.begin(); itr != cache_.end(); ++itr) {
    if (itr->second.second < oldest_time) {
      oldest_time = itr->second.second;
      oldest = itr;
    }
  }
  PrepackedMatrix &pmatrix = oldest->second.first;
  cache_size_ -= pmatrix.data_size;
  cache_size_ -= pmatrix.sums_size;
  allocator_.Free(pmatrix.data);
  allocator_.Free(pmatrix.sums);
  cache_.erase(oldest);
}

void PrepackedCache::AllocatePrepackedMatrix(PrepackedMatrix *pmatrix) {
  pmatrix->data = allocator_.Alloc(pmatrix->data_size);
  pmatrix->sums = allocator_.Alloc(pmatrix->sums_size);
}

void PrepackedCache::DoInsert(const CacheKey &key,
                              const PrepackedMatrix &matrix) {
  // TODO(talumbau) Profile timestamps on relevant models to see if
  // this level of granularity is sufficient. CoarseNow is cheap so
  // it would be nice to keep it.
  const TimePoint t = CoarseNow();
  const MatrixWithTimeStamp mts({matrix, t});
  cache_.insert({key, mts});
}

}  // namespace ruy
