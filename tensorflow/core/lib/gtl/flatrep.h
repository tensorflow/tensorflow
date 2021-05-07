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

#ifndef TENSORFLOW_CORE_LIB_GTL_FLATREP_H_
#define TENSORFLOW_CORE_LIB_GTL_FLATREP_H_

#include <string.h>
#include <utility>
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {
namespace internal {

// Internal representation for FlatMap and FlatSet.
//
// The representation is an open-addressed hash table.  Conceptually,
// the representation is a flat array of entries.  However we
// structure it as an array of buckets where each bucket holds
// kWidth entries along with metadata for the kWidth entries.  The
// metadata marker is
//
//  (a) kEmpty: the entry is empty
//  (b) kDeleted: the entry has been deleted
//  (c) other: the entry is occupied and has low-8 bits of its hash.
//      These hash bits can be used to avoid potentially expensive
//      key comparisons.
//
// FlatMap passes in a bucket that contains keys and values, FlatSet
// passes in a bucket that does not contain values.
template <typename Key, typename Bucket, class Hash, class Eq>
class FlatRep {
 public:
  // kWidth is the number of entries stored in a bucket.
  static constexpr uint32 kBase = 3;
  static constexpr uint32 kWidth = (1 << kBase);

  FlatRep(size_t N, const Hash& hf, const Eq& eq) : hash_(hf), equal_(eq) {
    Init(N);
  }
  FlatRep(const FlatRep& src) : hash_(src.hash_), equal_(src.equal_) {
    Init(src.size());
    CopyEntries(src.array_, src.end_, CopyEntry());
  }

  FlatRep(FlatRep&& src)
      // Copy rather than move src.hash_ and src.equal_.  This is necessary to
      // leave src in a valid state -- otherwise e.g. if hash_ is an
      // std::function, moving it would null it out.
      : hash_(src.hash_), equal_(src.equal_) {
    // TODO(jlebar): Init(1) still allocates some memory, so this isn't as cheap
    // as it could be.  The fundamental problem is that we need to leave src in
    // a valid state, and FlatRep *always* owns a nonzero amount of memory.
    Init(1);
    swap(src);
  }

  ~FlatRep() {
    clear_no_resize();
    delete[] array_;
  }

  // Simple accessors.
  size_t size() const { return not_empty_ - deleted_; }
  size_t bucket_count() const { return mask_ + 1; }
  Bucket* start() const { return array_; }
  Bucket* limit() const { return end_; }
  const Hash& hash_function() const { return hash_; }
  const Eq& key_eq() const { return equal_; }

  // Overwrite contents of *this with contents of src.
  void CopyFrom(const FlatRep& src) {
    if (this != &src) {
      clear_no_resize();
      delete[] array_;
      Init(src.size());
      CopyEntries(src.array_, src.end_, CopyEntry());
    }
  }

  void MoveFrom(FlatRep&& src) {
    if (this != &src) {
      swap(src);
    }
  }

  void clear_no_resize() {
    for (Bucket* b = array_; b != end_; b++) {
      for (uint32 i = 0; i < kWidth; i++) {
        if (b->marker[i] >= 2) {
          b->Destroy(i);
          b->marker[i] = kEmpty;
        }
      }
    }
    not_empty_ = 0;
    deleted_ = 0;
  }

  void clear() {
    clear_no_resize();
    grow_ = 0;  // Consider shrinking in MaybeResize()
    MaybeResize();
  }

  void swap(FlatRep& x) {
    using std::swap;
    swap(array_, x.array_);
    swap(end_, x.end_);
    swap(lglen_, x.lglen_);
    swap(mask_, x.mask_);
    swap(not_empty_, x.not_empty_);
    swap(deleted_, x.deleted_);
    swap(grow_, x.grow_);
    swap(shrink_, x.shrink_);
  }

  struct SearchResult {
    bool found;
    Bucket* b;
    uint32 index;
  };

  // Hash value is partitioned as follows:
  // 1. Bottom 8 bits are stored in bucket to help speed up comparisons.
  // 2. Next 3 bits give index inside bucket.
  // 3. Remaining bits give bucket number.

  // Find bucket/index for key k.
  SearchResult Find(const Key& k) const {
    size_t h = hash_(k);
    const uint32 marker = Marker(h & 0xff);
    size_t index = (h >> 8) & mask_;  // Holds bucket num and index-in-bucket
    uint32 num_probes = 1;            // Needed for quadratic probing
    while (true) {
      uint32 bi = index & (kWidth - 1);
      Bucket* b = &array_[index >> kBase];
      const uint32 x = b->marker[bi];
      if (x == marker && equal_(b->key(bi), k)) {
        return {true, b, bi};
      } else if (x == kEmpty) {
        return {false, nullptr, 0};
      }
      index = NextIndex(index, num_probes);
      num_probes++;
    }
  }

  // Find bucket/index for key k, creating a new one if necessary.
  //
  // KeyType is a template parameter so that k's type is deduced and it
  // becomes a universal reference which allows the key initialization
  // below to use an rvalue constructor if available.
  template <typename KeyType>
  SearchResult FindOrInsert(KeyType&& k) {
    size_t h = hash_(k);
    const uint32 marker = Marker(h & 0xff);
    size_t index = (h >> 8) & mask_;  // Holds bucket num and index-in-bucket
    uint32 num_probes = 1;            // Needed for quadratic probing
    Bucket* del = nullptr;            // First encountered deletion for kInsert
    uint32 di = 0;
    while (true) {
      uint32 bi = index & (kWidth - 1);
      Bucket* b = &array_[index >> kBase];
      const uint32 x = b->marker[bi];
      if (x == marker && equal_(b->key(bi), k)) {
        return {true, b, bi};
      } else if (!del && x == kDeleted) {
        // Remember deleted index to use for insertion.
        del = b;
        di = bi;
      } else if (x == kEmpty) {
        if (del) {
          // Store in the first deleted slot we encountered
          b = del;
          bi = di;
          deleted_--;  // not_empty_ does not change
        } else {
          not_empty_++;
        }
        b->marker[bi] = marker;
        new (&b->key(bi)) Key(std::forward<KeyType>(k));
        return {false, b, bi};
      }
      index = NextIndex(index, num_probes);
      num_probes++;
    }
  }

  void Erase(Bucket* b, uint32 i) {
    b->Destroy(i);
    b->marker[i] = kDeleted;
    deleted_++;
    grow_ = 0;  // Consider shrinking on next insert
  }

  void Prefetch(const Key& k) const {
    size_t h = hash_(k);
    size_t index = (h >> 8) & mask_;  // Holds bucket num and index-in-bucket
    uint32 bi = index & (kWidth - 1);
    Bucket* b = &array_[index >> kBase];
    port::prefetch<port::PREFETCH_HINT_T0>(&b->marker[bi]);
    port::prefetch<port::PREFETCH_HINT_T0>(&b->storage.key[bi]);
  }

  inline void MaybeResize() {
    if (not_empty_ < grow_) {
      return;  // Nothing to do
    }
    if (grow_ == 0) {
      // Special value set by erase to cause shrink on next insert.
      if (size() >= shrink_) {
        // Not small enough to shrink.
        grow_ = static_cast<size_t>(bucket_count() * 0.8);
        if (not_empty_ < grow_) return;
      }
    }
    Resize(size() + 1);
  }

  void Resize(size_t N) {
    Bucket* old = array_;
    Bucket* old_end = end_;
    Init(N);
    CopyEntries(old, old_end, MoveEntry());
    delete[] old;
  }

 private:
  enum { kEmpty = 0, kDeleted = 1 };  // Special markers for an entry.

  Hash hash_;         // User-supplied hasher
  Eq equal_;          // User-supplied comparator
  uint8 lglen_;       // lg(#buckets)
  Bucket* array_;     // array of length (1 << lglen_)
  Bucket* end_;       // Points just past last bucket in array_
  size_t mask_;       // (# of entries in table) - 1
  size_t not_empty_;  // Count of entries with marker != kEmpty
  size_t deleted_;    // Count of entries with marker == kDeleted
  size_t grow_;       // Grow array when not_empty_ >= grow_
  size_t shrink_;     // Shrink array when size() < shrink_

  // Avoid kEmpty and kDeleted markers when computing hash values to
  // store in Bucket::marker[].
  static uint32 Marker(uint32 hb) { return hb + (hb < 2 ? 2 : 0); }

  void Init(size_t N) {
    // Make enough room for N elements.
    size_t lg = 0;  // Smallest table is just one bucket.
    while (N >= 0.8 * ((1 << lg) * kWidth)) {
      lg++;
    }
    const size_t n = (1 << lg);
    Bucket* array = new Bucket[n];
    for (size_t i = 0; i < n; i++) {
      Bucket* b = &array[i];
      memset(b->marker, kEmpty, kWidth);
    }
    const size_t capacity = (1 << lg) * kWidth;
    lglen_ = lg;
    mask_ = capacity - 1;
    array_ = array;
    end_ = array + n;
    not_empty_ = 0;
    deleted_ = 0;
    grow_ = static_cast<size_t>(capacity * 0.8);
    if (lg == 0) {
      // Already down to one bucket; no more shrinking.
      shrink_ = 0;
    } else {
      shrink_ = static_cast<size_t>(grow_ * 0.4);  // Must be less than 0.5
    }
  }

  // Used by FreshInsert when we should copy from source.
  struct CopyEntry {
    inline void operator()(Bucket* dst, uint32 dsti, Bucket* src, uint32 srci) {
      dst->CopyFrom(dsti, src, srci);
    }
  };

  // Used by FreshInsert when we should move from source.
  struct MoveEntry {
    inline void operator()(Bucket* dst, uint32 dsti, Bucket* src, uint32 srci) {
      dst->MoveFrom(dsti, src, srci);
      src->Destroy(srci);
      src->marker[srci] = kDeleted;
    }
  };

  template <typename Copier>
  void CopyEntries(Bucket* start, Bucket* end, Copier copier) {
    for (Bucket* b = start; b != end; b++) {
      for (uint32 i = 0; i < kWidth; i++) {
        if (b->marker[i] >= 2) {
          FreshInsert(b, i, copier);
        }
      }
    }
  }

  // Create an entry for the key numbered src_index in *src and return
  // its bucket/index.  Used for insertion into a fresh table.  We
  // assume that there are no deletions, and k does not already exist
  // in the table.
  template <typename Copier>
  void FreshInsert(Bucket* src, uint32 src_index, Copier copier) {
    size_t h = hash_(src->key(src_index));
    const uint32 marker = Marker(h & 0xff);
    size_t index = (h >> 8) & mask_;  // Holds bucket num and index-in-bucket
    uint32 num_probes = 1;            // Needed for quadratic probing
    while (true) {
      uint32 bi = index & (kWidth - 1);
      Bucket* b = &array_[index >> kBase];
      const uint32 x = b->marker[bi];
      if (x == 0) {
        b->marker[bi] = marker;
        not_empty_++;
        copier(b, bi, src, src_index);
        return;
      }
      index = NextIndex(index, num_probes);
      num_probes++;
    }
  }

  inline size_t NextIndex(size_t i, uint32 num_probes) const {
    // Quadratic probing.
    return (i + num_probes) & mask_;
  }
};

}  // namespace internal
}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_FLATREP_H_
