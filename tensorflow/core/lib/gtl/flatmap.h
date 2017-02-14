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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_

#include <stddef.h>
#include <functional>
#include <utility>
#include "tensorflow/core/lib/gtl/flatrep.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {

// FlatMap<K,V,...> provides a map from K to V.
//
// The map is implemented using an open-addressed hash table.  A
// single array holds entire map contents and collisions are resolved
// by probing at a sequence of locations in the array.
template <typename Key, typename Val, class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
class FlatMap {
 private:
  // Forward declare some internal types needed in public section.
  struct Bucket;

 public:
  typedef Key key_type;
  typedef Val mapped_type;
  typedef Hash hasher;
  typedef Eq key_equal;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  // We cannot use std::pair<> since internal representation stores
  // keys and values in separate arrays, so we make a custom struct
  // that holds references to the internal key, value elements.
  struct value_type {
    typedef Key first_type;
    typedef Val second_type;

    const Key& first;
    Val& second;
    value_type(const Key& k, Val& v) : first(k), second(v) {}
  };
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;

  FlatMap() : FlatMap(1) {}

  explicit FlatMap(size_t N, const Hash& hf = Hash(), const Eq& eq = Eq())
      : rep_(N, hf, eq) {}

  FlatMap(const FlatMap& src) : rep_(src.rep_) {}

  template <typename InputIter>
  FlatMap(InputIter first, InputIter last, size_t N = 1,
          const Hash& hf = Hash(), const Eq& eq = Eq())
      : FlatMap(N, hf, eq) {
    insert(first, last);
  }

  FlatMap& operator=(const FlatMap& src) {
    rep_.CopyFrom(src.rep_);
    return *this;
  }

  ~FlatMap() {}

  void swap(FlatMap& x) { rep_.swap(x.rep_); }
  void clear_no_resize() { rep_.clear_no_resize(); }
  void clear() { rep_.clear(); }
  void reserve(size_t N) { rep_.Resize(std::max(N, size())); }
  void rehash(size_t N) { rep_.Resize(std::max(N, size())); }
  void resize(size_t N) { rep_.Resize(std::max(N, size())); }
  size_t size() const { return rep_.size(); }
  bool empty() const { return size() == 0; }
  size_t bucket_count() const { return rep_.bucket_count(); }
  hasher hash_function() const { return rep_.hash_function(); }
  key_equal key_eq() const { return rep_.key_eq(); }

  class iterator {
   public:
    iterator() : b_(nullptr), end_(nullptr), i_(0) {}

    // Make iterator pointing at first element at or after b.
    explicit iterator(Bucket* b, Bucket* end) : b_(b), end_(end), i_(0) {
      SkipUnused();
    }

    // Make iterator pointing exactly at ith element in b, which must exist.
    iterator(Bucket* b, Bucket* end, uint32 i) : b_(b), end_(end), i_(i) {
      FillValue();
    }

    value_type& operator*() { return *val(); }
    value_type* operator->() { return val(); }
    bool operator==(const iterator& x) const {
      return b_ == x.b_ && i_ == x.i_;
    }
    bool operator!=(const iterator& x) const { return !(*this == x); }
    iterator& operator++() {
      DCHECK(b_ != end_);
      i_++;
      SkipUnused();
      return *this;
    }

   private:
    friend class FlatMap;
    Bucket* b_;
    Bucket* end_;
    uint32 i_;
    char space_[sizeof(value_type)];

    value_type* val() { return reinterpret_cast<value_type*>(space_); }
    void FillValue() { new (space_) value_type(b_->key(i_), b_->val(i_)); }
    void SkipUnused() {
      while (b_ < end_) {
        if (i_ >= Rep::kWidth) {
          i_ = 0;
          b_++;
        } else if (b_->marker[i_] < 2) {
          i_++;
        } else {
          FillValue();
          break;
        }
      }
    }
  };

  class const_iterator {
   private:
    mutable iterator rep_;  // Share state and logic with non-const iterator.
   public:
    const_iterator() : rep_() {}
    explicit const_iterator(Bucket* start, Bucket* end) : rep_(start, end) {}
    const_iterator(Bucket* b, Bucket* end, uint32 i) : rep_(b, end, i) {}

    const value_type& operator*() const { return *rep_.val(); }
    const value_type* operator->() const { return rep_.val(); }
    bool operator==(const const_iterator& x) const { return rep_ == x.rep_; }
    bool operator!=(const const_iterator& x) const { return rep_ != x.rep_; }
    const_iterator& operator++() {
      ++rep_;
      return *this;
    }
  };

  iterator begin() { return iterator(rep_.start(), rep_.limit()); }
  iterator end() { return iterator(rep_.limit(), rep_.limit()); }
  const_iterator begin() const {
    return const_iterator(rep_.start(), rep_.limit());
  }
  const_iterator end() const {
    return const_iterator(rep_.limit(), rep_.limit());
  }

  size_t count(const Key& k) const { return rep_.Find(k).found ? 1 : 0; }
  iterator find(const Key& k) {
    auto r = rep_.Find(k);
    return r.found ? iterator(r.b, rep_.limit(), r.index) : end();
  }
  const_iterator find(const Key& k) const {
    auto r = rep_.Find(k);
    return r.found ? const_iterator(r.b, rep_.limit(), r.index) : end();
  }

  Val& at(const Key& k) {
    auto r = rep_.Find(k);
    DCHECK(r.found);
    return r.b->val(r.index);
  }
  const Val& at(const Key& k) const {
    auto r = rep_.Find(k);
    DCHECK(r.found);
    return r.b->val(r.index);
  }

  template <typename P>
  std::pair<iterator, bool> insert(const P& p) {
    return Insert(p.first, p.second);
  }
  std::pair<iterator, bool> insert(const std::pair<const Key, Val>& p) {
    return Insert(p.first, p.second);
  }
  template <typename InputIter>
  void insert(InputIter first, InputIter last) {
    for (; first != last; ++first) {
      insert(*first);
    }
  }

  Val& operator[](const Key& k) { return IndexOp(k); }
  Val& operator[](Key&& k) { return IndexOp(std::forward<Key>(k)); }

  template <typename... Args>
  std::pair<iterator, bool> emplace(Args&&... args) {
    return InsertPair(std::make_pair(std::forward<Args>(args)...));
  }

  size_t erase(const Key& k) {
    auto r = rep_.Find(k);
    if (!r.found) return 0;
    rep_.Erase(r.b, r.index);
    return 1;
  }
  iterator erase(iterator pos) {
    rep_.Erase(pos.b_, pos.i_);
    ++pos;
    return pos;
  }
  iterator erase(iterator pos, iterator last) {
    for (; pos != last; ++pos) {
      rep_.Erase(pos.b_, pos.i_);
    }
    return pos;
  }

  std::pair<iterator, iterator> equal_range(const Key& k) {
    auto pos = find(k);
    if (pos == end()) {
      return std::make_pair(pos, pos);
    } else {
      auto next = pos;
      ++next;
      return std::make_pair(pos, next);
    }
  }
  std::pair<const_iterator, const_iterator> equal_range(const Key& k) const {
    auto pos = find(k);
    if (pos == end()) {
      return std::make_pair(pos, pos);
    } else {
      auto next = pos;
      ++next;
      return std::make_pair(pos, next);
    }
  }

  bool operator==(const FlatMap& x) const {
    if (size() != x.size()) return false;
    for (auto& p : x) {
      auto i = find(p.first);
      if (i == end()) return false;
      if (i->second != p.second) return false;
    }
    return true;
  }
  bool operator!=(const FlatMap& x) const { return !(*this == x); }

  // If key exists in the table, prefetch the associated value.  This
  // is a hint, and may have no effect.
  void prefetch_value(const Key& key) const { rep_.Prefetch(key); }

 private:
  using Rep = internal::FlatRep<Key, Bucket, Hash, Eq>;

  // Bucket stores kWidth <marker, key, value> triples.
  // The data is organized as three parallel arrays to reduce padding.
  struct Bucket {
    uint8 marker[Rep::kWidth];

    // Wrap keys and values in union to control construction and destruction.
    union Storage {
      struct {
        Key key[Rep::kWidth];
        Val val[Rep::kWidth];
      };
      Storage() {}
      ~Storage() {}
    } storage;

    Key& key(uint32 i) {
      DCHECK_GE(marker[i], 2);
      return storage.key[i];
    }
    Val& val(uint32 i) {
      DCHECK_GE(marker[i], 2);
      return storage.val[i];
    }
    template <typename V>
    void InitVal(uint32 i, V&& v) {
      new (&storage.val[i]) Val(std::forward<V>(v));
    }
    void Destroy(uint32 i) {
      storage.key[i].Key::~Key();
      storage.val[i].Val::~Val();
    }
    void MoveFrom(uint32 i, Bucket* src, uint32 src_index) {
      new (&storage.key[i]) Key(std::move(src->storage.key[src_index]));
      new (&storage.val[i]) Val(std::move(src->storage.val[src_index]));
    }
    void CopyFrom(uint32 i, Bucket* src, uint32 src_index) {
      new (&storage.key[i]) Key(src->storage.key[src_index]);
      new (&storage.val[i]) Val(src->storage.val[src_index]);
    }
  };

  template <typename Pair>
  std::pair<iterator, bool> InsertPair(Pair&& p) {
    return Insert(std::forward<decltype(p.first)>(p.first),
                  std::forward<decltype(p.second)>(p.second));
  }

  template <typename K, typename V>
  std::pair<iterator, bool> Insert(K&& k, V&& v) {
    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<K>(k));
    const bool inserted = !r.found;
    if (inserted) {
      r.b->InitVal(r.index, std::forward<V>(v));
    }
    return {iterator(r.b, rep_.limit(), r.index), inserted};
  }

  template <typename K>
  Val& IndexOp(K&& k) {
    rep_.MaybeResize();
    auto r = rep_.FindOrInsert(std::forward<K>(k));
    Val* vptr = &r.b->val(r.index);
    if (!r.found) {
      new (vptr) Val();  // Initialize value in new slot.
    }
    return *vptr;
  }

  Rep rep_;
};

}  // namespace gtl
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_GTL_FLATMAP_H_
