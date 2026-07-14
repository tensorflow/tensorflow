/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_LIB_GTL_COMPACTPTRSET_H_
#define XLA_TSL_LIB_GTL_COMPACTPTRSET_H_

#include <cstdint>
#include <type_traits>

#include "xla/tsl/lib/gtl/flatset.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace tsl {
namespace gtl {

// CompactPointerSet<T> is like a std::unordered_set<T> but is optimized
// for small sets (<= 1 element).  T must be a pointer type.
template <typename T>
class CompactPointerSet {
 private:
  using BigRep = FlatSet<T>;

 public:
  using value_type = T;

  CompactPointerSet() : rep_(0) {}

  ~CompactPointerSet() {
    static_assert(
        std::is_pointer<T>::value,
        "CompactPointerSet<T> can only be used with T's that are pointers");
    if (isbig()) delete big();
  }

  CompactPointerSet(const CompactPointerSet& other) : rep_(0) { *this = other; }

  CompactPointerSet& operator=(const CompactPointerSet& other) {
    if (this == &other) return *this;
    if (other.isbig()) {
      // big => any
      if (!isbig()) MakeBig();
      *big() = *other.big();
    } else if (isbig()) {
      // !big => big
      big()->clear();
      if (other.rep_ != 0) {
        big()->insert(reinterpret_cast<T>(other.rep_));
      }
    } else {
      // !big => !big
      rep_ = other.rep_;
    }
    return *this;
  }

  class iterator {
   public:
    typedef ssize_t difference_type;
    typedef T value_type;
    typedef const T* pointer;
    typedef const T& reference;
    typedef ::std::forward_iterator_tag iterator_category;

    explicit iterator(uintptr_t rep)
        : bigrep_(false), single_(reinterpret_cast<T>(rep)) {}
    explicit iterator(typename BigRep::iterator iter)
        : bigrep_(true), single_(nullptr), iter_(iter) {}

    iterator& operator++() {
      if (bigrep_) {
        ++iter_;
      } else {
        DCHECK(single_ != nullptr);
        single_ = nullptr;
      }
      return *this;
    }
    // maybe post-increment?

    bool operator==(const iterator& other) const {
      if (bigrep_) {
        return iter_ == other.iter_;
      } else {
        return single_ == other.single_;
      }
    }
    bool operator!=(const iterator& other) const { return !(*this == other); }

    const T& operator*() const {
      if (bigrep_) {
        return *iter_;
      } else {
        DCHECK(single_ != nullptr);
        return single_;
      }
    }

   private:
    friend class CompactPointerSet;
    bool bigrep_;
    T single_;
    typename BigRep::iterator iter_;
  };
  using const_iterator = iterator;

  bool empty() const { return isbig() ? big()->empty() : (rep_ == 0); }
  size_t size() const { return isbig() ? big()->size() : (rep_ == 0 ? 0 : 1); }

  void clear() {
    if (isbig()) {
      delete big();
    }
    rep_ = 0;
  }

  std::pair<iterator, bool> insert(T elem) {
    if (!isbig()) {
      if (rep_ == 0) {
        uintptr_t v = safe_reinterpret_cast<std::uintptr_t>(elem);
        if (v == 0 || ((v & 0x3) != 0)) {
          // Cannot use small representation for nullptr.  Fall through.
        } else {
          rep_ = v;
          return {iterator(v), true};
        }
      }
      MakeBig();
    }
    auto p = big()->insert(elem);
    return {iterator(p.first), p.second};
  }

  template <typename InputIter>
  void insert(InputIter begin, InputIter end) {
    for (; begin != end; ++begin) {
      insert(*begin);
    }
  }

  const_iterator begin() const {
    return isbig() ? iterator(big()->begin()) : iterator(rep_);
  }
  const_iterator end() const {
    return isbig() ? iterator(big()->end()) : iterator(0);
  }

  iterator find(T elem) const {
    if (rep_ == safe_reinterpret_cast<std::uintptr_t>(elem)) {
      return iterator(rep_);
    } else if (!isbig()) {
      return iterator(0);
    } else {
      return iterator(big()->find(elem));
    }
  }

  size_t count(T elem) const { return find(elem) != end() ? 1 : 0; }

  size_t erase(T elem) {
    if (!isbig()) {
      if (rep_ == safe_reinterpret_cast<std::uintptr_t>(elem)) {
        rep_ = 0;
        return 1;
      } else {
        return 0;
      }
    } else {
      return big()->erase(elem);
    }
  }

 private:
  // Size         rep_
  // -------------------------------------------------------------------------
  // 0            0
  // 1            The pointer itself (bottom bits == 00)
  // large        Pointer to a BigRep (bottom bits == 01)
  uintptr_t rep_;

  bool isbig() const { return (rep_ & 0x3) == 1; }
  BigRep* big() const {
    DCHECK(isbig());
    return reinterpret_cast<BigRep*>(rep_ - 1);
  }

  void MakeBig() {
    DCHECK(!isbig());
    BigRep* big = new BigRep;
    if (rep_ != 0) {
      big->insert(reinterpret_cast<T>(rep_));
    }
    rep_ = safe_reinterpret_cast<std::uintptr_t>(big) + 0x1;
  }
};

}  // namespace gtl
}  // namespace tsl

#endif  // XLA_TSL_LIB_GTL_COMPACTPTRSET_H_
