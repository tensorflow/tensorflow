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

#ifndef TENSORFLOW_GRAPH_EDGESET_H_
#define TENSORFLOW_GRAPH_EDGESET_H_

#include <stddef.h>
#include <set>
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

class Edge;

// An unordered set of edges.  Uses very little memory for small sets.
// Unlike std::set, EdgeSet does NOT allow mutations during iteration.
class EdgeSet {
 public:
  EdgeSet();
  ~EdgeSet();

  typedef const Edge* key_type;
  typedef const Edge* value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  class const_iterator;
  typedef const_iterator iterator;

  bool empty() const;
  size_type size() const;
  void clear();
  std::pair<iterator, bool> insert(value_type value);
  size_type erase(key_type key);

  // Caller is not allowed to mutate the EdgeSet while iterating.
  const_iterator begin() const;
  const_iterator end() const;

 private:
  // Up to kInline elements are stored directly in ptrs_ (nullptr means none).
  // If ptrs_[0] == this then ptrs_[1] points to a set<const Edge*>.
  static const int kInline = 4;  // Must be >= 2.
  const void* ptrs_[kInline];

  std::set<const Edge*>* get_set() const {
    if (ptrs_[0] == this) {
      return static_cast<std::set<const Edge*>*>(const_cast<void*>(ptrs_[1]));
    } else {
      return nullptr;
    }
  }

// To detect mutations while iterating.
#ifdef NDEBUG
  void RegisterMutation() {}
#else
  uint32 mutations_ = 0;
  void RegisterMutation() { mutations_++; }
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(EdgeSet);
};

class EdgeSet::const_iterator {
 public:
  typedef typename EdgeSet::value_type value_type;
  typedef const typename EdgeSet::value_type& reference;
  typedef const typename EdgeSet::value_type* pointer;
  typedef typename EdgeSet::difference_type difference_type;
  typedef std::forward_iterator_tag iterator_category;

  const_iterator() {}

  const_iterator& operator++();
  const_iterator operator++(int /*unused*/);
  const value_type* operator->() const;
  value_type operator*() const;
  bool operator==(const const_iterator& other) const;
  bool operator!=(const const_iterator& other) const {
    return !(*this == other);
  }

 private:
  friend class EdgeSet;

  void const* const* array_iter_ = nullptr;
  typename std::set<const Edge*>::const_iterator tree_iter_;

#ifdef NDEBUG
  inline void Init(const EdgeSet* e) {}
  inline void CheckNoMutations() const {}
#else
  inline void Init(const EdgeSet* e) {
    owner_ = e;
    init_mutations_ = e->mutations_;
  }
  inline void CheckNoMutations() const {
    CHECK_EQ(init_mutations_, owner_->mutations_);
  }
  const EdgeSet* owner_ = nullptr;
  uint32 init_mutations_ = 0;
#endif
};

inline EdgeSet::EdgeSet() {
  for (int i = 0; i < kInline; i++) {
    ptrs_[i] = nullptr;
  }
}

inline EdgeSet::~EdgeSet() { delete get_set(); }

inline bool EdgeSet::empty() const { return size() == 0; }

inline EdgeSet::size_type EdgeSet::size() const {
  auto s = get_set();
  if (s) {
    return s->size();
  } else {
    size_t result = 0;
    for (int i = 0; i < kInline; i++) {
      if (ptrs_[i]) result++;
    }
    return result;
  }
}

inline void EdgeSet::clear() {
  RegisterMutation();
  delete get_set();
  for (int i = 0; i < kInline; i++) {
    ptrs_[i] = nullptr;
  }
}

inline EdgeSet::const_iterator EdgeSet::begin() const {
  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (s) {
    ci.tree_iter_ = s->begin();
  } else {
    ci.array_iter_ = &ptrs_[0];
  }
  return ci;
}

inline EdgeSet::const_iterator EdgeSet::end() const {
  const_iterator ci;
  ci.Init(this);
  auto s = get_set();
  if (s) {
    ci.tree_iter_ = s->end();
  } else {
    ci.array_iter_ = &ptrs_[size()];
  }
  return ci;
}

inline EdgeSet::const_iterator& EdgeSet::const_iterator::operator++() {
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    ++array_iter_;
  } else {
    ++tree_iter_;
  }
  return *this;
}

inline EdgeSet::const_iterator EdgeSet::const_iterator::operator++(
    int /*unused*/) {
  CheckNoMutations();
  const_iterator tmp = *this;
  operator++();
  return tmp;
}

// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.
inline const EdgeSet::const_iterator::value_type* EdgeSet::const_iterator::
operator->() const {
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return reinterpret_cast<const value_type*>(array_iter_);
  } else {
    return tree_iter_.operator->();
  }
}

// gcc's set and multiset always use const_iterator since it will otherwise
// allow modification of keys.
inline EdgeSet::const_iterator::value_type EdgeSet::const_iterator::operator*()
    const {
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return static_cast<value_type>(*array_iter_);
  } else {
    return *tree_iter_;
  }
}

inline bool EdgeSet::const_iterator::operator==(
    const const_iterator& other) const {
  DCHECK((array_iter_ == nullptr) == (other.array_iter_ == nullptr))
      << "Iterators being compared must be from same set that has not "
      << "been modified since the iterator was constructed";
  CheckNoMutations();
  if (array_iter_ != nullptr) {
    return array_iter_ == other.array_iter_;
  } else {
    return other.array_iter_ == nullptr && tree_iter_ == other.tree_iter_;
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_EDGESET_H_
