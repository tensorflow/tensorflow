/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_ITERATOR_UTIL_H_
#define XLA_ITERATOR_UTIL_H_

#include <cstddef>
#include <iterator>
#include <utility>

#include "tsl/lib/gtl/iterator_range.h"

namespace xla {

// UnwrappingIterator is a transforming iterator that calls get() on the
// elements it returns.
//
// Together with tsl::gtl::iterator_range, this lets classes which
// contain a collection of smart pointers expose a view of raw pointers to
// consumers.  For example:
//
//  class MyContainer {
//   public:
//    tsl::gtl::iterator_range<
//        UnwrappingIterator<std::vector<std::unique_ptr<Thing>>::iterator>>
//    things() {
//      return {MakeUnwrappingIterator(things_.begin()),
//              MakeUnwrappingIterator(things_.end())};
//    }
//
//    tsl::gtl::iterator_range<UnwrappingIterator<
//        std::vector<std::unique_ptr<Thing>>::const_iterator>>
//    things() const {
//      return {MakeUnwrappingIterator(things_.begin()),
//              MakeUnwrappingIterator(things_.end())};
//    }
//
//   private:
//    std::vector<std::unique_ptr<Thing>> things_;
//  };
//
//  MyContainer container = ...;
//  for (Thing* t : container.things()) {
//    ...
//  }
//
// For simplicity, UnwrappingIterator is currently unconditionally an
// input_iterator -- it doesn't inherit any superpowers NestedIterator may have.
template <typename NestedIter>
class UnwrappingIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = decltype(std::declval<NestedIter>()->get());
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit UnwrappingIterator(NestedIter iter) : iter_(std::move(iter)) {}

  auto operator*() -> value_type { return iter_->get(); }
  UnwrappingIterator& operator++() {
    ++iter_;
    return *this;
  }
  UnwrappingIterator operator++(int) {
    UnwrappingIterator temp(iter_);
    operator++();
    return temp;
  }

  friend bool operator==(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return !(a == b);
  }

 private:
  NestedIter iter_;
};

template <typename NestedIter>
UnwrappingIterator<NestedIter> MakeUnwrappingIterator(NestedIter iter) {
  return UnwrappingIterator<NestedIter>(std::move(iter));
}

// An iterator that filters out values where the predicate(value) evaluates to
// false. An unwrapping iterator can be nested inside a filtering iterator to
// also unwrap smart pointers.
template <typename NestedIter, typename UnaryPredicate>
class FilteringIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = decltype(*std::declval<NestedIter>());
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  FilteringIterator(NestedIter iter, NestedIter end_iter, UnaryPredicate pred)
      : iter_(std::move(iter)),
        end_iter_(std::move(end_iter)),
        pred_(std::move(pred)) {
    if (iter_ != end_iter_ && !pred_(**this)) {
      ++*this;
    }
  }

  auto operator*() -> value_type { return *iter_; }
  FilteringIterator& operator++() {
    do {
      ++iter_;
    } while (iter_ != end_iter_ && !pred_(**this));
    return *this;
  }
  FilteringIterator operator++(int) {
    FilteringIterator temp(iter_, end_iter_, pred_);
    operator++();
    return temp;
  }

  friend bool operator==(const FilteringIterator& a,
                         const FilteringIterator& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const FilteringIterator& a,
                         const FilteringIterator& b) {
    return !(a == b);
  }

 private:
  NestedIter iter_;
  NestedIter end_iter_;
  UnaryPredicate pred_;
};

template <typename NestedIter, typename UnaryPredicate>
using FilteringUnwrappingIterator =
    FilteringIterator<UnwrappingIterator<NestedIter>, UnaryPredicate>;

// Create and return a filtering unwrapping iterator.
template <typename NestedIter, typename UnaryPredicate>
FilteringUnwrappingIterator<NestedIter, UnaryPredicate>
MakeFilteringUnwrappingIterator(NestedIter iter, NestedIter end_iter,
                                UnaryPredicate pred) {
  return FilteringUnwrappingIterator<NestedIter, UnaryPredicate>(
      MakeUnwrappingIterator(iter), MakeUnwrappingIterator(end_iter),
      std::move(pred));
}

// Create and return a filtering unwrapping iterator range.
template <typename NestedIter, typename UnaryPredicate>
tsl::gtl::iterator_range<
    FilteringUnwrappingIterator<NestedIter, UnaryPredicate>>
MakeFilteringUnwrappingIteratorRange(NestedIter begin_iter, NestedIter end_iter,
                                     UnaryPredicate pred) {
  return {MakeFilteringUnwrappingIterator(begin_iter, end_iter, pred),
          MakeFilteringUnwrappingIterator(end_iter, end_iter, pred)};
}

}  // namespace xla

#endif  // XLA_ITERATOR_UTIL_H_
