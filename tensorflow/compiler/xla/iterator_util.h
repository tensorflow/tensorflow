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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_ITERATOR_UTIL_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_ITERATOR_UTIL_H_

#include <iterator>
#include <utility>

namespace xla {

// UnwrappingIterator is a transforming iterator that calls get() on the
// elements it returns.
//
// Together with tensorflow::gtl::iterator_range, this lets classes which
// contain a collection of smart pointers expose a view of raw pointers to
// consumers.  For example:
//
//  class MyContainer {
//   public:
//    tensorflow::gtl::iterator_range<
//        UnwrappingIterator<std::vector<std::unique_ptr<Thing>>::iterator>>
//    things() {
//      return {MakeUnwrappingIterator(things_.begin()),
//              MakeUnwrappingIterator(things_.end())};
//    }
//
//    tensorflow::gtl::iterator_range<UnwrappingIterator<
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
class UnwrappingIterator
    : public std::iterator<std::input_iterator_tag,
                           decltype(std::declval<NestedIter>()->get())> {
 private:
  NestedIter iter_;

 public:
  explicit UnwrappingIterator(NestedIter iter) : iter_(std::move(iter)) {}

  auto operator*() -> decltype(iter_->get()) { return iter_->get(); }
  auto operator-> () -> decltype(iter_->get()) { return iter_->get(); }
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
};

template <typename NestedIter>
UnwrappingIterator<NestedIter> MakeUnwrappingIterator(NestedIter iter) {
  return UnwrappingIterator<NestedIter>(std::move(iter));
}

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_ITERATOR_UTIL_H_
