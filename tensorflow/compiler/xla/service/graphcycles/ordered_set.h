/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/tsl/platform/logging.h"

namespace tensorflow {
// This is a set data structure that provides a deterministic iteration order.
// The iteration order of elements only depends on the sequence of
// inserts/deletes, so as long as the inserts/deletes happen in the same
// sequence, the set will have the same iteration order.
//
// Assumes that T can be cheaply copied for simplicity.
template <typename T>
class OrderedSet {
 public:
  // Inserts `value` into the ordered set.  Returns true if the value was not
  // present in the set before the insertion.
  bool Insert(T value) {
    bool new_insertion =
        value_to_index_.insert({value, value_sequence_.size()}).second;
    if (new_insertion) {
      value_sequence_.push_back(value);
    }
    return new_insertion;
  }

  // Removes `value` from the set.  Assumes `value` is already present in the
  // set.
  void Erase(T value) {
    auto it = value_to_index_.find(value);
    DCHECK(it != value_to_index_.end());

    // Since we don't want to move values around in `value_sequence_` we swap
    // the value in the last position and with value to be deleted and then
    // pop_back.
    value_to_index_[value_sequence_.back()] = it->second;
    std::swap(value_sequence_[it->second], value_sequence_.back());
    value_sequence_.pop_back();
    value_to_index_.erase(it);
  }

  void Reserve(size_t new_size) {
    value_to_index_.reserve(new_size);
    value_sequence_.reserve(new_size);
  }

  void Clear() {
    value_to_index_.clear();
    value_sequence_.clear();
  }

  bool Contains(T value) const { return value_to_index_.contains(value); }
  size_t Size() const { return value_sequence_.size(); }

  absl::Span<T const> GetSequence() const { return value_sequence_; }

 private:
  // The stable order that we maintain through insertions and deletions.
  std::vector<T> value_sequence_;

  // Maps values to their indices in `value_sequence_`.
  absl::flat_hash_map<T, int> value_to_index_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GRAPHCYCLES_ORDERED_SET_H_
