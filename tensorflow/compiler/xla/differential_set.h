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

#ifndef TENSORFLOW_COMPILER_XLA_DIFFERENTIAL_SET_H_
#define TENSORFLOW_COMPILER_XLA_DIFFERENTIAL_SET_H_

#include <unordered_set>

#include "tensorflow/core/platform/macros.h"

namespace xla {

// In the base case, the differential set is just a set.
// However, you can also point a differential set at another differential set to
// use as a "parent". This makes a chain of sets, which each node in the chain
// adds some number of elements to the "Contains" property.
//
// E.g. if the base set holds {1, 2}, you can create a derived set that holds
// {3}, and the derived set will tell you it contains {1, 2, 3} whereas the base
// will continue to tell you it holds only {1, 2}.
template <typename T>
class DifferentialSet {
 public:
  // Constructs a differential set capable of holding values. It may have an
  // ancestor link, which would it into a chain of sets.
  explicit DifferentialSet(const DifferentialSet* parent = nullptr)
      : parent_(parent) {}

  // Adds a value to be held directly by this set.
  void Add(T value) { held_.insert(value); }

  // Returns whether this set holds the given value, or any ancestor in the
  // chain of sets.
  bool Contains(T value) const {
    return held_.find(value) != held_.end() ||
           (parent_ != nullptr && parent_->Contains(value));
  }

 private:
  // Values held directly by this node in the chain of sets.
  std::unordered_set<T> held_;

  // Parent node in the chain of sets.
  const DifferentialSet* parent_;

  TF_DISALLOW_COPY_AND_ASSIGN(DifferentialSet);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_DIFFERENTIAL_SET_H_
