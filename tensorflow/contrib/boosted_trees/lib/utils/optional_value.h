// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_OPTIONAL_VALUE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_OPTIONAL_VALUE_H_

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// Utility class holding an optional value.
template <typename T>
class OptionalValue {
 public:
  OptionalValue() : value_(), has_value_(false) {}
  explicit OptionalValue(T value) : value_(value), has_value_(true) {}

  bool has_value() const { return has_value_; }
  const T& get_value() const {
    QCHECK(has_value());
    return value_;
  }

 private:
  T value_;
  bool has_value_;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_OPTIONAL_VALUE_H_
