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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIDE_PAIR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIDE_PAIR_H_

#include "tensorflow/lite/experimental/ruy/check_macros.h"

namespace ruy {

enum class Side { kLhs = 0, kRhs = 1 };

template <typename T>
class SidePair final {
 public:
  SidePair() {}
  SidePair(const T& a, const T& b) : elem_{a, b} {}
  const T& operator[](Side side) const {
    const int index = static_cast<int>(side);
    // Technically this check is vacuous, since other values would be
    // out-of-range for enum Side.
    RUY_DCHECK(index == 0 || index == 1);
    return elem_[index];
  }

  T& operator[](Side side) {
    const int index = static_cast<int>(side);
    // Technically this check is vacuous, since other values would be
    // out-of-range for enum Side.
    RUY_DCHECK(index == 0 || index == 1);
    return elem_[index];
  }

 private:
  static_assert(static_cast<int>(Side::kLhs) == 0, "");
  static_assert(static_cast<int>(Side::kRhs) == 1, "");
  T elem_[2];
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_SIDE_PAIR_H_
