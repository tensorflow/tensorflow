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

#ifndef TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_

#include <initializer_list>
#include <memory>
#include <random>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {

// A class which generates pseudorandom numbers of a given type within a given
// range. Not cryptographically secure and likely not perfectly evenly
// distributed across the range but sufficient for most tests.
template <typename NativeT>
class PseudorandomGenerator {
 public:
  explicit PseudorandomGenerator(NativeT min_value, NativeT max_value,
                                 uint32 seed)
      : min_(min_value), max_(max_value), generator_(seed) {}

  // Get a pseudorandom value.
  NativeT get() {
    std::uniform_real_distribution<> distribution;
    return static_cast<NativeT>(min_ +
                                (max_ - min_) * distribution(generator_));
  }

 private:
  NativeT min_;
  NativeT max_;
  std::mt19937 generator_;
};

// Generates fake data in a literal of the given shape, or returns an error
// status if the element type is currently unhandled for fake data generation.
StatusOr<std::unique_ptr<Literal>> MakeFakeLiteral(const Shape& shape);

// Generates a vector of arguments containing fake data. The number, shape and
// layout of the arguments is appropriate for given HLO module.
//
// Will handle special cases such as making sure that indices used for dynamic
// slices are bounded, reduces that call adds use 0 as an init value, etc.
StatusOr<std::vector<std::unique_ptr<Literal>>> MakeFakeArguments(
    HloModule* const module);

// Check that a given module satisfies various constraints before trying to
// execute it.
Status VerifyHloModule(HloModule* const module,
                       bool allow_mixed_precision = false);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
