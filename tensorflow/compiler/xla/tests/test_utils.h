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
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A class which generates pseudorandom numbers of a given type within a given
// range. Not cryptographically secure and likely not perfectly evenly
// distributed across the range but sufficient for most tests.
template <typename NativeT>
class PseudorandomGenerator {
 public:
  explicit PseudorandomGenerator(NativeT min_value, NativeT max_value,
                                 uint32_t seed)
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
// status if the element type is currently unhandled for fake data
// generation. See below for documentation of pseudo_random and use_large_range.
StatusOr<Literal> MakeFakeLiteral(const Shape& shape, bool pseudo_random = true,
                                  bool use_large_range = false);

// Generates a vector of arguments containing fake data. The number, shape and
// layout of the arguments is appropriate for given HLO module.
//
// A best-effort attempt is made to generate the data in a way which produce
// stable computation results across platforms. Specifically:
//
//  (1) Init values of reductions should be the identity of the reduction
//  computation.
//
//  (2) Indices of dynamic slices and update slices should be in bounds.
//
//  (3) Keys of key/value sorts should contain no duplicates.
//
// These constraints are best-effort only.
//
// If pseudo_random is true, the generated numbers will be generated
// deterministically in a pseudo random way unless the values are constrated to
// be e.g. init values as above. If pseudo_random is false, the returned values
// will be generated in a faster way that yields less interesting data, e.g. the
// values may all be just the same value.
//
// If use_large_range is false, the generated floating point numbers will be
// sampled from a small range of possible values. If use_large_range is true,
// the generated floating point numbers will be sampled from a uniform-log
// distribution of most possible floats, with a small chance to instead be
// sampled from a list of special floating point values (such as 0, inf, etc.).
//
// TODO(b/79942829): Make interesting argument generation fast enough that using
// pseudo_random does not save any noticeable amount of time so that the
// parameter can be removed.
StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, bool pseudo_random = true,
    bool use_large_range = false, bool treat_gte_as_data_formatting = false);

// Overload which accepts a random number generator. This enables generation of
// different random values with sequential calls to MakeFakeArguments by reusing
// the same generator.
StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, std::minstd_rand0* engine,
    bool use_large_range = false, bool treat_gte_as_data_formatting = false);

// Check that a given module satisfies various constraints before trying to
// execute it.
Status VerifyHloModule(HloModule* const module, bool layout_sensitive,
                       bool allow_mixed_precision);

// Creates a dot op with operands 'lhs' and 'rhs' that contracts dimension 1 of
// the LHS with dimension 0 of the RHS with no batch dimensions.
// Both LHS and the RHS must be of rank 2.
std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs);

// Checks whether MLIR lowering is enabled through XLA_FLAGS.
bool IsMlirLoweringEnabled();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
