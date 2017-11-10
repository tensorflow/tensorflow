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
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace test_utils {

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

// Convenience function for creating a rank-2 array with arbitrary layout.
template <typename NativeT>
std::unique_ptr<Literal> CreateR2LiteralWithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    tensorflow::gtl::ArraySlice<int64> minor_to_major) {
  auto literal = MakeUnique<Literal>();
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  literal.get()->PopulateWithValue<NativeT>(0, {d0, d1});
  *literal->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout(minor_to_major);
  TF_CHECK_OK(ShapeUtil::ValidateShape(literal->shape()));

  int64 dim0 = 0;
  for (auto inner_list : values) {
    int64 dim1 = 0;
    for (auto value : inner_list) {
      literal.get()->Set({dim0, dim1}, value);
      ++dim1;
    }
    ++dim0;
  }
  return literal;
}

// Convenience function for creating a rank-3 array with arbitrary layout.
template <typename NativeT>
std::unique_ptr<Literal> CreateR3LiteralWithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    tensorflow::gtl::ArraySlice<int64> minor_to_major) {
  auto literal = MakeUnique<Literal>();
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  const int64 d2 = values.begin()->begin()->size();
  literal.get()->PopulateWithValue<NativeT>(0, {d0, d1, d2});
  *literal->mutable_shape()->mutable_layout() =
      LayoutUtil::MakeLayout(minor_to_major);
  TF_CHECK_OK(ShapeUtil::ValidateShape(literal->shape()));

  int64 dim0 = 0;
  for (auto inner_list : values) {
    int64 dim1 = 0;
    for (auto inner_inner_list : inner_list) {
      int64 dim2 = 0;
      for (auto value : inner_inner_list) {
        literal.get()->Set({dim0, dim1, dim2}, value);
        ++dim2;
      }
      ++dim1;
    }
    ++dim0;
  }
  return literal;
}

}  // namespace test_utils
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TESTS_TEST_UTILS_H_
