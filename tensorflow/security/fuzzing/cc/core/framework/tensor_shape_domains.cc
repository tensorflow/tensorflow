/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tsl/platform/errors.h"

namespace tensorflow::fuzzing {
namespace {

using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::InRange;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

Domain<absl::StatusOr<TensorShape>> AnyStatusOrTensorShape(
    size_t max_rank, int64_t dim_lower_bound, int64_t dim_upper_bound) {
  return Map(
      [](std::vector<int64_t> v) -> absl::StatusOr<TensorShape> {
        TensorShape out;
        TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(v, &out));
        return out;
      },
      VectorOf(InRange(dim_lower_bound, dim_upper_bound))
          .WithMaxSize(max_rank));
}

}  // namespace

Domain<TensorShape> AnyValidTensorShape(
    size_t max_rank = std::numeric_limits<size_t>::max(),
    int64_t dim_lower_bound = std::numeric_limits<int64_t>::min(),
    int64_t dim_upper_bound = std::numeric_limits<int64_t>::max()) {
  return Map([](absl::StatusOr<TensorShape> t) { return *t; },
             Filter([](auto t_inner) { return t_inner.status().ok(); },
                    AnyStatusOrTensorShape(max_rank, dim_lower_bound,
                                           dim_upper_bound)));
}

}  // namespace tensorflow::fuzzing
