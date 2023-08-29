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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace fuzzing {
namespace {

void FuzzTensorShape(const std::vector<int64_t>& dim_sizes) {
  TensorShape out;
  Status status = TensorShape::BuildTensorShape(dim_sizes, &out);
  if (!dim_sizes.empty() && dim_sizes.size() < 5) {
    const auto [min, max] =
        std::minmax_element(dim_sizes.begin(), dim_sizes.end());
    if (*max < 10 && *min >= 0) {
      TF_EXPECT_OK(status);
      EXPECT_EQ(out.dims(), dim_sizes.size());
    }
  }
}
FUZZ_TEST(TensorShapeFuzz, FuzzTensorShape);

void FuzzPartialTensorShape(const std::vector<int64_t>& dim_sizes) {
  PartialTensorShape out;
  Status status = PartialTensorShape::BuildPartialTensorShape(dim_sizes, &out);
  if (!dim_sizes.empty() && dim_sizes.size() < 5) {
    const auto [min, max] =
        std::minmax_element(dim_sizes.begin(), dim_sizes.end());
    if (*max < 10 && *min >= -10) {
      TF_EXPECT_OK(status);
      EXPECT_EQ(out.dims(), dim_sizes.size());
    }
  }
}
FUZZ_TEST(TensorShapeFuzz, FuzzPartialTensorShape);

void FuzzSetDimWithStatus(TensorShape shape, int dim, int64_t value) {
  int initial_rank = shape.dims();
  bool should_be_ok = shape.dims() == 2 && shape.dim_size(0) <= 100 &&
                      shape.dim_size(1) <= 100 && dim < 2 && value < 100;
  Status status = shape.SetDimWithStatus(dim, value);
  if (status.ok()) {
    EXPECT_EQ(initial_rank, shape.dims());
    EXPECT_EQ(value, shape.dim_size(dim));
  } else {
    EXPECT_FALSE(should_be_ok);
  }
}
FUZZ_TEST(TensorShapeFuzz, FuzzSetDimWithStatus)
    .WithDomains(AnyValidTensorShape(), fuzztest::InRange<int>(0, 10),
                 fuzztest::InRange<int64_t>(0, 1000));

void FuzzRemoveDimWithStatus(TensorShape shape, int dim) {
  auto initial_rank = shape.dims();
  bool should_be_ok = shape.dims() == 2 && shape.dim_size(0) <= 100 &&
                      shape.dim_size(1) <= 100 && dim >= 0 && dim < 2;
  Status status = shape.RemoveDimWithStatus(dim);
  if (status.ok()) {
    EXPECT_EQ(shape.dims(), initial_rank - 1);
  } else {
    EXPECT_FALSE(should_be_ok);
  }
}
FUZZ_TEST(TensorShapeFuzz, FuzzRemoveDimWithStatus)
    .WithDomains(AnyValidTensorShape(), fuzztest::Arbitrary<int>());

}  // namespace
}  // namespace fuzzing
}  // namespace tensorflow
