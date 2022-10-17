/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

TEST(TensorUtilsTest, AllElementsPositive) {
  EXPECT_TRUE(AllElementsPositive<int32_t>(
      test::AsTensor<int32_t>({1, 2, 3, 4, 5}, {5})));
  EXPECT_FALSE(AllElementsPositive<int32_t>(
      test::AsTensor<int32_t>({1, 2, 0, 4, 5}, {5})));
  EXPECT_FALSE(AllElementsPositive<int32_t>(
      test::AsTensor<int32_t>({1, 2, -2, 4, 5}, {5})));
}

TEST(TensorUtilsTest, QuantizationAxisAndShapeValid) {
  TF_EXPECT_OK(QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                             /*scales_shape=*/{3},
                                             /*zero_points_shape=*/{3},
                                             /*quantization_axis=*/1));
  TF_EXPECT_OK(QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                             /*scales_shape=*/{},
                                             /*zero_points_shape=*/{},
                                             /*quantization_axis=*/-1));

  EXPECT_TRUE(errors::IsInvalidArgument(
      QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                    /*scales_shape=*/{3},
                                    /*zero_points_shape=*/{2},
                                    /*quantization_axis=*/1)));
  EXPECT_TRUE(errors::IsInvalidArgument(
      QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                    /*scales_shape=*/{3},
                                    /*zero_points_shape=*/{3},
                                    /*quantization_axis=*/3)));
  EXPECT_TRUE(errors::IsInvalidArgument(
      QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                    /*scales_shape=*/{3},
                                    /*zero_points_shape=*/{3},
                                    /*quantization_axis=*/-1)));
  EXPECT_TRUE(errors::IsInvalidArgument(
      QuantizationAxisAndShapeValid(/*data_shape=*/{2, 3, 4},
                                    /*scales_shape=*/{5},
                                    /*zero_points_shape=*/{5},
                                    /*quantization_axis=*/1)));
}

TEST(TensorUtilsTest, TransposedShape) {
  EXPECT_EQ(TransposedShape({2, 3, 4, 5}, {1, 2, 3, 0}),
            TensorShape({3, 4, 5, 2}));
}

TEST(TensorUtilsTest, Transpose) {
  const std::vector<int32_t> perm = {1, 2, 0};
  const TensorShape shape({2, 3, 4});
  const TensorShape transposed_shape = TransposedShape(shape, perm);
  Tensor transposed_tensor = test::AsTensor<int32_t>(
      std::vector<int32_t>(2 * 3 * 4, 0), transposed_shape);
  Transpose<int32_t>(
      test::AsTensor<int32_t>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                              shape),
      perm, transposed_tensor);
  test::ExpectTensorEqual<int32_t>(
      transposed_tensor,
      test::AsTensor<int32_t>({0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                               6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23},
                              transposed_shape));
}

}  // namespace tensorflow
