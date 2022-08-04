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

}  // namespace tensorflow
