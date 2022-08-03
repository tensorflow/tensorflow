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
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {

TEST(MathUtilsTest, AffineQuantize) {
  TensorShape shape({2, 2, 2});
  Tensor tensor = test::AsTensor<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 70.0f, 80.0f}, shape);
  Tensor quantized_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 0, 0, 0, 0}, shape);

  // Quantizes only the part [1:2, 0:2, 0:1].
  Eigen::DSizes<Eigen::Index, 3> start_indices{1, 0, 0};
  Eigen::DSizes<Eigen::Index, 3> sizes{1, 2, 2};
  auto tensor_slice = tensor.tensor<float, 3>().slice(start_indices, sizes);
  auto quantized_tensor_slice =
      quantized_tensor.tensor<qint8, 3>().slice(start_indices, sizes);

  AffineQuantize(tensor_slice, /*inv_scale=*/0.5f, /*zero_point=*/3,
                 /*quantization_min_val=*/-128, /*quantization_max_val=*/40,
                 quantized_tensor_slice);

  // 5.0f (scale=2.0, zero_point=3) is quantized to 6, since the formula is
  // floor((5.0f * 0.5) + 0.5) + 3. 80.0f is quantized to 40 (Not 43), since
  // quantization_max_val is set to 40.
  Tensor expected_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 6, 6, 38, 40}, shape);
  test::ExpectEqual(quantized_tensor, expected_tensor);
}

TEST(MathUtilsTest, AffineDequantize) {
  TensorShape shape({2, 2, 2});
  Tensor tensor = test::AsTensor<qint8>({10, 15, 20, 25, -10, -5, 0, 5}, shape);
  Tensor dequantized_tensor =
      test::AsTensor<float>({0, 0, 0, 0, 0, 0, 0, 0}, shape);

  // Dequantizes only the part [1:2, 0:2, 0:1].
  Eigen::DSizes<Eigen::Index, 3> start_indices{1, 0, 0};
  Eigen::DSizes<Eigen::Index, 3> sizes{1, 2, 2};
  auto tensor_slice = tensor.tensor<qint8, 3>().slice(start_indices, sizes);
  auto dequantized_tensor_slice =
      dequantized_tensor.tensor<float, 3>().slice(start_indices, sizes);

  AffineDequantize(tensor_slice, /*scale=*/2.0f, /*zero_point=*/3,
                   dequantized_tensor_slice);

  Tensor expected_tensor =
      test::AsTensor<float>({0, 0, 0, 0, -26.0, -16.0, -6.0, 4.0}, shape);
  test::ExpectTensorNear<float>(dequantized_tensor, expected_tensor, 1e-6);
}

TEST(MathUtilsTest, AsymmetricQuantize) {
  float scale;
  int32_t zero_point;
  TensorShape shape({2, 2, 2});
  Tensor quantized_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 0, 0, 0, 0}, shape);

  AsymmetricQuantize(
      test::AsTensor<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                            shape),
      /*apply_offset=*/4, /*apply_size=*/4, /*quantization_min_val=*/-128,
      /*quantization_max_val=*/127, scale, zero_point, quantized_tensor);

  // Only flattened_tensor[apply_offset : apply_offset + apply_size] is
  // quantized.
  Tensor expected_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 31, 63, 95, 127}, shape);
  test::ExpectEqual(quantized_tensor, expected_tensor);
  EXPECT_FLOAT_EQ(scale, 0.031372551f);
  EXPECT_EQ(zero_point, -128);
}

TEST(MathUtilsTest, AsymmetricQuantizeZeroValuesTensor) {
  float scale;
  int32_t zero_point;
  TensorShape shape({2, 2, 2});
  Tensor quantized_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 0, 0, 0, 0}, shape);

  AsymmetricQuantize(
      test::AsTensor<float>({0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f},
                            shape),
      /*apply_offset=*/0, /*apply_size=*/4, /*quantization_min_val=*/-128,
      /*quantization_max_val=*/127, scale, zero_point, quantized_tensor);

  // All values in flattened_tensor[apply_offset : apply_offset + apply_size]
  // are zero, Thus all the quantized output values are zero. Scale and
  // zero_point is set to 1.0f and 0 respectively.
  Tensor expected_tensor =
      test::AsTensor<qint8>({0, 0, 0, 0, 0, 0, 0, 0}, shape);
  test::ExpectEqual(quantized_tensor, expected_tensor);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(zero_point, 0);
}

}  // namespace tensorflow
