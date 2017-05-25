/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/winograd_transform.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void ComputeKroneckerProduct(const int rows, const int cols,
                                    const float* matrix, float* matrix_out) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float v = matrix[i * cols + j];
      const int output_index_base = cols * (i * rows * cols + j);
      for (int k = 0; k < rows; ++k) {
        for (int l = 0; l < cols; ++l) {
          const int input_index = k * cols + l;
          const int output_index = k * cols * cols + l;
          matrix_out[output_index_base + output_index] =
              matrix[input_index] * v;
        }
      }
    }
  }
}

TEST(DeepConv2DTransformTest, Basic) {
  // Tests kronecker product of the following matrix with itself:
  //
  // [1.0 2.0]
  // [3.0 4.0]
  //
  const int rows = 2;
  const int cols = 2;

  float transform_matrix[] = {1, 2, 3, 4};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;
  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[] = {1, 2, 2, 4, 3, 4,  6,  8,
                                   3, 6, 4, 8, 9, 12, 12, 16};

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradFilterTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix with itself:
  //
  //   [ 1    0   0   ]
  //   [ 1/2  1/2 1/2 ]
  //   [ 1/2 -1/2 1/2 ]
  //   [ 0    0   1   ]
  //
  const int rows = 4;
  const int cols = 3;

  float transform_matrix[] = {1, 0, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0, 0, 1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetFilterTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradInputTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix:
  //
  //   [1   0  -1   0]
  //   [0   1   1   0]
  //   [0  -1   1   0]
  //   [0   1   0  -1]
  //
  const int rows = 4;
  const int cols = 4;

  float transform_matrix[] = {1, 0,  -1, 0, 0, 1, 1, 0,
                              0, -1, 1,  0, 0, 1, 0, -1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetInputTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

TEST(DeepConv2DTransformTest, WingradOutputTransformMatrix) {
  // Test that the filter transform matrix returned is the kronecker product of
  // the following matrix:
  //
  //   [1  1  1  0]
  //   [0  1 -1 -1]
  //
  const int rows = 2;
  const int cols = 4;

  float transform_matrix[] = {1, 1, 1, 0, 0, 1, -1, -1};

  const int kron_rows = rows * rows;
  const int kron_cols = cols * cols;

  float transform_matrix_kron[kron_rows * kron_cols];

  ComputeKroneckerProduct(rows, cols, &transform_matrix[0],
                          &transform_matrix_kron[0]);

  float transform_matrix_test[kron_rows * kron_cols];
  WinogradTransform<float> t;
  t.GetOutputTransformMatrix(kron_rows, kron_cols, &transform_matrix_test[0]);

  for (int i = 0; i < kron_rows * kron_cols; ++i) {
    EXPECT_FLOAT_EQ(transform_matrix_kron[i], transform_matrix_test[i]);
  }
}

}  // namespace
}  // namespace tensorflow
