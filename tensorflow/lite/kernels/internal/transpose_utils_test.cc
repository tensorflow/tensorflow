/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/transpose_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_1DNoChanges) {
  RuntimeShape input_shape({9});
  RuntimeShape output_shape({9});

  TransposeParams params;
  params.perm_count = 1;
  params.perm[0] = 0;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({9}));
  EXPECT_EQ(output_shape, RuntimeShape({9}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_2DNoChanges) {
  RuntimeShape input_shape({9, 3});
  RuntimeShape output_shape({3, 9});

  TransposeParams params;
  params.perm_count = 2;
  params.perm[0] = 1;
  params.perm[1] = 0;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({9, 3}));
  EXPECT_EQ(output_shape, RuntimeShape({3, 9}));

  EXPECT_EQ(params.perm_count, 2);
  EXPECT_EQ(params.perm[0], 1);
  EXPECT_EQ(params.perm[1], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_2DShrinking) {
  RuntimeShape input_shape({9, 1});
  RuntimeShape output_shape({1, 9});

  TransposeParams params;
  params.perm_count = 2;
  params.perm[0] = 1;
  params.perm[1] = 0;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({9}));
  EXPECT_EQ(output_shape, RuntimeShape({9}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_3DNoChanges) {
  RuntimeShape input_shape({4, 3, 8});
  RuntimeShape output_shape({8, 4, 3});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 0;
  params.perm[2] = 1;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({4, 3, 8}));
  EXPECT_EQ(output_shape, RuntimeShape({8, 4, 3}));

  EXPECT_EQ(params.perm_count, 3);
  EXPECT_EQ(params.perm[0], 2);
  EXPECT_EQ(params.perm[1], 0);
  EXPECT_EQ(params.perm[2], 1);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_3DShrinkingOnce) {
  RuntimeShape input_shape({4, 1, 8});
  RuntimeShape output_shape({8, 4, 1});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 0;
  params.perm[2] = 1;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({4, 8}));
  EXPECT_EQ(output_shape, RuntimeShape({8, 4}));
  EXPECT_EQ(output_shape.Dims(1), 4);

  EXPECT_EQ(params.perm_count, 2);
  EXPECT_EQ(params.perm[0], 1);
  EXPECT_EQ(params.perm[1], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_3DShrinkingTwice) {
  RuntimeShape input_shape({4, 1, 1});
  RuntimeShape output_shape({1, 4, 1});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 0;
  params.perm[2] = 1;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({4}));
  EXPECT_EQ(output_shape, RuntimeShape({4}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_3DAllOnes) {
  RuntimeShape input_shape({1, 1, 1});
  RuntimeShape output_shape({1, 1, 1});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 0;
  params.perm[2] = 1;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({1}));
  EXPECT_EQ(output_shape, RuntimeShape({1}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_4DNoChanges) {
  RuntimeShape input_shape({9, 3, 2, 4});
  RuntimeShape output_shape({3, 9, 4, 2});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 1;
  params.perm[1] = 0;
  params.perm[2] = 3;
  params.perm[3] = 2;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({9, 3, 2, 4}));
  EXPECT_EQ(output_shape, RuntimeShape({3, 9, 4, 2}));

  EXPECT_EQ(params.perm_count, 4);
  EXPECT_EQ(params.perm[0], 1);
  EXPECT_EQ(params.perm[1], 0);
  EXPECT_EQ(params.perm[2], 3);
  EXPECT_EQ(params.perm[3], 2);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_4DShrinkingOnce) {
  RuntimeShape input_shape({9, 3, 1, 4});
  RuntimeShape output_shape({3, 9, 4, 1});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 1;
  params.perm[1] = 0;
  params.perm[2] = 3;
  params.perm[3] = 2;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({9, 3, 4}));
  EXPECT_EQ(output_shape, RuntimeShape({3, 9, 4}));

  EXPECT_EQ(params.perm_count, 3);
  EXPECT_EQ(params.perm[0], 1);
  EXPECT_EQ(params.perm[1], 0);
  EXPECT_EQ(params.perm[2], 2);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_4DShrinkingTwice) {
  RuntimeShape input_shape({1, 3, 1, 4});
  RuntimeShape output_shape({3, 1, 4, 1});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 1;
  params.perm[1] = 2;
  params.perm[2] = 3;
  params.perm[3] = 0;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({3, 4}));
  EXPECT_EQ(output_shape, RuntimeShape({3, 4}));

  EXPECT_EQ(params.perm_count, 2);
  EXPECT_EQ(params.perm[0], 0);
  EXPECT_EQ(params.perm[1], 1);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_4DShrinkingThirdTimes) {
  RuntimeShape input_shape({1, 1, 7, 1});
  RuntimeShape output_shape({1, 7, 1, 1});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 0;
  params.perm[1] = 2;
  params.perm[2] = 1;
  params.perm[3] = 3;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({7}));
  EXPECT_EQ(output_shape, RuntimeShape({7}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, RemoveOneSizeDimensions_4DAllOnes) {
  RuntimeShape input_shape({1, 1, 1, 1});
  RuntimeShape output_shape({1, 1, 1, 1});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 0;
  params.perm[1] = 2;
  params.perm[2] = 1;
  params.perm[3] = 3;

  transpose_utils::RemoveOneSizeDimensions(&input_shape, &output_shape,
                                           &params);

  EXPECT_EQ(input_shape, RuntimeShape({1}));
  EXPECT_EQ(output_shape, RuntimeShape({1}));

  EXPECT_EQ(params.perm_count, 1);
  EXPECT_EQ(params.perm[0], 0);
}

TEST(TransposeUtilsTest, Flatten3D) {
  RuntimeShape input_shape({3, 5, 7});
  RuntimeShape output_shape({3, 7, 5});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 0;
  params.perm[1] = 2;
  params.perm[2] = 1;

  RuntimeShape non_flatten_input_shape;
  RuntimeShape non_flatten_output_shape;
  TransposeParams non_flatten_params;
  size_t non_flatten_size = transpose_utils::Flatten(
      input_shape, output_shape, params, &non_flatten_input_shape,
      &non_flatten_output_shape, &non_flatten_params);

  EXPECT_EQ(non_flatten_input_shape, RuntimeShape({5, 7}));
  EXPECT_EQ(non_flatten_output_shape, RuntimeShape({7, 5}));
  EXPECT_EQ(non_flatten_size, 5 * 7);

  EXPECT_EQ(non_flatten_params.perm_count, 2);
  EXPECT_EQ(non_flatten_params.perm[0], 1);
  EXPECT_EQ(non_flatten_params.perm[1], 0);
}

TEST(TransposeUtilsTest, Flatten4DFlattenOnce) {
  RuntimeShape input_shape({3, 5, 7, 9});
  RuntimeShape output_shape({3, 7, 5, 9});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 0;
  params.perm[1] = 2;
  params.perm[2] = 1;
  params.perm[3] = 3;

  RuntimeShape non_flatten_input_shape;
  RuntimeShape non_flatten_output_shape;
  TransposeParams non_flatten_params;
  size_t non_flatten_size = transpose_utils::Flatten(
      input_shape, output_shape, params, &non_flatten_input_shape,
      &non_flatten_output_shape, &non_flatten_params);

  EXPECT_EQ(non_flatten_input_shape, RuntimeShape({5, 7, 9}));
  EXPECT_EQ(non_flatten_output_shape, RuntimeShape({7, 5, 9}));
  EXPECT_EQ(non_flatten_size, 5 * 7 * 9);

  EXPECT_EQ(non_flatten_params.perm_count, 3);
  EXPECT_EQ(non_flatten_params.perm[0], 1);
  EXPECT_EQ(non_flatten_params.perm[1], 0);
  EXPECT_EQ(non_flatten_params.perm[2], 2);
}

TEST(TransposeUtilsTest, Flatten4DFlattenTwice) {
  RuntimeShape input_shape({3, 5, 7, 9});
  RuntimeShape output_shape({3, 5, 9, 7});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 0;
  params.perm[1] = 1;
  params.perm[2] = 3;
  params.perm[3] = 2;

  RuntimeShape non_flatten_input_shape;
  RuntimeShape non_flatten_output_shape;
  TransposeParams non_flatten_params;
  size_t non_flatten_size = transpose_utils::Flatten(
      input_shape, output_shape, params, &non_flatten_input_shape,
      &non_flatten_output_shape, &non_flatten_params);

  EXPECT_EQ(non_flatten_input_shape, RuntimeShape({7, 9}));
  EXPECT_EQ(non_flatten_output_shape, RuntimeShape({9, 7}));
  EXPECT_EQ(non_flatten_size, 7 * 9);

  EXPECT_EQ(non_flatten_params.perm_count, 2);
  EXPECT_EQ(non_flatten_params.perm[0], 1);
  EXPECT_EQ(non_flatten_params.perm[1], 0);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable2D) {
  RuntimeShape input_shape({4, 5});

  TransposeParams params;
  params.perm_count = 2;
  params.perm[0] = 1;
  params.perm[1] = 0;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 4);
  EXPECT_EQ(dim1, 5);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable3DOne) {
  RuntimeShape input_shape({4, 5, 6});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 1;
  params.perm[1] = 2;
  params.perm[2] = 0;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 4);
  EXPECT_EQ(dim1, 30);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable3DTwo) {
  RuntimeShape input_shape({4, 5, 6});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 0;
  params.perm[2] = 1;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 20);
  EXPECT_EQ(dim1, 6);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable3DNotApplicable) {
  RuntimeShape input_shape({4, 5, 6});

  TransposeParams params;
  params.perm_count = 3;
  params.perm[0] = 2;
  params.perm[1] = 1;
  params.perm[2] = 0;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_FALSE(applicable);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable4DOne) {
  RuntimeShape input_shape({4, 5, 6, 7});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 1;
  params.perm[1] = 2;
  params.perm[2] = 3;
  params.perm[3] = 0;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 4);
  EXPECT_EQ(dim1, 210);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable4DTwo) {
  RuntimeShape input_shape({4, 5, 6, 7});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 2;
  params.perm[1] = 3;
  params.perm[2] = 0;
  params.perm[3] = 1;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 20);
  EXPECT_EQ(dim1, 42);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable4DThird) {
  RuntimeShape input_shape({4, 5, 6, 7});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 3;
  params.perm[1] = 0;
  params.perm[2] = 1;
  params.perm[3] = 2;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_TRUE(applicable);
  EXPECT_EQ(dim0, 120);
  EXPECT_EQ(dim1, 7);
}

TEST(TransposeUtilsTest, IsTranspose2DApplicable4DNotApplicable) {
  RuntimeShape input_shape({4, 5, 6, 7});

  TransposeParams params;
  params.perm_count = 4;
  params.perm[0] = 3;
  params.perm[1] = 2;
  params.perm[2] = 1;
  params.perm[3] = 0;

  int dim0, dim1;
  bool applicable = transpose_utils::IsTranspose2DApplicable(
      params, input_shape, &dim0, &dim1);

  EXPECT_FALSE(applicable);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
