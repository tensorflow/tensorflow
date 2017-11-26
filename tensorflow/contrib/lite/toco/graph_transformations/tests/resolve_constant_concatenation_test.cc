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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
//#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

namespace {
// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
std::vector<testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}
}  // namespace

// The following 3 tests make sure the concatenation operation on different axis
// values match TensorFlow results listed below:
//
// x0 = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
// x1 = [[[10, 11], [12, 13]], [[14, 15], [16, 17]]]
// x2 = [[[20, 21], [22, 23]], [[24, 25], [26, 27]]]
// x3 = [[[30, 31], [32, 33]], [[34, 35], [36, 37]]]
//
// ConcatAtAxis0 test:
// t0 = tf.concat([x0, x1, x2, x3], 0)
// [[[ 0  1]
//   [ 2  3]]
//
//  [[ 4  5]
//   [ 6  7]]
//
//  [[10 11]
//   [12 13]]
//
//  [[14 15]
//   [16 17]]
//
//  [[20 21]
//   [22 23]]
//
//  [[24 25]
//   [26 27]]
//
//  [[30 31]
//   [32 33]]
//
//  [[34 35]
//   [36 37]]]
//
// ConcatAtAxis1 test:
// t1 = tf.concat([x0, x1, x2, x3], 1)
// [[[ 0  1]
//   [ 2  3]
//   [10 11]
//   [12 13]
//   [20 21]
//   [22 23]
//   [30 31]
//   [32 33]]
//
//  [[ 4  5]
//   [ 6  7]
//   [14 15]
//   [16 17]
//   [24 25]
//   [26 27]
//   [34 35]
//   [36 37]]]
//
// ConcatAtAxis2 test:
// t2 = tf.concat([x0, x1, x2, x3], 2)
// [[[ 0  1 10 11 20 21 30 31]
//   [ 2  3 12 13 22 23 32 33]]
//
//  [[ 4  5 14 15 24 25 34 35]
//   [ 6  7 16 17 26 27 36 37]]]

class ResolveConstantConcatenationTest : public ::testing::Test {
 protected:
  ResolveConstantConcatenationTest() {}

  // Prepare a hypothetical TOCO model with one Concatenation operator in it
  // together with 4 arrays as its inputs.
  // It receives the dimension of concatenation as input.
  void PrepareModel(Model* model, int concat_dim) {
    std::vector<string> concat_input_names = {"array0", "array1", "array2",
                                              "array3"};

    const int kDim = 3;
    const int kElementPerDim = 2;
    const int kBufSize = 8;
    const int kNumArrays = 4;
    static float in_buf[kNumArrays][kBufSize] = {
        {0., 1., 2., 3., 4., 5., 6., 7.},
        {10., 11., 12., 13., 14., 15., 16., 17.},
        {20., 21., 22., 23., 24., 25., 26., 27.},
        {30., 31., 32., 33., 34., 35., 36., 37.}};
    int cnt = 0;
    for (const string& concat_input_name : concat_input_names) {
      Array& in_array = model->GetOrCreateArray(concat_input_name);
      in_array.data_type = ArrayDataType::kFloat;

      // Initialize shape for the input  array.
      Shape* in_array_shape = in_array.mutable_shape();
      std::vector<int>* in_array_shape_dim = in_array_shape->mutable_dims();
      for (int i = 0; i < kDim; i++) {
        in_array_shape_dim->push_back(kElementPerDim);
      }
      auto& in_array_buffer =
          in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>();
      in_array_buffer.data.resize(kBufSize);
      float* buf_ptr =
          in_array.GetMutableBuffer<toco::ArrayDataType::kFloat>().data.data();
      std::copy(in_buf[cnt], in_buf[cnt] + kBufSize, buf_ptr);
      cnt++;
    }
    auto* concatenation_op = new ConcatenationOperator;
    concatenation_op->concat_dim = concat_dim;
    concatenation_op->inputs = concat_input_names;
    concatenation_op->outputs = {"concat_op_outputs"};
    Array& out_array = model->GetOrCreateArray(concatenation_op->outputs[0]);
    out_array.data_type = ArrayDataType::kFloat;
    Shape* out_array_shape = out_array.mutable_shape();
    std::vector<int>* out_array_shape_dim = out_array_shape->mutable_dims();
    out_array_shape_dim->resize(kDim);
    for (int i = 0; i < kDim; i++) {
      if (i == concat_dim) {
        (*out_array_shape_dim)[i] = kNumArrays * kElementPerDim;
      } else {
        (*out_array_shape_dim)[i] = kElementPerDim;
      }
    }
    model->operators.push_back(std::unique_ptr<Operator>(concatenation_op));
  }
};

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis0) {
  Model model;
  const int concat_dim = 0;
  PrepareModel(&model, concat_dim);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.arrays.size(), 5);
  (*graph_transformation_set.begin())->Run(&model, /*op_index=*/0);
  EXPECT_THAT(model.arrays.size(), 1);

  auto& concatenated_array = (*model.arrays.begin()).second;
  EXPECT_THAT(concatenated_array->GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  10., 11., 12.,
                   13., 14., 15., 16., 17., 20., 21., 22., 23., 24., 25.,
                   26., 27., 30., 31., 32., 33., 34., 35., 36., 37.})));
}

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis1) {
  Model model;
  const int concat_dim = 1;
  PrepareModel(&model, concat_dim);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.arrays.size(), 5);
  (*graph_transformation_set.begin())->Run(&model, /*op_index=*/0);
  EXPECT_THAT(model.arrays.size(), 1);

  auto& concatenated_array = (*model.arrays.begin()).second;
  EXPECT_THAT(concatenated_array->GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  2.,  3.,  10., 11., 12., 13., 20., 21., 22.,
                   23., 30., 31., 32., 33., 4.,  5.,  6.,  7.,  14., 15.,
                   16., 17., 24., 25., 26., 27., 34., 35., 36., 37.})));
}

TEST_F(ResolveConstantConcatenationTest, ConcatAtAxis2) {
  Model model;
  const int concat_dim = 2;
  PrepareModel(&model, concat_dim);

  GraphTransformationsSet graph_transformation_set;
  graph_transformation_set.Add(new toco::ResolveConstantConcatenation);
  EXPECT_THAT(model.arrays.size(), 5);
  (*graph_transformation_set.begin())->Run(&model, /*op_index=*/0);
  EXPECT_THAT(model.arrays.size(), 1);

  auto& concatenated_array = (*model.arrays.begin()).second;
  EXPECT_THAT(concatenated_array->GetBuffer<toco::ArrayDataType::kFloat>().data,
              ElementsAreArray(ArrayFloatNear(
                  {0.,  1.,  10., 11., 20., 21., 30., 31., 2.,  3.,  12.,
                   13., 22., 23., 32., 33., 4.,  5.,  14., 15., 24., 25.,
                   34., 35., 6.,  7.,  16., 17., 26., 27., 36., 37.})));
}

}  // namespace toco
