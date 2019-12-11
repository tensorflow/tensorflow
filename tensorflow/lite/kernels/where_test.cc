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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseWhereOpModel : public SingleOpModel {
 public:
  BaseWhereOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_WHERE, BuiltinOptions_WhereOptions,
                 CreateWhereOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

class IntegerWhereOpModel : public BaseWhereOpModel {
 public:
  using BaseWhereOpModel::BaseWhereOpModel;

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
};

TEST(WhereOpTest, SelectFromVectorNoResult) {
  IntegerWhereOpModel m({TensorType_BOOL, {3}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {false, false, false});
  m.Invoke();
  EXPECT_THAT(m.GetOutput().size(), 0);
}

TEST(WhereOpTest, SelectFromVector) {
  IntegerWhereOpModel m({TensorType_BOOL, {3}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, false, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2}));
}

TEST(WhereOpTest, SelectFromMatrixNoResult) {
  IntegerWhereOpModel m({TensorType_BOOL, {3, 3}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {false, false, false,  //
                                     false, false, false,  //
                                     false, false, false});
  m.Invoke();
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TEST(WhereOpTest, SelectFromMatrix1) {
  IntegerWhereOpModel m({TensorType_BOOL, {3, 1}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, false, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               2, 0}));
}

TEST(WhereOpTest, SelectFromMatrix2) {
  IntegerWhereOpModel m({TensorType_BOOL, {3, 3}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, true, false,   //
                                     true, false, false,  //
                                     true, false, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 1,  //
                                               1, 0,  //
                                               2, 0,  //
                                               2, 2}));
}

TEST(WhereOpTest, SelectFromMatrix3) {
  IntegerWhereOpModel m({TensorType_BOOL, {3, 5}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, false, false, true, true,   //
                                     false, true, true, false, false,  //
                                     true, false, true, false, false});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0,  //
                                               0, 3,  //
                                               0, 4,  //
                                               1, 1,  //
                                               1, 2,  //
                                               2, 0,  //
                                               2, 2}));
}

TEST(WhereOpTest, SelectFromRank3TensorNoResult) {
  IntegerWhereOpModel m({TensorType_BOOL, {2, 2, 2}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {false, false, false, false,  //
                                     false, false, false, false});
  m.Invoke();
  EXPECT_EQ(m.GetOutput().size(), 0);
}

TEST(WhereOpTest, SelectFromRank3Tensor1) {
  IntegerWhereOpModel m({TensorType_BOOL, {2, 1, 3}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, false, true,  //
                                     false, false, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 2,  //
                                               1, 0, 2}));
}

TEST(WhereOpTest, SelectFromRank3Tensor2) {
  IntegerWhereOpModel m({TensorType_BOOL, {2, 2, 2}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, true, false, true,  //
                                     false, false, true, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 1, 1}));
}

TEST(WhereOpTest, SelectFromRank3Tensor3) {
  IntegerWhereOpModel m({TensorType_BOOL, {2, 3, 2}}, {TensorType_INT64, {}});
  m.PopulateTensor<bool>(m.input(), {true, true, false, true, false, false,  //
                                     false, false, true, false, true, true});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0,  //
                                               0, 0, 1,  //
                                               0, 1, 1,  //
                                               1, 1, 0,  //
                                               1, 2, 0,  //
                                               1, 2, 1}));
}

}  // namespace
}  // namespace tflite
