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
#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <vector>

#include "Eigen/Core"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/transpose_test_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class TransposeOpInt4Model : public SingleOpModel {
 public:
  TransposeOpInt4Model(std::initializer_list<int> input_shape,
                       std::initializer_list<int> perm_shape,
                       std::initializer_list<int> perm) {
    input_ = AddInput({TensorType_INT4, input_shape});
    perm_ = AddConstInput(TensorType_INT32, perm, perm_shape);
    output_ = AddOutput(TensorType_INT4);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(const std::vector<int8_t> data) {
    auto non_const = *const_cast<std::vector<int8_t>*>(&data);
    std::vector<int8_t> data_int8(non_const.size());
    std::copy(non_const.begin(), non_const.end(), data_int8.begin());
    PopulateTensor4bit(0, 0, data_int8.data(),
                       data_int8.data() + data_int8.size());
  }

  void SetPerm(std::initializer_list<int> data) {
    PopulateTensor<int>(perm_, data);
  }

  std::vector<int8_t> GetOutput() {
    const auto* tensor = interpreter_->tensor(output_);
    const std::vector<int8_t> data_int8 = std::vector<int8_t>(
        tensor->data.raw, tensor->data.raw + GetTensorSize(output_));
    int num_elements = 1;
    auto shape = GetTensorShape(output_);
    for (int i = 0; i < shape.size(); i++) {
      num_elements *= shape[i];
    }
    std::vector<int8_t> inflated_output(num_elements);
    tensor_utils::UnpackDenseInt4IntoInt8(data_int8.data(), num_elements,
                                          inflated_output.data());
    return inflated_output;
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int perm_;
  int output_;
};


template <typename input_type>
class TransposeOpModel : public SingleOpModel {
 public:
  void SetInput(std::vector<input_type> data) {
    PopulateTensor<input_type>(input_, data);
  }

  void SetPerm(std::initializer_list<int> data) {
    PopulateTensor<int>(perm_, data);
  }
  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int perm_;
  int output_;
};

// Tests case where perm is a const tensor.
//
// Example usage is as follows:
//    TransposeModel <input_type> m (input_shape, perm_shape, perm_data);
//    m.SetInput(input_data);
//    m.Invoke();
template <typename input_type>
class TransposeOpConstModel : public TransposeOpModel<input_type> {
 public:
  TransposeOpConstModel(std::initializer_list<int> input_shape,
                        std::initializer_list<int> perm_shape,
                        std::initializer_list<int> perm) {
    this->input_ = this->AddInput({GetTensorType<input_type>(), input_shape});
    this->perm_ = this->AddConstInput(TensorType_INT32, perm, perm_shape);
    this->output_ = this->AddOutput(GetTensorType<input_type>());
    this->SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(this->builder_).Union());
    this->BuildInterpreter({input_shape});
  }
};

// Tests case where perm is a non-const tensor.
//
// Example usage is as follows:
//    TransposeOpDynamicModel <input_type> m (input_shape, perm_shape);
//    m.SetInput(input_data);
//    m.SetPerm(perm_data);
//    m.Invoke();
template <typename input_type>
class TransposeOpDynamicModel : public TransposeOpModel <input_type> {
 public:
  TransposeOpDynamicModel(std::initializer_list<int> input_shape,
                          std::initializer_list<int> perm_shape) {
    this->input_ = this->AddInput(GetTensorType<input_type>());
    this->perm_ = this->AddInput(TensorType_INT32);
    this->output_ = this->AddOutput(GetTensorType<input_type>());
    this->SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(this->builder_).Union());
    this->BuildInterpreter({input_shape, perm_shape});
  }
};

template <typename T>
class TransposeTest : public ::testing::Test {};

using DataTypes = ::testing::Types<bool, Eigen::bfloat16, Eigen::half, int8_t,
                                   int16_t, int32_t, int64_t, uint8_t>;
TYPED_TEST_SUITE(TransposeTest, DataTypes);

template <typename T>
std::vector<T> CastVector(const std::vector<int>& input_data) {
  std::vector<T> casted_input(input_data.size());

  std::transform(input_data.begin(), input_data.end(), casted_input.begin(),
                   [](int x) { return static_cast<T>(x); });
  return casted_input;
}

#if GTEST_HAS_DEATH_TEST
TEST(TransposeTest, TestUnequalPermSize) {
  EXPECT_DEATH(TransposeOpConstModel<float>({1, 3, 3, 1}, {2}, {2, 2}), "2 != 4");
}
TEST(TransposeTest, TestPermOutOfBounds) {
  EXPECT_DEATH(TransposeOpConstModel<float>({1, 3, 3, 1}, {4}, {0, -1, -2, -5}),
               "Transpose op permutations array is out of bounds.");
  EXPECT_DEATH(TransposeOpConstModel<float>({1, 3, 3, 1}, {4}, {0, 1, 2, 4}),
               "Transpose op permutations array is out of bounds.");
}
#endif

TEST(TransposeTest, TestInt41DInputConstTensor) {
  TransposeOpInt4Model m({3}, {1}, {0});
  m.SetInput({1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAre(1, 2, 3));
}

TEST(TransposeTest, TestInt42DInputConstTensor) {
  TransposeOpInt4Model m({3, 2}, {2}, {1, 0});
  m.SetInput({0, 1, 2, 3, 4, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

TEST(TransposeTest, TestInt43DInputConstTensor) {
  TransposeOpInt4Model m({2, 3, 4}, {3}, {2, 0, 1});
  m.SetInput(
      {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 1, 1, 1, 0, 1, 5, 2, 2, 2, 1,
                                2, 6, 3, 3, 3, 2, 3, 0, 4, 0, 4, 3}));
}

TYPED_TEST(TransposeTest, Test1DInputConstTensor) {
  TransposeOpConstModel<TypeParam> m({3}, {1}, {0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TYPED_TEST(TransposeTest, Test1DInputDynamicTensor) {
  TransposeOpDynamicModel<TypeParam> m({3}, {1});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3});
  m.SetInput(input_data);
  m.SetPerm({0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TYPED_TEST(TransposeTest, Test2DInputConstTensor) {
  TransposeOpConstModel<TypeParam> m({3, 2}, {2}, {1, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1, 2, 3, 4, 5});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

TYPED_TEST(TransposeTest, Test2D4x4KernelTestLeftOverRightSide) {
  TransposeOpConstModel<TypeParam> m({4, 6}, {2}, {1, 0});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 4}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 6, 12, 18, 1, 7,  13, 19, 2, 8,  14, 20,
                                3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23}));
}

TYPED_TEST(TransposeTest, Test2D4x4KernelTest2LeftOverBottomSide) {
  TransposeOpConstModel<TypeParam> m({6, 4}, {2}, {1, 0});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 6}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TYPED_TEST(TransposeTest, Test2DInputDynamicTensor) {
  TransposeOpDynamicModel<TypeParam> m({3, 2}, {2});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1, 2, 3, 4, 5});
  m.SetInput(input_data);
  m.SetPerm({1, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

TYPED_TEST(TransposeTest, Test3DInputConstTensor) {
  TransposeOpConstModel<TypeParam> m({2, 3, 4}, {3}, {2, 0, 1});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TYPED_TEST(TransposeTest, Test3DInputDynamicTensor) {
  TransposeOpDynamicModel<TypeParam> m({2, 3, 4}, {3});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  m.SetPerm({2, 0, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TYPED_TEST(TransposeTest, Test1DNotShrinked) {
  TransposeOpConstModel<TypeParam> m({1}, {1}, {0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TYPED_TEST(TransposeTest, Test2DShrinkedOneTime) {
  TransposeOpConstModel<TypeParam> m({2, 1}, {2}, {1, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1}));
}

TYPED_TEST(TransposeTest, Test2DShrinkedTwoTimes) {
  TransposeOpConstModel<TypeParam> m({1, 1}, {2}, {1, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TYPED_TEST(TransposeTest, Test3DShrinkedOneTime) {
  TransposeOpConstModel<TypeParam> m({2, 1, 3}, {3}, {0, 2, 1});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1, 2, 3, 4, 5});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3, 4, 5}));
}

TYPED_TEST(TransposeTest, Test3DShrinkedTwoTimes) {
  TransposeOpConstModel<TypeParam> m({1, 1, 3}, {3}, {1, 2, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1, 2});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2}));
}

TYPED_TEST(TransposeTest, Test3DShrinkedAll) {
  TransposeOpConstModel<TypeParam> m({1, 1, 1}, {3}, {1, 2, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TYPED_TEST(TransposeTest, Test4DShrinkedOneTimes) {
  TransposeOpConstModel<TypeParam> m({2, 2, 3, 1}, {4}, {3, 0, 1, 2});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TYPED_TEST(TransposeTest, Test4DShrinkedTwoTimes) {
  TransposeOpConstModel<TypeParam> m({2, 1, 3, 1}, {4}, {0, 3, 1, 2});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1, 2, 3, 4, 5});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3, 4, 5}));
}

TYPED_TEST(TransposeTest, Test4DShrinkedThirdTimes) {
  TransposeOpConstModel<TypeParam> m({2, 1, 1, 1}, {4}, {3, 2, 1, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0, 1});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1}));
}

TYPED_TEST(TransposeTest, Test4DShrinkedFourTimes) {
  TransposeOpConstModel<TypeParam> m({1, 1, 1, 1}, {4}, {2, 3, 1, 0});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TYPED_TEST(TransposeTest, Test3DFlatten) {
  TransposeOpConstModel<TypeParam> m({2, 2, 3}, {3}, {0, 2, 1});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}));
}

TYPED_TEST(TransposeTest, Test4DFlattenOne) {
  TransposeOpConstModel<TypeParam> m({2, 2, 2, 2}, {4}, {0, 1, 3, 2});
  std::vector<TypeParam> input_data = CastVector<TypeParam>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9,
                                               11, 12, 14, 13, 15}));
}

TYPED_TEST(TransposeTest, Test4DFlattenTwo) {
  TransposeOpConstModel<TypeParam> m({2, 2, 2, 2}, {4}, {0, 2, 3, 1});
  std::vector<TypeParam> input_data = CastVector<TypeParam>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9,
                                               13, 10, 14, 11, 15}));
}

TEST(TransposeTest, 3DDividedIntoTwo2DsOne) {
  std::vector<float> out = RunTestPermutation<float>({2, 3, 4}, {1, 2, 0});
  TransposeOpConstModel<float> m({2, 3, 4}, {3}, {1, 2, 0});
  std::vector<float> input_data =
      CastVector<float>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 3DDividedIntoTwo2DsTwo) {
  std::vector<float> out = RunTestPermutation<float>({2, 3, 4}, {2, 0, 1});
  TransposeOpConstModel<float> m({2, 3, 4}, {3}, {2, 0, 1});
  std::vector<float> input_data =
      CastVector<float>({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsOne) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 4, 2}, {1, 2, 3, 0});
  TransposeOpConstModel<float> m({2, 3, 4, 2}, {4}, {1, 2, 3, 0});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsTwo) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 4, 2}, {2, 3, 0, 1});
  TransposeOpConstModel<float> m({2, 3, 4, 2}, {4}, {2, 3, 0, 1});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsThird) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 4, 2}, {3, 0, 1, 2});
  TransposeOpConstModel<float> m({2, 3, 4, 2}, {4}, {3, 0, 1, 2});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsOne) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 2, 2, 2}, {1, 4, 2, 3, 0});
  TransposeOpConstModel<float> m({2, 3, 2, 2, 2}, {5}, {1, 4, 2, 3, 0});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsTwo) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 2, 2, 2}, {2, 3, 0, 4, 1});
  TransposeOpConstModel<float> m({2, 3, 2, 2, 2}, {5}, {2, 3, 0, 4, 1});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsThird) {
  std::vector<float> out =
      RunTestPermutation<float>({2, 3, 2, 2, 2}, {3, 0, 4, 1, 2});
  TransposeOpConstModel<float> m({2, 3, 2, 2, 2}, {5}, {3, 0, 4, 1, 2});
  std::vector<float> input_data = CastVector<float>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutput(), out);
}

#if GTEST_HAS_DEATH_TEST
TEST(TransposeTest, Test7DInputTensor) {
  EXPECT_DEATH(
      TransposeOpConstModel<int32_t>({1, 2, 3, 4, 5, 6, 7}, {6}, {0, 1, 2, 3, 4, 5}),
      "Transpose op only supports 1D-6D input arrays.");
}
#endif

TYPED_TEST(TransposeTest, SimpleTestNoReorderConstTensor) {
  TransposeOpConstModel <TypeParam> m({1, 2, 3, 1}, {4}, {0, 1, 2, 3});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({1, 2, 3, 4, 5, 6});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(TransposeTest, SimpleTestNoReorderDynamicTensor) {
  TransposeOpDynamicModel <TypeParam> m({1, 2, 3, 1}, {4});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({{1, 2, 3, 4, 5, 6}});
  m.SetInput(input_data);
  m.SetPerm({0, 1, 2, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TYPED_TEST(TransposeTest, SimpleTestWithReorderConstTensor) {
  TransposeOpConstModel <TypeParam> m({1, 2, 3, 1}, {4}, {2, 1, 3, 0});
  std::vector<TypeParam> input_data =
      CastVector<TypeParam>({{1, 2, 3, 4, 5, 6}});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TYPED_TEST(TransposeTest, ComplexTestWithReorderConstTensor) {
  TransposeOpConstModel <TypeParam> m({2, 3, 4, 5}, {4}, {2, 0, 1, 3});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3, 5}));
  auto result = ElementsAreArray(
      {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
       60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
       5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
       65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
       10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
       70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
       15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
       75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119});
  EXPECT_THAT(m.GetOutput(), result);
}

TYPED_TEST(TransposeTest, ComplexTestWithReorderDynamicTensor) {
  TransposeOpDynamicModel <TypeParam> m({2, 3, 4, 5}, {4});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetInput(input_data);
  m.SetPerm({2, 0, 1, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3, 5}));
  auto result = ElementsAreArray(
      {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
       60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
       5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
       65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
       10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
       70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
       15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
       75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119});
  EXPECT_THAT(m.GetOutput(), result);
}

TYPED_TEST(TransposeTest, Complex5DTestWithReorderConstTensor) {
  TransposeOpConstModel <TypeParam> m({2, 3, 2, 2, 5}, {5}, {2, 0, 1, 4, 3});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetInput(input_data);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3, 5, 2}));
  auto result = ElementsAreArray(
      {0,  5,  1,  6,  2,  7,   3,   8,   4,   9,   20,  25,  21,  26,  22,
       27, 23, 28, 24, 29, 40,  45,  41,  46,  42,  47,  43,  48,  44,  49,
       60, 65, 61, 66, 62, 67,  63,  68,  64,  69,  80,  85,  81,  86,  82,
       87, 83, 88, 84, 89, 100, 105, 101, 106, 102, 107, 103, 108, 104, 109,
       10, 15, 11, 16, 12, 17,  13,  18,  14,  19,  30,  35,  31,  36,  32,
       37, 33, 38, 34, 39, 50,  55,  51,  56,  52,  57,  53,  58,  54,  59,
       70, 75, 71, 76, 72, 77,  73,  78,  74,  79,  90,  95,  91,  96,  92,
       97, 93, 98, 94, 99, 110, 115, 111, 116, 112, 117, 113, 118, 114, 119});
  EXPECT_THAT(m.GetOutput(), result);
}

TYPED_TEST(TransposeTest, Complex5DTestWithReorderDynamicTensor) {
  TransposeOpDynamicModel <TypeParam> m({2, 3, 2, 2, 5}, {5});
  std::vector<TypeParam> input_data = CastVector<TypeParam>({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetInput(input_data);
  m.SetPerm({2, 0, 1, 4, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 3, 5, 2}));
  auto result = ElementsAreArray(
      {0,  5,  1,  6,  2,  7,   3,   8,   4,   9,   20,  25,  21,  26,  22,
       27, 23, 28, 24, 29, 40,  45,  41,  46,  42,  47,  43,  48,  44,  49,
       60, 65, 61, 66, 62, 67,  63,  68,  64,  69,  80,  85,  81,  86,  82,
       87, 83, 88, 84, 89, 100, 105, 101, 106, 102, 107, 103, 108, 104, 109,
       10, 15, 11, 16, 12, 17,  13,  18,  14,  19,  30,  35,  31,  36,  32,
       37, 33, 38, 34, 39, 50,  55,  51,  56,  52,  57,  53,  58,  54,  59,
       70, 75, 71, 76, 72, 77,  73,  78,  74,  79,  90,  95,  91,  96,  92,
       97, 93, 98, 94, 99, 110, 115, 111, 116, 112, 117, 113, 118, 114, 119});
  EXPECT_THAT(m.GetOutput(), result);
}

}  // namespace
}  // namespace tflite
