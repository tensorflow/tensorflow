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

#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
void RunTestPermutation(const std::vector<int>& shape,
                        const std::vector<int>& perms,
                        std::vector<T>* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  std::vector<T> input(count);
  for (int i = 0; i < input.size(); i++) {
    input[i] = i;
  }

  // Make input and output shapes.
  const RuntimeShape input_shape = GetTensorShape(shape);
  RuntimeShape output_shape(perms.size());
  for (int i = 0; i < perms.size(); i++) {
    output_shape.SetDim(i, input_shape.Dims(perms[i]));
  }

  TransposeParams params;
  params.perm_count = perms.size();
  for (int i = 0; i < perms.size(); ++i) {
    params.perm[i] = perms[i];
  }

  reference_ops::Transpose<T>(params, input_shape, input.data(), output_shape,
                              input_transposed->data());
}

TEST(TransposeTest, TestRefOps1D) {
  // Basic 1D identity.
  std::vector<float> out;
  RunTestPermutation({3}, {0}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 1, 2}));
}

TEST(TransposeTest, TestRefOps2D) {
  std::vector<float> out;
  // Basic 2D.
  RunTestPermutation({3, 2}, {1, 0}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 2, 4, 1, 3, 5}));
  // Identity.
  RunTestPermutation({3, 2}, {0, 1}, &out);
  ASSERT_EQ(out, std::vector<float>({0, 1, 2, 3, 4, 5}));
}

TEST(TransposeTest, TestRefOps3D) {
  std::vector<float> out;
  {
    std::vector<float> ref({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                            2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{2, 0, 1}, &out);
    ASSERT_EQ(out, ref);
  }

  // Test 3 dimensional identity transform
  {
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{0, 1, 2}, &out);
    std::vector<float> ref(out.size());
    for (int k = 0; k < ref.size(); k++) ref[k] = k;
    ASSERT_EQ(out, ref);
  }

  /**
   * Additional tests that mimic first case, but with different perm.
   */
  {
    std::vector<float> ref({0, 12, 1, 13, 2, 14, 3, 15, 4,  16, 5,  17,
                            6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23});
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{1, 2, 0}, &out);
    ASSERT_EQ(out, ref);
  }

  {
    std::vector<float> ref({0,  4,  8,  1,  5,  9,  2,  6,  10, 3,  7,  11,
                            12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23});
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{0, 2, 1}, &out);
    ASSERT_EQ(out, ref);
  }

  {
    std::vector<float> ref({0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
                            16, 17, 18, 19, 8,  9,  10, 11, 20, 21, 22, 23});
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{1, 0, 2}, &out);
    ASSERT_EQ(out, ref);
  }

  {
    std::vector<float> ref({0, 12, 4, 16, 8,  20, 1, 13, 5, 17, 9,  21,
                            2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23});
    RunTestPermutation(/*shape=*/{2, 3, 4}, /*perms=*/{2, 1, 0}, &out);
    ASSERT_EQ(out, ref);
  }
}

TEST(TransposeTest, TestRefOps3D_OneInDimension) {
  std::vector<float> out;
  // Shape with 1 as first dim -> transposed.
  {
    std::vector<float> ref({0, 3, 1, 4, 2, 5});
    RunTestPermutation(/*shape=*/{1, 2, 3}, /*perms=*/{2, 0, 1}, &out);
    ASSERT_EQ(out, ref);
  }
  // Shape with 1 as first dim -> identity.
  {
    std::vector<float> ref({0, 1, 2, 3, 4, 5});
    RunTestPermutation(/*shape=*/{1, 2, 3}, /*perms=*/{1, 2, 0}, &out);
    ASSERT_EQ(out, ref);
  }
  // Shape with 1 as third dim -> transposed.
  {
    std::vector<float> ref({0, 3, 1, 4, 2, 5});
    RunTestPermutation(/*shape=*/{2, 3, 1}, /*perms=*/{1, 2, 0}, &out);
    ASSERT_EQ(out, ref);
  }
  // Shape with 1 as third dim -> identity.
  {
    std::vector<float> ref({0, 1, 2, 3, 4, 5});
    RunTestPermutation(/*shape=*/{2, 3, 1}, /*perms=*/{2, 0, 1}, &out);
    ASSERT_EQ(out, ref);
  }
}

TEST(TransposeTest, TestRefOps4D) {
  std::vector<float> out;
  // Basic 4d.
  RunTestPermutation({2, 3, 4, 5}, {2, 0, 1, 3}, &out);
  ASSERT_EQ(
      out,
      std::vector<float>(
          {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
           60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
           5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
           65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
           10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
           70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
           15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
           75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119}));
  RunTestPermutation({2, 3, 4, 5}, {0, 1, 2, 3}, &out);
  // Basic identity.
  std::vector<float> ref(out.size());
  for (int k = 0; k < ref.size(); k++) ref[k] = k;
  ASSERT_EQ(out, ref);
}

TEST(TransposeTest, TestRefOps4DInt8) {
  std::vector<int8_t> out;
  // Basic 4d.
  RunTestPermutation({2, 3, 4, 5}, {2, 0, 1, 3}, &out);
  ASSERT_EQ(
      out,
      std::vector<int8_t>(
          {0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
           60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
           5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
           65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
           10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
           70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
           15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
           75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119}));
  RunTestPermutation({2, 3, 4, 5}, {0, 1, 2, 3}, &out);
  // Basic identity.
  std::vector<int8_t> ref(out.size());
  for (int k = 0; k < ref.size(); k++) ref[k] = k;
  ASSERT_EQ(out, ref);
}

class TransposeOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetPerm(std::initializer_list<int> data) {
    PopulateTensor<int>(perm_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int perm_;
  int output_;
};

// Tests case where perm is a const tensor.
//
// Example usage is as follows:
//    SpaceToBatchNDOpConstModel m(input_shape, perm_shape, perm_data);
//    m.SetInput(input_data);
//    m.Invoke();
class TransposeOpConstModel : public TransposeOpModel {
 public:
  TransposeOpConstModel(std::initializer_list<int> input_shape,
                        std::initializer_list<int> perm_shape,
                        std::initializer_list<int> perm) {
    input_ = AddInput({TensorType_FLOAT32, input_shape});
    perm_ = AddConstInput(TensorType_INT32, perm, perm_shape);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }
};

// Tests case where perm is a non-const tensor.
//
// Example usage is as follows:
//    TransposeOpDynamicModel m(input_shape, perm_shape);
//    m.SetInput(input_data);
//    m.SetPerm(perm_data);
//    m.Invoke();
class TransposeOpDynamicModel : public TransposeOpModel {
 public:
  TransposeOpDynamicModel(std::initializer_list<int> input_shape,
                          std::initializer_list<int> perm_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    perm_ = AddInput(TensorType_INT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_TRANSPOSE, BuiltinOptions_TransposeOptions,
                 CreateTransposeOptions(builder_).Union());
    BuildInterpreter({input_shape, perm_shape});
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(TransposeTest, TestUnequalPermSize) {
  EXPECT_DEATH(TransposeOpConstModel({1, 3, 3, 1}, {2}, {2, 2}), "2 != 4");
}

TEST(TransposeTest, TestPermOutOfBounds) {
  EXPECT_DEATH(TransposeOpConstModel({1, 3, 3, 1}, {4}, {0, -1, -2, -3}),
               "Transpose op permutations array is out of bounds.");
  EXPECT_DEATH(TransposeOpConstModel({1, 3, 3, 1}, {4}, {0, 1, 2, 4}),
               "Transpose op permutations array is out of bounds.");
}
#endif

TEST(TransposeTest, Test1DInputConstTensor) {
  TransposeOpConstModel m({3}, {1}, {0});
  m.SetInput({1, 2, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TEST(TransposeTest, Test1DInputDynamicTensor) {
  TransposeOpDynamicModel m({3}, {1});
  m.SetInput({1, 2, 3});
  m.SetPerm({0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3}));
}

TEST(TransposeTest, Test2DInputConstTensor) {
  TransposeOpConstModel m({3, 2}, {2}, {1, 0});
  m.SetInput({0, 1, 2, 3, 4, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

TEST(TransposeTest, Test2D4x4KernelTestLeftOverRightSide) {
  TransposeOpConstModel m({4, 6}, {2}, {1, 0});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 4}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 6, 12, 18, 1, 7,  13, 19, 2, 8,  14, 20,
                                3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23}));
}

TEST(TransposeTest, Test2D4x4KernelTest2LeftOverBottomSide) {
  TransposeOpConstModel m({6, 4}, {2}, {1, 0});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 6}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TEST(TransposeTest, Test2DInputDynamicTensor) {
  TransposeOpDynamicModel m({3, 2}, {2});
  m.SetInput({0, 1, 2, 3, 4, 5});
  m.SetPerm({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 4, 1, 3, 5}));
}

TEST(TransposeTest, Test3DInputConstTensor) {
  TransposeOpConstModel m({2, 3, 4}, {3}, {2, 0, 1});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TEST(TransposeTest, Test3DInputDynamicTensor) {
  TransposeOpDynamicModel m({2, 3, 4}, {3});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.SetPerm({2, 0, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                                2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}));
}

TEST(TransposeTest, Test1DNotShrinked) {
  TransposeOpConstModel m({1}, {1}, {0});
  m.SetInput({0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TEST(TransposeTest, Test2DShrinkedOneTime) {
  TransposeOpConstModel m({2, 1}, {2}, {1, 0});
  m.SetInput({0, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1}));
}

TEST(TransposeTest, Test2DShrinkedTwoTimes) {
  TransposeOpConstModel m({1, 1}, {2}, {1, 0});
  m.SetInput({0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TEST(TransposeTest, Test3DShrinkedOneTime) {
  TransposeOpConstModel m({2, 1, 3}, {3}, {0, 2, 1});
  m.SetInput({0, 1, 2, 3, 4, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3, 4, 5}));
}

TEST(TransposeTest, Test3DShrinkedTwoTimes) {
  TransposeOpConstModel m({1, 1, 3}, {3}, {1, 2, 0});
  m.SetInput({0, 1, 2});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2}));
}

TEST(TransposeTest, Test3DShrinkedAll) {
  TransposeOpConstModel m({1, 1, 1}, {3}, {1, 2, 0});
  m.SetInput({0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TEST(TransposeTest, Test4DShrinkedOneTimes) {
  TransposeOpConstModel m({2, 2, 3, 1}, {4}, {3, 0, 1, 2});
  m.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

TEST(TransposeTest, Test4DShrinkedTwoTimes) {
  TransposeOpConstModel m({2, 1, 3, 1}, {4}, {0, 3, 1, 2});
  m.SetInput({0, 1, 2, 3, 4, 5});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 3}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1, 2, 3, 4, 5}));
}

TEST(TransposeTest, Test4DShrinkedThirdTimes) {
  TransposeOpConstModel m({2, 1, 1, 1}, {4}, {3, 2, 1, 0});
  m.SetInput({0, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 1}));
}

TEST(TransposeTest, Test4DShrinkedFourTimes) {
  TransposeOpConstModel m({1, 1, 1, 1}, {4}, {2, 3, 1, 0});
  m.SetInput({0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0}));
}

TEST(TransposeTest, Test3DFlatten) {
  TransposeOpConstModel m({2, 2, 3}, {3}, {0, 2, 1});
  m.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}));
}

TEST(TransposeTest, Test4DFlattenOne) {
  TransposeOpConstModel m({2, 2, 2, 2}, {4}, {0, 1, 3, 2});
  m.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9,
                                               11, 12, 14, 13, 15}));
}

TEST(TransposeTest, Test4DFlattenTwo) {
  TransposeOpConstModel m({2, 2, 2, 2}, {4}, {0, 2, 3, 1});
  m.SetInput({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9,
                                               13, 10, 14, 11, 15}));
}

TEST(TransposeTest, 3DDividedIntoTwo2DsOne) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 4}, {1, 2, 0}, &out);
  TransposeOpConstModel m({2, 3, 4}, {3}, {1, 2, 0});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 3DDividedIntoTwo2DsTwo) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 4}, {2, 0, 1}, &out);
  TransposeOpConstModel m({2, 3, 4}, {3}, {2, 0, 1});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsOne) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 4, 2}, {1, 2, 3, 0}, &out);
  TransposeOpConstModel m({2, 3, 4, 2}, {4}, {1, 2, 3, 0});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsTwo) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 4, 2}, {2, 3, 0, 1}, &out);
  TransposeOpConstModel m({2, 3, 4, 2}, {4}, {2, 3, 0, 1});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 4DDividedIntoTwo2DsThird) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 4, 2}, {3, 0, 1, 2}, &out);
  TransposeOpConstModel m({2, 3, 4, 2}, {4}, {3, 0, 1, 2});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsOne) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 2, 2, 2}, {1, 4, 2, 3, 0}, &out);
  TransposeOpConstModel m({2, 3, 2, 2, 2}, {5}, {1, 4, 2, 3, 0});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsTwo) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 2, 2, 2}, {2, 3, 0, 4, 1}, &out);
  TransposeOpConstModel m({2, 3, 2, 2, 2}, {5}, {2, 3, 0, 4, 1});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

TEST(TransposeTest, 5DDividedIntoTwo2DsThird) {
  std::vector<float> out;
  RunTestPermutation({2, 3, 2, 2, 2}, {3, 0, 4, 1, 2}, &out);
  TransposeOpConstModel m({2, 3, 2, 2, 2}, {5}, {3, 0, 4, 1, 2});
  m.SetInput({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47});
  m.Invoke();
  EXPECT_EQ(m.GetOutput(), out);
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(TransposeTest, Test6DInputTensor) {
  EXPECT_DEATH(TransposeOpConstModel({1, 2, 3, 4, 5, 6}, {5}, {0, 1, 2, 3, 4}),
               "Transpose op only supports 1D-5D input arrays.");
}
#endif

TEST(TransposeTest, SimpleTestNoReorderConstTensor) {
  TransposeOpConstModel m({1, 2, 3, 1}, {4}, {0, 1, 2, 3});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(TransposeTest, SimpleTestNoReorderDynamicTensor) {
  TransposeOpDynamicModel m({1, 2, 3, 1}, {4});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPerm({0, 1, 2, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(TransposeTest, SimpleTestWithReorderConstTensor) {
  TransposeOpConstModel m({1, 2, 3, 1}, {4}, {2, 1, 3, 0});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 4, 2, 5, 3, 6}));
}

TEST(TransposeTest, ComplexTestWithReorderConstTensor) {
  TransposeOpConstModel m({2, 3, 4, 5}, {4}, {2, 0, 1, 3});
  m.SetInput({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.Invoke();

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

TEST(TransposeTest, ComplexTestWithReorderDynamicTensor) {
  TransposeOpDynamicModel m({2, 3, 4, 5}, {4});
  m.SetInput({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetPerm({2, 0, 1, 3});
  m.Invoke();

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

TEST(TransposeTest, Complex5DTestWithReorderConstTensor) {
  TransposeOpConstModel m({2, 3, 2, 2, 5}, {5}, {2, 0, 1, 4, 3});
  m.SetInput({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.Invoke();

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

TEST(TransposeTest, Complex5DTestWithReorderDynamicTensor) {
  TransposeOpDynamicModel m({2, 3, 2, 2, 5}, {5});
  m.SetInput({0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,
              12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
              36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
              48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
              60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
              84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
              96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
              108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119});
  m.SetPerm({2, 0, 1, 4, 3});
  m.Invoke();

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
