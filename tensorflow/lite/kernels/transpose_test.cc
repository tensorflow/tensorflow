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
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

void RunTestPermutation(const std::vector<int>& shape,
                        const std::vector<int>& perms,
                        std::vector<float>* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (auto factor : shape) count *= factor;
  input_transposed->resize(count);

  // Create the dummy data
  std::vector<float> input(count);
  for (int i = 0; i < input.size(); i++) {
    input[i] = i;
  }

  // Create reversed and padded perms.
  int reversed_perms[4];
  for (int output_k = 0, input_k = shape.size() - 1; output_k < shape.size();
       output_k++, input_k--) {
    reversed_perms[output_k] = shape.size() - perms[input_k] - 1;
  }
  // Unused dimensions should not be permuted so pad with identity transform
  // subset.
  for (int k = shape.size(); k < 4; k++) {
    reversed_perms[k] = k;
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

  reference_ops::Transpose<float>(params, input_shape, input.data(),
                                  output_shape, input_transposed->data());
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
  // Test 3 dimensional
  {
    std::vector<float> ref({0, 4, 8,  12, 16, 20, 1, 5, 9,  13, 17, 21,
                            2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});
    RunTestPermutation({2, 3, 4}, {2, 0, 1}, &out);
    ASSERT_EQ(out, ref);
  }
  // Test 3 dimensional identity transform
  {
    RunTestPermutation({2, 3, 4}, {0, 1, 2}, &out);
    std::vector<float> ref(out.size());
    for (int k = 0; k < ref.size(); k++) ref[k] = k;
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
    input_ = AddInput(TensorType_FLOAT32);
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

#ifdef GTEST_HAS_DEATH_TEST
TEST(TransposeTest, Test5DInputTensor) {
  EXPECT_DEATH(TransposeOpConstModel({1, 2, 3, 4, 5}, {5}, {0, 1, 2, 3, 4}),
               "Transpose op only supports 1D-4D input arrays.");
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

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
