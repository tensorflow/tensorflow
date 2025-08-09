/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "Eigen/Core"
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

enum class TestType {
  kConst = 0,
  kDynamic = 1,
};

template <typename input_type, typename index_type>
class SliceOpModel : public SingleOpModel {
 public:
  SliceOpModel(std::initializer_list<int> input_shape,
               std::initializer_list<int> begin_shape,
               std::initializer_list<index_type> begin_data,
               std::initializer_list<int> size_shape,
               std::initializer_list<index_type> size_data,
               TensorType tensor_index_type, TensorType tensor_input_type,
               TestType input_tensor_types,
               std::initializer_list<int> output_shape = {}) {
    input_ = AddInput(tensor_input_type);
    if (input_tensor_types == TestType::kDynamic) {
      begin_ = AddInput(tensor_index_type);
      size_ = AddInput(tensor_index_type);
    } else {
      begin_ =
          AddConstInput(GetTensorType<index_type>(), begin_data, begin_shape);
      size_ = AddConstInput(GetTensorType<index_type>(), size_data, size_shape);
    }
    output_ = AddOutput(TensorData(tensor_input_type, output_shape));
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({input_shape, begin_shape, size_shape});

    if (input_tensor_types == TestType::kDynamic) {
      PopulateTensor<index_type>(begin_, begin_data);
      PopulateTensor<index_type>(size_, size_data);
    }
  }

  void SetInput(std::initializer_list<input_type> data) {
    PopulateTensor<input_type>(input_, data);
  }
  void SetStringInput(std::vector<string> data) {
    PopulateStringTensor(input_, data);
  }

  std::vector<input_type> GetOutput() {
    return ExtractVector<input_type>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  const TfLiteTensor* GetOutputTensor() {
    return interpreter_->tensor(output_);
  }

 private:
  int input_;
  int begin_;
  int size_;
  int output_;
};

class SliceOpTest : public ::testing::TestWithParam<TestType> {};

TEST_P(SliceOpTest, In1D) {
  SliceOpModel<float, int32_t> m({4}, {1}, {1}, {1}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3}));
}

TEST_P(SliceOpTest, In2D) {
  SliceOpModel<float, int32_t> m({2, 3}, {2}, {1, 0}, {2}, {1, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
}

TEST_P(SliceOpTest, In3D) {
  SliceOpModel<float, int32_t> m({2, 3, 2}, {3}, {0, 0, 0}, {3}, {2, 3, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 3, 2}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
}

TEST_P(SliceOpTest, In5D) {
  SliceOpModel<float, int32_t> m({5, 1, 1, 1, 1}, {5}, {1, 0, 0, 0, 0}, {5},
                                 {3, 1, 1, 1, 1}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4, 5});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST_P(SliceOpTest, InputFloat) {
  SliceOpModel<float, int32_t> m({4, 1, 1, 1}, {4}, {1, 0, 0, 0}, {4},
                                 {3, 1, 1, 1}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

TEST_P(SliceOpTest, IndexInt64) {
  SliceOpModel<float, int64_t> m({4, 1, 1, 1}, {4}, {1, 0, 0, 0}, {4},
                                 {3, 1, 1, 1}, TensorType_INT64,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 1, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({2, 3, 4}));
}

// See these test cases under:
// https://www.tensorflow.org/versions/master/api_docs/python/tf/slice
TEST_P(SliceOpTest, InputInteger1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {1, 1, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3}));
}

TEST_P(SliceOpTest, InputInteger2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {1, 2, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 4, 4, 4}));
}

TEST_P(SliceOpTest, InputInteger3) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, 3, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SizeMinus1) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis1) {
  SliceOpModel<int32_t, int32_t> m({3, 3, 2, 1}, {4}, {1, 1, 0, 0}, {4},
                                   {2, -1, 1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6, 8, 9}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis2) {
  SliceOpModel<int32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 1, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis3) {
  SliceOpModel<int32_t, int32_t> m({3, 1, 2, 3}, {4}, {1, 0, 0, 1}, {4},
                                   {2, 1, 1, -1}, TensorType_INT32,
                                   TensorType_INT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 5, 5}));
}

TEST_P(SliceOpTest, SliceUint8) {
  SliceOpModel<uint8_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_UINT8, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceUint32) {
  SliceOpModel<uint32_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                    {2, 1, -1, 1}, TensorType_INT32,
                                    TensorType_UINT32, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceInt8) {
  SliceOpModel<int8_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_INT8, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceInt16) {
  SliceOpModel<int16_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT16, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceString) {
  SliceOpModel<string, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_STRING, GetParam());
  m.SetStringInput({"0,0,0,0", "0,0,1,0", "0,0,2,0",  //
                    "0,1,0,0", "0,1,1,0", "0,1,2,0",  //
                    "1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                    "1,1,0,0", "1,1,1,0", "1,1,2,0",  //
                    "2,0,0,0", "2,0,1,0", "2,0,2,0",  //
                    "2,1,0,0", "2,1,1,0", "2,1,2,0"});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({"1,0,0,0", "1,0,1,0", "1,0,2,0",  //
                                "2,0,0,0", "2,0,1,0", "2,0,2,0"}));
}

TEST_P(SliceOpTest, SliceInt64) {
  SliceOpModel<int64_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT64, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceInt64StaticOutput) {
  SliceOpModel<int64_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                   {2, 1, -1, 1}, TensorType_INT32,
                                   TensorType_INT64, GetParam(), {2, 1, 3, 1});
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
  EXPECT_NE(m.GetOutputTensor()->allocation_type, kTfLiteDynamic);
}

TEST_P(SliceOpTest, SliceBool) {
  SliceOpModel<bool, int32_t> m({2, 3}, {2}, {1, 0}, {2}, {-1, 2},
                                TensorType_INT32, TensorType_BOOL, GetParam());
  m.SetInput({true, false, true, false, true, true});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({false, true}));
}

TEST_P(SliceOpTest, SliceFloat16) {
  SliceOpModel<Eigen::half, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                       {2, 1, -1, 1}, TensorType_INT32,
                                       TensorType_FLOAT16, GetParam());
  m.SetInput({Eigen::half(1), Eigen::half(1), Eigen::half(1), Eigen::half(2),
              Eigen::half(2), Eigen::half(2), Eigen::half(3), Eigen::half(3),
              Eigen::half(3), Eigen::half(4), Eigen::half(4), Eigen::half(4),
              Eigen::half(5), Eigen::half(5), Eigen::half(5), Eigen::half(6),
              Eigen::half(6), Eigen::half(6)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({Eigen::half(3), Eigen::half(3), Eigen::half(3),
                        Eigen::half(5), Eigen::half(5), Eigen::half(5)}));
}

TEST_P(SliceOpTest, SliceBFloat16) {
  SliceOpModel<Eigen::bfloat16, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                           {2, 1, -1, 1}, TensorType_INT32,
                                           TensorType_BFLOAT16, GetParam());
  m.SetInput({Eigen::bfloat16(1), Eigen::bfloat16(1), Eigen::bfloat16(1),
              Eigen::bfloat16(2), Eigen::bfloat16(2), Eigen::bfloat16(2),
              Eigen::bfloat16(3), Eigen::bfloat16(3), Eigen::bfloat16(3),
              Eigen::bfloat16(4), Eigen::bfloat16(4), Eigen::bfloat16(4),
              Eigen::bfloat16(5), Eigen::bfloat16(5), Eigen::bfloat16(5),
              Eigen::bfloat16(6), Eigen::bfloat16(6), Eigen::bfloat16(6)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({Eigen::bfloat16(3), Eigen::bfloat16(3),
                                Eigen::bfloat16(3), Eigen::bfloat16(5),
                                Eigen::bfloat16(5), Eigen::bfloat16(5)}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis1Float16) {
  SliceOpModel<Eigen::half, int32_t> m({3, 3, 2, 1}, {4}, {1, 1, 0, 0}, {4},
                                       {2, -1, 1, 1}, TensorType_INT32,
                                       TensorType_FLOAT16, GetParam());
  m.SetInput({Eigen::half(1), Eigen::half(1), Eigen::half(2), Eigen::half(2),
              Eigen::half(3), Eigen::half(3), Eigen::half(4), Eigen::half(4),
              Eigen::half(5), Eigen::half(5), Eigen::half(6), Eigen::half(6),
              Eigen::half(7), Eigen::half(7), Eigen::half(8), Eigen::half(8),
              Eigen::half(9), Eigen::half(9)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({Eigen::half(5), Eigen::half(6), Eigen::half(8),
                                Eigen::half(9)}));
}

TEST_P(SliceOpTest, BeginNonZeroSizeMinus1Axis1BFloat16) {
  SliceOpModel<Eigen::bfloat16, int32_t> m({3, 3, 2, 1}, {4}, {1, 1, 0, 0}, {4},
                                           {2, -1, 1, 1}, TensorType_INT32,
                                           TensorType_BFLOAT16, GetParam());
  m.SetInput({Eigen::bfloat16(1), Eigen::bfloat16(1), Eigen::bfloat16(2),
              Eigen::bfloat16(2), Eigen::bfloat16(3), Eigen::bfloat16(3),
              Eigen::bfloat16(4), Eigen::bfloat16(4), Eigen::bfloat16(5),
              Eigen::bfloat16(5), Eigen::bfloat16(6), Eigen::bfloat16(6),
              Eigen::bfloat16(7), Eigen::bfloat16(7), Eigen::bfloat16(8),
              Eigen::bfloat16(8), Eigen::bfloat16(9), Eigen::bfloat16(9)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({Eigen::bfloat16(5), Eigen::bfloat16(6),
                                Eigen::bfloat16(8), Eigen::bfloat16(9)}));
}

INSTANTIATE_TEST_SUITE_P(SliceOpTest, SliceOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
