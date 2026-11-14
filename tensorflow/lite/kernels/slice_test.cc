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
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace ops {
namespace builtin {
TfLiteRegistration* Register_SLICE_REF();
}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

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
               std::initializer_list<int> output_shape = {},
               bool allocate_and_delegate = true,
               bool use_reference_kernel = false) {
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
    if (use_reference_kernel) {
      SetResolver(std::make_unique<SingleOpResolver>(
          BuiltinOperator_SLICE, ops::builtin::Register_SLICE_REF()));
    }
    BuildInterpreter({input_shape, begin_shape, size_shape}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, allocate_and_delegate);

    if (allocate_and_delegate && input_tensor_types == TestType::kDynamic) {
      PopulateTensor<index_type>(begin_, begin_data);
      PopulateTensor<index_type>(size_, size_data);
    }
  }

  void SetInput(std::initializer_list<input_type> data) {
    if constexpr (std::is_same_v<input_type, int8_t>) {
      if (interpreter_->tensor(input_)->type == kTfLiteInt4) {
        PopulateTensor4bit(input_, 0, data.begin(), data.end());
        return;
      }
    }
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

template <typename begin_type, typename size_type>
class MismatchedIndexSliceOpModel : public SingleOpModel {
 public:
  MismatchedIndexSliceOpModel(std::initializer_list<int> input_shape,
                              std::initializer_list<int> begin_shape,
                              std::initializer_list<begin_type> begin_data,
                              std::initializer_list<int> size_shape,
                              std::initializer_list<size_type> size_data,
                              TestType input_tensor_types) {
    input_ = AddInput(TensorType_FLOAT32);
    if (input_tensor_types == TestType::kDynamic) {
      begin_ = AddInput(GetTensorType<begin_type>());
      size_ = AddInput(GetTensorType<size_type>());
    } else {
      begin_ =
          AddConstInput(GetTensorType<begin_type>(), begin_data, begin_shape);
      size_ = AddConstInput(GetTensorType<size_type>(), size_data, size_shape);
    }
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_SLICE, BuiltinOptions_SliceOptions,
                 CreateSliceOptions(builder_).Union());
    BuildInterpreter({input_shape, begin_shape, size_shape}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true,
                     /*allocate_and_delegate=*/false);
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

TEST_P(SliceOpTest, In6D) {
  SliceOpModel<float, int32_t> m({2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1},
                                 {6}, {1, 1, 1, 1, 1, 2}, TensorType_INT32,
                                 TensorType_FLOAT32, GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
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

TEST_P(SliceOpTest, SliceInt4) {
  SliceOpModel<int8_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_INT4, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  const TfLiteTensor* output_tensor = m.GetOutputTensor();
  int num_elements = NumElements(output_tensor);
  std::vector<int8_t> unpacked_output(num_elements);
  tensor_utils::UnpackPackedIntToInt8(GetTensorData<int8_t>(output_tensor),
                                      num_elements,
                                      /*bit_width=*/4, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({3, 3, 3, 5, 5, 5}));
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
  SliceOpModel<half, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                {2, 1, -1, 1}, TensorType_INT32,
                                TensorType_FLOAT16, GetParam());
  m.SetInput({half(1), half(1), half(1), half(2), half(2), half(2), half(3),
              half(3), half(3), half(4), half(4), half(4), half(5), half(5),
              half(5), half(6), half(6), half(6)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({half(3), half(3), half(3),
                                               half(5), half(5), half(5)}));
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
  SliceOpModel<half, int32_t> m({3, 3, 2, 1}, {4}, {1, 1, 0, 0}, {4},
                                {2, -1, 1, 1}, TensorType_INT32,
                                TensorType_FLOAT16, GetParam());
  m.SetInput({half(1), half(1), half(2), half(2), half(3), half(3), half(4),
              half(4), half(5), half(5), half(6), half(6), half(7), half(7),
              half(8), half(8), half(9), half(9)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 1, 1}));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({half(5), half(6), half(8), half(9)}));
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

TEST_P(SliceOpTest, SliceUint16) {
  SliceOpModel<uint16_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                    {2, 1, -1, 1}, TensorType_INT32,
                                    TensorType_UINT16, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceUint64) {
  SliceOpModel<uint64_t, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                    {2, 1, -1, 1}, TensorType_INT32,
                                    TensorType_UINT64, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, SliceFloat64) {
  SliceOpModel<double, int32_t> m({3, 2, 3, 1}, {4}, {1, 0, 0, 0}, {4},
                                  {2, 1, -1, 1}, TensorType_INT32,
                                  TensorType_FLOAT64, GetParam());
  m.SetInput({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 3, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 3, 3, 5, 5, 5}));
}

TEST_P(SliceOpTest, AllDataTypesWithInt64Indices) {
  {
    SliceOpModel<uint8_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                     TensorType_INT64, TensorType_UINT8,
                                     GetParam());
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<int8_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                    TensorType_INT64, TensorType_INT8,
                                    GetParam());
    m.SetInput({-1, -2, -3, -4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<bool, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                  TensorType_INT64, TensorType_BOOL,
                                  GetParam());
    m.SetInput({true, false, true, false, true, false});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({true, false}));
  }
  {
    SliceOpModel<int16_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                     TensorType_INT64, TensorType_INT16,
                                     GetParam());
    m.SetInput({10, 20, 30, 40, -50, 60});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({-50, 60}));
  }
  {
    SliceOpModel<uint16_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                      TensorType_INT64, TensorType_UINT16,
                                      GetParam());
    m.SetInput({10, 20, 30, 40, 50, 60});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({50, 60}));
  }
  {
    SliceOpModel<half, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                  TensorType_INT64, TensorType_FLOAT16,
                                  GetParam());
    m.SetInput({half(1), half(2), half(3), half(4), half(5), half(6)});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({half(5), half(6)}));
  }
  {
    SliceOpModel<Eigen::bfloat16, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                             TensorType_INT64,
                                             TensorType_BFLOAT16, GetParam());
    m.SetInput({Eigen::bfloat16(1), Eigen::bfloat16(2), Eigen::bfloat16(3),
                Eigen::bfloat16(4), Eigen::bfloat16(5), Eigen::bfloat16(6)});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(),
                ElementsAreArray({Eigen::bfloat16(5), Eigen::bfloat16(6)}));
  }
  {
    SliceOpModel<int32_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                     TensorType_INT64, TensorType_INT32,
                                     GetParam());
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<uint32_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                      TensorType_INT64, TensorType_UINT32,
                                      GetParam());
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<int64_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                     TensorType_INT64, TensorType_INT64,
                                     GetParam());
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<uint64_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                      TensorType_INT64, TensorType_UINT64,
                                      GetParam());
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
  }
  {
    SliceOpModel<double, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                    TensorType_INT64, TensorType_FLOAT64,
                                    GetParam());
    m.SetInput({1.0, 2.0, 3.0, 4.0, 5.5, 6.5});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({5.5, 6.5}));
  }
  {
    SliceOpModel<int8_t, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                    TensorType_INT64, TensorType_INT4,
                                    GetParam());
    m.SetInput({1, 2, 3, 4, -5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    const TfLiteTensor* output_tensor = m.GetOutputTensor();
    const int num_elements = NumElements(output_tensor);
    std::vector<int8_t> unpacked_output(num_elements);
    tensor_utils::UnpackPackedIntToInt8(GetTensorData<int8_t>(output_tensor),
                                        num_elements, /*bit_width=*/4,
                                        unpacked_output.data());
    EXPECT_THAT(unpacked_output, ElementsAreArray({-5, 6}));
  }
  {
    SliceOpModel<string, int64_t> m({2, 3}, {2}, {1, 1}, {2}, {1, -1},
                                    TensorType_INT64, TensorType_STRING,
                                    GetParam());
    m.SetStringInput({"a", "b", "c", "d", "e", "f"});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({"e", "f"}));
  }
}

// Covers the `input_dims > kMaxDim` (6D) dynamic-rank reference path in Eval()
// across all supported bit widths (8, 16, 32, 64), INT4, and STRING, for both
// optimized and reference registrations.
TEST_P(SliceOpTest, In6DAllDataTypesAndKernelVariants) {
  for (bool use_reference : {false, true}) {
    {
      SliceOpModel<float, int64_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, -1},
          TensorType_INT64, TensorType_FLOAT32, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<int32_t, int64_t> m(
          {2, 2, 1, 1, 2, 2}, {6}, {0, 1, 0, 0, 0, 1}, {6}, {2, 1, 1, 1, -1, 1},
          TensorType_INT64, TensorType_INT32, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 1, 1, 2, 1}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({6, 8, 14, 16}));
    }
    {
      SliceOpModel<uint32_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_UINT32, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<int64_t, int64_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT64, TensorType_INT64, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<uint64_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_UINT64, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<double, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_FLOAT64, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5.0, 6.0}));
    }
    {
      SliceOpModel<int16_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_INT16, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<uint16_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_UINT16, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<half, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_FLOAT16, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({half(1), half(2), half(3), half(4), half(5), half(6)});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({half(5), half(6)}));
    }
    {
      SliceOpModel<Eigen::bfloat16, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_BFLOAT16, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({Eigen::bfloat16(1), Eigen::bfloat16(2), Eigen::bfloat16(3),
                  Eigen::bfloat16(4), Eigen::bfloat16(5), Eigen::bfloat16(6)});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(),
                  ElementsAreArray({Eigen::bfloat16(5), Eigen::bfloat16(6)}));
    }
    {
      SliceOpModel<int8_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_INT8, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<uint8_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_UINT8, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, 5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 6}));
    }
    {
      SliceOpModel<bool, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_BOOL, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({true, false, true, false, true, false});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({true, false}));
    }
    {
      SliceOpModel<int8_t, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_INT4, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetInput({1, 2, 3, 4, -5, 6});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      const TfLiteTensor* output_tensor = m.GetOutputTensor();
      const int num_elements = NumElements(output_tensor);
      std::vector<int8_t> unpacked_output(num_elements);
      tensor_utils::UnpackPackedIntToInt8(GetTensorData<int8_t>(output_tensor),
                                          num_elements, /*bit_width=*/4,
                                          unpacked_output.data());
      EXPECT_THAT(unpacked_output, ElementsAreArray({-5, 6}));
    }
    {
      SliceOpModel<string, int32_t> m(
          {2, 1, 1, 1, 1, 3}, {6}, {1, 0, 0, 0, 0, 1}, {6}, {1, 1, 1, 1, 1, 2},
          TensorType_INT32, TensorType_STRING, GetParam(), {},
          /*allocate_and_delegate=*/true, use_reference);
      m.SetStringInput({"a", "b", "c", "d", "e", "f"});
      ASSERT_EQ(m.Invoke(), kTfLiteOk);
      EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 1, 1, 1, 2}));
      EXPECT_THAT(m.GetOutput(), ElementsAreArray({"e", "f"}));
    }
  }
}

TEST_P(SliceOpTest, ReferenceKernelUpTo5D) {
  {
    SliceOpModel<float, int32_t> m(
        {2, 3}, {2}, {1, 0}, {2}, {1, 2}, TensorType_INT32, TensorType_FLOAT32,
        GetParam(), {}, /*allocate_and_delegate=*/true,
        /*use_reference_kernel=*/true);
    m.SetInput({1, 2, 3, 4, 5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({4, 5}));
  }
  {
    SliceOpModel<int8_t, int32_t> m(
        {2, 3}, {2}, {1, 1}, {2}, {1, -1}, TensorType_INT32, TensorType_INT4,
        GetParam(), {}, /*allocate_and_delegate=*/true,
        /*use_reference_kernel=*/true);
    m.SetInput({1, 2, 3, 4, -5, 6});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    const TfLiteTensor* output_tensor = m.GetOutputTensor();
    const int num_elements = NumElements(output_tensor);
    std::vector<int8_t> unpacked_output(num_elements);
    tensor_utils::UnpackPackedIntToInt8(GetTensorData<int8_t>(output_tensor),
                                        num_elements, /*bit_width=*/4,
                                        unpacked_output.data());
    EXPECT_THAT(unpacked_output, ElementsAreArray({-5, 6}));
  }
  {
    SliceOpModel<string, int32_t> m(
        {2, 3}, {2}, {1, 1}, {2}, {1, -1}, TensorType_INT32, TensorType_STRING,
        GetParam(), {}, /*allocate_and_delegate=*/true,
        /*use_reference_kernel=*/true);
    m.SetStringInput({"a", "b", "c", "d", "e", "f"});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
    EXPECT_THAT(m.GetOutput(), ElementsAreArray({"e", "f"}));
  }
}

TEST_P(SliceOpTest, ZeroSizeSlice) {
  SliceOpModel<float, int32_t> m({2, 3}, {2}, {1, 0}, {2}, {0, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam());
  m.SetInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({0, 2}));
  EXPECT_THAT(m.GetOutput(), IsEmpty());
}

// A statically shaped output must keep its static allocation. Prepare() returns
// early for such an output; this asserts that validating it does not convert it
// to a dynamic tensor, which would change arena planning for valid models.
TEST(SliceOpTest, StaticOutputKeepsStaticAllocation) {
  SliceOpModel<float, int32_t> m({4}, {1}, {1}, {1}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32, TestType::kConst, {2});
  m.SetInput({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_NE(m.GetOutputTensor()->allocation_type, kTfLiteDynamic);
}

// `begin` and `size` must have matching index types (`INT32` or `INT64`).
TEST_P(SliceOpTest, RejectsMismatchedBeginAndSizeIndexTypes) {
  {
    MismatchedIndexSliceOpModel<int32_t, int64_t> m({2, 3}, {2}, {0, 0}, {2},
                                                    {1, 2}, GetParam());
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    MismatchedIndexSliceOpModel<int64_t, int32_t> m({2, 3}, {2}, {0, 0}, {2},
                                                    {1, 2}, GetParam());
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
}

TEST_P(SliceOpTest, RejectsMismatchedBeginAndSizeElementCounts) {
  SliceOpModel<float, int32_t> m({2, 3}, {1}, {0}, {2}, {1, 2},
                                 TensorType_INT32, TensorType_FLOAT32,
                                 GetParam(), {},
                                 /*allocate_and_delegate=*/false);
  EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
}

// `begin` and `size` supplied at run time are not visible to Prepare(), which
// returns early because the output shape is fully specified. Without validation
// in Eval() the kernel reads past the end of the input: begin 3 + size 2
// exceeds the input extent of 4.
TEST(SliceOpTest, RuntimeIndicesOutOfBoundsWithStaticOutput) {
  SliceOpModel<float, int32_t> m({4}, {1}, {3}, {1}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32, TestType::kDynamic, {2});
  m.SetInput({1, 2, 3, 4});
  EXPECT_NE(m.Invoke(), kTfLiteOk);
}

// The same case with int64 indices, so both supported index widths are covered.
TEST(SliceOpTest, RuntimeIndicesOutOfBoundsWithStaticOutputInt64) {
  SliceOpModel<float, int64_t> m({4}, {1}, {3}, {1}, {2}, TensorType_INT64,
                                 TensorType_FLOAT32, TestType::kDynamic, {2});
  m.SetInput({1, 2, 3, 4});
  EXPECT_NE(m.Invoke(), kTfLiteOk);
}

// A negative `begin` is rejected before it is used in the bounds arithmetic.
TEST(SliceOpTest, NegativeBeginWithStaticOutput) {
  SliceOpModel<float, int32_t> m({4}, {1}, {-1}, {1}, {2}, TensorType_INT32,
                                 TensorType_FLOAT32, TestType::kDynamic, {2});
  m.SetInput({1, 2, 3, 4});
  EXPECT_NE(m.Invoke(), kTfLiteOk);
}

TEST(SliceOpTest, NegativeBeginRejectedForConstAndDynamic) {
  {
    SliceOpModel<float, int32_t> m({4}, {1}, {-1}, {1}, {2}, TensorType_INT32,
                                   TensorType_FLOAT32, TestType::kConst, {},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({4}, {1}, {-1}, {1}, {2}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kConst, {2},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({4}, {1}, {-1}, {1}, {2}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kDynamic);
    m.SetInput({1, 2, 3, 4});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
}

TEST(SliceOpTest, BeginGreaterThanInputDimensionRejected) {
  {
    SliceOpModel<float, int32_t> m({4}, {1}, {5}, {1}, {0}, TensorType_INT32,
                                   TensorType_FLOAT32, TestType::kConst, {},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({4}, {1}, {5}, {1}, {0}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kDynamic);
    m.SetInput({1, 2, 3, 4});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
}

TEST(SliceOpTest, SizeLessThanMinusOneRejected) {
  {
    SliceOpModel<float, int32_t> m({4}, {1}, {0}, {1}, {-2}, TensorType_INT32,
                                   TensorType_FLOAT32, TestType::kConst, {},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({4}, {1}, {0}, {1}, {-2}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kDynamic, {2});
    m.SetInput({1, 2, 3, 4});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
}

TEST(SliceOpTest, BeginPlusSizeOverflowRejected) {
  {
    SliceOpModel<float, int32_t> m(
        {4}, {1}, {2}, {1}, {std::numeric_limits<int32_t>::max()},
        TensorType_INT32, TensorType_FLOAT32, TestType::kDynamic, {2});
    m.SetInput({1, 2, 3, 4});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m(
        {4}, {1}, {2}, {1}, {std::numeric_limits<int64_t>::max()},
        TensorType_INT64, TensorType_FLOAT32, TestType::kDynamic, {2});
    m.SetInput({1, 2, 3, 4});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
}

// CalculateOutputShapeVector() indexes `begin` and `size` by input dimension.
// Prepare() only checks the two against each other, so an index vector shorter
// or longer than the input rank is rejected rather than read out of bounds.
TEST(SliceOpTest, IndexVectorShorterThanInputRank) {
  {
    SliceOpModel<float, int32_t> m({2, 3}, {1}, {0}, {1}, {1}, TensorType_INT32,
                                   TensorType_FLOAT32, TestType::kDynamic,
                                   {1, 1});
    m.SetInput({1, 2, 3, 4, 5, 6});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({2, 3}, {1}, {0}, {1}, {1}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kDynamic);
    m.SetInput({1, 2, 3, 4, 5, 6});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int32_t> m({2, 3}, {1}, {0}, {1}, {1}, TensorType_INT32,
                                   TensorType_FLOAT32, TestType::kConst, {},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({2, 3}, {1}, {0}, {1}, {1}, TensorType_INT64,
                                   TensorType_FLOAT32, TestType::kConst, {1, 1},
                                   /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
}

TEST(SliceOpTest, IndexVectorLongerThanInputRank) {
  {
    SliceOpModel<float, int32_t> m(
        {2}, {2}, {0, 0}, {2}, {1, 1}, TensorType_INT32, TensorType_FLOAT32,
        TestType::kConst, {}, /*allocate_and_delegate=*/false);
    EXPECT_NE(m.AllocateTensors(), kTfLiteOk);
  }
  {
    SliceOpModel<float, int64_t> m({2}, {2}, {0, 0}, {2}, {1, 1},
                                   TensorType_INT64, TensorType_FLOAT32,
                                   TestType::kDynamic, {1});
    m.SetInput({1, 2});
    EXPECT_NE(m.Invoke(), kTfLiteOk);
  }
}

// A valid window whose computed shape disagrees with the declared static output
// must fail cleanly rather than write into a mismatched buffer.
TEST(SliceOpTest, ComputedShapeMustMatchDeclaredStaticOutput) {
  SliceOpModel<float, int32_t> m({4}, {1}, {0}, {1}, {3}, TensorType_INT32,
                                 TensorType_FLOAT32, TestType::kDynamic, {2});
  m.SetInput({1, 2, 3, 4});
  EXPECT_NE(m.Invoke(), kTfLiteOk);
}

INSTANTIATE_TEST_SUITE_P(SliceOpTest, SliceOpTest,
                         ::testing::Values(TestType::kConst,
                                           TestType::kDynamic));

}  // namespace
}  // namespace tflite
