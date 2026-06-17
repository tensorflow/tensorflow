/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::tflite::variants::TensorArray;

std::vector<uint8_t> CustomOptionsToRaw(const std::vector<int32_t>& options) {
  std::vector<uint8_t> raw(options.size() * sizeof(int32_t));
  std::memcpy(raw.data(), options.data(), raw.size());
  return raw;
}

class ListReserveModel : public SingleOpModel {
 public:
  explicit ListReserveModel(TensorType element_type) {
    element_shape_input_ = AddInput({TensorType_INT32, {1}});
    list_len_input_ = AddInput({TensorType_INT32, {}});
    reserve_output_ = AddOutput({TensorType_VARIANT, {}});
    SetCustomOp("ListReserve", CustomOptionsToRaw({element_type}),
                Register_LIST_RESERVE);
    BuildInterpreter({{1}, {}});
  }
  const TfLiteTensor* GetOutputTensor(int index) {
    return interpreter_->tensor(index);
  }
  int list_len_input_;
  int reserve_output_;
  int element_shape_input_;
};

TEST(ListReserveTest, NonZeroNumElements_StaticShape) {
  ListReserveModel m(TensorType_INT32);
  m.PopulateTensor(m.list_len_input_, {5});
  m.PopulateTensor(m.element_shape_input_, {2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* tensor = m.GetOutputTensor(m.reserve_output_);
  EXPECT_EQ(tensor->type, kTfLiteVariant);
  EXPECT_EQ(tensor->allocation_type, kTfLiteVariantObject);
  TensorArray* arr = reinterpret_cast<TensorArray*>(tensor->data.data);
  EXPECT_EQ(arr->ElementType(), kTfLiteInt32);
  EXPECT_EQ(arr->ElementShape()->size, 1);
  ASSERT_EQ(arr->ElementShape()->data[0], 2);
  ASSERT_EQ(arr->NumElements(), 5);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(arr->At(i), nullptr);
  }
}

TEST(ListReserveTest, NegativeNumElements_Fails) {
  ListReserveModel m(TensorType_INT32);
  m.PopulateTensor(m.list_len_input_, {-1});
  m.PopulateTensor(m.element_shape_input_, {2});
  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListReserveTest, NumElements0_StaticShape_Succeeds) {
  ListReserveModel m(TensorType_INT32);
  m.PopulateTensor(m.list_len_input_, {0});
  m.PopulateTensor(m.element_shape_input_, {2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* tensor = m.GetOutputTensor(m.reserve_output_);
  TensorArray* arr = reinterpret_cast<TensorArray*>(tensor->data.data);
  EXPECT_EQ(arr->NumElements(), 0);
  EXPECT_EQ(arr->ElementType(), kTfLiteInt32);
}

TEST(ListReserveTest, NumElements0_StaticShape_FloatType) {
  ListReserveModel m(TensorType_FLOAT32);
  m.PopulateTensor(m.list_len_input_, {0});
  m.PopulateTensor(m.element_shape_input_, {2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* tensor = m.GetOutputTensor(m.reserve_output_);
  TensorArray* arr = reinterpret_cast<TensorArray*>(tensor->data.data);
  EXPECT_EQ(arr->NumElements(), 0);
  EXPECT_EQ(arr->ElementType(), kTfLiteFloat32);
}

TEST(ListReserveTest, UnsupportedDataType_Fails) {
  ListReserveModel m(TensorType_COMPLEX64);
  m.PopulateTensor(m.list_len_input_, {0});
  m.PopulateTensor(m.element_shape_input_, {2});
  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
