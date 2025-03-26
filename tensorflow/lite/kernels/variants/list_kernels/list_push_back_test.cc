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
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::testing::AllOf;

template <typename T>
class PushBackWithTypeTest : public ::testing::Test {};

class ListPushBackModel : public ListOpModel {
 public:
  explicit ListPushBackModel(TensorData item_data) {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    tensor_input_ = AddInput(item_data);

    list_output_ = AddOutput({TensorType_VARIANT, {}});

    SetCustomOp("ListPushBack", {}, Register_LIST_PUSH_BACK);

    BuildInterpreter({{}, item_data.shape});

    interpreter_->input_tensor(0)->allocation_type = kTfLiteVariantObject;
  }

  const TensorArray* GetOutputTensorArray() {
    TfLiteTensor* tensor = interpreter_->tensor(list_output_);
    TFLITE_CHECK(tensor != nullptr && tensor->type == kTfLiteVariant &&
                 tensor->allocation_type == kTfLiteVariantObject);
    return static_cast<const TensorArray*>(
        static_cast<const VariantData*>(tensor->data.data));
  }

  int list_input_;
  int tensor_input_;
  int list_output_;
};

constexpr int kNumElements = 4;

TYPED_TEST_SUITE_P(PushBackWithTypeTest);

TYPED_TEST_P(PushBackWithTypeTest, PushBackOnEmptyTensorList_ListShapeDefined) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListPushBackModel m({tensor_type.value(), {2, 2}});

  m.PopulateListTensor(m.list_input_, {2, 2}, kNumElements, tfl_type);
  m.PopulateTensor<TypeParam>(m.tensor_input_, {0, 0, 0, 0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* const arr = m.GetOutputTensorArray();

  ASSERT_EQ(arr->NumElements(), kNumElements + 1);
  ASSERT_EQ(arr->ElementType(), tfl_type);

  for (int i = 0; i < arr->NumElements() - 1; ++i) {
    EXPECT_EQ(arr->At(i), nullptr);
  }

  // Sometimes sanitizers will fail if this is inlined within macro.
  const int last_ind = arr->NumElements() - 1;
  EXPECT_THAT(arr->At(last_ind), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                       FilledWith<TypeParam>(0)));
}

TYPED_TEST_P(PushBackWithTypeTest,
             PushBackOnEmptyTensorList_ListShapeUnranked) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListPushBackModel m({tensor_type.value(), {2, 2}});

  m.PopulateListTensor(m.list_input_, {}, kNumElements, tfl_type);
  m.PopulateTensor<TypeParam>(m.tensor_input_, {0, 0, 0, 0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* const arr = m.GetOutputTensorArray();

  ASSERT_EQ(arr->NumElements(), kNumElements + 1);
  ASSERT_EQ(arr->ElementType(), tfl_type);

  for (int i = 0; i < arr->NumElements() - 1; ++i) {
    EXPECT_EQ(arr->At(i), nullptr);
  }
  // Sometimes sanitizers will fail if this is inlined within macro.
  const int last_ind = arr->NumElements() - 1;
  EXPECT_THAT(arr->At(last_ind), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                       FilledWith<TypeParam>(0)));
}

TYPED_TEST_P(PushBackWithTypeTest,
             PushBackOnNonEmptyListAtEmptyIndex_ItemsSameShape) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListPushBackModel m({tensor_type.value(), {2, 2}});

  m.PopulateListTensor(m.list_input_, {}, kNumElements, tfl_type);

  TypeParam init_item_data[4] = {1, 1, 1, 1};
  m.ListSetItem(m.list_input_, 0, {2, 2}, tfl_type, init_item_data);

  m.PopulateTensor<TypeParam>(m.tensor_input_, {0, 0, 0, 0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* const arr = m.GetOutputTensorArray();

  ASSERT_EQ(arr->NumElements(), kNumElements + 1);
  ASSERT_EQ(arr->ElementType(), tfl_type);

  for (int i = 1; i < arr->NumElements() - 1; ++i) {
    EXPECT_EQ(arr->At(i), nullptr);
  }

  EXPECT_THAT(arr->At(0), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                FilledWith<TypeParam>(1)));
  // Sometimes sanitizers will fail if this is inlined within macro.
  const int last_ind = arr->NumElements() - 1;
  EXPECT_THAT(arr->At(last_ind), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                       FilledWith<TypeParam>(0)));
}

REGISTER_TYPED_TEST_SUITE_P(PushBackWithTypeTest,
                            PushBackOnEmptyTensorList_ListShapeDefined,
                            PushBackOnEmptyTensorList_ListShapeUnranked,
                            PushBackOnNonEmptyListAtEmptyIndex_ItemsSameShape);

using ValidTypes = ::testing::Types<int, int64_t, bool, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(PushBackTests, PushBackWithTypeTest, ValidTypes);

TEST(ListPushBackTest, ItemNotSameTypeAsList_Fails) {
  ListPushBackModel m{{TensorType_INT32, {2, 2}}};

  m.PopulateListTensor(m.list_input_, {2, 2}, kNumElements, kTfLiteInt64);

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
