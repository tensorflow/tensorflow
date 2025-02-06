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

class ListAddNModel : public ListOpModel {
 public:
  explicit ListAddNModel(int num_inputs) {
    std::vector<std::vector<int>> input_shapes(num_inputs, std::vector<int>{});
    for (int i = 0; i < num_inputs; ++i) {
      input_inds_.push_back(AddInput({TensorType_VARIANT, {}}));
    }
    output_ind_ = AddOutput({TensorType_VARIANT, {}});
    SetCustomOp("VariantAddN", {}, Register_VARIANT_ADD_N);
    BuildInterpreter(input_shapes);
  }

  const TensorArray* GetOutput() {
    TfLiteTensor* tensor = interpreter_->tensor(output_ind_);
    TFLITE_CHECK(tensor != nullptr && tensor->type == kTfLiteVariant &&
                 tensor->allocation_type == kTfLiteVariantObject);
    return static_cast<const TensorArray*>(
        static_cast<const VariantData*>(tensor->data.data));
  }

  int GetIndOfInput(int input) { return input_inds_[input]; }

 private:
  std::vector<int> input_inds_;
  int output_ind_;
};

template <typename T>
class ListAddNTest : public ::testing::Test {};
TYPED_TEST_SUITE_P(ListAddNTest);

TYPED_TEST_P(ListAddNTest, TwoInputs_AllItemsPresent_AllSameShape) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListAddNModel m(2);
  for (const int i : {0, 1}) {
    const int list_ind = m.GetIndOfInput(i);
    m.PopulateListTensor(list_ind, {}, 2, tfl_type);
    m.ListSetItem(list_ind, 0, {2, 2}, tfl_type,
                  std::vector<TypeParam>(4, 1).data());
    m.ListSetItem(list_ind, 1, {2, 2}, tfl_type,
                  std::vector<TypeParam>(4, 1).data());
  }

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TensorArray* const arr = m.GetOutput();
  ASSERT_EQ(arr->NumElements(), 2);
  ASSERT_EQ(arr->ElementType(), tfl_type);
  EXPECT_THAT(arr->At(0), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                FilledWith<TypeParam>(2)));
  EXPECT_THAT(arr->At(1), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                FilledWith<TypeParam>(2)));
}

TYPED_TEST_P(ListAddNTest, TwoInputs_AllItemsPresent_ListsContainMixedShapes) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListAddNModel m(2);
  for (const int i : {0, 1}) {
    const int list_ind = m.GetIndOfInput(i);
    m.PopulateListTensor(list_ind, {}, 2, tfl_type);
    m.ListSetItem(list_ind, 0, {2, 2}, tfl_type,
                  std::vector<TypeParam>(4, 1).data());
    m.ListSetItem(list_ind, 1, {3, 3}, tfl_type,
                  std::vector<TypeParam>(9, 1).data());
  }

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TensorArray* const arr = m.GetOutput();
  ASSERT_EQ(arr->NumElements(), 2);
  ASSERT_EQ(arr->ElementType(), tfl_type);
  EXPECT_THAT(arr->At(0), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                FilledWith<TypeParam>(2)));
  EXPECT_THAT(arr->At(1), AllOf(IsAllocatedAs(tfl_type), DimsAre({3, 3}),
                                FilledWith<TypeParam>(2)));
}

TYPED_TEST_P(ListAddNTest, TwoInputs_NoItemsPresent_ListShapesMerge) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListAddNModel m(2);
  m.PopulateListTensor(m.GetIndOfInput(0), {2, -1}, 1, tfl_type);
  m.PopulateListTensor(m.GetIndOfInput(1), {-1, 2}, 1, tfl_type);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TensorArray* const arr = m.GetOutput();
  ASSERT_EQ(arr->NumElements(), 1);
  ASSERT_EQ(arr->ElementType(), tfl_type);
  EXPECT_THAT(arr->At(0), AllOf(IsAllocatedAs(tfl_type), DimsAre({2, 2}),
                                FilledWith<TypeParam>(0)));
}

TYPED_TEST_P(ListAddNTest, TwoInputs_NoItemsPresent_ListShapesUndefinedError) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListAddNModel m(2);
  m.PopulateListTensor(m.GetIndOfInput(0), {2, -1}, 1, tfl_type);
  m.PopulateListTensor(m.GetIndOfInput(1), {-1, -1}, 1, tfl_type);

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TYPED_TEST_P(ListAddNTest, TwoInputs_SomeItemsPresent_UsesElementShape) {
  TfLiteType tfl_type = typeToTfLiteType<TypeParam>();
  std::optional<TensorType> tensor_type = TflToTensorType(tfl_type);
  ASSERT_TRUE(tensor_type.has_value());

  ListAddNModel m(2);
  m.PopulateListTensor(m.GetIndOfInput(0), {}, 1, tfl_type);
  m.PopulateListTensor(m.GetIndOfInput(1), {}, 1, tfl_type);
  m.ListSetItem(m.GetIndOfInput(0), 0, {3, 3}, tfl_type,
                std::vector<TypeParam>(9, 1).data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TensorArray* const arr = m.GetOutput();
  ASSERT_EQ(arr->NumElements(), 1);
  ASSERT_EQ(arr->ElementType(), tfl_type);

  EXPECT_THAT(arr->At(0), AllOf(IsAllocatedAs(tfl_type), DimsAre({3, 3}),
                                FilledWith<TypeParam>(1)));
}

REGISTER_TYPED_TEST_SUITE_P(ListAddNTest,
                            TwoInputs_AllItemsPresent_AllSameShape,
                            TwoInputs_AllItemsPresent_ListsContainMixedShapes,
                            TwoInputs_NoItemsPresent_ListShapesMerge,
                            TwoInputs_SomeItemsPresent_UsesElementShape,
                            TwoInputs_NoItemsPresent_ListShapesUndefinedError);

using ValidTypes = ::testing::Types<int, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(ListAddNTests, ListAddNTest, ValidTypes);

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
