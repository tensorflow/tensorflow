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
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::testing::ElementsAre;

class ListPopBackModel : public ListOpModel {
 public:
  ListPopBackModel(const std::vector<int>& list_element_shape, int num_elements,
                   const std::vector<int>& tensor_element_shape,
                   TensorData output = {TensorType_INT32, {}}) {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    const std::vector<int> tensor_element_shape_shape =
        tensor_element_shape.empty()
            ? std::vector<int>()
            : std::vector<int>{static_cast<int>(tensor_element_shape.size())};
    element_shape_input_ =
        AddInput({TensorType_INT32, tensor_element_shape_shape});

    list_output_ = AddOutput({TensorType_VARIANT, {}});
    output_ = AddOutput(output);

    SetCustomOp("ListPopBack", /*custom_option=*/{}, Register_LIST_POP_BACK);

    BuildInterpreter({{}, tensor_element_shape_shape});

    PopulateListTensor(list_input_, list_element_shape, num_elements,
                       kTfLiteInt32);
    PopulateTensor(element_shape_input_, tensor_element_shape);
  }

  const TfLiteTensor* GetTensorOutput() {
    return interpreter_->tensor(output_);
  }
  const TensorArray* GetTensorListOutput() {
    return reinterpret_cast<const TensorArray*>(
        interpreter_->tensor(list_output_)->data.data);
  }

  int list_input_;
  int index_input_;
  int element_shape_input_;
  int output_;
  int list_output_;
};

TEST(ListPopBackTest, ZeroLenListFails) {
  ListPopBackModel m({2, 2}, 0, {});
  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListPopBackTest, GetUnsetItem_InferShapeFromListShape_Dynamic) {
  ListPopBackModel m({2, 2}, 2, {});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

TEST(ListPopBackTest, GetUnsetItem_InferShapeFromGivenShape_Dynamic) {
  ListPopBackModel m({}, 2, {2, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

TEST(ListPopBackTest, GetUnsetItem_InferShapeFromOtherElements_Dynamic) {
  ListPopBackModel m({}, 3, {});

  m.ListSetItem(m.list_input_, 0, {2, 2}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4}.data());
  m.ListSetItem(m.list_input_, 1, {2, 2}, kTfLiteInt32,
                std::vector<int>{5, 6, 7, 8}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 2);

  const TfLiteTensor* const item0 = arr->At(0);
  ASSERT_NE(item0, nullptr);
  ASSERT_THAT(item0, DimsAre({2, 2}));
  ASSERT_THAT(std::tuple(GetTensorData<int>(item0), 4),
              ElementsAre(1, 2, 3, 4));

  const TfLiteTensor* const item1 = arr->At(1);
  ASSERT_NE(item1, nullptr);
  ASSERT_THAT(item1, DimsAre({2, 2}));
  ASSERT_THAT(std::tuple(GetTensorData<int>(item1), 4),
              ElementsAre(5, 6, 7, 8));
}

TEST(ListPopBackTest,
     GetUnsetItem_InferShapeFromMergedListShapeGivenShape_Dynamic) {
  ListPopBackModel m({2, -1}, 2, {-1, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* const output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);
  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

TEST(ListPopBackTest, GetPresentItem_ReturnsElement_ScalarFallsBackDynamic) {
  ListPopBackModel m({}, 2, {});

  m.ListSetItem(m.list_input_, 1, {}, kTfLiteInt32, std::vector<int>{1}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 1), ElementsAre(1));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

TEST(ListPopBackTest, GetPresentItem_ReturnsElement_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListPopBackModel m({2, 2}, 2, {}, output_spec);

  m.ListSetItem(m.list_input_, 1, {2, 2}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteArenaRw);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(1, 2, 3, 4));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

TEST(ListPopBackTest, GetPresentItem_OutputShapeMismatched_Fails_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListPopBackModel m({}, 2, {}, output_spec);

  m.ListSetItem(m.list_input_, 1, {3, 3}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListPopBackTest, GetUnsetItem_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListPopBackModel m({}, 2, {}, output_spec);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetTensorOutput();
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteArenaRw);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));

  const TensorArray* const arr = m.GetTensorListOutput();
  EXPECT_EQ(arr->NumElements(), 1);
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
