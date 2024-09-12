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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::testing::ElementsAreArray;
using ::tflite::variants::ops::Register_LIST_STACK;

namespace tflite {
namespace {

class ListStackModel : public ListOpModel {
 public:
  explicit ListStackModel(TensorData output_data) {
    tensor_id_ = AddOutput(output_data);
    list_id_ = AddInput({TensorType_VARIANT, {}});
    shape_id_ = AddInput({TensorType_INT32, {1}});
    SetCustomOp("ListStack", {}, Register_LIST_STACK);
    BuildInterpreter({{}, {1}});
  }

  ListStackModel(TensorData output_data, TensorData shape_input_data) {
    tensor_id_ = AddOutput(output_data);
    list_id_ = AddInput({TensorType_VARIANT, {}});
    shape_id_ = AddInput(shape_input_data);
    SetCustomOp("ListStack", {}, Register_LIST_STACK);
    BuildInterpreter({{}, shape_input_data.shape});
  }

  const TfLiteTensor* GetOutputTensor(int tensor_id) {
    return interpreter_->tensor(tensor_id);
  }

  int tensor_id_;
  int shape_id_;
  int list_id_;
};

TEST(ListStackTest, MismatchedListShapeInputShape_Fails) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {1}, 2, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {3});

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListStackTest, MismatchedShapeOfElementsAndInput_Fails) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 4, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {2});

  m.ListSetItem(m.list_id_, 0, {1}, kTfLiteInt32, std::vector<int>{0}.data());
  m.ListSetItem(m.list_id_, 1, {1}, kTfLiteInt32, std::vector<int>{1}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListStackTest, ElementsNotSameShape_Fails) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {2});

  m.ListSetItem(m.list_id_, 0, {2}, kTfLiteInt32,
                std::vector<int>{2, 2}.data());
  m.ListSetItem(m.list_id_, 1, {1}, kTfLiteInt32, std::vector<int>{3}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListStackTest, NoElementsNoShape_Fails) {
  ListStackModel m({TensorType_INT32, {4}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {-1});

  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListStackTest, ListElementTypeNotEqualOutputType_Fails) {
  ListStackModel m({TensorType_INT32, {4}});

  m.PopulateListTensor(m.list_id_, {}, 0, kTfLiteInt64);
  m.PopulateTensor<int>(m.shape_id_, {-1});

  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListStackTest, ScalarElementShape_FullList_Returns1D) {
  ListStackModel m({TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {1});

  m.ListSetItem(m.list_id_, 0, {1}, kTfLiteInt32, std::vector<int>{2}.data());
  m.ListSetItem(m.list_id_, 1, {1}, kTfLiteInt32, std::vector<int>{3}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2}));
  ASSERT_THAT(output->type, kTfLiteInt32);
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 2),
              ElementsAreArray({2, 3}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, ScalarElementShape_PartialFilledList_Returns1DWithZeroed) {
  ListStackModel m({TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {1});

  m.ListSetItem(m.list_id_, 0, {1}, kTfLiteInt32, std::vector<int>{2}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2}));
  ASSERT_THAT(output->type, kTfLiteInt32);
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 2),
              ElementsAreArray({2, 0}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, ScalarElementShape_EmptyList_Returns1DAllZeroed) {
  ListStackModel m({TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor(m.shape_id_, {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2}));
  ASSERT_THAT(output->type, kTfLiteInt32);
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 2),
              ElementsAreArray({0, 0}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, VectorElementShape_FilledList_Returns2D) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2});

  m.ListSetItem(m.list_id_, 0, {2}, kTfLiteInt32,
                std::vector<int>{2, 2}.data());
  m.ListSetItem(m.list_id_, 1, {2}, kTfLiteInt32,
                std::vector<int>{3, 3}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 4),
              ElementsAreArray({2, 2, 3, 3}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, VectorElementShape_PartialFilledList_Returns2DWithZeroed) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2});

  m.ListSetItem(m.list_id_, 0, {2}, kTfLiteInt32,
                std::vector<int>{2, 2}.data());

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 4),
              ElementsAreArray({2, 2, 0, 0}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, VectorElementShape_EmptyList_Returns2DAllZeroed) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 4),
              ElementsAreArray({0, 0, 0, 0}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, NoShapeArguments_ZeroSizeList_InfersShapeFromElements) {
  ListStackModel m({TensorType_INT32, {2, 2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {-1});

  m.ListSetItem(m.list_id_, 0, {2}, kTfLiteInt32,
                std::vector<int>{2, 2}.data());

  EXPECT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_THAT(std::vector<int>(output->data.i32, output->data.i32 + 4),
              ElementsAreArray({2, 2, 0, 0}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
}

TEST(ListStackTest, ListFirstDimZero_ReturnsEmptyTensor) {
  ListStackModel m({TensorType_INT32, {0, 2}});

  m.PopulateListTensor(m.list_id_, {}, 0, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  EXPECT_THAT(output, DimsAre({0, 2}));
}

TEST(ListStackTest, MismatchedOutput_ReturnsResizedOutput1D) {
  ListStackModel m({TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 4, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  EXPECT_THAT(output, DimsAre({4}));
}

TEST(ListStackTest, MismatchedOutput_ReturnsResizedOutput2D) {
  ListStackModel m({TensorType_INT32, std::vector<int>{}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  EXPECT_THAT(output, DimsAre({2, 2}));
}

TEST(ListStackTest, Trailing0DimInElementShape1D_NonZeroLen_Returns2DNoData) {
  ListStackModel m({TensorType_INT32, std::vector<int>{}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 0}));
  EXPECT_EQ(output->bytes, 0);
}

TEST(ListStackTest, Trailing0DimInElementShape2D_NonZeroLen_Returns3DNoData) {
  ListStackModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 2, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2, 0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({2, 2, 0}));
  EXPECT_EQ(output->bytes, 0);
}

TEST(ListStackTest, Trailing0DimInElementShape1D_ZeroLen_Returns2DNoData) {
  ListStackModel m({TensorType_INT32, {}}, {TensorType_INT32, {1}});

  m.PopulateListTensor(m.list_id_, {}, 0, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({0, 0}));
  EXPECT_EQ(output->bytes, 0);
}

TEST(ListStackTest, Trailing0DimInElementShape2D_ZeroLen_Returns3DNoData) {
  ListStackModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}});

  m.PopulateListTensor(m.list_id_, {}, 0, kTfLiteInt32);
  m.PopulateTensor<int>(m.shape_id_, {2, 0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TfLiteTensor* output = m.GetOutputTensor(m.tensor_id_);

  ASSERT_THAT(output, DimsAre({0, 2, 0}));
  EXPECT_EQ(output->bytes, 0);
}

}  // namespace
}  // namespace tflite
