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
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::testing::ElementsAre;

// TODO add output param
class ListGetItemModel : public ListOpModel {
 public:
  ListGetItemModel(TensorData index, TensorData element_shape,
                   TensorData output) {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    index_input_ = AddInput(index);
    element_shape_input_ = AddInput(element_shape);

    output_ = AddOutput(output);

    SetCustomOp("TensorListGetItem", /*custom_option=*/{},
                Register_LIST_GET_ITEM);

    BuildInterpreter({{}, index.shape, element_shape.shape});
  }

  // Simplified constructor for creating valid models.
  ListGetItemModel(int index, const std::vector<int>& element_shape) {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    index_input_ = AddInput({TensorType_INT32, {1}});
    element_shape_input_ =
        AddInput({TensorType_INT32, {static_cast<int>(element_shape.size())}});

    output_ = AddOutput({TensorType_INT32, element_shape});

    SetCustomOp("TensorListGetItem", /*custom_option=*/{},
                Register_LIST_GET_ITEM);

    BuildInterpreter({{}, {1}, {static_cast<int>(element_shape.size())}});

    PopulateListTensor(list_input_, {}, 2, kTfLiteInt32);
    PopulateTensor(index_input_, {index});
    PopulateTensor(element_shape_input_, element_shape);
  }

  const TfLiteTensor* GetOutput(int idx) { return interpreter_->tensor(idx); }

  int list_input_;
  int index_input_;
  int element_shape_input_;
  int output_;
};

TEST(ListGetItemTest, IndexOOB_Fails) {
  ListGetItemModel m(-1, {2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListGetItemTest, GetPresentItem_ReturnsElement_Dynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {2, 2}, 3, kTfLiteInt32);
  m.PopulateTensor(m.element_shape_input_, {2, 2});
  m.PopulateTensor(m.index_input_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

TEST(ListGetItemTest, GetUnsetItem_InferShapeFromListShape_Dynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {2, 2}, 2, kTfLiteInt32);
  m.PopulateTensor(m.index_input_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

TEST(ListGetItemTest, GetUnsetItem_InferShapeFromGivenShape_Dynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {}, 2, kTfLiteInt32);
  m.PopulateTensor(m.index_input_, {0});
  m.PopulateTensor(m.element_shape_input_, {2, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

TEST(ListGetItemTest, GetUnsetItem_InferShapeFromOtherElements_Dynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {}, 3, kTfLiteInt32);
  m.ListSetItem(m.list_input_, 1, {2, 2}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4}.data());
  m.ListSetItem(m.list_input_, 2, {2, 2}, kTfLiteInt32,
                std::vector<int>{5, 6, 7, 8}.data());
  m.PopulateTensor(m.index_input_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

TEST(ListGetItemTest,
     GetUnsetItem_InferShapeFromMergedListShapeGivenShape_Dynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {2, -1}, 3, kTfLiteInt32);
  m.PopulateTensor(m.element_shape_input_, {-1, 2});
  m.PopulateTensor(m.index_input_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

TEST(ListGetItemTest, GetPresentItem_ReturnsElement_ScalarFallsBackDynamic) {
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {}},
                     {TensorType_INT32, {}});

  m.PopulateListTensor(m.list_input_, {}, 3, kTfLiteInt32);
  m.ListSetItem(m.list_input_, 1, {}, kTfLiteInt32, std::vector<int>{1}.data());
  m.PopulateTensor(m.index_input_, {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({}));
  EXPECT_EQ(output->allocation_type, kTfLiteDynamic);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 1), ElementsAre(1));
}

TEST(ListGetItemTest, GetPresentItem_ReturnsElement_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}},
                     output_spec);

  m.PopulateListTensor(m.list_input_, {2, 2}, 3, kTfLiteInt32);
  m.ListSetItem(m.list_input_, 1, {2, 2}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4}.data());
  m.PopulateTensor(m.element_shape_input_, {2, 2});
  m.PopulateTensor(m.index_input_, {1});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteArenaRw);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(1, 2, 3, 4));
}

TEST(ListGetItemTest, GetPresentItem_OutputShapeMismatched_Fails_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {}},
                     output_spec);

  m.PopulateListTensor(m.list_input_, {}, 3, kTfLiteInt32);
  m.ListSetItem(m.list_input_, 1, {3, 3}, kTfLiteInt32,
                std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}.data());
  m.PopulateTensor(m.index_input_, {1});

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListGetItemTest, GetUnsetItem_Static) {
  TensorData output_spec({TensorType_INT32, {2, 2}});
  output_spec.shape_signature = {2, 2};
  ListGetItemModel m({TensorType_INT32, {}}, {TensorType_INT32, {2}},
                     output_spec);

  m.PopulateListTensor(m.list_input_, {2, 2}, 3, kTfLiteInt32);
  m.PopulateTensor(m.element_shape_input_, {2, 2});
  m.PopulateTensor(m.index_input_, {0});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TfLiteTensor* output = m.GetOutput(m.output_);
  ASSERT_EQ(output->type, kTfLiteInt32);

  ASSERT_THAT(output, DimsAre({2, 2}));
  EXPECT_EQ(output->allocation_type, kTfLiteArenaRw);
  EXPECT_THAT(std::tuple(GetTensorData<int>(output), 4),
              ElementsAre(0, 0, 0, 0));
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
