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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::testing::ElementsAre;

namespace tflite {
namespace variants {
namespace ops {
namespace {

class ListFromTensorModel : public SingleOpModel {
 public:
  ListFromTensorModel(TensorData tensor_data, TensorData shape_data) {
    tensor_id_ = AddInput(tensor_data);
    shape_id_ = AddInput(shape_data);
    list_id_ = AddOutput({TensorType_VARIANT, {1}});
    SetCustomOp("TensorListFromTensor", /*custom_option=*/{},
                Register_LIST_FROM_TENSOR);
    BuildInterpreter({tensor_data.shape, shape_data.shape});
  }

  const TensorArray* GetOutputTensorArray(int tensor_id) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    TFLITE_CHECK(tensor != nullptr && tensor->type == kTfLiteVariant &&
                 tensor->allocation_type == kTfLiteVariantObject);
    return static_cast<const TensorArray*>(
        static_cast<const VariantData*>(tensor->data.data));
  }

  int tensor_id_;
  int shape_id_;
  int list_id_;
};

TEST(ListFromTensorTest, MatrixInput_ReturnsListWithVectorElements) {
  ListFromTensorModel m({TensorType_INT32, {2, 2}}, {TensorType_INT32, {1}});

  m.PopulateTensor<int>(m.tensor_id_, {1, 2, 3, 4});
  m.PopulateTensor<int>(m.shape_id_, {2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* arr = m.GetOutputTensorArray(m.list_id_);

  ASSERT_EQ(arr->NumElements(), 2);
  ASSERT_THAT(arr->ElementShape(), DimsAre({2}));
  ASSERT_EQ(arr->ElementType(), kTfLiteInt32);

  {
    const TfLiteTensor* element = arr->At(0);
    ASSERT_THAT(element, DimsAre({2}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 2),
                ElementsAre(1, 2));
  }

  {
    const TfLiteTensor* element = arr->At(1);
    ASSERT_THAT(element, DimsAre({2}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 2),
                ElementsAre(3, 4));
  }
}

TEST(ListFromTensorTest, VectorInput_ReturnsListWithScalarElements) {
  ListFromTensorModel m({TensorType_INT32, {2}}, {TensorType_INT32, {0}});

  m.PopulateTensor<int>(m.tensor_id_, {1, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* arr = m.GetOutputTensorArray(m.list_id_);

  ASSERT_EQ(arr->NumElements(), 2);
  ASSERT_THAT(arr->ElementShape(), DimsAre({}));
  ASSERT_EQ(arr->ElementType(), kTfLiteInt32);

  {
    const TfLiteTensor* element = arr->At(0);
    ASSERT_THAT(element, DimsAre({}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 1),
                ElementsAre(1));
  }

  {
    const TfLiteTensor* element = arr->At(1);
    ASSERT_THAT(element, DimsAre({}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 1),
                ElementsAre(2));
  }
}

TEST(ListFromTensorTest, 3DInput_ReturnsListWithMatrixElements) {
  ListFromTensorModel m({TensorType_INT32, {2, 2, 2}}, {TensorType_INT32, {2}});

  m.PopulateTensor<int>(m.tensor_id_, {1, 2, 3, 4, 5, 6, 7, 8});
  m.PopulateTensor<int>(m.shape_id_, {2, 2});

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* arr = m.GetOutputTensorArray(m.list_id_);

  ASSERT_EQ(arr->NumElements(), 2);
  ASSERT_THAT(arr->ElementShape(), DimsAre({2, 2}));
  ASSERT_EQ(arr->ElementType(), kTfLiteInt32);

  {
    const TfLiteTensor* element = arr->At(0);
    ASSERT_THAT(element, DimsAre({2, 2}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 4),
                ElementsAre(1, 2, 3, 4));
  }

  {
    const TfLiteTensor* element = arr->At(1);
    ASSERT_THAT(element, DimsAre({2, 2}));

    EXPECT_THAT(std::make_tuple(GetTensorData<int>(element), 4),
                ElementsAre(5, 6, 7, 8));
  }
}

TEST(ListFromTensorTest, MismatchedShapeInputTensorShape_Fails) {
  ListFromTensorModel m({TensorType_INT32, {2, 2, 2}}, {TensorType_INT32, {2}});

  m.PopulateTensor<int>(m.shape_id_, {2, 3});

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

TEST(ListFromTensorTest, ScalarInput_Fails) {
  ListFromTensorModel m({TensorType_INT32, {}}, {TensorType_INT32, {}});

  ASSERT_EQ(m.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
