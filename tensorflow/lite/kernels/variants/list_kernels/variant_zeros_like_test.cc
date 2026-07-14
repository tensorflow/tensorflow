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
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_kernels/test_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

using ::testing::AllOf;
using ::testing::Combine;
using ::testing::ValuesIn;
using ::tflite::variants::TensorArray;

class VariantZerosLikeModel : public ListOpModel {
 public:
  explicit VariantZerosLikeModel() {
    list_input_ = AddInput({TensorType_VARIANT, {}});
    list_output_ = AddOutput({TensorType_VARIANT, {}});
    SetCustomOp("VariantZerosLike", {}, Register_VARIANT_ZEROS_LIKE);
    BuildInterpreter({{}});
  }

  const TensorArray* GetOutputTensorArray() {
    TfLiteTensor* tensor = interpreter_->tensor(list_output_);
    TFLITE_CHECK(tensor != nullptr && tensor->type == kTfLiteVariant &&
                 tensor->allocation_type == kTfLiteVariantObject);
    return static_cast<const TensorArray*>(
        static_cast<const VariantData*>(tensor->data.data));
  }

  int list_input_;
  int list_output_;
};

using VariantZerosLikeTestParam = std::tuple<std::vector<int>, TfLiteType, int>;
class VariantZerosLikeTest
    : public testing::TestWithParam<VariantZerosLikeTestParam> {
 public:
  enum { kShape, kType, kLen };
};

TEST_P(VariantZerosLikeTest, OutputsEmptyListWithSameAttrs) {
  const auto& param = GetParam();
  const std::vector<int>& shape = std::get<kShape>(param);
  const TfLiteType t = std::get<kType>(param);
  const int len = std::get<kLen>(param);
  VariantZerosLikeModel m;
  m.PopulateListTensor(m.list_input_, shape, len, t);

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const TensorArray* const out = m.GetOutputTensorArray();
  ASSERT_EQ(out->NumElements(), len);
  ASSERT_EQ(out->ElementType(), t);
  ASSERT_THAT(out->ElementShape(), DimsAre(shape));
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(out->At(i), nullptr);
  }
}

using VariantZerosLikeItemTestParam = std::tuple<int, std::vector<int>>;
class VariantZerosLikeItemTest
    : public testing::TestWithParam<VariantZerosLikeItemTestParam> {
 public:
  enum { kLen, kShape };
};

TEST_P(VariantZerosLikeItemTest, OutputsEmptyListContainsZeroedElement) {
  const auto& param = GetParam();
  const int len = std::get<kLen>(param);
  const std::vector<int>& item_shape = std::get<kShape>(param);
  VariantZerosLikeModel m;
  m.PopulateListTensor(m.list_input_, {}, len, kTfLiteInt32);
  const int num_elements = NumElements(item_shape.data(), item_shape.size());
  m.ListSetItem(m.list_input_, 0, item_shape, kTfLiteInt32,
                std::vector<int>(num_elements, 1).data());
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const TensorArray* const out = m.GetOutputTensorArray();
  ASSERT_EQ(out->NumElements(), len);
  ASSERT_EQ(out->ElementType(), kTfLiteInt32);
  ASSERT_THAT(out->ElementShape(), DimsAre({}));
  const TfLiteTensor* const zero = out->At(0);
  ASSERT_NE(zero, nullptr);
  EXPECT_THAT(zero, AllOf(DimsAre(item_shape), IsAllocatedAs(kTfLiteInt32),
                          FilledWith<int>(0)));
  for (int i = 1; i < len; ++i) {
    EXPECT_EQ(out->At(i), nullptr);
  }
}

INSTANTIATE_TEST_SUITE_P(VariantZerosLikeTests, VariantZerosLikeTest,
                         Combine(ValuesIn(std::vector<std::vector<int>>{
                                     {}, {-1}, {2, 2}, {3, 3, 3}}),
                                 ValuesIn({kTfLiteInt32, kTfLiteInt64,
                                           kTfLiteFloat32, kTfLiteBool}),
                                 ValuesIn({0, 2, 10})));

INSTANTIATE_TEST_SUITE_P(VariantZerosLikeTests, VariantZerosLikeItemTest,
                         Combine(ValuesIn({1, 2, 10}),
                                 ValuesIn(std::vector<std::vector<int>>{
                                     {1}, {2, 2}, {3, 3, 3}})));

}  // namespace
}  // namespace ops
}  // namespace variants
}  // namespace tflite
