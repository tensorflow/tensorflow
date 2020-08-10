/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class PackOpModel : public SingleOpModelWithHexagon {
 public:
  PackOpModel(const TensorData& input_template, int axis, int values_count) {
    std::vector<std::vector<int>> all_input_shapes;
    for (int i = 0; i < values_count; ++i) {
      all_input_shapes.push_back(input_template.shape);
      AddInput(input_template);
    }
    output_ = AddOutput({input_template.type, /*shape=*/{}, input_template.min,
                         input_template.max});
    SetBuiltinOp(BuiltinOperator_PACK, BuiltinOptions_PackOptions,
                 CreatePackOptions(builder_, values_count, axis).Union());
    BuildInterpreter(all_input_shapes);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename integer_type>
  void SetInput(int index, std::initializer_list<float> data) {
    QuantizeAndPopulate<integer_type>(index, data);
  }

  template <typename integer_type>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_type>(ExtractVector<integer_type>(output_),
                                    GetScale(output_), GetZeroPoint(output_));
  }

 private:
  int output_;
};

template <typename InputType>
struct PackOpTest : public ::testing::Test {
  using TypeToTest = InputType;
  TensorType TENSOR_TYPE =
      (std::is_same<InputType, int16_t>::value
           ? TensorType_INT16
           : (std::is_same<InputType, uint8_t>::value ? TensorType_UINT8
                                                      : TensorType_INT8));
};

using TestTypes = testing::Types<int8_t, uint8_t>;
TYPED_TEST_CASE(PackOpTest, TestTypes);

TYPED_TEST(PackOpTest, ThreeInputs) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, 0, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  model.Invoke();
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, ThreeInputsDifferentAxis) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, 1, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  model.Invoke();
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, ThreeInputsNegativeAxis) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2}, -10, 10}, -1, 3);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 4});
  model.SetInput<typename TestFixture::TypeToTest>(1, {2, 5});
  model.SetInput<typename TestFixture::TypeToTest>(2, {3, 6});
  model.Invoke();
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

TYPED_TEST(PackOpTest, MultilDimensions) {
  PackOpModel model({TestFixture::TENSOR_TYPE, {2, 3}, -10, 20}, 1, 2);
  model.SetInput<typename TestFixture::TypeToTest>(0, {1, 2, 3, 4, 5, 6});
  model.SetInput<typename TestFixture::TypeToTest>(1, {7, 8, 9, 10, 11, 12});
  model.Invoke();
  auto ref_output_shape = model.GetOutputShape();
  auto ref_output =
      model.GetDequantizedOutput<typename TestFixture::TypeToTest>();
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray(ref_output_shape));
  EXPECT_THAT(model.GetDequantizedOutput<typename TestFixture::TypeToTest>(),
              ElementsAreArray(ArrayFloatNear(ref_output)));
}

}  // namespace tflite
