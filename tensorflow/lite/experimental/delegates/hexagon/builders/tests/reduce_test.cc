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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

// TODO(b/148390890): Reduce Sum tests are disabled, enable after fix is
// available and op is enabled.
class ReduceOpModel : public SingleOpModelWithHexagon {
 public:
  ReduceOpModel(BuiltinOperator type, const TensorData& input,
                const TensorData& output, std::initializer_list<int> axis_shape,
                std::initializer_list<int> axis, bool keep_dims) {
    input_ = AddInput(input);
    axis_ = AddConstInput(TensorType_INT32, axis, axis_shape);
    output_ = AddOutput(output);
    SetBuiltinOp(type, BuiltinOptions_ReducerOptions,
                 CreateReducerOptions(builder_, keep_dims).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int Input() { return input_; }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 private:
  int input_;
  int axis_;
  int output_;
};

template <TensorType Tensor_Type, typename input_type>
void TestMeanImpl() {
  float kQuantizedTolerance = 2.0 / 255;
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  ReduceOpModel m(BuiltinOperator_MEAN, {Tensor_Type, {1, 1, 3, 2}, -1.0, 1.0},
                  {Tensor_Type, {2}, -1.0, 1.0}, {1}, {2}, false);
  m.QuantizeAndPopulate<input_type>(m.Input(), data);
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<input_type>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2}));
  EXPECT_THAT(
      m.GetDequantizedOutput<input_type>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(ReduceOpModel, MeanNotKeepDims_Uint8) {
  TestMeanImpl<TensorType_UINT8, uint8_t>();
}

TEST(ReduceOpModel, MeanNotKeepDims_Int8) {
  TestMeanImpl<TensorType_INT8, int8_t>();
}

template <TensorType Tensor_Type, typename input_type>
void TestMeanKeppDimsImpl() {
  float kQuantizedTolerance = 2.0 / 255;
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  ReduceOpModel m(BuiltinOperator_MEAN, {Tensor_Type, {1, 1, 3, 2}, -1.0, 1.0},
                  {Tensor_Type, {3}, -1.0, 1.0}, {1}, {3}, true);
  m.QuantizeAndPopulate<input_type>(m.Input(), data);
  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<input_type>();
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<input_type>(),
      ElementsAreArray(ArrayFloatNear(reference_output, kQuantizedTolerance)));
}

TEST(ReduceOpModel, MeanKeepDims_Int8) {
  TestMeanKeppDimsImpl<TensorType_INT8, int8_t>();
}

TEST(ReduceOpModel, MeanKeepDims_Uint8) {
  TestMeanKeppDimsImpl<TensorType_UINT8, uint8_t>();
}

TEST(ReduceOpModel, DISABLED_SumNotKeepDims) {
  float kQuantizedTolerance = 2.0 / 255;
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  ReduceOpModel m(BuiltinOperator_SUM,
                  {TensorType_UINT8, {1, 1, 3, 2}, -1.0, 1.0},
                  {TensorType_UINT8, {2}, -1.0, 1.0}, {1}, {2}, false);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 2}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(
                  ArrayFloatNear({-0.823529, -0.815686}, kQuantizedTolerance)));
}

TEST(ReduceOpModel, DISABLED_SumKeepDims) {
  float kQuantizedTolerance = 2.0 / 255;
  std::vector<float> data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  ReduceOpModel m(BuiltinOperator_SUM,
                  {TensorType_UINT8, {1, 1, 3, 2}, -1.0, 1.0},
                  {TensorType_UINT8, {3}, -1.0, 1.0}, {1}, {3}, true);
  m.QuantizeAndPopulate<uint8_t>(m.Input(), data);
  m.ApplyDelegateAndInvoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 1, 3, 1}));
  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({-0.407843, -0.313726, 0.0941177},
                                              kQuantizedTolerance)));
}

}  // namespace tflite
