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
using testing::ElementsAre;
using testing::ElementsAreArray;

class FullyConnectedOpModel : public SingleOpModelWithHexagon {
 public:
  FullyConnectedOpModel(int units, int batches, const TensorData& input,
                        const TensorData& output, bool optional_bias = false,
                        bool const_weights = true)
      : batches_(batches), units_(units) {
    int total_input_size = 1;
    for (size_t i = 0; i < input.shape.size(); ++i) {
      total_input_size *= input.shape[i];
    }
    input_size_ = total_input_size / batches_;

    input_ = AddInput(input);
    weights_ =
        AddInput({input.type, {units_, input_size_}, input.min, input.max});

    if (optional_bias) {
      bias_ = AddNullInput();
    } else {
      auto bias_scale = GetScale(input_) * GetScale(weights_);
      TensorData bias{TensorType_INT32, {units_}, 0, 0, bias_scale};
      bias_ = AddInput(bias);
    }

    output_ = AddOutput(output);

    SetBuiltinOp(
        BuiltinOperator_FULLY_CONNECTED, BuiltinOptions_FullyConnectedOptions,
        CreateFullyConnectedOptions(builder_, ActivationFunctionType_NONE,
                                    FullyConnectedOptionsWeightsFormat_DEFAULT,
                                    /*keep_num_dims=*/false)
            .Union());
    BuildInterpreter({GetShape(input_), GetShape(weights_)});

    // Weights & bias tensors need to be constant.
    // We don't use AddConstInput to allow setting filter values later.
    if (const_weights) {
      auto* weights_tensor = interpreter_->tensor(weights_);
      weights_tensor->allocation_type = kTfLiteMmapRo;
    }
    if (!optional_bias) {
      auto* bias_tensor = interpreter_->tensor(bias_);
      bias_tensor->allocation_type = kTfLiteMmapRo;
    }
  }

  void SetBias(const std::vector<float>& data) {
    QuantizeAndPopulate<int32_t>(bias_, data);
  }

  template <typename T>
  void SetWeights(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(weights_, data);
  }

  template <typename T>
  void SetInput(const std::vector<float>& data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

 protected:
  int input_;
  int weights_;
  int bias_;
  int output_;

  int batches_;
  int units_;
  int input_size_;
};

TEST(QuantizedFullyConnectedOpTest, TestQuantizedInt8) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128});

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear({24, 25, 26, 58, 59, 60})));
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAre(23, 24, 25, 57, 58, 59));
}

TEST(QuantizedFullyConnectedOpTest, TestQuantizedUint8) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128});

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  //
                  58, 59, 60,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(151, 152, 153, 185, 186, 187));
}

TEST(QuantizedFullyConnectedOpTest, TestQuantizedUint8_NoBias) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128}, /*optional_bias*/ true);

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<uint8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST(QuantizedFullyConnectedOpTest, TestQuantizedInt8_NoBias) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128},
                          /*optional_bias*/ true);

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST(QuantizedFullyConnectedOpTest, TestQuantizedInt8_NonConstWeights) {
  FullyConnectedOpModel m(/*units=*/3, /*batches*/ 2,
                          /*input=*/{TensorType_INT8, {2, 10}, -63.5, 64},
                          /*output=*/{TensorType_INT8, {}, -127, 128},
                          /*optional_bias=*/false, /*const_weights=*/false);

  m.SetWeights<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<int8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.Invoke();
  auto reference_output = m.GetDequantizedOutput<int8_t>();

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<int8_t>(),
              ElementsAreArray(ArrayFloatNear(reference_output)));
}

TEST(QuantizedFullyConnectedOpTest, TestQuantizedUint8_NonConstWeights) {
  FullyConnectedOpModel m(
      /*units=*/3, /*batches*/ 2,
      /*input=*/{TensorType_UINT8, {2, 10}, -63.5, 64},
      /*output=*/{TensorType_UINT8, {}, -127, 128}, /*optional_bias=*/false,
      /*const_weights=*/false);

  m.SetWeights<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 2
  });
  m.SetBias({1, 2, 3});

  m.SetInput<uint8_t>({
      1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
      1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
  });

  m.ApplyDelegateAndInvoke();

  EXPECT_THAT(m.GetDequantizedOutput<uint8_t>(),
              ElementsAreArray(ArrayFloatNear({
                  24, 25, 26,  //
                  58, 59, 60,  //
              })));
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAre(151, 152, 153, 185, 186, 187));
}

}  // namespace tflite
