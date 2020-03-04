/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <initializer_list>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class QuantizedLSTMOpModel : public MultiOpModel {
 public:
  QuantizedLSTMOpModel(int numBatches, int inputSize, float weightsScale,
                       int32_t weightsZeroPoint, int outputSize,
                       std::initializer_list<uint8_t> weights,
                       std::initializer_list<int32_t> biases,
                       // If true the LTSM node will be preceded by a noop
                       // one (add to 0)
                       bool prepend_noop) {
    std::vector<uint32_t> inputs;

    input_size_ = inputSize;
    output_size_ = outputSize;
    prepend_noop_ = prepend_noop;

    std::vector<int> input_shape{numBatches, inputSize};
    std::vector<int> output_shape{numBatches, outputSize};
    std::vector<int> weight_shape{4 * outputSize, outputSize + inputSize};
    std::vector<int> state_shape{numBatches, outputSize};
    std::vector<int> bias_shape{4 * outputSize};

    std::vector<int> lstm_inputs;

    const TensorData input_tensor_data{
        TensorType_UINT8, input_shape, 0.0f, 0.0f, 1. / 128., 128};

    if (prepend_noop) {
      zero_input_ = AddInput(input_tensor_data);
    } else {
      zero_input_ = 0;
    }

    input_ = AddInput(input_tensor_data);

    prev_output_ =
        AddInput({TensorType_UINT8, output_shape, 0.0f, 0.0f, 1. / 128., 128});
    // Biases and Weights have to be constant in order to allow NNAPI
    // delegation
    weights_ = AddConstInput<uint8_t>({TensorType_UINT8, weight_shape, 0.0f,
                                       0.0f, weightsScale, weightsZeroPoint},
                                      weights);
    biases_ = AddConstInput<int32_t>(
        {TensorType_INT32, bias_shape, 0.0f, 0.0f, weightsScale / 128, 0},
        biases);
    prev_cell_state_ =
        AddInput({TensorType_INT16, state_shape, 0.0f, 0.0f, 1. / 2048., 0});

    sum_out_ = AddOutput(input_tensor_data);

    output_ =
        AddOutput({TensorType_UINT8, output_shape, 0.0f, 0.0f, 1. / 128., 128});
    cell_state_out_ =
        AddOutput({TensorType_INT16, state_shape, 0.0f, 0.0f, 1. / 2048., 0});
    output_concat_temp_ =
        AddOutput({TensorType_UINT8, output_shape, 0.0f, 0.0f, 1. / 128., 128});
    output_activation_temp_ =
        AddOutput({TensorType_INT16, output_shape, 0.0f, 0.0f, 1. / 128., 128});

    if (prepend_noop) {
      AddBuiltinOp(
          BuiltinOperator_ADD, BuiltinOptions_AddOptions,
          CreateAddOptions(builder_, ActivationFunctionType_NONE).Union(),
          {zero_input_, input_}, {sum_out_});

      lstm_inputs.push_back(sum_out_);
    } else {
      lstm_inputs.push_back(input_);
    }

    lstm_inputs.push_back(prev_output_);
    lstm_inputs.push_back(weights_);
    lstm_inputs.push_back(biases_);
    lstm_inputs.push_back(prev_cell_state_);

    std::vector<int> lstm_outputs{output_, cell_state_out_, output_concat_temp_,
                                  output_activation_temp_};

    AddBuiltinOp(BuiltinOperator_LSTM, BuiltinOptions_LSTMOptions,
                 CreateLSTMOptions(builder_, ActivationFunctionType_TANH, 0.0,
                                   0.0, LSTMKernelType_BASIC)
                     .Union(),
                 lstm_inputs, lstm_outputs);

    if (prepend_noop) {
      BuildInterpreter({GetShape(input_), GetShape(zero_input_),
                        GetShape(prev_output_), GetShape(weights_),
                        GetShape(biases_), GetShape(prev_cell_state_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(prev_output_),
                        GetShape(weights_), GetShape(biases_),
                        GetShape(prev_cell_state_)});
    }
    // init feedback inputs to zero
    std::vector<int16_t> initial_state(GetTensorSize(cell_state_out_), 0);
    PopulateTensor(prev_cell_state_, initial_state);
    std::vector<uint8_t> initial_prev_output(GetTensorSize(output_), 0);
    PopulateTensor(prev_output_, initial_prev_output);
  }

  int inputSize() { return input_size_; }

  int outputSize() { return output_size_; }

  void setInput(const std::vector<uint8_t>& input) {
    PopulateTensor(input_, input);
    if (prepend_noop_) {
      std::vector<uint8_t> zero(GetTensorSize(zero_input_), 128);
      PopulateTensor(zero_input_, zero);
    }
  }

  std::vector<uint8_t> getOutput() { return ExtractVector<uint8_t>(output_); }

 private:
  // Inputs
  int input_;
  int weights_;
  int biases_;
  int prev_cell_state_;
  int prev_output_;
  // Outputs
  int cell_state_out_;
  int output_;
  int output_concat_temp_;
  int output_activation_temp_;

  int input_size_;
  int output_size_;
  bool prepend_noop_;
  int zero_input_;
  int sum_out_;
};

class QuantizedLstmTest : public ::testing::Test,
                          public testing::WithParamInterface<bool> {
 protected:
  void VerifyGoldens(const std::vector<std::vector<uint8_t>>& input,
                     const std::vector<std::vector<uint8_t>>& output,
                     QuantizedLSTMOpModel* lstm) {
    const int numBatches = input.size();
    ASSERT_GT(numBatches, 0);
    const int inputSize = lstm->inputSize();
    ASSERT_GT(inputSize, 0);
    const int inputSequenceSize = input[0].size() / inputSize;
    ASSERT_GT(inputSequenceSize, 0);
    for (int i = 0; i < inputSequenceSize; ++i) {
      std::vector<uint8_t> inputStep;
      for (int b = 0; b < numBatches; ++b) {
        const uint8_t* batchStart = input[b].data() + i * inputSize;
        const uint8_t* batchEnd = batchStart + inputSize;
        inputStep.insert(inputStep.end(), batchStart, batchEnd);
      }
      lstm->setInput(inputStep);
      lstm->Invoke();

      const int outputSize = lstm->outputSize();
      std::vector<float> expected;
      for (int b = 0; b < numBatches; ++b) {
        const uint8_t* goldenBatchStart = output[b].data() + i * outputSize;
        const uint8_t* goldenBatchEnd = goldenBatchStart + outputSize;
        expected.insert(expected.end(), goldenBatchStart, goldenBatchEnd);
      }
      EXPECT_THAT(lstm->getOutput(), ElementsAreArray(expected));
    }
  }
};

// Inputs and weights in this test are random and the test only checks that the
// outputs are equal to outputs obtained from running TF Lite version of
// quantized LSTM on the same inputs.
TEST_P(QuantizedLstmTest, BasicQuantizedLstmTest) {
  const int numBatches = 2;
  const int inputSize = 2;
  const int outputSize = 4;

  float weightsScale = 0.00408021;
  int weightsZeroPoint = 100;

  bool prepend_dummy_node = GetParam();

  QuantizedLSTMOpModel lstm(
      numBatches, inputSize, weightsScale, weightsZeroPoint, outputSize,

      // This data are copied from QuantizedLSTMTest.cpp in NNAPI source code
      // I have to recompose the weight matrix before passing it to the model

      // recurrentToInputWeights   inputToInputWeights
      {254, 206, 77, 168, 146, 250, 71, 20, 215, 6, 235, 171, 223, 7, 118, 225,
       10, 218, 59, 130, 174, 26, 171, 108,

       // recurrentToCellWeights     inputToCellWeights
       172, 60, 205, 65, 133, 34, 14, 0, 140, 168, 29, 49, 240, 223, 133, 56,
       206, 109, 142, 64, 246, 216, 54, 183,

       // recurrentToForgetWeights   inputToForgetWeights
       137, 240, 103, 52, 24, 50, 68, 51, 237, 112, 132, 179, 0, 220, 89, 23,
       158, 110, 69, 4, 207, 253, 3, 169,

       // recurrentToOutputWeights  inputToOutputWeights
       106, 214, 67, 23, 195, 187, 59, 158, 45, 3, 11, 99, 119, 132, 49, 205,
       109, 10, 129, 218, 11, 98, 218, 48},

      // inputGateBias
      {-7876, 13488, -726, 32839,
       // cellGateBias
       39481, 48624, 48976, -21419,
       // forgetGateBias
       9206, -46884, -11693, -38724,
       // outputGateBias
       -58999, -17050, -41852, -40538},
      prepend_dummy_node);
  // clang-format on

  // LSTM input is stored as numBatches x (sequenceLength x inputSize) vector.
  std::vector<std::vector<uint8_t>> lstmInput;
  // clang-format off
    lstmInput = {{154, 166,
                  166, 179,
                  141, 141},
                 {100, 200,
                  50,  150,
                  111, 222}};
  // clang-format on

  // LSTM output is stored as numBatches x (sequenceLength x outputSize) vector.
  std::vector<std::vector<uint8_t>> lstmGoldenOutput;
  /*
    This is the output used in NNAPI's QuantizedLSTMTest.cpp
    I get slightly different values that are consistent running with or
    without acceleration

    lstmGoldenOutput = {{136, 150, 140, 115,
                         140, 151, 146, 112,
                         139, 153, 146, 114},
                        {135, 152, 138, 112,
                         136, 156, 142, 112,
                         141, 154, 146, 108}};
   */

  // clang-format off
    lstmGoldenOutput = {{131, 152, 136, 109,
                         138, 150, 145, 111,
                         139, 152, 146, 113},
                        {131, 153, 135, 107,
                         134, 154, 140, 111,
                         140, 154, 145, 108}};
  // clang-format on
  VerifyGoldens(lstmInput, lstmGoldenOutput, &lstm);
}

INSTANTIATE_TEST_SUITE_P(QuantizedLstmTest, QuantizedLstmTest,
                         testing::Values(false, true));

}  // namespace
}  // namespace tflite
