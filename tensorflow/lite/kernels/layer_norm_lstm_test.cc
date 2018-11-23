/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// Unit test for TFLite Layer Norm LSTM op.

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_LAYER_NORM_LSTM();

namespace {

using ::testing::ElementsAreArray;

class LayerNormLSTMOpModel : public SingleOpModel {
 public:
  LayerNormLSTMOpModel(int n_batch, int n_input, int n_cell, int n_output,
                       bool use_cifg, bool use_peephole,
                       bool use_projection_weights, bool use_projection_bias,
                       float cell_clip, float proj_clip,
                       const std::vector<std::vector<int>>& input_shapes,
                       const TensorType& weight_type = TensorType_FLOAT32)
      : n_batch_(n_batch),
        n_input_(n_input),
        n_cell_(n_cell),
        n_output_(n_output) {
    input_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      input_to_input_weights_ = AddNullInput();
    } else {
      input_to_input_weights_ = AddInput(weight_type);
    }

    input_to_forget_weights_ = AddInput(weight_type);
    input_to_cell_weights_ = AddInput(weight_type);
    input_to_output_weights_ = AddInput(weight_type);

    if (use_cifg) {
      recurrent_to_input_weights_ = AddNullInput();
    } else {
      recurrent_to_input_weights_ = AddInput(weight_type);
    }

    recurrent_to_forget_weights_ = AddInput(weight_type);
    recurrent_to_cell_weights_ = AddInput(weight_type);
    recurrent_to_output_weights_ = AddInput(weight_type);

    if (use_peephole) {
      if (use_cifg) {
        cell_to_input_weights_ = AddNullInput();
      } else {
        cell_to_input_weights_ = AddInput(weight_type);
      }
      cell_to_forget_weights_ = AddInput(weight_type);
      cell_to_output_weights_ = AddInput(weight_type);
    } else {
      cell_to_input_weights_ = AddNullInput();
      cell_to_forget_weights_ = AddNullInput();
      cell_to_output_weights_ = AddNullInput();
    }

    input_layer_norm_weights_ = AddInput(TensorType_FLOAT32);
    forget_layer_norm_weights_ = AddInput(TensorType_FLOAT32);
    cell_layer_norm_weights_ = AddInput(TensorType_FLOAT32);
    output_layer_norm_weights_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      input_gate_bias_ = AddNullInput();
    } else {
      input_gate_bias_ = AddInput(TensorType_FLOAT32);
    }
    forget_gate_bias_ = AddInput(TensorType_FLOAT32);
    cell_bias_ = AddInput(TensorType_FLOAT32);
    output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      projection_weights_ = AddInput(weight_type);
      if (use_projection_bias) {
        projection_bias_ = AddInput(TensorType_FLOAT32);
      } else {
        projection_bias_ = AddNullInput();
      }
    } else {
      projection_weights_ = AddNullInput();
      projection_bias_ = AddNullInput();
    }

    // Adding the 2 state tensors.
    output_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_output_ * n_batch_}}, true);
    cell_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_cell_ * n_batch_}}, true);

    output_ = AddOutput(TensorType_FLOAT32);

    // Set up and pass in custom options using flexbuffer.
    flexbuffers::Builder fbb;
    fbb.Map([&]() {
      fbb.Int("cell_clip", cell_clip);
      fbb.Int("proj_clip", proj_clip);
      fbb.String("fused_activation_function", "TANH");
    });
    fbb.Finish();
    SetCustomOp("LAYER_NORM_LSTM", fbb.GetBuffer(), Register_LAYER_NORM_LSTM);
    BuildInterpreter(input_shapes);
  }

  void SetInputToInputWeights(std::vector<float> f) {
    PopulateTensor(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(std::vector<float> f) {
    PopulateTensor(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(std::vector<float> f) {
    PopulateTensor(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(std::vector<float> f) {
    PopulateTensor(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(std::vector<float> f) {
    PopulateTensor(recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(std::vector<float> f) {
    PopulateTensor(recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(std::vector<float> f) {
    PopulateTensor(recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(std::vector<float> f) {
    PopulateTensor(recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(std::vector<float> f) {
    PopulateTensor(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(std::vector<float> f) {
    PopulateTensor(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(std::vector<float> f) {
    PopulateTensor(cell_to_output_weights_, f);
  }

  void SetInputLayerNormWeights(std::vector<float> f) {
    PopulateTensor(input_layer_norm_weights_, f);
  }

  void SetForgetLayerNormWeights(std::vector<float> f) {
    PopulateTensor(forget_layer_norm_weights_, f);
  }

  void SetCellLayerNormWeights(std::vector<float> f) {
    PopulateTensor(cell_layer_norm_weights_, f);
  }

  void SetOutputLayerNormWeights(std::vector<float> f) {
    PopulateTensor(output_layer_norm_weights_, f);
  }

  void SetInputGateBias(std::vector<float> f) {
    PopulateTensor(input_gate_bias_, f);
  }

  void SetForgetGateBias(std::vector<float> f) {
    PopulateTensor(forget_gate_bias_, f);
  }

  void SetCellBias(std::vector<float> f) { PopulateTensor(cell_bias_, f); }

  void SetOutputGateBias(std::vector<float> f) {
    PopulateTensor(output_gate_bias_, f);
  }

  void SetProjectionWeights(std::vector<float> f) {
    PopulateTensor(projection_weights_, f);
  }

  void SetProjectionBias(std::vector<float> f) {
    PopulateTensor(projection_bias_, f);
  }

  void SetInput(int offset, const float* begin, const float* end) {
    PopulateTensor(input_, offset, const_cast<float*>(begin),
                   const_cast<float*>(end));
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int num_inputs() { return n_input_; }
  int num_outputs() { return n_output_; }
  int num_cells() { return n_cell_; }
  int num_batches() { return n_batch_; }

 protected:
  int input_;
  int input_to_input_weights_;
  int input_to_forget_weights_;
  int input_to_cell_weights_;
  int input_to_output_weights_;

  int recurrent_to_input_weights_;
  int recurrent_to_forget_weights_;
  int recurrent_to_cell_weights_;
  int recurrent_to_output_weights_;

  int cell_to_input_weights_;
  int cell_to_forget_weights_;
  int cell_to_output_weights_;

  int input_layer_norm_weights_;
  int forget_layer_norm_weights_;
  int cell_layer_norm_weights_;
  int output_layer_norm_weights_;

  int input_gate_bias_;
  int forget_gate_bias_;
  int cell_bias_;
  int output_gate_bias_;

  int projection_weights_;
  int projection_bias_;

  int output_state_;
  int cell_state_;

  int output_;

  int n_batch_;
  int n_input_;
  int n_cell_;
  int n_output_;
};

class HybridLayerNormLSTMOpModel : public LayerNormLSTMOpModel {
 public:
  HybridLayerNormLSTMOpModel(int n_batch, int n_input, int n_cell, int n_output,
                             bool use_cifg, bool use_peephole,
                             bool use_projection_weights,
                             bool use_projection_bias, float cell_clip,
                             float proj_clip,
                             const std::vector<std::vector<int>>& input_shapes)
      : LayerNormLSTMOpModel(n_batch, n_input, n_cell, n_output, use_cifg,
                             use_peephole, use_projection_weights,
                             use_projection_bias, cell_clip, proj_clip,
                             input_shapes, TensorType_UINT8) {}

  void SetInputToInputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(cell_to_output_weights_, f);
  }

  void SetInputLayerNormWeights(std::vector<float> f) {
    PopulateTensor(input_layer_norm_weights_, f);
  }

  void SetForgetLayerNormWeights(std::vector<float> f) {
    PopulateTensor(forget_layer_norm_weights_, f);
  }

  void SetCellLayerNormWeights(std::vector<float> f) {
    PopulateTensor(cell_layer_norm_weights_, f);
  }

  void SetOutputLayerNormWeights(std::vector<float> f) {
    PopulateTensor(output_layer_norm_weights_, f);
  }

  void SetProjectionWeights(std::vector<float> f) {
    SymmetricQuantizeAndPopulate(projection_weights_, f);
  }
};

class BaseLayerNormLstmTest : public ::testing::Test {
 protected:
  // Weights of the Layer Norm LSTM model. Some are optional.
  std::vector<float> input_to_input_weights_;
  std::vector<float> input_to_cell_weights_;
  std::vector<float> input_to_forget_weights_;
  std::vector<float> input_to_output_weights_;
  std::vector<float> input_gate_bias_;
  std::vector<float> cell_gate_bias_;
  std::vector<float> forget_gate_bias_;
  std::vector<float> output_gate_bias_;
  std::vector<float> recurrent_to_input_weights_;
  std::vector<float> recurrent_to_cell_weights_;
  std::vector<float> recurrent_to_forget_weights_;
  std::vector<float> recurrent_to_output_weights_;
  std::vector<float> cell_to_input_weights_;
  std::vector<float> cell_to_forget_weights_;
  std::vector<float> cell_to_output_weights_;
  std::vector<float> input_layer_norm_weights_;
  std::vector<float> forget_layer_norm_weights_;
  std::vector<float> cell_layer_norm_weights_;
  std::vector<float> output_layer_norm_weights_;
  std::vector<float> projection_weights_;

  // Layer Norm LSTM input is stored as num_batch x num_inputs vector.
  std::vector<std::vector<float>> layer_norm_lstm_input_;

  // Compares output up to tolerance to the result of the layer_norm_lstm given
  // the input.
  void VerifyGoldens(const std::vector<std::vector<float>>& input,
                     const std::vector<std::vector<float>>& output,
                     LayerNormLSTMOpModel* layer_norm_lstm,
                     float tolerance = 1e-5) {
    const int num_batches = input.size();
    EXPECT_GT(num_batches, 0);
    const int num_inputs = layer_norm_lstm->num_inputs();
    EXPECT_GT(num_inputs, 0);
    const int input_sequence_size = input[0].size() / num_inputs;
    EXPECT_GT(input_sequence_size, 0);
    for (int i = 0; i < input_sequence_size; ++i) {
      for (int b = 0; b < num_batches; ++b) {
        const float* batch_start = input[b].data() + i * num_inputs;
        const float* batch_end = batch_start + num_inputs;

        layer_norm_lstm->SetInput(b * layer_norm_lstm->num_inputs(),
                                  batch_start, batch_end);
      }

      layer_norm_lstm->Invoke();

      const int num_outputs = layer_norm_lstm->num_outputs();
      std::vector<float> expected;
      for (int b = 0; b < num_batches; ++b) {
        const float* golden_start_batch = output[b].data() + i * num_outputs;
        const float* golden_end_batch = golden_start_batch + num_outputs;
        expected.insert(expected.end(), golden_start_batch, golden_end_batch);
      }
      EXPECT_THAT(layer_norm_lstm->GetOutput(),
                  ElementsAreArray(ArrayFloatNear(expected, tolerance)));
    }
  }
};

class NoCifgPeepholeProjectionNoClippingLayerNormLstmTest
    : public BaseLayerNormLstmTest {
  void SetUp() override {
    input_to_input_weights_ = {0.5,  0.6,  0.7,  -0.8, -0.9, 0.1,  0.2,
                               0.3,  -0.4, 0.5,  -0.8, 0.7,  -0.6, 0.5,
                               -0.4, -0.5, -0.4, -0.3, -0.2, -0.1};

    input_to_forget_weights_ = {-0.6, -0.1, 0.3,  0.2,  0.9,  -0.5, -0.2,
                                -0.4, 0.3,  -0.8, -0.4, 0.3,  -0.5, -0.4,
                                -0.6, 0.3,  -0.4, -0.6, -0.5, -0.5};

    input_to_cell_weights_ = {-0.4, -0.3, -0.2, -0.1, -0.5, 0.5,  -0.2,
                              -0.3, -0.2, -0.6, 0.6,  -0.1, -0.4, -0.3,
                              -0.7, 0.7,  -0.9, -0.5, 0.8,  0.6};

    input_to_output_weights_ = {-0.8, -0.4, -0.2, -0.9, -0.1, -0.7, 0.3,
                                -0.3, -0.8, -0.2, 0.6,  -0.2, 0.4,  -0.7,
                                -0.3, -0.5, 0.1,  0.5,  -0.6, -0.4};

    input_gate_bias_ = {0.03, 0.15, 0.22, 0.38};

    forget_gate_bias_ = {0.1, -0.3, -0.2, 0.1};

    cell_gate_bias_ = {-0.05, 0.72, 0.25, 0.08};

    output_gate_bias_ = {0.05, -0.01, 0.2, 0.1};

    recurrent_to_input_weights_ = {-0.2, -0.3, 0.4,  0.1,  -0.5, 0.9,
                                   -0.2, -0.3, -0.7, 0.05, -0.2, -0.6};

    recurrent_to_cell_weights_ = {-0.3, 0.2, 0.1, -0.3, 0.8,  -0.08,
                                  -0.2, 0.3, 0.8, -0.6, -0.1, 0.2};

    recurrent_to_forget_weights_ = {-0.5, -0.3, -0.5, -0.2, 0.6, 0.4,
                                    0.9,  0.3,  -0.1, 0.2,  0.5, 0.2};

    recurrent_to_output_weights_ = {0.3,  -0.1, 0.1,  -0.2, -0.5, -0.7,
                                    -0.2, -0.6, -0.1, -0.4, -0.7, -0.2};

    cell_to_input_weights_ = {0.05, 0.1, 0.25, 0.15};

    cell_to_forget_weights_ = {-0.02, -0.15, -0.25, -0.03};

    cell_to_output_weights_ = {0.1, -0.1, -0.5, 0.05};

    input_layer_norm_weights_ = {0.1, 0.2, 0.3, 0.5};
    forget_layer_norm_weights_ = {0.2, 0.2, 0.4, 0.3};
    cell_layer_norm_weights_ = {0.7, 0.2, 0.3, 0.8};
    output_layer_norm_weights_ = {0.6, 0.2, 0.2, 0.5};

    projection_weights_ = {-0.1, 0.2,  0.01, -0.2, 0.1,  0.5,
                           0.3,  0.08, 0.07, 0.2,  -0.4, 0.2};

    layer_norm_lstm_input_ = {
        {// Batch0: 3 (input_sequence_size) * 5 (n_input)
         0.7, 0.8, 0.1, 0.2, 0.3,   // seq 0
         0.8, 0.1, 0.2, 0.4, 0.5,   // seq 1
         0.2, 0.7, 0.7, 0.1, 0.7},  // seq 2

        {// Batch1: 3 (input_sequence_size) * 5 (n_input)
         0.3, 0.2, 0.9, 0.8, 0.1,   // seq 0
         0.1, 0.5, 0.2, 0.4, 0.2,   // seq 1
         0.6, 0.9, 0.2, 0.5, 0.7},  // seq 2
    };
  }
};

TEST_F(NoCifgPeepholeProjectionNoClippingLayerNormLstmTest,
       LayerNormLstmBlackBoxTest) {
  const int n_batch = 2;
  const int n_input = 5;
  const int n_cell = 4;
  const int n_output = 3;
  const float ceil_clip = 0.0;
  const float proj_clip = 0.0;

  LayerNormLSTMOpModel layer_norm_lstm(
      n_batch, n_input, n_cell, n_output,
      /*use_cifg=*/false, /*use_peephole=*/true,
      /*use_projection_weights=*/true,
      /*use_projection_bias=*/false, ceil_clip, proj_clip,
      {
          {n_batch, n_input},  // input tensor

          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {n_cell},  // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {n_cell},  // input_layer_norm_weight tensor
          {n_cell},  // forget_layer_norm_weight tensor
          {n_cell},  // cell_layer_norm_weight tensor
          {n_cell},  // output_layer_norm_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor
      });

  layer_norm_lstm.SetInputToInputWeights(input_to_input_weights_);
  layer_norm_lstm.SetInputToCellWeights(input_to_cell_weights_);
  layer_norm_lstm.SetInputToForgetWeights(input_to_forget_weights_);
  layer_norm_lstm.SetInputToOutputWeights(input_to_output_weights_);

  layer_norm_lstm.SetInputGateBias(input_gate_bias_);
  layer_norm_lstm.SetCellBias(cell_gate_bias_);
  layer_norm_lstm.SetForgetGateBias(forget_gate_bias_);
  layer_norm_lstm.SetOutputGateBias(output_gate_bias_);

  layer_norm_lstm.SetRecurrentToInputWeights(recurrent_to_input_weights_);
  layer_norm_lstm.SetRecurrentToCellWeights(recurrent_to_cell_weights_);
  layer_norm_lstm.SetRecurrentToForgetWeights(recurrent_to_forget_weights_);
  layer_norm_lstm.SetRecurrentToOutputWeights(recurrent_to_output_weights_);

  layer_norm_lstm.SetCellToInputWeights(cell_to_input_weights_);
  layer_norm_lstm.SetCellToForgetWeights(cell_to_forget_weights_);
  layer_norm_lstm.SetCellToOutputWeights(cell_to_output_weights_);

  layer_norm_lstm.SetInputLayerNormWeights(input_layer_norm_weights_);
  layer_norm_lstm.SetForgetLayerNormWeights(forget_layer_norm_weights_);
  layer_norm_lstm.SetCellLayerNormWeights(cell_layer_norm_weights_);
  layer_norm_lstm.SetOutputLayerNormWeights(output_layer_norm_weights_);

  layer_norm_lstm.SetProjectionWeights(projection_weights_);

  // Verify the final output.
  const std::vector<std::vector<float>> layer_norm_lstm_golden_output = {
      {
          // Batch0: 3 (input_sequence_size) * 3 (n_output)
          0.0244077, 0.128027, -0.00170918,  // seq 0
          0.0137642, 0.140751, 0.0395835,    // seq 1
          -0.00459231, 0.155278, 0.0837377,  // seq 2
      },
      {
          // Batch1: 3 (input_sequence_size) * 3 (n_output)
          -0.00692428, 0.0848741, 0.063445,  // seq 0
          -0.00403912, 0.139963, 0.072681,   // seq 1
          0.00752706, 0.161903, 0.0561371,   // seq 2
      }};

  VerifyGoldens(layer_norm_lstm_input_, layer_norm_lstm_golden_output,
                &layer_norm_lstm);
}

TEST_F(NoCifgPeepholeProjectionNoClippingLayerNormLstmTest,
       HybridLayerNormLstmBlackBoxTest) {
  const int n_batch = 2;
  const int n_input = 5;
  const int n_cell = 4;
  const int n_output = 3;
  const float ceil_clip = 0.0;
  const float proj_clip = 0.0;

  HybridLayerNormLSTMOpModel layer_norm_lstm(
      n_batch, n_input, n_cell, n_output,
      /*use_cifg=*/false, /*use_peephole=*/true,
      /*use_projection_weights=*/true,
      /*use_projection_bias=*/false, ceil_clip, proj_clip,
      {
          {n_batch, n_input},  // input tensor

          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {n_cell},  // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {n_cell},  // input_layer_norm_weight tensor
          {n_cell},  // forget_layer_norm_weight tensor
          {n_cell},  // cell_layer_norm_weight tensor
          {n_cell},  // output_layer_norm_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor
      });

  layer_norm_lstm.SetInputToInputWeights(input_to_input_weights_);
  layer_norm_lstm.SetInputToCellWeights(input_to_cell_weights_);
  layer_norm_lstm.SetInputToForgetWeights(input_to_forget_weights_);
  layer_norm_lstm.SetInputToOutputWeights(input_to_output_weights_);

  layer_norm_lstm.SetInputGateBias(input_gate_bias_);
  layer_norm_lstm.SetCellBias(cell_gate_bias_);
  layer_norm_lstm.SetForgetGateBias(forget_gate_bias_);
  layer_norm_lstm.SetOutputGateBias(output_gate_bias_);

  layer_norm_lstm.SetRecurrentToInputWeights(recurrent_to_input_weights_);
  layer_norm_lstm.SetRecurrentToCellWeights(recurrent_to_cell_weights_);
  layer_norm_lstm.SetRecurrentToForgetWeights(recurrent_to_forget_weights_);
  layer_norm_lstm.SetRecurrentToOutputWeights(recurrent_to_output_weights_);

  layer_norm_lstm.SetCellToInputWeights(cell_to_input_weights_);
  layer_norm_lstm.SetCellToForgetWeights(cell_to_forget_weights_);
  layer_norm_lstm.SetCellToOutputWeights(cell_to_output_weights_);

  layer_norm_lstm.SetInputLayerNormWeights(input_layer_norm_weights_);
  layer_norm_lstm.SetForgetLayerNormWeights(forget_layer_norm_weights_);
  layer_norm_lstm.SetCellLayerNormWeights(cell_layer_norm_weights_);
  layer_norm_lstm.SetOutputLayerNormWeights(output_layer_norm_weights_);

  layer_norm_lstm.SetProjectionWeights(projection_weights_);

  const std::vector<std::vector<float>> layer_norm_lstm_golden_output = {
      {
          // Batch0: 3 (input_sequence_size) * 3 (n_output)
          0.0244576, 0.127847, -0.00181765,  // seq 0
          0.0137518, 0.140892, 0.0402234,    // seq 1
          -0.0048839, 0.155096, 0.0840309,   // seq 2
      },
      {
          // Batch1: 3 (input_sequence_size) * 3 (n_output)
          -0.00728636, 0.0843957, 0.0634786,  // seq 0
          -0.00448382, 0.139278, 0.0737372,   // seq 1
          0.00734616, 0.161793, 0.0560238,    // seq 2
      }};

  VerifyGoldens(layer_norm_lstm_input_, layer_norm_lstm_golden_output,
                &layer_norm_lstm);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
