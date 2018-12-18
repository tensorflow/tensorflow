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
// Unit test for TFLite Bidirectional LSTM op.

#include <initializer_list>
#include <iomanip>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BidirectionalLSTMOpModel : public SingleOpModel {
 public:
  BidirectionalLSTMOpModel(int n_batch, int n_input, int n_cell, int n_output,
                           int sequence_length, bool use_cifg,
                           bool use_peephole, bool use_projection_weights,
                           bool use_projection_bias, bool merge_outputs,
                           float cell_clip, float proj_clip,
                           bool quantize_weights, bool time_major,
                           const std::vector<std::vector<int>>& input_shapes)
      : n_batch_(n_batch),
        n_input_(n_input),
        n_fw_cell_(n_cell),
        n_bw_cell_(n_cell),
        n_fw_output_(n_output),
        n_bw_output_(n_output),
        sequence_length_(sequence_length),
        quantize_weights_(quantize_weights) {
    input_ = AddInput(TensorType_FLOAT32);
    const auto weight_type =
        quantize_weights_ ? TensorType_UINT8 : TensorType_FLOAT32;

    if (use_cifg) {
      fw_input_to_input_weights_ = AddNullInput();
    } else {
      fw_input_to_input_weights_ = AddInput(weight_type);
    }

    fw_input_to_forget_weights_ = AddInput(weight_type);
    fw_input_to_cell_weights_ = AddInput(weight_type);
    fw_input_to_output_weights_ = AddInput(weight_type);

    if (use_cifg) {
      fw_recurrent_to_input_weights_ = AddNullInput();
    } else {
      fw_recurrent_to_input_weights_ = AddInput(weight_type);
    }

    fw_recurrent_to_forget_weights_ = AddInput(weight_type);
    fw_recurrent_to_cell_weights_ = AddInput(weight_type);
    fw_recurrent_to_output_weights_ = AddInput(weight_type);

    if (use_peephole) {
      if (use_cifg) {
        fw_cell_to_input_weights_ = AddNullInput();
      } else {
        fw_cell_to_input_weights_ = AddInput(weight_type);
      }
      fw_cell_to_forget_weights_ = AddInput(weight_type);
      fw_cell_to_output_weights_ = AddInput(weight_type);
    } else {
      fw_cell_to_input_weights_ = AddNullInput();
      fw_cell_to_forget_weights_ = AddNullInput();
      fw_cell_to_output_weights_ = AddNullInput();
    }

    if (use_cifg) {
      fw_input_gate_bias_ = AddNullInput();
    } else {
      fw_input_gate_bias_ = AddInput(TensorType_FLOAT32);
    }
    fw_forget_gate_bias_ = AddInput(TensorType_FLOAT32);
    fw_cell_bias_ = AddInput(TensorType_FLOAT32);
    fw_output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      fw_projection_weights_ = AddInput(TensorType_FLOAT32);
      if (use_projection_bias) {
        fw_projection_bias_ = AddInput(TensorType_FLOAT32);
      } else {
        fw_projection_bias_ = AddNullInput();
      }
    } else {
      fw_projection_weights_ = AddNullInput();
      fw_projection_bias_ = AddNullInput();
    }

    if (use_cifg) {
      bw_input_to_input_weights_ = AddNullInput();
    } else {
      bw_input_to_input_weights_ = AddInput(weight_type);
    }

    bw_input_to_forget_weights_ = AddInput(weight_type);
    bw_input_to_cell_weights_ = AddInput(weight_type);
    bw_input_to_output_weights_ = AddInput(weight_type);

    if (use_cifg) {
      bw_recurrent_to_input_weights_ = AddNullInput();
    } else {
      bw_recurrent_to_input_weights_ = AddInput(weight_type);
    }

    bw_recurrent_to_forget_weights_ = AddInput(weight_type);
    bw_recurrent_to_cell_weights_ = AddInput(weight_type);
    bw_recurrent_to_output_weights_ = AddInput(weight_type);

    if (use_peephole) {
      if (use_cifg) {
        bw_cell_to_input_weights_ = AddNullInput();
      } else {
        bw_cell_to_input_weights_ = AddInput(weight_type);
      }
      bw_cell_to_forget_weights_ = AddInput(weight_type);
      bw_cell_to_output_weights_ = AddInput(weight_type);
    } else {
      bw_cell_to_input_weights_ = AddNullInput();
      bw_cell_to_forget_weights_ = AddNullInput();
      bw_cell_to_output_weights_ = AddNullInput();
    }

    if (use_cifg) {
      bw_input_gate_bias_ = AddNullInput();
    } else {
      bw_input_gate_bias_ = AddInput(TensorType_FLOAT32);
    }
    bw_forget_gate_bias_ = AddInput(TensorType_FLOAT32);
    bw_cell_bias_ = AddInput(TensorType_FLOAT32);
    bw_output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      bw_projection_weights_ = AddInput(weight_type);
      if (use_projection_bias) {
        bw_projection_bias_ = AddInput(TensorType_FLOAT32);
      } else {
        bw_projection_bias_ = AddNullInput();
      }
    } else {
      bw_projection_weights_ = AddNullInput();
      bw_projection_bias_ = AddNullInput();
    }

    // Adding the 2 input state tensors.
    fw_input_activation_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_fw_output_ * n_batch_}},
                 /*is_variable=*/true);
    fw_input_cell_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_fw_cell_ * n_batch_}},
                 /*is_variable=*/true);

    // Adding the 2 input state tensors.
    bw_input_activation_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_bw_output_ * n_batch_}},
                 /*is_variable=*/true);
    bw_input_cell_state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_bw_cell_ * n_batch_}},
                 /*is_variable=*/true);

    fw_output_ = AddOutput(TensorType_FLOAT32);

    if (!merge_outputs) {
      bw_output_ = AddOutput(TensorType_FLOAT32);
    }

    aux_input_ = AddNullInput();
    fw_aux_input_to_input_weights_ = AddNullInput();
    fw_aux_input_to_forget_weights_ = AddNullInput();
    fw_aux_input_to_cell_weights_ = AddNullInput();
    fw_aux_input_to_output_weights_ = AddNullInput();
    bw_aux_input_to_input_weights_ = AddNullInput();
    bw_aux_input_to_forget_weights_ = AddNullInput();
    bw_aux_input_to_cell_weights_ = AddNullInput();
    bw_aux_input_to_output_weights_ = AddNullInput();

    SetBuiltinOp(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM,
                 BuiltinOptions_BidirectionalSequenceLSTMOptions,
                 CreateBidirectionalSequenceLSTMOptions(
                     builder_, ActivationFunctionType_TANH, cell_clip,
                     proj_clip, merge_outputs, time_major)
                     .Union());
    BuildInterpreter(input_shapes);
  }

  void PopulateWeightTensor(int tensor_id, const std::vector<float>& f) {
    if (quantize_weights_) {
      SymmetricQuantizeAndPopulate(tensor_id, f);
    } else {
      PopulateTensor(tensor_id, f);
    }
  }

  // Set weights in forward and backward cells to be the same.
  void SetInputToInputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_input_to_input_weights_, f);
    PopulateWeightTensor(bw_input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_input_to_forget_weights_, f);
    PopulateWeightTensor(bw_input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_input_to_cell_weights_, f);
    PopulateWeightTensor(bw_input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_input_to_output_weights_, f);
    PopulateWeightTensor(bw_input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_recurrent_to_input_weights_, f);
    PopulateWeightTensor(bw_recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_recurrent_to_forget_weights_, f);
    PopulateWeightTensor(bw_recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_recurrent_to_cell_weights_, f);
    PopulateWeightTensor(bw_recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_recurrent_to_output_weights_, f);
    PopulateWeightTensor(bw_recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_cell_to_input_weights_, f);
    PopulateWeightTensor(bw_cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_cell_to_forget_weights_, f);
    PopulateWeightTensor(bw_cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_cell_to_output_weights_, f);
    PopulateWeightTensor(bw_cell_to_output_weights_, f);
  }

  void SetInputGateBias(const std::vector<float>& f) {
    PopulateTensor(fw_input_gate_bias_, f);
    PopulateTensor(bw_input_gate_bias_, f);
  }

  void SetForgetGateBias(const std::vector<float>& f) {
    PopulateTensor(fw_forget_gate_bias_, f);
    PopulateTensor(bw_forget_gate_bias_, f);
  }

  void SetCellBias(const std::vector<float>& f) {
    PopulateTensor(fw_cell_bias_, f);
    PopulateTensor(bw_cell_bias_, f);
  }

  void SetOutputGateBias(const std::vector<float>& f) {
    PopulateTensor(fw_output_gate_bias_, f);
    PopulateTensor(bw_output_gate_bias_, f);
  }

  void SetProjectionWeights(const std::vector<float>& f) {
    PopulateWeightTensor(fw_projection_weights_, f);
    PopulateWeightTensor(bw_projection_weights_, f);
  }

  void SetProjectionBias(const std::vector<float>& f) {
    PopulateTensor(fw_projection_bias_, f);
    PopulateTensor(bw_projection_bias_, f);
  }

  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetFwOutput() { return ExtractVector<float>(fw_output_); }
  std::vector<float> GetBwOutput() { return ExtractVector<float>(bw_output_); }

  int num_inputs() { return n_input_; }
  int num_fw_outputs() { return n_fw_output_; }
  int num_bw_outputs() { return n_bw_output_; }
  int num_fw_cells() { return n_fw_cell_; }
  int num_bw_cells() { return n_bw_cell_; }
  int num_batches() { return n_batch_; }
  int sequence_length() { return sequence_length_; }

 private:
  int input_;
  int fw_input_to_input_weights_;
  int fw_input_to_forget_weights_;
  int fw_input_to_cell_weights_;
  int fw_input_to_output_weights_;

  int fw_recurrent_to_input_weights_;
  int fw_recurrent_to_forget_weights_;
  int fw_recurrent_to_cell_weights_;
  int fw_recurrent_to_output_weights_;

  int fw_cell_to_input_weights_;
  int fw_cell_to_forget_weights_;
  int fw_cell_to_output_weights_;

  int fw_input_gate_bias_;
  int fw_forget_gate_bias_;
  int fw_cell_bias_;
  int fw_output_gate_bias_;

  int fw_projection_weights_;
  int fw_projection_bias_;

  int bw_input_to_input_weights_;
  int bw_input_to_forget_weights_;
  int bw_input_to_cell_weights_;
  int bw_input_to_output_weights_;

  int bw_recurrent_to_input_weights_;
  int bw_recurrent_to_forget_weights_;
  int bw_recurrent_to_cell_weights_;
  int bw_recurrent_to_output_weights_;

  int bw_cell_to_input_weights_;
  int bw_cell_to_forget_weights_;
  int bw_cell_to_output_weights_;

  int bw_input_gate_bias_;
  int bw_forget_gate_bias_;
  int bw_cell_bias_;
  int bw_output_gate_bias_;

  int bw_projection_weights_;
  int bw_projection_bias_;

  int fw_input_activation_state_;
  int fw_input_cell_state_;
  int bw_input_activation_state_;
  int bw_input_cell_state_;

  int fw_output_;
  int bw_output_;

  int aux_input_;
  int fw_aux_input_to_input_weights_;
  int fw_aux_input_to_forget_weights_;
  int fw_aux_input_to_cell_weights_;
  int fw_aux_input_to_output_weights_;
  int bw_aux_input_to_input_weights_;
  int bw_aux_input_to_forget_weights_;
  int bw_aux_input_to_cell_weights_;
  int bw_aux_input_to_output_weights_;

  int n_batch_;
  int n_input_;
  int n_fw_cell_;
  int n_bw_cell_;
  int n_fw_output_;
  int n_bw_output_;
  int sequence_length_;

  bool quantize_weights_;
};

// Declare LSTMOpTest as a parameterized test, where the parameter is a boolean
// indicating whether to use quantization or not.
class LSTMOpTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_CASE_P(QuantizationOrNot, LSTMOpTest, ::testing::Bool());

TEST_P(LSTMOpTest, BlackBoxTestNoCifgNoPeepholeNoProjectionNoClipping) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;
  const bool quantize_weights = GetParam();

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/false,
      /*use_peephole=*/false, /*use_projection_weights=*/false,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, quantize_weights, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

          // Forward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          // Backward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToInputWeights({-0.45018822, -0.02338299, -0.0870589,
                               -0.34550029, 0.04266912, -0.15680569,
                               -0.34856534, 0.43890524});

  lstm.SetInputToCellWeights({-0.50013041, 0.1370284, 0.11810488, 0.2013163,
                              -0.20583314, 0.44344562, 0.22077113,
                              -0.29909778});

  lstm.SetInputToForgetWeights({0.09701663, 0.20334584, -0.50592935,
                                -0.31343272, -0.40032279, 0.44781327,
                                0.01387155, -0.35593212});

  lstm.SetInputToOutputWeights({-0.25065863, -0.28290087, 0.04613829,
                                0.40525138, 0.44272184, 0.03897077, -0.1556896,
                                0.19487578});

  lstm.SetInputGateBias({0., 0., 0., 0.});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToInputWeights(
      {-0.0063535, -0.2042388, 0.31454784, -0.35746509, 0.28902304, 0.08183324,
       -0.16555229, 0.02286911, -0.13566875, 0.03034258, 0.48091322,
       -0.12528998, 0.24077177, -0.51332325, -0.33502164, 0.10629296});

  lstm.SetRecurrentToCellWeights(
      {-0.3407414, 0.24443203, -0.2078532, 0.26320225, 0.05695659, -0.00123841,
       -0.4744786, -0.35869038, -0.06418842, -0.13502428, -0.501764, 0.22830659,
       -0.46367589, 0.26016325, -0.03894562, -0.16368064});

  lstm.SetRecurrentToForgetWeights(
      {-0.48684245, -0.06655136, 0.42224967, 0.2112639, 0.27654213, 0.20864892,
       -0.07646349, 0.45877004, 0.00141793, -0.14609534, 0.36447752, 0.09196436,
       0.28053468, 0.01560611, -0.20127171, -0.01140004});

  lstm.SetRecurrentToOutputWeights(
      {0.43385774, -0.17194885, 0.2718237, 0.09215671, 0.24107647, -0.39835793,
       0.18212086, 0.01301402, 0.48572797, -0.50656658, 0.20047462, -0.20607421,
       -0.51818722, -0.15390486, 0.0468148, 0.39922136});

  // Input should have n_input * sequence_length many values.
  static float lstm_input[] = {2., 3., 3., 4., 1., 1.};
  static float lstm_fw_golden_output[] = {
      -0.02973187, 0.1229473,  0.20885126, -0.15358765,
      -0.03716109, 0.12507336, 0.41193449, -0.20860538,
      -0.15053082, 0.09120187, 0.24278517, -0.12222792};
  static float lstm_bw_golden_output[] = {
      -0.0806187, 0.139077, 0.400476,   -0.197842, -0.0332076, 0.123838,
      0.309777,   -0.17621, -0.0490733, 0.0739237, 0.067706,   -0.0208124};

  float* batch0_start = lstm_input;
  float* batch0_end = batch0_start + lstm.num_inputs() * lstm.sequence_length();

  lstm.SetInput(0, batch0_start, batch0_end);

  lstm.Invoke();

  float* fw_golden_start = lstm_fw_golden_output;
  float* fw_golden_end =
      fw_golden_start + lstm.num_fw_outputs() * lstm.sequence_length();
  std::vector<float> fw_expected;
  fw_expected.insert(fw_expected.end(), fw_golden_start, fw_golden_end);
  EXPECT_THAT(lstm.GetFwOutput(),
              ElementsAreArray(
                  ArrayFloatNear(fw_expected, quantize_weights ? 1e-2 : 1e-5)));

  float* bw_golden_start = lstm_bw_golden_output;
  float* bw_golden_end =
      bw_golden_start + lstm.num_bw_outputs() * lstm.sequence_length();
  std::vector<float> bw_expected;
  bw_expected.insert(bw_expected.end(), bw_golden_start, bw_golden_end);
  EXPECT_THAT(lstm.GetBwOutput(),
              ElementsAreArray(
                  ArrayFloatNear(bw_expected, quantize_weights ? 1e-2 : 1e-5)));
}

// Same as the previous test, yet with a single merged output tensor and n_batch
// of 2.
TEST_P(LSTMOpTest, BlackBoxTestMergedOutput) {
  const int n_batch = 2;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;
  const bool quantize_weights = GetParam();

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/false,
      /*use_peephole=*/false, /*use_projection_weights=*/false,
      /*use_projection_bias=*/false, /*merge_outputs=*/true, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, quantize_weights, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

          // Forward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          // Backward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToInputWeights({-0.45018822, -0.02338299, -0.0870589,
                               -0.34550029, 0.04266912, -0.15680569,
                               -0.34856534, 0.43890524});

  lstm.SetInputToCellWeights({-0.50013041, 0.1370284, 0.11810488, 0.2013163,
                              -0.20583314, 0.44344562, 0.22077113,
                              -0.29909778});

  lstm.SetInputToForgetWeights({0.09701663, 0.20334584, -0.50592935,
                                -0.31343272, -0.40032279, 0.44781327,
                                0.01387155, -0.35593212});

  lstm.SetInputToOutputWeights({-0.25065863, -0.28290087, 0.04613829,
                                0.40525138, 0.44272184, 0.03897077, -0.1556896,
                                0.19487578});

  lstm.SetInputGateBias({0., 0., 0., 0.});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToInputWeights(
      {-0.0063535, -0.2042388, 0.31454784, -0.35746509, 0.28902304, 0.08183324,
       -0.16555229, 0.02286911, -0.13566875, 0.03034258, 0.48091322,
       -0.12528998, 0.24077177, -0.51332325, -0.33502164, 0.10629296});

  lstm.SetRecurrentToCellWeights(
      {-0.3407414, 0.24443203, -0.2078532, 0.26320225, 0.05695659, -0.00123841,
       -0.4744786, -0.35869038, -0.06418842, -0.13502428, -0.501764, 0.22830659,
       -0.46367589, 0.26016325, -0.03894562, -0.16368064});

  lstm.SetRecurrentToForgetWeights(
      {-0.48684245, -0.06655136, 0.42224967, 0.2112639, 0.27654213, 0.20864892,
       -0.07646349, 0.45877004, 0.00141793, -0.14609534, 0.36447752, 0.09196436,
       0.28053468, 0.01560611, -0.20127171, -0.01140004});

  lstm.SetRecurrentToOutputWeights(
      {0.43385774, -0.17194885, 0.2718237, 0.09215671, 0.24107647, -0.39835793,
       0.18212086, 0.01301402, 0.48572797, -0.50656658, 0.20047462, -0.20607421,
       -0.51818722, -0.15390486, 0.0468148, 0.39922136});

  // Input should have n_input * sequence_length many values.
  static float lstm_input[] = {2., 3., 2., 3., 3., 4., 3., 4., 1., 1., 1., 1.};
  static float lstm_fw_golden_output[] = {
      -0.02973187, 0.1229473,   0.20885126,  -0.15358765, -0.02973187,
      0.1229473,   0.20885126,  -0.15358765, -0.03716109, 0.12507336,
      0.41193449,  -0.20860538, -0.03716109, 0.12507336,  0.41193449,
      -0.20860538, -0.15053082, 0.09120187,  0.24278517,  -0.12222792,
      -0.15053082, 0.09120187,  0.24278517,  -0.12222792};
  static float lstm_bw_golden_output[] = {
      -0.0806187, 0.139077,   0.400476,   -0.197842, -0.0806187, 0.139077,
      0.400476,   -0.197842,  -0.0332076, 0.123838,  0.309777,   -0.17621,
      -0.0332076, 0.123838,   0.309777,   -0.17621,  -0.0490733, 0.0739237,
      0.067706,   -0.0208124, -0.0490733, 0.0739237, 0.067706,   -0.0208124};

  float* batch0_start = lstm_input;
  float* batch0_end = batch0_start + lstm.num_inputs() * lstm.num_batches() *
                                         lstm.sequence_length();

  lstm.SetInput(0, batch0_start, batch0_end);

  lstm.Invoke();

  std::vector<float> merged_expected;
  for (int k = 0; k < lstm.sequence_length() * lstm.num_batches(); k++) {
    merged_expected.insert(
        merged_expected.end(),
        lstm_fw_golden_output + k * lstm.num_fw_outputs(),
        lstm_fw_golden_output + (k + 1) * lstm.num_fw_outputs());
    merged_expected.insert(
        merged_expected.end(),
        lstm_bw_golden_output + k * lstm.num_bw_outputs(),
        lstm_bw_golden_output + (k + 1) * lstm.num_bw_outputs());
  }
  EXPECT_THAT(lstm.GetFwOutput(),
              ElementsAreArray(ArrayFloatNear(merged_expected,
                                              quantize_weights ? 1e-2 : 1e-5)));
}

TEST(LSTMOpTest, BlackBoxTestNoCifgNoPeepholeNoProjectionNoClippingReverse) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/false,
      /*use_peephole=*/false, /*use_projection_weights=*/false,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, /*quantize_weights=*/false, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

          // Forward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          // Backward cell
          {n_cell, n_input},  // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {n_cell, n_output},  // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},  // cell_to_input_weight tensor
          {0},  // cell_to_forget_weight tensor
          {0},  // cell_to_output_weight tensor

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToInputWeights({-0.45018822, -0.02338299, -0.0870589,
                               -0.34550029, 0.04266912, -0.15680569,
                               -0.34856534, 0.43890524});

  lstm.SetInputToCellWeights({-0.50013041, 0.1370284, 0.11810488, 0.2013163,
                              -0.20583314, 0.44344562, 0.22077113,
                              -0.29909778});

  lstm.SetInputToForgetWeights({0.09701663, 0.20334584, -0.50592935,
                                -0.31343272, -0.40032279, 0.44781327,
                                0.01387155, -0.35593212});

  lstm.SetInputToOutputWeights({-0.25065863, -0.28290087, 0.04613829,
                                0.40525138, 0.44272184, 0.03897077, -0.1556896,
                                0.19487578});

  lstm.SetInputGateBias({0., 0., 0., 0.});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToInputWeights(
      {-0.0063535, -0.2042388, 0.31454784, -0.35746509, 0.28902304, 0.08183324,
       -0.16555229, 0.02286911, -0.13566875, 0.03034258, 0.48091322,
       -0.12528998, 0.24077177, -0.51332325, -0.33502164, 0.10629296});

  lstm.SetRecurrentToCellWeights(
      {-0.3407414, 0.24443203, -0.2078532, 0.26320225, 0.05695659, -0.00123841,
       -0.4744786, -0.35869038, -0.06418842, -0.13502428, -0.501764, 0.22830659,
       -0.46367589, 0.26016325, -0.03894562, -0.16368064});

  lstm.SetRecurrentToForgetWeights(
      {-0.48684245, -0.06655136, 0.42224967, 0.2112639, 0.27654213, 0.20864892,
       -0.07646349, 0.45877004, 0.00141793, -0.14609534, 0.36447752, 0.09196436,
       0.28053468, 0.01560611, -0.20127171, -0.01140004});

  lstm.SetRecurrentToOutputWeights(
      {0.43385774, -0.17194885, 0.2718237, 0.09215671, 0.24107647, -0.39835793,
       0.18212086, 0.01301402, 0.48572797, -0.50656658, 0.20047462, -0.20607421,
       -0.51818722, -0.15390486, 0.0468148, 0.39922136});

  // Input should have n_input * sequence_length many values.
  // Check reversed inputs.
  static float lstm_input_reversed[] = {1., 1., 3., 4., 2., 3.};
  static float lstm_fw_golden_output[] = {
      -0.02973187, 0.1229473,  0.20885126, -0.15358765,
      -0.03716109, 0.12507336, 0.41193449, -0.20860538,
      -0.15053082, 0.09120187, 0.24278517, -0.12222792};
  static float lstm_bw_golden_output[] = {
      -0.0806187, 0.139077, 0.400476,   -0.197842, -0.0332076, 0.123838,
      0.309777,   -0.17621, -0.0490733, 0.0739237, 0.067706,   -0.0208124};

  float* batch0_start = lstm_input_reversed;
  float* batch0_end = batch0_start + lstm.num_inputs() * lstm.sequence_length();

  lstm.SetInput(0, batch0_start, batch0_end);

  lstm.Invoke();

  std::vector<float> fw_expected;
  for (int s = 0; s < lstm.sequence_length(); s++) {
    float* fw_golden_start = lstm_fw_golden_output + s * lstm.num_fw_outputs();
    float* fw_golden_end = fw_golden_start + lstm.num_fw_outputs();
    fw_expected.insert(fw_expected.begin(), fw_golden_start, fw_golden_end);
  }
  EXPECT_THAT(lstm.GetBwOutput(),
              ElementsAreArray(ArrayFloatNear(fw_expected)));

  std::vector<float> bw_expected;
  for (int s = 0; s < lstm.sequence_length(); s++) {
    float* bw_golden_start = lstm_bw_golden_output + s * lstm.num_bw_outputs();
    float* bw_golden_end = bw_golden_start + lstm.num_bw_outputs();
    bw_expected.insert(bw_expected.begin(), bw_golden_start, bw_golden_end);
  }
  EXPECT_THAT(lstm.GetFwOutput(),
              ElementsAreArray(ArrayFloatNear(bw_expected)));
}

TEST(LSTMOpTest, BlackBoxTestWithCifgWithPeepholeNoProjectionNoClipping) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/true,
      /*use_peephole=*/true, /*use_projection_weights=*/false,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, /*quantize_weights=*/false, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

          {0, 0},             // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {0, 0},              // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},       // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {0},       // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {0, 0},             // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {0, 0},              // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},       // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {0},       // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToCellWeights({-0.49770179, -0.27711356, -0.09624726, 0.05100781,
                              0.04717243, 0.48944736, -0.38535351,
                              -0.17212132});

  lstm.SetInputToForgetWeights({-0.55291498, -0.42866567, 0.13056988,
                                -0.3633365, -0.22755712, 0.28253698, 0.24407166,
                                0.33826375});

  lstm.SetInputToOutputWeights({0.10725588, -0.02335852, -0.55932593,
                                -0.09426838, -0.44257352, 0.54939759,
                                0.01533556, 0.42751634});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToCellWeights(
      {0.54066205, -0.32668582, -0.43562764, -0.56094903, 0.42957711,
       0.01841056, -0.32764608, -0.33027974, -0.10826075, 0.20675004,
       0.19069612, -0.03026325, -0.54532051, 0.33003211, 0.44901288,
       0.21193194});

  lstm.SetRecurrentToForgetWeights(
      {-0.13832897, -0.0515101, -0.2359007, -0.16661474, -0.14340827,
       0.36986142, 0.23414481, 0.55899, 0.10798943, -0.41174671, 0.17751795,
       -0.34484994, -0.35874045, -0.11352962, 0.27268326, 0.54058349});

  lstm.SetRecurrentToOutputWeights(
      {0.41613156, 0.42610586, -0.16495961, -0.5663873, 0.30579174, -0.05115908,
       -0.33941799, 0.23364776, 0.11178309, 0.09481031, -0.26424935, 0.46261835,
       0.50248802, 0.26114327, -0.43736315, 0.33149987});

  lstm.SetCellToForgetWeights(
      {0.47485286, -0.51955009, -0.24458408, 0.31544167});
  lstm.SetCellToOutputWeights(
      {-0.17135078, 0.82760304, 0.85573703, -0.77109635});

  static float lstm_input[] = {2., 3., 3., 4., 1., 1.};
  static float lstm_fw_golden_output[] = {
      -0.36444446, -0.00352185, 0.12886585, -0.05163646,
      -0.42312205, -0.01218222, 0.24201041, -0.08124574,
      -0.358325,   -0.04621704, 0.21641694, -0.06471302};
  static float lstm_bw_golden_output[] = {
      -0.401685, -0.0232794, 0.288642,  -0.123074,   -0.42915,  -0.00871577,
      0.20912,   -0.103567,  -0.166398, -0.00486649, 0.0697471, -0.0537578};

  float* batch0_start = lstm_input;
  float* batch0_end = batch0_start + lstm.num_inputs() * lstm.sequence_length();

  lstm.SetInput(0, batch0_start, batch0_end);

  lstm.Invoke();

  float* fw_golden_start = lstm_fw_golden_output;
  float* fw_golden_end =
      fw_golden_start + lstm.num_fw_outputs() * lstm.sequence_length();
  std::vector<float> fw_expected;
  fw_expected.insert(fw_expected.end(), fw_golden_start, fw_golden_end);
  EXPECT_THAT(lstm.GetFwOutput(),
              ElementsAreArray(ArrayFloatNear(fw_expected)));

  float* bw_golden_start = lstm_bw_golden_output;
  float* bw_golden_end =
      bw_golden_start + lstm.num_bw_outputs() * lstm.sequence_length();
  std::vector<float> bw_expected;
  bw_expected.insert(bw_expected.end(), bw_golden_start, bw_golden_end);
  EXPECT_THAT(lstm.GetBwOutput(),
              ElementsAreArray(ArrayFloatNear(bw_expected)));
}

TEST(LSTMOpTest,
     BlackBoxTestWithCifgWithPeepholeNoProjectionNoClippingReversed) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;
  const int sequence_length = 3;

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/true,
      /*use_peephole=*/true, /*use_projection_weights=*/false,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, /*quantize_weights=*/false, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

          {0, 0},             // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {0, 0},              // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},       // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {0},       // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {0, 0},             // input_to_input_weight tensor
          {n_cell, n_input},  // input_to_forget_weight tensor
          {n_cell, n_input},  // input_to_cell_weight tensor
          {n_cell, n_input},  // input_to_output_weight tensor

          {0, 0},              // recurrent_to_input_weight tensor
          {n_cell, n_output},  // recurrent_to_forget_weight tensor
          {n_cell, n_output},  // recurrent_to_cell_weight tensor
          {n_cell, n_output},  // recurrent_to_output_weight tensor

          {0},       // cell_to_input_weight tensor
          {n_cell},  // cell_to_forget_weight tensor
          {n_cell},  // cell_to_output_weight tensor

          {0},       // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {0, 0},  // projection_weight tensor
          {0},     // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToCellWeights({-0.49770179, -0.27711356, -0.09624726, 0.05100781,
                              0.04717243, 0.48944736, -0.38535351,
                              -0.17212132});

  lstm.SetInputToForgetWeights({-0.55291498, -0.42866567, 0.13056988,
                                -0.3633365, -0.22755712, 0.28253698, 0.24407166,
                                0.33826375});

  lstm.SetInputToOutputWeights({0.10725588, -0.02335852, -0.55932593,
                                -0.09426838, -0.44257352, 0.54939759,
                                0.01533556, 0.42751634});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToCellWeights(
      {0.54066205, -0.32668582, -0.43562764, -0.56094903, 0.42957711,
       0.01841056, -0.32764608, -0.33027974, -0.10826075, 0.20675004,
       0.19069612, -0.03026325, -0.54532051, 0.33003211, 0.44901288,
       0.21193194});

  lstm.SetRecurrentToForgetWeights(
      {-0.13832897, -0.0515101, -0.2359007, -0.16661474, -0.14340827,
       0.36986142, 0.23414481, 0.55899, 0.10798943, -0.41174671, 0.17751795,
       -0.34484994, -0.35874045, -0.11352962, 0.27268326, 0.54058349});

  lstm.SetRecurrentToOutputWeights(
      {0.41613156, 0.42610586, -0.16495961, -0.5663873, 0.30579174, -0.05115908,
       -0.33941799, 0.23364776, 0.11178309, 0.09481031, -0.26424935, 0.46261835,
       0.50248802, 0.26114327, -0.43736315, 0.33149987});

  lstm.SetCellToForgetWeights(
      {0.47485286, -0.51955009, -0.24458408, 0.31544167});
  lstm.SetCellToOutputWeights(
      {-0.17135078, 0.82760304, 0.85573703, -0.77109635});

  static float lstm_input_reversed[] = {1., 1., 3., 4., 2., 3.};
  static float lstm_fw_golden_output[] = {
      -0.36444446, -0.00352185, 0.12886585, -0.05163646,
      -0.42312205, -0.01218222, 0.24201041, -0.08124574,
      -0.358325,   -0.04621704, 0.21641694, -0.06471302};
  static float lstm_bw_golden_output[] = {
      -0.401685, -0.0232794, 0.288642,  -0.123074,   -0.42915,  -0.00871577,
      0.20912,   -0.103567,  -0.166398, -0.00486649, 0.0697471, -0.0537578};

  float* batch0_start = lstm_input_reversed;
  float* batch0_end = batch0_start + lstm.num_inputs() * lstm.sequence_length();

  lstm.SetInput(0, batch0_start, batch0_end);

  lstm.Invoke();

  std::vector<float> fw_expected;
  for (int s = 0; s < lstm.sequence_length(); s++) {
    float* fw_golden_start = lstm_fw_golden_output + s * lstm.num_fw_outputs();
    float* fw_golden_end = fw_golden_start + lstm.num_fw_outputs();
    fw_expected.insert(fw_expected.begin(), fw_golden_start, fw_golden_end);
  }
  EXPECT_THAT(lstm.GetBwOutput(),
              ElementsAreArray(ArrayFloatNear(fw_expected)));

  std::vector<float> bw_expected;
  for (int s = 0; s < lstm.sequence_length(); s++) {
    float* bw_golden_start = lstm_bw_golden_output + s * lstm.num_bw_outputs();
    float* bw_golden_end = bw_golden_start + lstm.num_bw_outputs();
    bw_expected.insert(bw_expected.begin(), bw_golden_start, bw_golden_end);
  }
  EXPECT_THAT(lstm.GetFwOutput(),
              ElementsAreArray(ArrayFloatNear(bw_expected)));
}

TEST(LSTMOpTest, BlackBoxTestWithPeepholeWithProjectionNoClipping) {
  const int n_batch = 2;
  const int n_input = 5;
  const int n_cell = 20;
  const int n_output = 16;
  const int sequence_length = 4;

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/false,
      /*use_peephole=*/true, /*use_projection_weights=*/true,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, /*quantize_weights=*/false, /*time_major=*/true,
      {
          {sequence_length, n_batch, n_input},  // input tensor

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

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor

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

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          // TODO(b/121134029): Update tests so tensor shapes after state tensor
          // are used. They are currently ignored by test_util.
          {sequence_length, n_batch, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToInputWeights(
      {0.021393683,  0.06124551,    0.046905167,  -0.014657677,  -0.03149463,
       0.09171803,   0.14647801,    0.10797193,   -0.0057968358, 0.0019193048,
       -0.2726754,   0.10154029,    -0.018539885, 0.080349885,   -0.10262385,
       -0.022599787, -0.09121155,   -0.008675967, -0.045206103,  -0.0821282,
       -0.008045952, 0.015478081,   0.055217247,  0.038719587,   0.044153627,
       -0.06453243,  0.05031825,    -0.046935108, -0.008164439,  0.014574226,
       -0.1671009,   -0.15519552,   -0.16819797,  -0.13971269,   -0.11953059,
       0.25005487,   -0.22790983,   0.009855087,  -0.028140958,  -0.11200698,
       0.11295408,   -0.0035217577, 0.054485075,  0.05184695,    0.064711206,
       0.10989193,   0.11674786,    0.03490607,   0.07727357,    0.11390585,
       -0.1863375,   -0.1034451,    -0.13945189,  -0.049401227,  -0.18767063,
       0.042483903,  0.14233552,    0.13832581,   0.18350165,    0.14545603,
       -0.028545704, 0.024939531,   0.050929718,  0.0076203286,  -0.0029723682,
       -0.042484224, -0.11827596,   -0.09171104,  -0.10808628,   -0.16327988,
       -0.2273378,   -0.0993647,    -0.017155107, 0.0023917493,  0.049272764,
       0.0038534778, 0.054764505,   0.089753784,  0.06947234,    0.08014476,
       -0.04544234,  -0.0497073,    -0.07135631,  -0.048929106,  -0.004042012,
       -0.009284026, 0.018042054,   0.0036860977, -0.07427302,   -0.11434604,
       -0.018995456, 0.031487543,   0.012834908,  0.019977754,   0.044256654,
       -0.39292613,  -0.18519334,   -0.11651281,  -0.06809892,   0.011373677});

  lstm.SetInputToForgetWeights(
      {-0.0018401089, -0.004852237,  0.03698424,   0.014181704,   0.028273236,
       -0.016726194,  -0.05249759,   -0.10204261,  0.00861066,    -0.040979505,
       -0.009899187,  0.01923892,    -0.028177269, -0.08535103,   -0.14585495,
       0.10662567,    -0.01909731,   -0.017883534, -0.0047269356, -0.045103323,
       0.0030784295,  0.076784775,   0.07463696,   0.094531395,   0.0814421,
       -0.12257899,   -0.033945758,  -0.031303465, 0.045630626,   0.06843887,
       -0.13492945,   -0.012480007,  -0.0811829,   -0.07224499,   -0.09628791,
       0.045100946,   0.0012300825,  0.013964662,  0.099372394,   0.02543059,
       0.06958324,    0.034257296,   0.0482646,    0.06267997,    0.052625068,
       0.12784666,    0.07077897,    0.025725935,  0.04165009,    0.07241905,
       0.018668644,   -0.037377294,  -0.06277783,  -0.08833636,   -0.040120605,
       -0.011405586,  -0.007808335,  -0.010301386, -0.005102167,  0.027717464,
       0.05483423,    0.11449111,    0.11289652,   0.10939839,    0.13396506,
       -0.08402166,   -0.01901462,   -0.044678304, -0.07720565,   0.014350063,
       -0.11757958,   -0.0652038,    -0.08185733,  -0.076754324,  -0.092614375,
       0.10405491,    0.052960336,   0.035755895,  0.035839386,   -0.012540553,
       0.036881298,   0.02913376,    0.03420159,   0.05448447,    -0.054523353,
       0.02582715,    0.02327355,    -0.011857179, -0.0011980024, -0.034641717,
       -0.026125094,  -0.17582615,   -0.15923657,  -0.27486774,   -0.0006143371,
       0.0001771948,  -8.470171e-05, 0.02651807,   0.045790765,   0.06956496});

  lstm.SetInputToCellWeights(
      {-0.04580283,   -0.09549462,   -0.032418985,  -0.06454633,
       -0.043528453,  0.043018587,   -0.049152344,  -0.12418144,
       -0.078985475,  -0.07596889,   0.019484362,   -0.11434962,
       -0.0074034138, -0.06314844,   -0.092981495,  0.0062155537,
       -0.025034338,  -0.0028890965, 0.048929527,   0.06235075,
       0.10665918,    -0.032036792,  -0.08505916,   -0.10843358,
       -0.13002433,   -0.036816437,  -0.02130134,   -0.016518239,
       0.0047691227,  -0.0025825808, 0.066017866,   0.029991534,
       -0.10652836,   -0.1037554,    -0.13056071,   -0.03266643,
       -0.033702414,  -0.006473424,  -0.04611692,   0.014419339,
       -0.025174323,  0.0396852,     0.081777506,   0.06157468,
       0.10210095,    -0.009658194,  0.046511717,   0.03603906,
       0.0069369148,  0.015960095,   -0.06507666,   0.09551598,
       0.053568836,   0.06408714,    0.12835667,    -0.008714329,
       -0.20211966,   -0.12093674,   0.029450472,   0.2849013,
       -0.029227901,  0.1164364,     -0.08560263,   0.09941786,
       -0.036999565,  -0.028842626,  -0.0033637602, -0.017012902,
       -0.09720865,   -0.11193351,   -0.029155117,  -0.017936034,
       -0.009768936,  -0.04223324,   -0.036159635,  0.06505112,
       -0.021742892,  -0.023377212,  -0.07221364,   -0.06430552,
       0.05453865,    0.091149814,   0.06387331,    0.007518393,
       0.055960953,   0.069779344,   0.046411168,   0.10509911,
       0.07463894,    0.0075130584,  0.012850982,   0.04555431,
       0.056955688,   0.06555285,    0.050801456,   -0.009862683,
       0.00826772,    -0.026555609,  -0.0073611983, -0.0014897042});

  lstm.SetInputToOutputWeights(
      {-0.0998932,   -0.07201956,  -0.052803773,  -0.15629593,  -0.15001918,
       -0.07650751,  0.02359855,   -0.075155355,  -0.08037709,  -0.15093534,
       0.029517552,  -0.04751393,  0.010350531,   -0.02664851,  -0.016839722,
       -0.023121163, 0.0077019283, 0.012851257,   -0.05040649,  -0.0129761,
       -0.021737747, -0.038305793, -0.06870586,   -0.01481247,  -0.001285394,
       0.10124236,   0.083122835,  0.053313006,   -0.062235646, -0.075637154,
       -0.027833903, 0.029774971,  0.1130802,     0.09218906,   0.09506135,
       -0.086665764, -0.037162706, -0.038880914,  -0.035832845, -0.014481564,
       -0.09825003,  -0.12048569,  -0.097665586,  -0.05287633,  -0.0964047,
       -0.11366429,  0.035777505,  0.13568819,    0.052451383,  0.050649304,
       0.05798951,   -0.021852335, -0.099848844,  0.014740475,  -0.078897946,
       0.04974699,   0.014160473,  0.06973932,    0.04964942,   0.033364646,
       0.08190124,   0.025535367,  0.050893165,   0.048514254,  0.06945813,
       -0.078907564, -0.06707616,  -0.11844508,   -0.09986688,  -0.07509403,
       0.06263226,   0.14925587,   0.20188436,    0.12098451,   0.14639415,
       0.0015017595, -0.014267382, -0.03417257,   0.012711468,  0.0028300495,
       -0.024758482, -0.05098548,  -0.0821182,    0.014225672,  0.021544158,
       0.08949725,   0.07505268,   -0.0020780868, 0.04908258,   0.06476295,
       -0.022907063, 0.027562456,  0.040185735,   0.019567577,  -0.015598739,
       -0.049097303, -0.017121866, -0.083368234,  -0.02332002,  -0.0840956});

  lstm.SetInputGateBias(
      {0.02234832,  0.14757581,   0.18176508,  0.10380666,  0.053110216,
       -0.06928846, -0.13942584,  -0.11816189, 0.19483899,  0.03652339,
       -0.10250295, 0.036714908,  -0.18426876, 0.036065217, 0.21810818,
       0.02383196,  -0.043370757, 0.08690144,  -0.04444982, 0.00030581196});

  lstm.SetForgetGateBias({0.035185695, -0.042891346, -0.03032477, 0.23027696,
                          0.11098921,  0.15378423,   0.09263801,  0.09790885,
                          0.09508917,  0.061199076,  0.07665568,  -0.015443159,
                          -0.03499149, 0.046190713,  0.08895977,  0.10899629,
                          0.40694186,  0.06030037,   0.012413437, -0.06108739});

  lstm.SetCellBias({-0.024379363, 0.0055531194, 0.23377132,   0.033463873,
                    -0.1483596,   -0.10639995,  -0.091433935, 0.058573797,
                    -0.06809782,  -0.07889636,  -0.043246906, -0.09829136,
                    -0.4279842,   0.034901652,  0.18797937,   0.0075234566,
                    0.016178843,  0.1749513,    0.13975595,   0.92058027});

  lstm.SetOutputGateBias(
      {0.046159424,  -0.0012809046, 0.03563469,   0.12648113, 0.027195795,
       0.35373217,   -0.018957434,  0.008907322,  -0.0762701, 0.12018895,
       0.04216877,   0.0022856654,  0.040952638,  0.3147856,  0.08225149,
       -0.057416286, -0.14995944,   -0.008040261, 0.13208859, 0.029760877});

  lstm.SetRecurrentToInputWeights(
      {-0.001374326,   -0.078856036,   0.10672688,    0.029162422,
       -0.11585556,    0.02557986,     -0.13446963,   -0.035785314,
       -0.01244275,    0.025961924,    -0.02337298,   -0.044228926,
       -0.055839065,   -0.046598054,   -0.010546039,  -0.06900766,
       0.027239809,    0.022582639,    -0.013296484,  -0.05459212,
       0.08981,        -0.045407712,   0.08682226,    -0.06867011,
       -0.14390695,    -0.02916037,    0.000996957,   0.091420636,
       0.14283475,     -0.07390571,    -0.06402044,   0.062524505,
       -0.093129106,   0.04860203,     -0.08364217,   -0.08119002,
       0.009352075,    0.22920375,     0.0016303885,  0.11583097,
       -0.13732095,    0.012405723,    -0.07551853,   0.06343048,
       0.12162708,     -0.031923793,   -0.014335606,  0.01790974,
       -0.10650317,    -0.0724401,     0.08554849,    -0.05727212,
       0.06556731,     -0.042729504,   -0.043227166,  0.011683251,
       -0.013082158,   -0.029302018,   -0.010899579,  -0.062036745,
       -0.022509435,   -0.00964907,    -0.01567329,   0.04260106,
       -0.07787477,    -0.11576462,    0.017356863,   0.048673786,
       -0.017577527,   -0.05527947,    -0.082487635,  -0.040137455,
       -0.10820036,    -0.04666372,    0.022746278,   -0.07851417,
       0.01068115,     0.032956902,    0.022433773,   0.0026891115,
       0.08944216,     -0.0685835,     0.010513544,   0.07228705,
       0.02032331,     -0.059686817,   -0.0005566496, -0.086984694,
       0.040414046,    -0.1380399,     0.094208956,   -0.05722982,
       0.012092817,    -0.04989123,    -0.086576,     -0.003399834,
       -0.04696032,    -0.045747425,   0.10091314,    0.048676282,
       -0.029037097,   0.031399418,    -0.0040285117, 0.047237843,
       0.09504992,     0.041799378,    -0.049185462,  -0.031518843,
       -0.10516937,    0.026374253,    0.10058866,    -0.0033195973,
       -0.041975245,   0.0073591834,   0.0033782164,  -0.004325073,
       -0.10167381,    0.042500053,    -0.01447153,   0.06464186,
       -0.017142897,   0.03312627,     0.009205989,   0.024138335,
       -0.011337001,   0.035530265,    -0.010912711,  0.0706555,
       -0.005894094,   0.051841937,    -0.1401738,    -0.02351249,
       0.0365468,      0.07590991,     0.08838724,    0.021681072,
       -0.10086113,    0.019608743,    -0.06195883,   0.077335775,
       0.023646897,    -0.095322326,   0.02233014,    0.09756986,
       -0.048691444,   -0.009579111,   0.07595467,    0.11480546,
       -0.09801813,    0.019894179,    0.08502348,    0.004032281,
       0.037211012,    0.068537936,    -0.048005626,  -0.091520436,
       -0.028379958,   -0.01556313,    0.06554592,    -0.045599163,
       -0.01672207,    -0.020169014,   -0.011877351,  -0.20212261,
       0.010889619,    0.0047078193,   0.038385306,   0.08540671,
       -0.017140968,   -0.0035865551,  0.016678626,   0.005633034,
       0.015963363,    0.00871737,     0.060130805,   0.028611384,
       0.10109069,     -0.015060172,   -0.07894427,   0.06401885,
       0.011584063,    -0.024466386,   0.0047652307,  -0.09041358,
       0.030737216,    -0.0046374933,  0.14215417,    -0.11823516,
       0.019899689,    0.006106124,    -0.027092824,  0.0786356,
       0.05052217,     -0.058925,      -0.011402121,  -0.024987547,
       -0.0013661642,  -0.06832946,    -0.015667673,  -0.1083353,
       -0.00096863037, -0.06988685,    -0.053350925,  -0.027275559,
       -0.033664223,   -0.07978348,    -0.025200296,  -0.017207067,
       -0.058403496,   -0.055697463,   0.005798788,   0.12965427,
       -0.062582195,   0.0013350133,   -0.10482091,   0.0379771,
       0.072521195,    -0.0029455067,  -0.13797039,   -0.03628521,
       0.013806405,    -0.017858358,   -0.01008298,   -0.07700066,
       -0.017081132,   0.019358726,    0.0027079724,  0.004635139,
       0.062634714,    -0.02338735,    -0.039547626,  -0.02050681,
       0.03385117,     -0.083611414,   0.002862572,   -0.09421313,
       0.058618143,    -0.08598433,    0.00972939,    0.023867095,
       -0.053934585,   -0.023203006,   0.07452513,    -0.048767887,
       -0.07314807,    -0.056307215,   -0.10433547,   -0.06440842,
       0.04328182,     0.04389765,     -0.020006588,  -0.09076438,
       -0.11652589,    -0.021705797,   0.03345259,    -0.010329105,
       -0.025767034,   0.013057034,    -0.07316461,   -0.10145612,
       0.06358255,     0.18531723,     0.07759293,    0.12006465,
       0.1305557,      0.058638252,    -0.03393652,   0.09622831,
       -0.16253184,    -2.4580743e-06, 0.079869635,   -0.070196845,
       -0.005644518,   0.06857898,     -0.12598175,   -0.035084512,
       0.03156317,     -0.12794146,    -0.031963028,  0.04692781,
       0.030070418,    0.0071660685,   -0.095516115,  -0.004643372,
       0.040170413,    -0.062104587,   -0.0037324072, 0.0554317,
       0.08184801,     -0.019164372,   0.06791302,    0.034257166,
       -0.10307039,    0.021943003,    0.046745934,   0.0790918,
       -0.0265588,     -0.007824208,   0.042546265,   -0.00977924,
       -0.0002440307,  -0.017384544,   -0.017990116,  0.12252321,
       -0.014512694,   -0.08251313,    0.08861942,    0.13589665,
       0.026351685,    0.012641483,    0.07466548,    0.044301085,
       -0.045414884,   -0.051112458,   0.03444247,    -0.08502782,
       -0.04106223,    -0.028126027,   0.028473156,   0.10467447});

  lstm.SetRecurrentToForgetWeights(
      {-0.057784554,  -0.026057621,  -0.068447545,   -0.022581743,
       0.14811787,    0.10826372,    0.09471067,     0.03987225,
       -0.0039523416, 0.00030638507, 0.053185795,    0.10572994,
       0.08414449,    -0.022036452,  -0.00066928595, -0.09203576,
       0.032950465,   -0.10985798,   -0.023809856,   0.0021431844,
       -0.02196096,   -0.00326074,   0.00058621005,  -0.074678116,
       -0.06193199,   0.055729095,   0.03736828,     0.020123724,
       0.061878487,   -0.04729229,   0.034919553,    -0.07585433,
       -0.04421272,   -0.044019096,  0.085488975,    0.04058006,
       -0.06890133,   -0.030951202,  -0.024628663,   -0.07672815,
       0.034293607,   0.08556707,    -0.05293577,    -0.033561368,
       -0.04899627,   0.0241671,     0.015736353,    -0.095442444,
       -0.029564252,  0.016493602,   -0.035026584,   0.022337519,
       -0.026871363,  0.004780428,   0.0077918363,   -0.03601621,
       0.016435321,   -0.03263031,   -0.09543275,    -0.047392778,
       0.013454138,   0.028934088,   0.01685226,     -0.086110644,
       -0.046250615,  -0.01847454,   0.047608484,    0.07339695,
       0.034546845,   -0.04881143,   0.009128804,    -0.08802852,
       0.03761666,    0.008096139,   -0.014454086,   0.014361001,
       -0.023502491,  -0.0011840804, -0.07607001,    0.001856849,
       -0.06509276,   -0.006021153,  -0.08570962,    -0.1451793,
       0.060212336,   0.055259194,   0.06974018,     0.049454916,
       -0.027794661,  -0.08077226,   -0.016179763,   0.1169753,
       0.17213494,    -0.0056326236, -0.053934924,   -0.0124349,
       -0.11520337,   0.05409887,    0.088759385,    0.0019655675,
       0.0042065294,  0.03881498,    0.019844765,    0.041858196,
       -0.05695512,   0.047233116,   0.038937137,    -0.06542224,
       0.014429736,   -0.09719407,   0.13908425,     -0.05379757,
       0.012321099,   0.082840554,   -0.029899208,   0.044217527,
       0.059855383,   0.07711018,    -0.045319796,   0.0948846,
       -0.011724666,  -0.0033288454, -0.033542685,   -0.04764985,
       -0.13873616,   0.040668588,   0.034832682,    -0.015319203,
       -0.018715994,  0.046002675,   0.0599172,      -0.043107376,
       0.0294216,     -0.002314414,  -0.022424703,   0.0030315618,
       0.0014641669,  0.0029166266,  -0.11878115,    0.013738511,
       0.12375372,    -0.0006038222, 0.029104086,    0.087442465,
       0.052958444,   0.07558703,    0.04817258,     0.044462286,
       -0.015213451,  -0.08783778,   -0.0561384,     -0.003008196,
       0.047060397,   -0.002058388,  0.03429439,     -0.018839769,
       0.024734668,   0.024614193,   -0.042046934,   0.09597743,
       -0.0043254104, 0.04320769,    0.0064070094,   -0.0019131786,
       -0.02558259,   -0.022822596,  -0.023273505,   -0.02464396,
       -0.10991725,   -0.006240552,  0.0074488563,   0.024044557,
       0.04383914,    -0.046476185,  0.028658995,    0.060410924,
       0.050786525,   0.009452605,   -0.0073054377,  -0.024810238,
       0.0052906186,  0.0066939713,  -0.0020913032,  0.014515517,
       0.015898481,   0.021362653,   -0.030262267,   0.016587038,
       -0.011442813,  0.041154444,   -0.007631438,   -0.03423484,
       -0.010977775,  0.036152758,   0.0066366293,   0.11915515,
       0.02318443,    -0.041350313,  0.021485701,    -0.10906167,
       -0.028218046,  -0.00954771,   0.020531068,    -0.11995105,
       -0.03672871,   0.024019798,   0.014255957,    -0.05221243,
       -0.00661567,   -0.04630967,   0.033188973,    0.10107534,
       -0.014027541,  0.030796422,   -0.10270911,    -0.035999842,
       0.15443139,    0.07684145,    0.036571592,    -0.035900835,
       -0.0034699554, 0.06209149,    0.015920248,    -0.031122351,
       -0.03858649,   0.01849943,    0.13872518,     0.01503974,
       0.069941424,   -0.06948533,   -0.0088794185,  0.061282158,
       -0.047401894,  0.03100163,    -0.041533746,   -0.10430945,
       0.044574402,   -0.01425562,   -0.024290353,   0.034563623,
       0.05866852,    0.023947537,   -0.09445152,    0.035450947,
       0.02247216,    -0.0042998926, 0.061146557,    -0.10250651,
       0.020881841,   -0.06747029,   0.10062043,     -0.0023941975,
       0.03532124,    -0.016341697,  0.09685456,     -0.016764693,
       0.051808182,   0.05875331,    -0.04536488,    0.001626336,
       -0.028892258,  -0.01048663,   -0.009793449,   -0.017093895,
       0.010987891,   0.02357273,    -0.00010856845, 0.0099760275,
       -0.001845119,  -0.03551521,   0.0018358806,   0.05763657,
       -0.01769146,   0.040995963,   0.02235177,     -0.060430344,
       0.11475477,    -0.023854522,  0.10071741,     0.0686208,
       -0.014250481,  0.034261297,   0.047418304,    0.08562733,
       -0.030519066,  0.0060542435,  0.014653856,    -0.038836084,
       0.04096551,    0.032249358,   -0.08355519,    -0.026823482,
       0.056386515,   -0.010401743,  -0.028396193,   0.08507674,
       0.014410365,   0.020995233,   0.17040324,     0.11511526,
       0.02459721,    0.0066619175,  0.025853224,    -0.023133837,
       -0.081302024,  0.017264642,   -0.009585969,   0.09491168,
       -0.051313367,  0.054532815,   -0.014298593,   0.10657464,
       0.007076659,   0.10964551,    0.0409152,      0.008275321,
       -0.07283536,   0.07937492,    0.04192024,     -0.1075027});

  lstm.SetRecurrentToCellWeights(
      {-0.037322544,   0.018592842,   0.0056175636,  -0.06253426,
       0.055647098,    -0.05713207,   -0.05626563,   0.005559383,
       0.03375411,     -0.025757805,  -0.088049285,  0.06017052,
       -0.06570978,    0.007384076,   0.035123326,   -0.07920549,
       0.053676967,    0.044480428,   -0.07663568,   0.0071805613,
       0.08089997,     0.05143358,    0.038261272,   0.03339287,
       -0.027673481,   0.044746667,   0.028349208,   0.020090483,
       -0.019443132,   -0.030755889,  -0.0040000007, 0.04465846,
       -0.021585021,   0.0031670958,  0.0053199246,  -0.056117613,
       -0.10893326,    0.076739706,   -0.08509834,   -0.027997585,
       0.037871376,    0.01449768,    -0.09002357,   -0.06111149,
       -0.046195522,   0.0422062,     -0.005683705,  -0.1253618,
       -0.012925729,   -0.04890792,   0.06985068,    0.037654128,
       0.03398274,     -0.004781977,  0.007032333,   -0.031787455,
       0.010868644,    -0.031489216,  0.09525667,    0.013939797,
       0.0058680447,   0.0167067,     0.02668468,    -0.04797466,
       -0.048885044,   -0.12722108,   0.035304096,   0.06554885,
       0.00972396,     -0.039238118,  -0.05159735,   -0.11329045,
       0.1613692,      -0.03750952,   0.06529313,    -0.071974665,
       -0.11769596,    0.015524369,   -0.0013754242, -0.12446318,
       0.02786344,     -0.014179351,  0.005264273,   0.14376344,
       0.015983658,    0.03406988,    -0.06939408,   0.040699873,
       0.02111075,     0.09669095,    0.041345075,   -0.08316494,
       -0.07684199,    -0.045768797,  0.032298047,   -0.041805092,
       0.0119405,      0.0061010392,  0.12652606,    0.0064572375,
       -0.024950314,   0.11574242,    0.04508852,    -0.04335324,
       0.06760663,     -0.027437469,  0.07216407,    0.06977076,
       -0.05438599,    0.034033038,   -0.028602652,  0.05346137,
       0.043184172,    -0.037189785,  0.10420091,    0.00882477,
       -0.054019816,   -0.074273005,  -0.030617684,  -0.0028467078,
       0.024302477,    -0.0038869337, 0.005332455,   0.0013399826,
       0.04361412,     -0.007001822,  0.09631092,    -0.06702025,
       -0.042049985,   -0.035070654,  -0.04103342,   -0.10273396,
       0.0544271,      0.037184782,   -0.13150354,   -0.0058036847,
       -0.008264958,   0.042035464,   0.05891794,    0.029673764,
       0.0063542654,   0.044788733,   0.054816857,   0.062257513,
       -0.00093483756, 0.048938446,   -0.004952862,  -0.007730018,
       -0.04043371,    -0.017094059,  0.07229206,    -0.023670016,
       -0.052195564,   -0.025616996,  -0.01520939,   0.045104615,
       -0.007376126,   0.003533447,   0.006570588,   0.056037236,
       0.12436656,     0.051817212,   0.028532185,   -0.08686856,
       0.11868599,     0.07663395,    -0.07323171,   0.03463402,
       -0.050708205,   -0.04458982,   -0.11590894,   0.021273347,
       0.1251325,      -0.15313013,   -0.12224372,   0.17228661,
       0.023029093,    0.086124025,   0.006445803,   -0.03496501,
       0.028332196,    0.04449512,    -0.042436164,  -0.026587414,
       -0.006041347,   -0.09292539,   -0.05678812,   0.03897832,
       0.09465633,     0.008115513,   -0.02171956,   0.08304309,
       0.071401566,    0.019622514,   0.032163795,   -0.004167056,
       0.02295182,     0.030739572,   0.056506045,   0.004612461,
       0.06524936,     0.059999723,   0.046395954,   -0.0045512207,
       -0.1335546,     -0.030136576,  0.11584653,    -0.014678886,
       0.0020118146,   -0.09688814,   -0.0790206,    0.039770417,
       -0.0329582,     0.07922767,    0.029322514,   0.026405897,
       0.04207835,     -0.07073373,   0.063781224,   0.0859677,
       -0.10925287,    -0.07011058,   0.048005477,   0.03438226,
       -0.09606514,    -0.006669445,  -0.043381985,  0.04240257,
       -0.06955775,    -0.06769346,   0.043903265,   -0.026784198,
       -0.017840602,   0.024307009,   -0.040079936,  -0.019946516,
       0.045318738,    -0.12233574,   0.026170589,   0.0074471775,
       0.15978073,     0.10185836,    0.10298046,    -0.015476589,
       -0.039390966,   -0.072174534,  0.0739445,     -0.1211869,
       -0.0347889,     -0.07943156,   0.014809798,   -0.12412325,
       -0.0030663363,  0.039695457,   0.0647603,     -0.08291318,
       -0.018529687,   -0.004423833,  0.0037507233,  0.084633216,
       -0.01514876,    -0.056505352,  -0.012800942,  -0.06994386,
       0.012962922,    -0.031234352,  0.07029052,    0.016418684,
       0.03618972,     0.055686004,   -0.08663945,   -0.017404709,
       -0.054761406,   0.029065743,   0.052404847,   0.020238016,
       0.0048197987,   -0.0214882,    0.07078733,    0.013016777,
       0.06262858,     0.009184685,   0.020785125,   -0.043904778,
       -0.0270329,     -0.03299152,   -0.060088247,  -0.015162964,
       -0.001828936,   0.12642565,    -0.056757294,  0.013586685,
       0.09232601,     -0.035886683,  0.06000002,    0.05229691,
       -0.052580316,   -0.082029596,  -0.010794592,  0.012947712,
       -0.036429964,   -0.085508935,  -0.13127148,   -0.017744139,
       0.031502828,    0.036232427,   -0.031581745,  0.023051167,
       -0.05325106,    -0.03421577,   0.028793324,   -0.034633752,
       -0.009881397,   -0.043551125,  -0.018609839,  0.0019097115,
       -0.008799762,   0.056595087,   0.0022273948,  0.055752404});

  lstm.SetRecurrentToOutputWeights({
      0.025825322,   -0.05813119,  0.09495884,   -0.045984812,   -0.01255415,
      -0.0026479573, -0.08196161,  -0.054914974, -0.0046604523,  -0.029587349,
      -0.044576716,  -0.07480124,  -0.082868785, 0.023254942,    0.027502948,
      -0.0039728214, -0.08683098,  -0.08116779,  -0.014675607,   -0.037924774,
      -0.023314456,  -0.007401714, -0.09255757,  0.029460307,    -0.08829125,
      -0.005139627,  -0.08989442,  -0.0555066,   0.13596267,     -0.025062224,
      -0.048351806,  -0.03850004,  0.07266485,   -0.022414139,   0.05940088,
      0.075114764,   0.09597592,   -0.010211725, -0.0049794707,  -0.011523867,
      -0.025980417,  0.072999895,  0.11091378,   -0.081685916,   0.014416728,
      0.043229222,   0.034178585,  -0.07530371,  0.035837382,    -0.085607,
      -0.007721233,  -0.03287832,  -0.043848954, -0.06404588,    -0.06632928,
      -0.073643476,  0.008214239,  -0.045984086, 0.039764922,    0.03474462,
      0.060612556,   -0.080590084, 0.049127717,  0.04151091,     -0.030063879,
      0.008801774,   -0.023021035, -0.019558564, 0.05158114,     -0.010947698,
      -0.011825728,  0.0075720972, 0.0699727,    -0.0039981045,  0.069350146,
      0.08799282,    0.016156472,  0.035502106,  0.11695009,     0.006217345,
      0.13392477,    -0.037875112, 0.025745004,  0.08940699,     -0.00924166,
      0.0046702605,  -0.036598757, -0.08811812,  0.10522024,     -0.032441203,
      0.008176899,   -0.04454919,  0.07058152,   0.0067963637,   0.039206743,
      0.03259838,    0.03725492,   -0.09515802,  0.013326398,    -0.052055415,
      -0.025676316,  0.03198509,   -0.015951829, -0.058556724,   0.036879618,
      0.043357447,   0.028362012,  -0.05908629,  0.0059240665,   -0.04995891,
      -0.019187413,  0.0276265,    -0.01628143,  0.0025863599,   0.08800015,
      0.035250366,   -0.022165963, -0.07328642,  -0.009415526,   -0.07455109,
      0.11690406,    0.0363299,    0.07411125,   0.042103454,    -0.009660886,
      0.019076364,   0.018299393,  -0.046004917, 0.08891175,     0.0431396,
      -0.026327137,  -0.051502608, 0.08979574,   -0.051670972,   0.04940282,
      -0.07491107,   -0.021240504, 0.022596184,  -0.034280192,   0.060163025,
      -0.058211457,  -0.051837247, -0.01349775,  -0.04639988,    -0.035936575,
      -0.011681591,  0.064818054,  0.0073146066, -0.021745546,   -0.043124277,
      -0.06471268,   -0.07053354,  -0.029321948, -0.05330136,    0.016933719,
      -0.053782392,  0.13747959,   -0.1361751,   -0.11569455,    0.0033329215,
      0.05693899,    -0.053219706, 0.063698,     0.07977434,     -0.07924483,
      0.06936997,    0.0034815092, -0.007305279, -0.037325785,   -0.07251102,
      -0.033633437,  -0.08677009,  0.091591336,  -0.14165086,    0.021752775,
      0.019683983,   0.0011612234, -0.058154266, 0.049996935,    0.0288841,
      -0.0024567875, -0.14345716,  0.010955264,  -0.10234828,    0.1183656,
      -0.0010731248, -0.023590032, -0.072285876, -0.0724771,     -0.026382286,
      -0.0014920527, 0.042667855,  0.0018776858, 0.02986552,     0.009814309,
      0.0733756,     0.12289186,   0.018043943,  -0.0458958,     0.049412545,
      0.033632483,   0.05495232,   0.036686596,  -0.013781798,   -0.010036754,
      0.02576849,    -0.08307328,  0.010112348,  0.042521734,    -0.05869831,
      -0.071689695,  0.03876447,   -0.13275425,  -0.0352966,     -0.023077697,
      0.10285965,    0.084736146,  0.15568255,   -0.00040734606, 0.027835453,
      -0.10292561,   -0.032401145, 0.10053256,   -0.026142767,   -0.08271222,
      -0.0030240538, -0.016368777, 0.1070414,    0.042672627,    0.013456989,
      -0.0437609,    -0.022309763, 0.11576483,   0.04108048,     0.061026827,
      -0.0190714,    -0.0869359,   0.037901703,  0.0610107,      0.07202949,
      0.01675338,    0.086139716,  -0.08795751,  -0.014898893,   -0.023771819,
      -0.01965048,   0.007955471,  -0.043740474, 0.03346837,     -0.10549954,
      0.090567775,   0.042013682,  -0.03176985,  0.12569028,     -0.02421228,
      -0.029526481,  0.023851605,  0.031539805,  0.05292009,     -0.02344001,
      -0.07811758,   -0.08834428,  0.10094801,   0.16594367,     -0.06861939,
      -0.021256343,  -0.041093912, -0.06669611,  0.035498552,    0.021757556,
      -0.09302526,   -0.015403468, -0.06614931,  -0.051798206,   -0.013874718,
      0.03630673,    0.010412845,  -0.08077351,  0.046185967,    0.0035662893,
      0.03541868,    -0.094149634, -0.034814864, 0.003128424,    -0.020674974,
      -0.03944324,   -0.008110165, -0.11113267,  0.08484226,     0.043586485,
      0.040582247,   0.0968012,    -0.065249965, -0.028036479,   0.0050708856,
      0.0017462453,  0.0326779,    0.041296225,  0.09164146,     -0.047743853,
      -0.015952192,  -0.034451712, 0.084197424,  -0.05347844,    -0.11768019,
      0.085926116,   -0.08251791,  -0.045081906, 0.0948852,      0.068401024,
      0.024856757,   0.06978981,   -0.057309967, -0.012775832,   -0.0032452994,
      0.01977615,    -0.041040014, -0.024264973, 0.063464895,    0.05431621,
  });

  lstm.SetCellToInputWeights(
      {0.040369894, 0.030746894,  0.24704495,  0.018586371,  -0.037586458,
       -0.15312155, -0.11812848,  -0.11465643, 0.20259799,   0.11418174,
       -0.10116027, -0.011334949, 0.12411352,  -0.076769054, -0.052169047,
       0.21198851,  -0.38871562,  -0.09061183, -0.09683246,  -0.21929175});

  lstm.SetCellToForgetWeights(
      {-0.01998659,  -0.15568835,  -0.24248174,   -0.012770197, 0.041331276,
       -0.072311886, -0.052123554, -0.0066330447, -0.043891653, 0.036225766,
       -0.047248036, 0.021479502,  0.033189066,   0.11952997,   -0.020432774,
       0.64658105,   -0.06650122,  -0.03467612,   0.095340036,  0.23647355});

  lstm.SetCellToOutputWeights(
      {0.08286371,  -0.08261836, -0.51210177, 0.002913762, 0.17764764,
       -0.5495371,  -0.08460716, -0.24552552, 0.030037103, 0.04123544,
       -0.11940523, 0.007358328, 0.1890978,   0.4833202,   -0.34441817,
       0.36312827,  -0.26375428, 0.1457655,   -0.19724406, 0.15548733});

  lstm.SetProjectionWeights(
      {-0.009802181,  0.09401916,    0.0717386,     -0.13895074,  0.09641832,
       0.060420845,   0.08539281,    0.054285463,   0.061395317,  0.034448683,
       -0.042991187,  0.019801661,   -0.16840284,   -0.015726732, -0.23041931,
       -0.024478018,  -0.10959692,   -0.013875541,  0.18600968,   -0.061274476,
       0.0138165,     -0.08160894,   -0.07661644,   0.032372914,  0.16169067,
       0.22465782,    -0.03993472,   -0.004017731,  0.08633481,   -0.28869787,
       0.08682067,    0.17240396,    0.014975425,   0.056431185,  0.031037588,
       0.16702051,    0.0077946745,  0.15140012,    0.29405436,   0.120285,
       -0.188994,     -0.027265169,  0.043389652,   -0.022061434, 0.014777949,
       -0.20203483,   0.094781205,   0.19100232,    0.13987629,   -0.036132768,
       -0.06426278,   -0.05108664,   0.13221376,    0.009441198,  -0.16715929,
       0.15859416,    -0.040437475,  0.050779544,   -0.022187516, 0.012166504,
       0.027685808,   -0.07675938,   -0.0055694645, -0.09444123,  0.0046453946,
       0.050794356,   0.10770313,    -0.20790008,   -0.07149004,  -0.11425117,
       0.008225835,   -0.035802525,  0.14374903,    0.15262283,   0.048710253,
       0.1847461,     -0.007487823,  0.11000021,    -0.09542012,  0.22619456,
       -0.029149994,  0.08527916,    0.009043713,   0.0042746216, 0.016261552,
       0.022461696,   0.12689082,    -0.043589946,  -0.12035478,  -0.08361797,
       -0.050666027,  -0.1248618,    -0.1275799,    -0.071875185, 0.07377272,
       0.09944291,    -0.18897448,   -0.1593054,    -0.06526116,  -0.040107165,
       -0.004618631,  -0.067624845,  -0.007576253,  0.10727444,   0.041546922,
       -0.20424393,   0.06907816,    0.050412357,   0.00724631,   0.039827548,
       0.12449835,    0.10747581,    0.13708383,    0.09134148,   -0.12617786,
       -0.06428341,   0.09956831,    0.1208086,     -0.14676677,  -0.0727722,
       0.1126304,     0.010139365,   0.015571211,   -0.038128063, 0.022913318,
       -0.042050496,  0.16842307,    -0.060597885,  0.10531834,   -0.06411776,
       -0.07451711,   -0.03410368,   -0.13393489,   0.06534304,   0.003620307,
       0.04490757,    0.05970546,    0.05197996,    0.02839995,   0.10434969,
       -0.013699693,  -0.028353551,  -0.07260381,   0.047201227,  -0.024575593,
       -0.036445823,  0.07155557,    0.009672501,   -0.02328883,  0.009533515,
       -0.03606021,   -0.07421458,   -0.028082801,  -0.2678904,   -0.13221288,
       0.18419984,    -0.13012612,   -0.014588381,  -0.035059117, -0.04824723,
       0.07830115,    -0.056184657,  0.03277091,    0.025466874,  0.14494097,
       -0.12522776,   -0.098633975,  -0.10766018,   -0.08317623,  0.08594209,
       0.07749552,    0.039474737,   0.1776665,     -0.07409566,  -0.0477268,
       0.29323658,    0.10801441,    0.1154011,     0.013952499,  0.10739139,
       0.10708251,    -0.051456142,  0.0074137426,  -0.10430189,  0.10034707,
       0.045594677,   0.0635285,     -0.0715442,    -0.089667566, -0.10811871,
       0.00026344223, 0.08298446,    -0.009525053,  0.006585689,  -0.24567553,
       -0.09450807,   0.09648481,    0.026996298,   -0.06419476,  -0.04752702,
       -0.11063944,   -0.23441927,   -0.17608605,   -0.052156363, 0.067035615,
       0.19271925,    -0.0032889997, -0.043264326,  0.09663576,   -0.057112187,
       -0.10100678,   0.0628376,     0.04447668,    0.017961001,  -0.10094388,
       -0.10190601,   0.18335468,    0.10494553,    -0.052095775, -0.0026118709,
       0.10539724,    -0.04383912,   -0.042349473,  0.08438151,   -0.1947263,
       0.02251204,    0.11216432,    -0.10307853,   0.17351969,   -0.039091777,
       0.08066188,    -0.00561982,   0.12633002,    0.11335965,   -0.0088127935,
       -0.019777594,  0.06864014,    -0.059751723,  0.016233567,  -0.06894641,
       -0.28651384,   -0.004228674,  0.019708522,   -0.16305895,  -0.07468996,
       -0.0855457,    0.099339016,   -0.07580735,   -0.13775392,  0.08434318,
       0.08330512,    -0.12131499,   0.031935584,   0.09180414,   -0.08876437,
       -0.08049874,   0.008753825,   0.03498998,    0.030215185,  0.03907079,
       0.089751154,   0.029194152,   -0.03337423,   -0.019092513, 0.04331237,
       0.04299654,    -0.036394123,  -0.12915532,   0.09793732,   0.07512415,
       -0.11319543,   -0.032502122,  0.15661901,    0.07671967,   -0.005491124,
       -0.19379048,   -0.218606,     0.21448623,    0.017840758,  0.1416943,
       -0.07051762,   0.19488361,    0.02664691,    -0.18104725,  -0.09334311,
       0.15026465,    -0.15493552,   -0.057762887,  -0.11604192,  -0.262013,
       -0.01391798,   0.012185008,   0.11156489,    -0.07483202,  0.06693364,
       -0.26151478,   0.046425626,   0.036540434,   -0.16435726,  0.17338543,
       -0.21401681,   -0.11385144,   -0.08283257,   -0.069031075, 0.030635102,
       0.010969227,   0.11109743,    0.010919218,   0.027526086,  0.13519906,
       0.01891392,    -0.046839405,  -0.040167913,  0.017953383,  -0.09700955,
       0.0061885654,  -0.07000971,   0.026893595,   -0.038844477, 0.14543656});

  static float lstm_input[][20] = {
      {// Batch0: 4 (input_sequence_size) * 5 (n_input)
       0.787926, 0.151646, 0.071352, 0.118426, 0.458058, 0.596268, 0.998386,
       0.568695, 0.864524, 0.571277, 0.073204, 0.296072, 0.743333, 0.069199,
       0.045348, 0.867394, 0.291279, 0.013714, 0.482521, 0.626339},

      {// Batch1: 4 (input_sequence_size) * 5 (n_input)
       0.295743, 0.544053, 0.690064, 0.858138, 0.497181, 0.642421, 0.524260,
       0.134799, 0.003639, 0.162482, 0.640394, 0.930399, 0.050782, 0.432485,
       0.988078, 0.082922, 0.563329, 0.865614, 0.333232, 0.259916}};

  static float lstm_fw_golden_output[][64] = {
      {// Batch0: 4 (input_sequence_size) * 16 (n_output)
       -0.00396806, 0.029352,     -0.00279226, 0.0159977,   -0.00835576,
       -0.0211779,  0.0283512,    -0.0114597,  0.00907307,  -0.0244004,
       -0.0152191,  -0.0259063,   0.00914318,  0.00415118,  0.017147,
       0.0134203,   -0.0166936,   0.0381209,   0.000889694, 0.0143363,
       -0.0328911,  -0.0234288,   0.0333051,   -0.012229,   0.0110322,
       -0.0457725,  -0.000832209, -0.0202817,  0.0327257,   0.0121308,
       0.0155969,   0.0312091,    -0.0213783,  0.0350169,   0.000324794,
       0.0276012,   -0.0263374,   -0.0371449,  0.0446149,   -0.0205474,
       0.0103729,   -0.0576349,   -0.0150052,  -0.0292043,  0.0376827,
       0.0136115,   0.0243435,    0.0354492,   -0.0189322,  0.0464512,
       -0.00251373, 0.0225745,    -0.0308346,  -0.0317124,  0.0460407,
       -0.0189395,  0.0149363,    -0.0530162,  -0.0150767,  -0.0340193,
       0.0286833,   0.00824207,   0.0264887,   0.0305169},
      {// Batch1: 4 (input_sequence_size) * 16 (n_output)
       -0.013869,    0.0287268,   -0.00334693, 0.00733398,  -0.0287926,
       -0.0186926,   0.0193662,   -0.0115437,  0.00422612,  -0.0345232,
       0.00223253,   -0.00957321, 0.0210624,   0.013331,    0.0150954,
       0.02168,      -0.0141913,  0.0322082,   0.00227024,  0.0260507,
       -0.0188721,   -0.0296489,  0.0399134,   -0.0160509,  0.0116039,
       -0.0447318,   -0.0150515,  -0.0277406,  0.0316596,   0.0118233,
       0.0214762,    0.0293641,   -0.0204549,  0.0450315,   -0.00117378,
       0.0167673,    -0.0375007,  -0.0238314,  0.038784,    -0.0174034,
       0.0131743,    -0.0506589,  -0.0048447,  -0.0240239,  0.0325789,
       0.00790065,   0.0220157,   0.0333314,   -0.0264787,  0.0387855,
       -0.000764675, 0.0217599,   -0.037537,   -0.0335206,  0.0431679,
       -0.0211424,   0.010203,    -0.062785,   -0.00832363, -0.025181,
       0.0412031,    0.0118723,   0.0239643,   0.0394009}};

  static float lstm_combined_golden_output[][64] = {
      {-0.022014, 0.073544,  -0.002235, 0.040068,  -0.037136, -0.052788,
       0.075325,  -0.029378, 0.024298,  -0.07733,  -0.030674, -0.060229,
       0.040599,  0.011608,  0.042005,  0.045977,  -0.039225, 0.076294,
       0.000735,  0.032852,  -0.069869, -0.053312, 0.073527,  -0.028136,
       0.021585,  -0.102679, -0.004327, -0.043304, 0.072861,  0.027077,
       0.034558,  0.068292,  -0.036292, 0.069832,  -0.003032, 0.053829,
       -0.043821, -0.072713, 0.085029,  -0.040374, 0.020014,  -0.104521,
       -0.034504, -0.059759, 0.062569,  0.025652,  0.049306,  0.061189,
       -0.025146, 0.079643,  -0.005188, 0.033080,  -0.048079, -0.048082,
       0.069369,  -0.028900, 0.024572,  -0.077547, -0.022517, -0.054477,
       0.038857,  0.013336,  0.043234,  0.044788},
      {-0.039186, 0.070792,  -0.005913, 0.02642,   -0.068274, -0.05022,
       0.061444,  -0.031241, 0.014996,  -0.094544, -0.004146, -0.03464,
       0.058981,  0.026097,  0.039781,  0.058408,  -0.031887, 0.069252,
       0.00576,   0.054062,  -0.042801, -0.059974, 0.085272,  -0.034453,
       0.026097,  -0.0959,   -0.031164, -0.058699, 0.06839,   0.020512,
       0.044727,  0.063609,  -0.039863, 0.084819,  -0.003909, 0.028666,
       -0.075677, -0.045125, 0.070379,  -0.033895, 0.022111,  -0.097184,
       -0.004921, -0.040851, 0.062316,  0.017435,  0.041437,  0.064568,
       -0.039656, 0.060726,  -0.003402, 0.036854,  -0.056503, -0.058554,
       0.068588,  -0.034879, 0.01352,   -0.09962,  -0.01434,  -0.039505,
       0.065133,  0.024321,  0.038473,  0.062438}};

  for (int i = 0; i < lstm.sequence_length(); i++) {
    float* batch0_start = lstm_input[0] + i * lstm.num_inputs();
    float* batch0_end = batch0_start + lstm.num_inputs();

    lstm.SetInput(2 * i * lstm.num_inputs(), batch0_start, batch0_end);

    float* batch1_start = lstm_input[1] + i * lstm.num_inputs();
    float* batch1_end = batch1_start + lstm.num_inputs();
    lstm.SetInput((2 * i + 1) * lstm.num_inputs(), batch1_start, batch1_end);
  }

  lstm.Invoke();

  std::vector<float> expected;
  for (int i = 0; i < lstm.sequence_length(); i++) {
    float* golden_start_batch0 =
        lstm_fw_golden_output[0] + i * lstm.num_fw_outputs();
    float* golden_end_batch0 = golden_start_batch0 + lstm.num_fw_outputs();
    float* golden_start_batch1 =
        lstm_fw_golden_output[1] + i * lstm.num_fw_outputs();
    float* golden_end_batch1 = golden_start_batch1 + lstm.num_fw_outputs();
    expected.insert(expected.end(), golden_start_batch0, golden_end_batch0);
    expected.insert(expected.end(), golden_start_batch1, golden_end_batch1);
  }
  EXPECT_THAT(lstm.GetFwOutput(), ElementsAreArray(ArrayFloatNear(expected)));

  // Check if the sum of forward backward matches the golden.
  expected.clear();
  for (int i = 0; i < lstm.sequence_length(); i++) {
    float* golden_start_batch0 =
        lstm_combined_golden_output[0] + i * lstm.num_fw_outputs();
    float* golden_end_batch0 = golden_start_batch0 + lstm.num_fw_outputs();
    float* golden_start_batch1 =
        lstm_combined_golden_output[1] + i * lstm.num_fw_outputs();
    float* golden_end_batch1 = golden_start_batch1 + lstm.num_fw_outputs();
    expected.insert(expected.end(), golden_start_batch0, golden_end_batch0);
    expected.insert(expected.end(), golden_start_batch1, golden_end_batch1);
  }

  std::vector<float> combined;
  for (int i = 0; i < lstm.GetFwOutput().size(); ++i) {
    combined.push_back(lstm.GetFwOutput()[i] + lstm.GetBwOutput()[i]);
  }
  EXPECT_THAT(combined, ElementsAreArray(ArrayFloatNear(expected)));
}

// Same as above but with batch_major input/output.
TEST(LSTMOpTest, BlackBoxTestWithPeepholeWithProjectionNoClippingBatchMajor) {
  const int n_batch = 2;
  const int n_input = 5;
  const int n_cell = 20;
  const int n_output = 16;
  const int sequence_length = 4;

  BidirectionalLSTMOpModel lstm(
      n_batch, n_input, n_cell, n_output, sequence_length, /*use_cifg=*/false,
      /*use_peephole=*/true, /*use_projection_weights=*/true,
      /*use_projection_bias=*/false, /*merge_outputs=*/false, /*cell_clip=*/0.0,
      /*proj_clip=*/0.0, /*quantize_weights=*/false, /*time_major=*/false,
      {
          {n_batch, sequence_length, n_input},  // input tensor

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

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor

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

          {n_cell},  // input_gate_bias tensor
          {n_cell},  // forget_gate_bias tensor
          {n_cell},  // cell_bias tensor
          {n_cell},  // output_gate_bias tensor

          {n_output, n_cell},  // projection_weight tensor
          {0},                 // projection_bias tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, n_output},  // activation_state tensor
          {n_batch, n_cell},    // cell_state tensor

          {n_batch, sequence_length, 0},  // aux_input tensor
          {n_cell, 0},                    // aux_fw_input_to_input tensor
          {n_cell, 0},                    // aux_fw_input_to_forget tensor
          {n_cell, 0},                    // aux_fw_input_to_cell tensor
          {n_cell, 0},                    // aux_fw_input_to_output tensor
          {n_cell, 0},                    // aux_bw_input_to_input tensor
          {n_cell, 0},                    // aux_bw_input_to_forget tensor
          {n_cell, 0},                    // aux_bw_input_to_cell tensor
          {n_cell, 0},                    // aux_bw_input_to_output tensor
      });

  lstm.SetInputToInputWeights(
      {0.021393683,  0.06124551,    0.046905167,  -0.014657677,  -0.03149463,
       0.09171803,   0.14647801,    0.10797193,   -0.0057968358, 0.0019193048,
       -0.2726754,   0.10154029,    -0.018539885, 0.080349885,   -0.10262385,
       -0.022599787, -0.09121155,   -0.008675967, -0.045206103,  -0.0821282,
       -0.008045952, 0.015478081,   0.055217247,  0.038719587,   0.044153627,
       -0.06453243,  0.05031825,    -0.046935108, -0.008164439,  0.014574226,
       -0.1671009,   -0.15519552,   -0.16819797,  -0.13971269,   -0.11953059,
       0.25005487,   -0.22790983,   0.009855087,  -0.028140958,  -0.11200698,
       0.11295408,   -0.0035217577, 0.054485075,  0.05184695,    0.064711206,
       0.10989193,   0.11674786,    0.03490607,   0.07727357,    0.11390585,
       -0.1863375,   -0.1034451,    -0.13945189,  -0.049401227,  -0.18767063,
       0.042483903,  0.14233552,    0.13832581,   0.18350165,    0.14545603,
       -0.028545704, 0.024939531,   0.050929718,  0.0076203286,  -0.0029723682,
       -0.042484224, -0.11827596,   -0.09171104,  -0.10808628,   -0.16327988,
       -0.2273378,   -0.0993647,    -0.017155107, 0.0023917493,  0.049272764,
       0.0038534778, 0.054764505,   0.089753784,  0.06947234,    0.08014476,
       -0.04544234,  -0.0497073,    -0.07135631,  -0.048929106,  -0.004042012,
       -0.009284026, 0.018042054,   0.0036860977, -0.07427302,   -0.11434604,
       -0.018995456, 0.031487543,   0.012834908,  0.019977754,   0.044256654,
       -0.39292613,  -0.18519334,   -0.11651281,  -0.06809892,   0.011373677});

  lstm.SetInputToForgetWeights(
      {-0.0018401089, -0.004852237,  0.03698424,   0.014181704,   0.028273236,
       -0.016726194,  -0.05249759,   -0.10204261,  0.00861066,    -0.040979505,
       -0.009899187,  0.01923892,    -0.028177269, -0.08535103,   -0.14585495,
       0.10662567,    -0.01909731,   -0.017883534, -0.0047269356, -0.045103323,
       0.0030784295,  0.076784775,   0.07463696,   0.094531395,   0.0814421,
       -0.12257899,   -0.033945758,  -0.031303465, 0.045630626,   0.06843887,
       -0.13492945,   -0.012480007,  -0.0811829,   -0.07224499,   -0.09628791,
       0.045100946,   0.0012300825,  0.013964662,  0.099372394,   0.02543059,
       0.06958324,    0.034257296,   0.0482646,    0.06267997,    0.052625068,
       0.12784666,    0.07077897,    0.025725935,  0.04165009,    0.07241905,
       0.018668644,   -0.037377294,  -0.06277783,  -0.08833636,   -0.040120605,
       -0.011405586,  -0.007808335,  -0.010301386, -0.005102167,  0.027717464,
       0.05483423,    0.11449111,    0.11289652,   0.10939839,    0.13396506,
       -0.08402166,   -0.01901462,   -0.044678304, -0.07720565,   0.014350063,
       -0.11757958,   -0.0652038,    -0.08185733,  -0.076754324,  -0.092614375,
       0.10405491,    0.052960336,   0.035755895,  0.035839386,   -0.012540553,
       0.036881298,   0.02913376,    0.03420159,   0.05448447,    -0.054523353,
       0.02582715,    0.02327355,    -0.011857179, -0.0011980024, -0.034641717,
       -0.026125094,  -0.17582615,   -0.15923657,  -0.27486774,   -0.0006143371,
       0.0001771948,  -8.470171e-05, 0.02651807,   0.045790765,   0.06956496});

  lstm.SetInputToCellWeights(
      {-0.04580283,   -0.09549462,   -0.032418985,  -0.06454633,
       -0.043528453,  0.043018587,   -0.049152344,  -0.12418144,
       -0.078985475,  -0.07596889,   0.019484362,   -0.11434962,
       -0.0074034138, -0.06314844,   -0.092981495,  0.0062155537,
       -0.025034338,  -0.0028890965, 0.048929527,   0.06235075,
       0.10665918,    -0.032036792,  -0.08505916,   -0.10843358,
       -0.13002433,   -0.036816437,  -0.02130134,   -0.016518239,
       0.0047691227,  -0.0025825808, 0.066017866,   0.029991534,
       -0.10652836,   -0.1037554,    -0.13056071,   -0.03266643,
       -0.033702414,  -0.006473424,  -0.04611692,   0.014419339,
       -0.025174323,  0.0396852,     0.081777506,   0.06157468,
       0.10210095,    -0.009658194,  0.046511717,   0.03603906,
       0.0069369148,  0.015960095,   -0.06507666,   0.09551598,
       0.053568836,   0.06408714,    0.12835667,    -0.008714329,
       -0.20211966,   -0.12093674,   0.029450472,   0.2849013,
       -0.029227901,  0.1164364,     -0.08560263,   0.09941786,
       -0.036999565,  -0.028842626,  -0.0033637602, -0.017012902,
       -0.09720865,   -0.11193351,   -0.029155117,  -0.017936034,
       -0.009768936,  -0.04223324,   -0.036159635,  0.06505112,
       -0.021742892,  -0.023377212,  -0.07221364,   -0.06430552,
       0.05453865,    0.091149814,   0.06387331,    0.007518393,
       0.055960953,   0.069779344,   0.046411168,   0.10509911,
       0.07463894,    0.0075130584,  0.012850982,   0.04555431,
       0.056955688,   0.06555285,    0.050801456,   -0.009862683,
       0.00826772,    -0.026555609,  -0.0073611983, -0.0014897042});

  lstm.SetInputToOutputWeights(
      {-0.0998932,   -0.07201956,  -0.052803773,  -0.15629593,  -0.15001918,
       -0.07650751,  0.02359855,   -0.075155355,  -0.08037709,  -0.15093534,
       0.029517552,  -0.04751393,  0.010350531,   -0.02664851,  -0.016839722,
       -0.023121163, 0.0077019283, 0.012851257,   -0.05040649,  -0.0129761,
       -0.021737747, -0.038305793, -0.06870586,   -0.01481247,  -0.001285394,
       0.10124236,   0.083122835,  0.053313006,   -0.062235646, -0.075637154,
       -0.027833903, 0.029774971,  0.1130802,     0.09218906,   0.09506135,
       -0.086665764, -0.037162706, -0.038880914,  -0.035832845, -0.014481564,
       -0.09825003,  -0.12048569,  -0.097665586,  -0.05287633,  -0.0964047,
       -0.11366429,  0.035777505,  0.13568819,    0.052451383,  0.050649304,
       0.05798951,   -0.021852335, -0.099848844,  0.014740475,  -0.078897946,
       0.04974699,   0.014160473,  0.06973932,    0.04964942,   0.033364646,
       0.08190124,   0.025535367,  0.050893165,   0.048514254,  0.06945813,
       -0.078907564, -0.06707616,  -0.11844508,   -0.09986688,  -0.07509403,
       0.06263226,   0.14925587,   0.20188436,    0.12098451,   0.14639415,
       0.0015017595, -0.014267382, -0.03417257,   0.012711468,  0.0028300495,
       -0.024758482, -0.05098548,  -0.0821182,    0.014225672,  0.021544158,
       0.08949725,   0.07505268,   -0.0020780868, 0.04908258,   0.06476295,
       -0.022907063, 0.027562456,  0.040185735,   0.019567577,  -0.015598739,
       -0.049097303, -0.017121866, -0.083368234,  -0.02332002,  -0.0840956});

  lstm.SetInputGateBias(
      {0.02234832,  0.14757581,   0.18176508,  0.10380666,  0.053110216,
       -0.06928846, -0.13942584,  -0.11816189, 0.19483899,  0.03652339,
       -0.10250295, 0.036714908,  -0.18426876, 0.036065217, 0.21810818,
       0.02383196,  -0.043370757, 0.08690144,  -0.04444982, 0.00030581196});

  lstm.SetForgetGateBias({0.035185695, -0.042891346, -0.03032477, 0.23027696,
                          0.11098921,  0.15378423,   0.09263801,  0.09790885,
                          0.09508917,  0.061199076,  0.07665568,  -0.015443159,
                          -0.03499149, 0.046190713,  0.08895977,  0.10899629,
                          0.40694186,  0.06030037,   0.012413437, -0.06108739});

  lstm.SetCellBias({-0.024379363, 0.0055531194, 0.23377132,   0.033463873,
                    -0.1483596,   -0.10639995,  -0.091433935, 0.058573797,
                    -0.06809782,  -0.07889636,  -0.043246906, -0.09829136,
                    -0.4279842,   0.034901652,  0.18797937,   0.0075234566,
                    0.016178843,  0.1749513,    0.13975595,   0.92058027});

  lstm.SetOutputGateBias(
      {0.046159424,  -0.0012809046, 0.03563469,   0.12648113, 0.027195795,
       0.35373217,   -0.018957434,  0.008907322,  -0.0762701, 0.12018895,
       0.04216877,   0.0022856654,  0.040952638,  0.3147856,  0.08225149,
       -0.057416286, -0.14995944,   -0.008040261, 0.13208859, 0.029760877});

  lstm.SetRecurrentToInputWeights(
      {-0.001374326,   -0.078856036,   0.10672688,    0.029162422,
       -0.11585556,    0.02557986,     -0.13446963,   -0.035785314,
       -0.01244275,    0.025961924,    -0.02337298,   -0.044228926,
       -0.055839065,   -0.046598054,   -0.010546039,  -0.06900766,
       0.027239809,    0.022582639,    -0.013296484,  -0.05459212,
       0.08981,        -0.045407712,   0.08682226,    -0.06867011,
       -0.14390695,    -0.02916037,    0.000996957,   0.091420636,
       0.14283475,     -0.07390571,    -0.06402044,   0.062524505,
       -0.093129106,   0.04860203,     -0.08364217,   -0.08119002,
       0.009352075,    0.22920375,     0.0016303885,  0.11583097,
       -0.13732095,    0.012405723,    -0.07551853,   0.06343048,
       0.12162708,     -0.031923793,   -0.014335606,  0.01790974,
       -0.10650317,    -0.0724401,     0.08554849,    -0.05727212,
       0.06556731,     -0.042729504,   -0.043227166,  0.011683251,
       -0.013082158,   -0.029302018,   -0.010899579,  -0.062036745,
       -0.022509435,   -0.00964907,    -0.01567329,   0.04260106,
       -0.07787477,    -0.11576462,    0.017356863,   0.048673786,
       -0.017577527,   -0.05527947,    -0.082487635,  -0.040137455,
       -0.10820036,    -0.04666372,    0.022746278,   -0.07851417,
       0.01068115,     0.032956902,    0.022433773,   0.0026891115,
       0.08944216,     -0.0685835,     0.010513544,   0.07228705,
       0.02032331,     -0.059686817,   -0.0005566496, -0.086984694,
       0.040414046,    -0.1380399,     0.094208956,   -0.05722982,
       0.012092817,    -0.04989123,    -0.086576,     -0.003399834,
       -0.04696032,    -0.045747425,   0.10091314,    0.048676282,
       -0.029037097,   0.031399418,    -0.0040285117, 0.047237843,
       0.09504992,     0.041799378,    -0.049185462,  -0.031518843,
       -0.10516937,    0.026374253,    0.10058866,    -0.0033195973,
       -0.041975245,   0.0073591834,   0.0033782164,  -0.004325073,
       -0.10167381,    0.042500053,    -0.01447153,   0.06464186,
       -0.017142897,   0.03312627,     0.009205989,   0.024138335,
       -0.011337001,   0.035530265,    -0.010912711,  0.0706555,
       -0.005894094,   0.051841937,    -0.1401738,    -0.02351249,
       0.0365468,      0.07590991,     0.08838724,    0.021681072,
       -0.10086113,    0.019608743,    -0.06195883,   0.077335775,
       0.023646897,    -0.095322326,   0.02233014,    0.09756986,
       -0.048691444,   -0.009579111,   0.07595467,    0.11480546,
       -0.09801813,    0.019894179,    0.08502348,    0.004032281,
       0.037211012,    0.068537936,    -0.048005626,  -0.091520436,
       -0.028379958,   -0.01556313,    0.06554592,    -0.045599163,
       -0.01672207,    -0.020169014,   -0.011877351,  -0.20212261,
       0.010889619,    0.0047078193,   0.038385306,   0.08540671,
       -0.017140968,   -0.0035865551,  0.016678626,   0.005633034,
       0.015963363,    0.00871737,     0.060130805,   0.028611384,
       0.10109069,     -0.015060172,   -0.07894427,   0.06401885,
       0.011584063,    -0.024466386,   0.0047652307,  -0.09041358,
       0.030737216,    -0.0046374933,  0.14215417,    -0.11823516,
       0.019899689,    0.006106124,    -0.027092824,  0.0786356,
       0.05052217,     -0.058925,      -0.011402121,  -0.024987547,
       -0.0013661642,  -0.06832946,    -0.015667673,  -0.1083353,
       -0.00096863037, -0.06988685,    -0.053350925,  -0.027275559,
       -0.033664223,   -0.07978348,    -0.025200296,  -0.017207067,
       -0.058403496,   -0.055697463,   0.005798788,   0.12965427,
       -0.062582195,   0.0013350133,   -0.10482091,   0.0379771,
       0.072521195,    -0.0029455067,  -0.13797039,   -0.03628521,
       0.013806405,    -0.017858358,   -0.01008298,   -0.07700066,
       -0.017081132,   0.019358726,    0.0027079724,  0.004635139,
       0.062634714,    -0.02338735,    -0.039547626,  -0.02050681,
       0.03385117,     -0.083611414,   0.002862572,   -0.09421313,
       0.058618143,    -0.08598433,    0.00972939,    0.023867095,
       -0.053934585,   -0.023203006,   0.07452513,    -0.048767887,
       -0.07314807,    -0.056307215,   -0.10433547,   -0.06440842,
       0.04328182,     0.04389765,     -0.020006588,  -0.09076438,
       -0.11652589,    -0.021705797,   0.03345259,    -0.010329105,
       -0.025767034,   0.013057034,    -0.07316461,   -0.10145612,
       0.06358255,     0.18531723,     0.07759293,    0.12006465,
       0.1305557,      0.058638252,    -0.03393652,   0.09622831,
       -0.16253184,    -2.4580743e-06, 0.079869635,   -0.070196845,
       -0.005644518,   0.06857898,     -0.12598175,   -0.035084512,
       0.03156317,     -0.12794146,    -0.031963028,  0.04692781,
       0.030070418,    0.0071660685,   -0.095516115,  -0.004643372,
       0.040170413,    -0.062104587,   -0.0037324072, 0.0554317,
       0.08184801,     -0.019164372,   0.06791302,    0.034257166,
       -0.10307039,    0.021943003,    0.046745934,   0.0790918,
       -0.0265588,     -0.007824208,   0.042546265,   -0.00977924,
       -0.0002440307,  -0.017384544,   -0.017990116,  0.12252321,
       -0.014512694,   -0.08251313,    0.08861942,    0.13589665,
       0.026351685,    0.012641483,    0.07466548,    0.044301085,
       -0.045414884,   -0.051112458,   0.03444247,    -0.08502782,
       -0.04106223,    -0.028126027,   0.028473156,   0.10467447});

  lstm.SetRecurrentToForgetWeights(
      {-0.057784554,  -0.026057621,  -0.068447545,   -0.022581743,
       0.14811787,    0.10826372,    0.09471067,     0.03987225,
       -0.0039523416, 0.00030638507, 0.053185795,    0.10572994,
       0.08414449,    -0.022036452,  -0.00066928595, -0.09203576,
       0.032950465,   -0.10985798,   -0.023809856,   0.0021431844,
       -0.02196096,   -0.00326074,   0.00058621005,  -0.074678116,
       -0.06193199,   0.055729095,   0.03736828,     0.020123724,
       0.061878487,   -0.04729229,   0.034919553,    -0.07585433,
       -0.04421272,   -0.044019096,  0.085488975,    0.04058006,
       -0.06890133,   -0.030951202,  -0.024628663,   -0.07672815,
       0.034293607,   0.08556707,    -0.05293577,    -0.033561368,
       -0.04899627,   0.0241671,     0.015736353,    -0.095442444,
       -0.029564252,  0.016493602,   -0.035026584,   0.022337519,
       -0.026871363,  0.004780428,   0.0077918363,   -0.03601621,
       0.016435321,   -0.03263031,   -0.09543275,    -0.047392778,
       0.013454138,   0.028934088,   0.01685226,     -0.086110644,
       -0.046250615,  -0.01847454,   0.047608484,    0.07339695,
       0.034546845,   -0.04881143,   0.009128804,    -0.08802852,
       0.03761666,    0.008096139,   -0.014454086,   0.014361001,
       -0.023502491,  -0.0011840804, -0.07607001,    0.001856849,
       -0.06509276,   -0.006021153,  -0.08570962,    -0.1451793,
       0.060212336,   0.055259194,   0.06974018,     0.049454916,
       -0.027794661,  -0.08077226,   -0.016179763,   0.1169753,
       0.17213494,    -0.0056326236, -0.053934924,   -0.0124349,
       -0.11520337,   0.05409887,    0.088759385,    0.0019655675,
       0.0042065294,  0.03881498,    0.019844765,    0.041858196,
       -0.05695512,   0.047233116,   0.038937137,    -0.06542224,
       0.014429736,   -0.09719407,   0.13908425,     -0.05379757,
       0.012321099,   0.082840554,   -0.029899208,   0.044217527,
       0.059855383,   0.07711018,    -0.045319796,   0.0948846,
       -0.011724666,  -0.0033288454, -0.033542685,   -0.04764985,
       -0.13873616,   0.040668588,   0.034832682,    -0.015319203,
       -0.018715994,  0.046002675,   0.0599172,      -0.043107376,
       0.0294216,     -0.002314414,  -0.022424703,   0.0030315618,
       0.0014641669,  0.0029166266,  -0.11878115,    0.013738511,
       0.12375372,    -0.0006038222, 0.029104086,    0.087442465,
       0.052958444,   0.07558703,    0.04817258,     0.044462286,
       -0.015213451,  -0.08783778,   -0.0561384,     -0.003008196,
       0.047060397,   -0.002058388,  0.03429439,     -0.018839769,
       0.024734668,   0.024614193,   -0.042046934,   0.09597743,
       -0.0043254104, 0.04320769,    0.0064070094,   -0.0019131786,
       -0.02558259,   -0.022822596,  -0.023273505,   -0.02464396,
       -0.10991725,   -0.006240552,  0.0074488563,   0.024044557,
       0.04383914,    -0.046476185,  0.028658995,    0.060410924,
       0.050786525,   0.009452605,   -0.0073054377,  -0.024810238,
       0.0052906186,  0.0066939713,  -0.0020913032,  0.014515517,
       0.015898481,   0.021362653,   -0.030262267,   0.016587038,
       -0.011442813,  0.041154444,   -0.007631438,   -0.03423484,
       -0.010977775,  0.036152758,   0.0066366293,   0.11915515,
       0.02318443,    -0.041350313,  0.021485701,    -0.10906167,
       -0.028218046,  -0.00954771,   0.020531068,    -0.11995105,
       -0.03672871,   0.024019798,   0.014255957,    -0.05221243,
       -0.00661567,   -0.04630967,   0.033188973,    0.10107534,
       -0.014027541,  0.030796422,   -0.10270911,    -0.035999842,
       0.15443139,    0.07684145,    0.036571592,    -0.035900835,
       -0.0034699554, 0.06209149,    0.015920248,    -0.031122351,
       -0.03858649,   0.01849943,    0.13872518,     0.01503974,
       0.069941424,   -0.06948533,   -0.0088794185,  0.061282158,
       -0.047401894,  0.03100163,    -0.041533746,   -0.10430945,
       0.044574402,   -0.01425562,   -0.024290353,   0.034563623,
       0.05866852,    0.023947537,   -0.09445152,    0.035450947,
       0.02247216,    -0.0042998926, 0.061146557,    -0.10250651,
       0.020881841,   -0.06747029,   0.10062043,     -0.0023941975,
       0.03532124,    -0.016341697,  0.09685456,     -0.016764693,
       0.051808182,   0.05875331,    -0.04536488,    0.001626336,
       -0.028892258,  -0.01048663,   -0.009793449,   -0.017093895,
       0.010987891,   0.02357273,    -0.00010856845, 0.0099760275,
       -0.001845119,  -0.03551521,   0.0018358806,   0.05763657,
       -0.01769146,   0.040995963,   0.02235177,     -0.060430344,
       0.11475477,    -0.023854522,  0.10071741,     0.0686208,
       -0.014250481,  0.034261297,   0.047418304,    0.08562733,
       -0.030519066,  0.0060542435,  0.014653856,    -0.038836084,
       0.04096551,    0.032249358,   -0.08355519,    -0.026823482,
       0.056386515,   -0.010401743,  -0.028396193,   0.08507674,
       0.014410365,   0.020995233,   0.17040324,     0.11511526,
       0.02459721,    0.0066619175,  0.025853224,    -0.023133837,
       -0.081302024,  0.017264642,   -0.009585969,   0.09491168,
       -0.051313367,  0.054532815,   -0.014298593,   0.10657464,
       0.007076659,   0.10964551,    0.0409152,      0.008275321,
       -0.07283536,   0.07937492,    0.04192024,     -0.1075027});

  lstm.SetRecurrentToCellWeights(
      {-0.037322544,   0.018592842,   0.0056175636,  -0.06253426,
       0.055647098,    -0.05713207,   -0.05626563,   0.005559383,
       0.03375411,     -0.025757805,  -0.088049285,  0.06017052,
       -0.06570978,    0.007384076,   0.035123326,   -0.07920549,
       0.053676967,    0.044480428,   -0.07663568,   0.0071805613,
       0.08089997,     0.05143358,    0.038261272,   0.03339287,
       -0.027673481,   0.044746667,   0.028349208,   0.020090483,
       -0.019443132,   -0.030755889,  -0.0040000007, 0.04465846,
       -0.021585021,   0.0031670958,  0.0053199246,  -0.056117613,
       -0.10893326,    0.076739706,   -0.08509834,   -0.027997585,
       0.037871376,    0.01449768,    -0.09002357,   -0.06111149,
       -0.046195522,   0.0422062,     -0.005683705,  -0.1253618,
       -0.012925729,   -0.04890792,   0.06985068,    0.037654128,
       0.03398274,     -0.004781977,  0.007032333,   -0.031787455,
       0.010868644,    -0.031489216,  0.09525667,    0.013939797,
       0.0058680447,   0.0167067,     0.02668468,    -0.04797466,
       -0.048885044,   -0.12722108,   0.035304096,   0.06554885,
       0.00972396,     -0.039238118,  -0.05159735,   -0.11329045,
       0.1613692,      -0.03750952,   0.06529313,    -0.071974665,
       -0.11769596,    0.015524369,   -0.0013754242, -0.12446318,
       0.02786344,     -0.014179351,  0.005264273,   0.14376344,
       0.015983658,    0.03406988,    -0.06939408,   0.040699873,
       0.02111075,     0.09669095,    0.041345075,   -0.08316494,
       -0.07684199,    -0.045768797,  0.032298047,   -0.041805092,
       0.0119405,      0.0061010392,  0.12652606,    0.0064572375,
       -0.024950314,   0.11574242,    0.04508852,    -0.04335324,
       0.06760663,     -0.027437469,  0.07216407,    0.06977076,
       -0.05438599,    0.034033038,   -0.028602652,  0.05346137,
       0.043184172,    -0.037189785,  0.10420091,    0.00882477,
       -0.054019816,   -0.074273005,  -0.030617684,  -0.0028467078,
       0.024302477,    -0.0038869337, 0.005332455,   0.0013399826,
       0.04361412,     -0.007001822,  0.09631092,    -0.06702025,
       -0.042049985,   -0.035070654,  -0.04103342,   -0.10273396,
       0.0544271,      0.037184782,   -0.13150354,   -0.0058036847,
       -0.008264958,   0.042035464,   0.05891794,    0.029673764,
       0.0063542654,   0.044788733,   0.054816857,   0.062257513,
       -0.00093483756, 0.048938446,   -0.004952862,  -0.007730018,
       -0.04043371,    -0.017094059,  0.07229206,    -0.023670016,
       -0.052195564,   -0.025616996,  -0.01520939,   0.045104615,
       -0.007376126,   0.003533447,   0.006570588,   0.056037236,
       0.12436656,     0.051817212,   0.028532185,   -0.08686856,
       0.11868599,     0.07663395,    -0.07323171,   0.03463402,
       -0.050708205,   -0.04458982,   -0.11590894,   0.021273347,
       0.1251325,      -0.15313013,   -0.12224372,   0.17228661,
       0.023029093,    0.086124025,   0.006445803,   -0.03496501,
       0.028332196,    0.04449512,    -0.042436164,  -0.026587414,
       -0.006041347,   -0.09292539,   -0.05678812,   0.03897832,
       0.09465633,     0.008115513,   -0.02171956,   0.08304309,
       0.071401566,    0.019622514,   0.032163795,   -0.004167056,
       0.02295182,     0.030739572,   0.056506045,   0.004612461,
       0.06524936,     0.059999723,   0.046395954,   -0.0045512207,
       -0.1335546,     -0.030136576,  0.11584653,    -0.014678886,
       0.0020118146,   -0.09688814,   -0.0790206,    0.039770417,
       -0.0329582,     0.07922767,    0.029322514,   0.026405897,
       0.04207835,     -0.07073373,   0.063781224,   0.0859677,
       -0.10925287,    -0.07011058,   0.048005477,   0.03438226,
       -0.09606514,    -0.006669445,  -0.043381985,  0.04240257,
       -0.06955775,    -0.06769346,   0.043903265,   -0.026784198,
       -0.017840602,   0.024307009,   -0.040079936,  -0.019946516,
       0.045318738,    -0.12233574,   0.026170589,   0.0074471775,
       0.15978073,     0.10185836,    0.10298046,    -0.015476589,
       -0.039390966,   -0.072174534,  0.0739445,     -0.1211869,
       -0.0347889,     -0.07943156,   0.014809798,   -0.12412325,
       -0.0030663363,  0.039695457,   0.0647603,     -0.08291318,
       -0.018529687,   -0.004423833,  0.0037507233,  0.084633216,
       -0.01514876,    -0.056505352,  -0.012800942,  -0.06994386,
       0.012962922,    -0.031234352,  0.07029052,    0.016418684,
       0.03618972,     0.055686004,   -0.08663945,   -0.017404709,
       -0.054761406,   0.029065743,   0.052404847,   0.020238016,
       0.0048197987,   -0.0214882,    0.07078733,    0.013016777,
       0.06262858,     0.009184685,   0.020785125,   -0.043904778,
       -0.0270329,     -0.03299152,   -0.060088247,  -0.015162964,
       -0.001828936,   0.12642565,    -0.056757294,  0.013586685,
       0.09232601,     -0.035886683,  0.06000002,    0.05229691,
       -0.052580316,   -0.082029596,  -0.010794592,  0.012947712,
       -0.036429964,   -0.085508935,  -0.13127148,   -0.017744139,
       0.031502828,    0.036232427,   -0.031581745,  0.023051167,
       -0.05325106,    -0.03421577,   0.028793324,   -0.034633752,
       -0.009881397,   -0.043551125,  -0.018609839,  0.0019097115,
       -0.008799762,   0.056595087,   0.0022273948,  0.055752404});

  lstm.SetRecurrentToOutputWeights({
      0.025825322,   -0.05813119,  0.09495884,   -0.045984812,   -0.01255415,
      -0.0026479573, -0.08196161,  -0.054914974, -0.0046604523,  -0.029587349,
      -0.044576716,  -0.07480124,  -0.082868785, 0.023254942,    0.027502948,
      -0.0039728214, -0.08683098,  -0.08116779,  -0.014675607,   -0.037924774,
      -0.023314456,  -0.007401714, -0.09255757,  0.029460307,    -0.08829125,
      -0.005139627,  -0.08989442,  -0.0555066,   0.13596267,     -0.025062224,
      -0.048351806,  -0.03850004,  0.07266485,   -0.022414139,   0.05940088,
      0.075114764,   0.09597592,   -0.010211725, -0.0049794707,  -0.011523867,
      -0.025980417,  0.072999895,  0.11091378,   -0.081685916,   0.014416728,
      0.043229222,   0.034178585,  -0.07530371,  0.035837382,    -0.085607,
      -0.007721233,  -0.03287832,  -0.043848954, -0.06404588,    -0.06632928,
      -0.073643476,  0.008214239,  -0.045984086, 0.039764922,    0.03474462,
      0.060612556,   -0.080590084, 0.049127717,  0.04151091,     -0.030063879,
      0.008801774,   -0.023021035, -0.019558564, 0.05158114,     -0.010947698,
      -0.011825728,  0.0075720972, 0.0699727,    -0.0039981045,  0.069350146,
      0.08799282,    0.016156472,  0.035502106,  0.11695009,     0.006217345,
      0.13392477,    -0.037875112, 0.025745004,  0.08940699,     -0.00924166,
      0.0046702605,  -0.036598757, -0.08811812,  0.10522024,     -0.032441203,
      0.008176899,   -0.04454919,  0.07058152,   0.0067963637,   0.039206743,
      0.03259838,    0.03725492,   -0.09515802,  0.013326398,    -0.052055415,
      -0.025676316,  0.03198509,   -0.015951829, -0.058556724,   0.036879618,
      0.043357447,   0.028362012,  -0.05908629,  0.0059240665,   -0.04995891,
      -0.019187413,  0.0276265,    -0.01628143,  0.0025863599,   0.08800015,
      0.035250366,   -0.022165963, -0.07328642,  -0.009415526,   -0.07455109,
      0.11690406,    0.0363299,    0.07411125,   0.042103454,    -0.009660886,
      0.019076364,   0.018299393,  -0.046004917, 0.08891175,     0.0431396,
      -0.026327137,  -0.051502608, 0.08979574,   -0.051670972,   0.04940282,
      -0.07491107,   -0.021240504, 0.022596184,  -0.034280192,   0.060163025,
      -0.058211457,  -0.051837247, -0.01349775,  -0.04639988,    -0.035936575,
      -0.011681591,  0.064818054,  0.0073146066, -0.021745546,   -0.043124277,
      -0.06471268,   -0.07053354,  -0.029321948, -0.05330136,    0.016933719,
      -0.053782392,  0.13747959,   -0.1361751,   -0.11569455,    0.0033329215,
      0.05693899,    -0.053219706, 0.063698,     0.07977434,     -0.07924483,
      0.06936997,    0.0034815092, -0.007305279, -0.037325785,   -0.07251102,
      -0.033633437,  -0.08677009,  0.091591336,  -0.14165086,    0.021752775,
      0.019683983,   0.0011612234, -0.058154266, 0.049996935,    0.0288841,
      -0.0024567875, -0.14345716,  0.010955264,  -0.10234828,    0.1183656,
      -0.0010731248, -0.023590032, -0.072285876, -0.0724771,     -0.026382286,
      -0.0014920527, 0.042667855,  0.0018776858, 0.02986552,     0.009814309,
      0.0733756,     0.12289186,   0.018043943,  -0.0458958,     0.049412545,
      0.033632483,   0.05495232,   0.036686596,  -0.013781798,   -0.010036754,
      0.02576849,    -0.08307328,  0.010112348,  0.042521734,    -0.05869831,
      -0.071689695,  0.03876447,   -0.13275425,  -0.0352966,     -0.023077697,
      0.10285965,    0.084736146,  0.15568255,   -0.00040734606, 0.027835453,
      -0.10292561,   -0.032401145, 0.10053256,   -0.026142767,   -0.08271222,
      -0.0030240538, -0.016368777, 0.1070414,    0.042672627,    0.013456989,
      -0.0437609,    -0.022309763, 0.11576483,   0.04108048,     0.061026827,
      -0.0190714,    -0.0869359,   0.037901703,  0.0610107,      0.07202949,
      0.01675338,    0.086139716,  -0.08795751,  -0.014898893,   -0.023771819,
      -0.01965048,   0.007955471,  -0.043740474, 0.03346837,     -0.10549954,
      0.090567775,   0.042013682,  -0.03176985,  0.12569028,     -0.02421228,
      -0.029526481,  0.023851605,  0.031539805,  0.05292009,     -0.02344001,
      -0.07811758,   -0.08834428,  0.10094801,   0.16594367,     -0.06861939,
      -0.021256343,  -0.041093912, -0.06669611,  0.035498552,    0.021757556,
      -0.09302526,   -0.015403468, -0.06614931,  -0.051798206,   -0.013874718,
      0.03630673,    0.010412845,  -0.08077351,  0.046185967,    0.0035662893,
      0.03541868,    -0.094149634, -0.034814864, 0.003128424,    -0.020674974,
      -0.03944324,   -0.008110165, -0.11113267,  0.08484226,     0.043586485,
      0.040582247,   0.0968012,    -0.065249965, -0.028036479,   0.0050708856,
      0.0017462453,  0.0326779,    0.041296225,  0.09164146,     -0.047743853,
      -0.015952192,  -0.034451712, 0.084197424,  -0.05347844,    -0.11768019,
      0.085926116,   -0.08251791,  -0.045081906, 0.0948852,      0.068401024,
      0.024856757,   0.06978981,   -0.057309967, -0.012775832,   -0.0032452994,
      0.01977615,    -0.041040014, -0.024264973, 0.063464895,    0.05431621,
  });

  lstm.SetCellToInputWeights(
      {0.040369894, 0.030746894,  0.24704495,  0.018586371,  -0.037586458,
       -0.15312155, -0.11812848,  -0.11465643, 0.20259799,   0.11418174,
       -0.10116027, -0.011334949, 0.12411352,  -0.076769054, -0.052169047,
       0.21198851,  -0.38871562,  -0.09061183, -0.09683246,  -0.21929175});

  lstm.SetCellToForgetWeights(
      {-0.01998659,  -0.15568835,  -0.24248174,   -0.012770197, 0.041331276,
       -0.072311886, -0.052123554, -0.0066330447, -0.043891653, 0.036225766,
       -0.047248036, 0.021479502,  0.033189066,   0.11952997,   -0.020432774,
       0.64658105,   -0.06650122,  -0.03467612,   0.095340036,  0.23647355});

  lstm.SetCellToOutputWeights(
      {0.08286371,  -0.08261836, -0.51210177, 0.002913762, 0.17764764,
       -0.5495371,  -0.08460716, -0.24552552, 0.030037103, 0.04123544,
       -0.11940523, 0.007358328, 0.1890978,   0.4833202,   -0.34441817,
       0.36312827,  -0.26375428, 0.1457655,   -0.19724406, 0.15548733});

  lstm.SetProjectionWeights(
      {-0.009802181,  0.09401916,    0.0717386,     -0.13895074,  0.09641832,
       0.060420845,   0.08539281,    0.054285463,   0.061395317,  0.034448683,
       -0.042991187,  0.019801661,   -0.16840284,   -0.015726732, -0.23041931,
       -0.024478018,  -0.10959692,   -0.013875541,  0.18600968,   -0.061274476,
       0.0138165,     -0.08160894,   -0.07661644,   0.032372914,  0.16169067,
       0.22465782,    -0.03993472,   -0.004017731,  0.08633481,   -0.28869787,
       0.08682067,    0.17240396,    0.014975425,   0.056431185,  0.031037588,
       0.16702051,    0.0077946745,  0.15140012,    0.29405436,   0.120285,
       -0.188994,     -0.027265169,  0.043389652,   -0.022061434, 0.014777949,
       -0.20203483,   0.094781205,   0.19100232,    0.13987629,   -0.036132768,
       -0.06426278,   -0.05108664,   0.13221376,    0.009441198,  -0.16715929,
       0.15859416,    -0.040437475,  0.050779544,   -0.022187516, 0.012166504,
       0.027685808,   -0.07675938,   -0.0055694645, -0.09444123,  0.0046453946,
       0.050794356,   0.10770313,    -0.20790008,   -0.07149004,  -0.11425117,
       0.008225835,   -0.035802525,  0.14374903,    0.15262283,   0.048710253,
       0.1847461,     -0.007487823,  0.11000021,    -0.09542012,  0.22619456,
       -0.029149994,  0.08527916,    0.009043713,   0.0042746216, 0.016261552,
       0.022461696,   0.12689082,    -0.043589946,  -0.12035478,  -0.08361797,
       -0.050666027,  -0.1248618,    -0.1275799,    -0.071875185, 0.07377272,
       0.09944291,    -0.18897448,   -0.1593054,    -0.06526116,  -0.040107165,
       -0.004618631,  -0.067624845,  -0.007576253,  0.10727444,   0.041546922,
       -0.20424393,   0.06907816,    0.050412357,   0.00724631,   0.039827548,
       0.12449835,    0.10747581,    0.13708383,    0.09134148,   -0.12617786,
       -0.06428341,   0.09956831,    0.1208086,     -0.14676677,  -0.0727722,
       0.1126304,     0.010139365,   0.015571211,   -0.038128063, 0.022913318,
       -0.042050496,  0.16842307,    -0.060597885,  0.10531834,   -0.06411776,
       -0.07451711,   -0.03410368,   -0.13393489,   0.06534304,   0.003620307,
       0.04490757,    0.05970546,    0.05197996,    0.02839995,   0.10434969,
       -0.013699693,  -0.028353551,  -0.07260381,   0.047201227,  -0.024575593,
       -0.036445823,  0.07155557,    0.009672501,   -0.02328883,  0.009533515,
       -0.03606021,   -0.07421458,   -0.028082801,  -0.2678904,   -0.13221288,
       0.18419984,    -0.13012612,   -0.014588381,  -0.035059117, -0.04824723,
       0.07830115,    -0.056184657,  0.03277091,    0.025466874,  0.14494097,
       -0.12522776,   -0.098633975,  -0.10766018,   -0.08317623,  0.08594209,
       0.07749552,    0.039474737,   0.1776665,     -0.07409566,  -0.0477268,
       0.29323658,    0.10801441,    0.1154011,     0.013952499,  0.10739139,
       0.10708251,    -0.051456142,  0.0074137426,  -0.10430189,  0.10034707,
       0.045594677,   0.0635285,     -0.0715442,    -0.089667566, -0.10811871,
       0.00026344223, 0.08298446,    -0.009525053,  0.006585689,  -0.24567553,
       -0.09450807,   0.09648481,    0.026996298,   -0.06419476,  -0.04752702,
       -0.11063944,   -0.23441927,   -0.17608605,   -0.052156363, 0.067035615,
       0.19271925,    -0.0032889997, -0.043264326,  0.09663576,   -0.057112187,
       -0.10100678,   0.0628376,     0.04447668,    0.017961001,  -0.10094388,
       -0.10190601,   0.18335468,    0.10494553,    -0.052095775, -0.0026118709,
       0.10539724,    -0.04383912,   -0.042349473,  0.08438151,   -0.1947263,
       0.02251204,    0.11216432,    -0.10307853,   0.17351969,   -0.039091777,
       0.08066188,    -0.00561982,   0.12633002,    0.11335965,   -0.0088127935,
       -0.019777594,  0.06864014,    -0.059751723,  0.016233567,  -0.06894641,
       -0.28651384,   -0.004228674,  0.019708522,   -0.16305895,  -0.07468996,
       -0.0855457,    0.099339016,   -0.07580735,   -0.13775392,  0.08434318,
       0.08330512,    -0.12131499,   0.031935584,   0.09180414,   -0.08876437,
       -0.08049874,   0.008753825,   0.03498998,    0.030215185,  0.03907079,
       0.089751154,   0.029194152,   -0.03337423,   -0.019092513, 0.04331237,
       0.04299654,    -0.036394123,  -0.12915532,   0.09793732,   0.07512415,
       -0.11319543,   -0.032502122,  0.15661901,    0.07671967,   -0.005491124,
       -0.19379048,   -0.218606,     0.21448623,    0.017840758,  0.1416943,
       -0.07051762,   0.19488361,    0.02664691,    -0.18104725,  -0.09334311,
       0.15026465,    -0.15493552,   -0.057762887,  -0.11604192,  -0.262013,
       -0.01391798,   0.012185008,   0.11156489,    -0.07483202,  0.06693364,
       -0.26151478,   0.046425626,   0.036540434,   -0.16435726,  0.17338543,
       -0.21401681,   -0.11385144,   -0.08283257,   -0.069031075, 0.030635102,
       0.010969227,   0.11109743,    0.010919218,   0.027526086,  0.13519906,
       0.01891392,    -0.046839405,  -0.040167913,  0.017953383,  -0.09700955,
       0.0061885654,  -0.07000971,   0.026893595,   -0.038844477, 0.14543656});

  static float lstm_input[][20] = {
      {// Batch0: 4 (input_sequence_size) * 5 (n_input)
       0.787926, 0.151646, 0.071352, 0.118426, 0.458058, 0.596268, 0.998386,
       0.568695, 0.864524, 0.571277, 0.073204, 0.296072, 0.743333, 0.069199,
       0.045348, 0.867394, 0.291279, 0.013714, 0.482521, 0.626339},

      {// Batch1: 4 (input_sequence_size) * 5 (n_input)
       0.295743, 0.544053, 0.690064, 0.858138, 0.497181, 0.642421, 0.524260,
       0.134799, 0.003639, 0.162482, 0.640394, 0.930399, 0.050782, 0.432485,
       0.988078, 0.082922, 0.563329, 0.865614, 0.333232, 0.259916}};

  static float lstm_fw_golden_output[][64] = {
      {// Batch0: 4 (input_sequence_size) * 16 (n_output)
       -0.00396806, 0.029352,     -0.00279226, 0.0159977,   -0.00835576,
       -0.0211779,  0.0283512,    -0.0114597,  0.00907307,  -0.0244004,
       -0.0152191,  -0.0259063,   0.00914318,  0.00415118,  0.017147,
       0.0134203,   -0.0166936,   0.0381209,   0.000889694, 0.0143363,
       -0.0328911,  -0.0234288,   0.0333051,   -0.012229,   0.0110322,
       -0.0457725,  -0.000832209, -0.0202817,  0.0327257,   0.0121308,
       0.0155969,   0.0312091,    -0.0213783,  0.0350169,   0.000324794,
       0.0276012,   -0.0263374,   -0.0371449,  0.0446149,   -0.0205474,
       0.0103729,   -0.0576349,   -0.0150052,  -0.0292043,  0.0376827,
       0.0136115,   0.0243435,    0.0354492,   -0.0189322,  0.0464512,
       -0.00251373, 0.0225745,    -0.0308346,  -0.0317124,  0.0460407,
       -0.0189395,  0.0149363,    -0.0530162,  -0.0150767,  -0.0340193,
       0.0286833,   0.00824207,   0.0264887,   0.0305169},
      {// Batch1: 4 (input_sequence_size) * 16 (n_output)
       -0.013869,    0.0287268,   -0.00334693, 0.00733398,  -0.0287926,
       -0.0186926,   0.0193662,   -0.0115437,  0.00422612,  -0.0345232,
       0.00223253,   -0.00957321, 0.0210624,   0.013331,    0.0150954,
       0.02168,      -0.0141913,  0.0322082,   0.00227024,  0.0260507,
       -0.0188721,   -0.0296489,  0.0399134,   -0.0160509,  0.0116039,
       -0.0447318,   -0.0150515,  -0.0277406,  0.0316596,   0.0118233,
       0.0214762,    0.0293641,   -0.0204549,  0.0450315,   -0.00117378,
       0.0167673,    -0.0375007,  -0.0238314,  0.038784,    -0.0174034,
       0.0131743,    -0.0506589,  -0.0048447,  -0.0240239,  0.0325789,
       0.00790065,   0.0220157,   0.0333314,   -0.0264787,  0.0387855,
       -0.000764675, 0.0217599,   -0.037537,   -0.0335206,  0.0431679,
       -0.0211424,   0.010203,    -0.062785,   -0.00832363, -0.025181,
       0.0412031,    0.0118723,   0.0239643,   0.0394009}};

  static float lstm_combined_golden_output[][64] = {
      {-0.022014, 0.073544,  -0.002235, 0.040068,  -0.037136, -0.052788,
       0.075325,  -0.029378, 0.024298,  -0.07733,  -0.030674, -0.060229,
       0.040599,  0.011608,  0.042005,  0.045977,  -0.039225, 0.076294,
       0.000735,  0.032852,  -0.069869, -0.053312, 0.073527,  -0.028136,
       0.021585,  -0.102679, -0.004327, -0.043304, 0.072861,  0.027077,
       0.034558,  0.068292,  -0.036292, 0.069832,  -0.003032, 0.053829,
       -0.043821, -0.072713, 0.085029,  -0.040374, 0.020014,  -0.104521,
       -0.034504, -0.059759, 0.062569,  0.025652,  0.049306,  0.061189,
       -0.025146, 0.079643,  -0.005188, 0.033080,  -0.048079, -0.048082,
       0.069369,  -0.028900, 0.024572,  -0.077547, -0.022517, -0.054477,
       0.038857,  0.013336,  0.043234,  0.044788},
      {-0.039186, 0.070792,  -0.005913, 0.02642,   -0.068274, -0.05022,
       0.061444,  -0.031241, 0.014996,  -0.094544, -0.004146, -0.03464,
       0.058981,  0.026097,  0.039781,  0.058408,  -0.031887, 0.069252,
       0.00576,   0.054062,  -0.042801, -0.059974, 0.085272,  -0.034453,
       0.026097,  -0.0959,   -0.031164, -0.058699, 0.06839,   0.020512,
       0.044727,  0.063609,  -0.039863, 0.084819,  -0.003909, 0.028666,
       -0.075677, -0.045125, 0.070379,  -0.033895, 0.022111,  -0.097184,
       -0.004921, -0.040851, 0.062316,  0.017435,  0.041437,  0.064568,
       -0.039656, 0.060726,  -0.003402, 0.036854,  -0.056503, -0.058554,
       0.068588,  -0.034879, 0.01352,   -0.09962,  -0.01434,  -0.039505,
       0.065133,  0.024321,  0.038473,  0.062438}};

  const int input_sequence_size = lstm.sequence_length() * lstm.num_inputs();
  EXPECT_EQ(input_sequence_size, 20);
  float* batch0_start = lstm_input[0];
  float* batch0_end = batch0_start + input_sequence_size;
  lstm.SetInput(0, batch0_start, batch0_end);

  float* batch1_start = lstm_input[1];
  float* batch1_end = batch1_start + input_sequence_size;
  lstm.SetInput(input_sequence_size, batch1_start, batch1_end);

  lstm.Invoke();

  const int output_sequence_size =
      lstm.sequence_length() * lstm.num_fw_outputs();
  EXPECT_EQ(output_sequence_size, 64);
  std::vector<float> expected;
  const float* golden_start_batch0 = lstm_fw_golden_output[0];
  const float* golden_end_batch0 = golden_start_batch0 + output_sequence_size;
  expected.insert(expected.end(), golden_start_batch0, golden_end_batch0);

  const float* golden_start_batch1 = lstm_fw_golden_output[1];
  const float* golden_end_batch1 = golden_start_batch1 + output_sequence_size;
  expected.insert(expected.end(), golden_start_batch1, golden_end_batch1);
  EXPECT_THAT(lstm.GetFwOutput(), ElementsAreArray(ArrayFloatNear(expected)));

  // Check if the sum of forward backward matches the golden.
  expected.clear();
  golden_start_batch0 = lstm_combined_golden_output[0];
  golden_end_batch0 = golden_start_batch0 + output_sequence_size;
  expected.insert(expected.end(), golden_start_batch0, golden_end_batch0);

  golden_start_batch1 = lstm_combined_golden_output[1];
  golden_end_batch1 = golden_start_batch1 + output_sequence_size;
  expected.insert(expected.end(), golden_start_batch1, golden_end_batch1);

  std::vector<float> combined;
  for (int i = 0; i < lstm.GetFwOutput().size(); ++i) {
    combined.push_back(lstm.GetFwOutput()[i] + lstm.GetBwOutput()[i]);
  }
  EXPECT_THAT(combined, ElementsAreArray(ArrayFloatNear(expected)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
