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
// Unit test for TFLite LSTM op.

#include <iomanip>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class LSTMOpModel : public SingleOpModel {
 public:
  LSTMOpModel(int n_batch, int n_input, int n_cell, int n_output, bool use_cifg,
              bool use_peephole, bool use_projection_weights,
              bool use_projection_bias, float cell_clip, float proj_clip,
              const std::vector<std::vector<int>>& input_shapes)
      : n_batch_(n_batch),
        n_input_(n_input),
        n_cell_(n_cell),
        n_output_(n_output) {
    input_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      input_to_input_weights_ = AddNullInput();
    } else {
      input_to_input_weights_ = AddInput(TensorType_FLOAT32);
    }

    input_to_forget_weights_ = AddInput(TensorType_FLOAT32);
    input_to_cell_weights_ = AddInput(TensorType_FLOAT32);
    input_to_output_weights_ = AddInput(TensorType_FLOAT32);

    if (use_cifg) {
      recurrent_to_input_weights_ = AddNullInput();
    } else {
      recurrent_to_input_weights_ = AddInput(TensorType_FLOAT32);
    }

    recurrent_to_forget_weights_ = AddInput(TensorType_FLOAT32);
    recurrent_to_cell_weights_ = AddInput(TensorType_FLOAT32);
    recurrent_to_output_weights_ = AddInput(TensorType_FLOAT32);

    if (use_peephole) {
      if (use_cifg) {
        cell_to_input_weights_ = AddNullInput();
      } else {
        cell_to_input_weights_ = AddInput(TensorType_FLOAT32);
      }
      cell_to_forget_weights_ = AddInput(TensorType_FLOAT32);
      cell_to_output_weights_ = AddInput(TensorType_FLOAT32);
    } else {
      cell_to_input_weights_ = AddNullInput();
      cell_to_forget_weights_ = AddNullInput();
      cell_to_output_weights_ = AddNullInput();
    }

    if (use_cifg) {
      input_gate_bias_ = AddNullInput();
    } else {
      input_gate_bias_ = AddInput(TensorType_FLOAT32);
    }
    forget_gate_bias_ = AddInput(TensorType_FLOAT32);
    cell_bias_ = AddInput(TensorType_FLOAT32);
    output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      projection_weights_ = AddInput(TensorType_FLOAT32);
      if (use_projection_bias) {
        projection_bias_ = AddInput(TensorType_FLOAT32);
      } else {
        projection_bias_ = AddNullInput();
      }
    } else {
      projection_weights_ = AddNullInput();
      projection_bias_ = AddNullInput();
    }

    scratch_buffer_ = AddOutput(TensorType_FLOAT32);
    // TODO(ghodrat): Modify these states when we have a permanent solution for
    // persistent buffer.
    output_state_ = AddOutput(TensorType_FLOAT32);
    cell_state_ = AddOutput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_LSTM, BuiltinOptions_LSTMOptions,
                 CreateLSTMOptions(builder_, ActivationFunctionType_TANH,
                                   cell_clip, proj_clip)
                     .Union());
    BuildInterpreter(input_shapes);
  }

  void SetInputToInputWeights(std::initializer_list<float> f) {
    PopulateTensor(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(std::initializer_list<float> f) {
    PopulateTensor(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(std::initializer_list<float> f) {
    PopulateTensor(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(std::initializer_list<float> f) {
    PopulateTensor(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(std::initializer_list<float> f) {
    PopulateTensor(recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(std::initializer_list<float> f) {
    PopulateTensor(recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(std::initializer_list<float> f) {
    PopulateTensor(recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(std::initializer_list<float> f) {
    PopulateTensor(recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(std::initializer_list<float> f) {
    PopulateTensor(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(std::initializer_list<float> f) {
    PopulateTensor(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(std::initializer_list<float> f) {
    PopulateTensor(cell_to_output_weights_, f);
  }

  void SetInputGateBias(std::initializer_list<float> f) {
    PopulateTensor(input_gate_bias_, f);
  }

  void SetForgetGateBias(std::initializer_list<float> f) {
    PopulateTensor(forget_gate_bias_, f);
  }

  void SetCellBias(std::initializer_list<float> f) {
    PopulateTensor(cell_bias_, f);
  }

  void SetOutputGateBias(std::initializer_list<float> f) {
    PopulateTensor(output_gate_bias_, f);
  }

  void SetProjectionWeights(std::initializer_list<float> f) {
    PopulateTensor(projection_weights_, f);
  }

  void SetProjectionBias(std::initializer_list<float> f) {
    PopulateTensor(projection_bias_, f);
  }

  void ResetOutputState() {
    const int zero_buffer_size = n_cell_ * n_batch_;
    std::unique_ptr<float[]> zero_buffer(new float[zero_buffer_size]);
    memset(zero_buffer.get(), 0, zero_buffer_size * sizeof(float));
    PopulateTensor(output_state_, 0, zero_buffer.get(),
                   zero_buffer.get() + zero_buffer_size);
  }

  void ResetCellState() {
    const int zero_buffer_size = n_cell_ * n_batch_;
    std::unique_ptr<float[]> zero_buffer(new float[zero_buffer_size]);
    memset(zero_buffer.get(), 0, zero_buffer_size * sizeof(float));
    PopulateTensor(cell_state_, 0, zero_buffer.get(),
                   zero_buffer.get() + zero_buffer_size);
  }

  void SetInput(int offset, float* begin, float* end) {
    PopulateTensor(input_, offset, begin, end);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int num_inputs() { return n_input_; }
  int num_outputs() { return n_output_; }
  int num_cells() { return n_cell_; }
  int num_batches() { return n_batch_; }

 private:
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

  int input_gate_bias_;
  int forget_gate_bias_;
  int cell_bias_;
  int output_gate_bias_;

  int projection_weights_;
  int projection_bias_;

  int output_;
  int output_state_;
  int cell_state_;
  int scratch_buffer_;

  int n_batch_;
  int n_input_;
  int n_cell_;
  int n_output_;
};

TEST(LSTMOpTest, BlackBoxTestNoCifgNoPeepholeNoProjectionNoClipping) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;

  LSTMOpModel lstm(n_batch, n_input, n_cell, n_output,
                   /*use_cifg=*/false, /*use_peephole=*/false,
                   /*use_projection_weights=*/false,
                   /*use_projection_bias=*/false,
                   /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
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

                       {0},  // cell_to_input_weight tensor
                       {0},  // cell_to_forget_weight tensor
                       {0},  // cell_to_output_weight tensor

                       {n_cell},  // input_gate_bias tensor
                       {n_cell},  // forget_gate_bias tensor
                       {n_cell},  // cell_bias tensor
                       {n_cell},  // output_gate_bias tensor

                       {0, 0},  // projection_weight tensor
                       {0},     // projection_bias tensor
                   });

  lstm.SetInputToInputWeights({-0.45018822f, -0.02338299f, -0.0870589f,
                               -0.34550029f, 0.04266912f, -0.15680569f,
                               -0.34856534f, 0.43890524f});

  lstm.SetInputToCellWeights({-0.50013041f, 0.1370284f, 0.11810488f, 0.2013163f,
                              -0.20583314f, 0.44344562f, 0.22077113f,
                              -0.29909778f});

  lstm.SetInputToForgetWeights({0.09701663f, 0.20334584f, -0.50592935f,
                                -0.31343272f, -0.40032279f, 0.44781327f,
                                0.01387155f, -0.35593212f});

  lstm.SetInputToOutputWeights({-0.25065863f, -0.28290087f, 0.04613829f,
                                0.40525138f, 0.44272184f, 0.03897077f, -0.1556896f,
                                0.19487578f});

  lstm.SetInputGateBias({0., 0., 0., 0.});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToInputWeights(
      {-0.0063535f, -0.2042388f, 0.31454784f, -0.35746509f, 0.28902304f, 0.08183324f,
       -0.16555229f, 0.02286911f, -0.13566875f, 0.03034258f, 0.48091322f,
       -0.12528998f, 0.24077177f, -0.51332325f, -0.33502164f, 0.10629296f});

  lstm.SetRecurrentToCellWeights(
      {-0.3407414f, 0.24443203f, -0.2078532f, 0.26320225f, 0.05695659f, -0.00123841f,
       -0.4744786f, -0.35869038f, -0.06418842f, -0.13502428f, -0.501764f, 0.22830659f,
       -0.46367589f, 0.26016325f, -0.03894562f, -0.16368064f});

  lstm.SetRecurrentToForgetWeights(
      {-0.48684245f, -0.06655136f, 0.42224967f, 0.2112639f, 0.27654213f, 0.20864892f,
       -0.07646349f, 0.45877004f, 0.00141793f, -0.14609534f, 0.36447752f, 0.09196436f,
       0.28053468f, 0.01560611f, -0.20127171f, -0.01140004f});

  lstm.SetRecurrentToOutputWeights(
      {0.43385774f, -0.17194885f, 0.2718237f, 0.09215671f, 0.24107647f, -0.39835793f,
       0.18212086f, 0.01301402f, 0.48572797f, -0.50656658f, 0.20047462f, -0.20607421f,
       -0.51818722f, -0.15390486f, 0.0468148f, 0.39922136f});

  static float lstm_input[] = {2., 3., 3., 4., 1., 1.};
  static float lstm_golden_output[] = {-0.02973187f, 0.1229473f,   0.20885126f,
                                       -0.15358765f, -0.03716109f, 0.12507336f,
                                       0.41193449f,  -0.20860538f, -0.15053082f,
                                       0.09120187f,  0.24278517f,  -0.12222792f};

  // Resetting cell_state and output_state
  lstm.ResetCellState();
  lstm.ResetOutputState();

  const int input_sequence_size =
      sizeof(lstm_input) / sizeof(float) / (lstm.num_inputs());
  for (int i = 0; i < input_sequence_size; i++) {
    float* batch0_start = lstm_input + i * lstm.num_inputs();
    float* batch0_end = batch0_start + lstm.num_inputs();

    lstm.SetInput(0, batch0_start, batch0_end);

    lstm.Invoke();

    float* golden_start = lstm_golden_output + i * lstm.num_outputs();
    float* golden_end = golden_start + lstm.num_outputs();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    EXPECT_THAT(lstm.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

TEST(LSTMOpTest, BlackBoxTestWithCifgWithPeepholeNoProjectionNoClipping) {
  const int n_batch = 1;
  const int n_input = 2;
  // n_cell and n_output have the same size when there is no projection.
  const int n_cell = 4;
  const int n_output = 4;

  LSTMOpModel lstm(n_batch, n_input, n_cell, n_output,
                   /*use_cifg=*/true, /*use_peephole=*/true,
                   /*use_projection_weights=*/false,
                   /*use_projection_bias=*/false,
                   /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
                   {
                       {n_batch, n_input},  // input tensor

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
                   });

  lstm.SetInputToCellWeights({-0.49770179f, -0.27711356f, -0.09624726f, 0.05100781f,
                              0.04717243f, 0.48944736f, -0.38535351f,
                              -0.17212132f});

  lstm.SetInputToForgetWeights({-0.55291498f, -0.42866567f, 0.13056988f,
                                -0.3633365f, -0.22755712f, 0.28253698f, 0.24407166f,
                                0.33826375f});

  lstm.SetInputToOutputWeights({0.10725588f, -0.02335852f, -0.55932593f,
                                -0.09426838f, -0.44257352f, 0.54939759f,
                                0.01533556f, 0.42751634f});

  lstm.SetCellBias({0., 0., 0., 0.});

  lstm.SetForgetGateBias({1., 1., 1., 1.});

  lstm.SetOutputGateBias({0., 0., 0., 0.});

  lstm.SetRecurrentToCellWeights(
      {0.54066205f, -0.32668582f, -0.43562764f, -0.56094903f, 0.42957711f,
       0.01841056f, -0.32764608f, -0.33027974f, -0.10826075f, 0.20675004f,
       0.19069612f, -0.03026325f, -0.54532051f, 0.33003211f, 0.44901288f,
       0.21193194f});

  lstm.SetRecurrentToForgetWeights(
      {-0.13832897f, -0.0515101f, -0.2359007f, -0.16661474f, -0.14340827f,
       0.36986142f, 0.23414481f, 0.55899f, 0.10798943f, -0.41174671f, 0.17751795f,
       -0.34484994f, -0.35874045f, -0.11352962f, 0.27268326f, 0.54058349f});

  lstm.SetRecurrentToOutputWeights(
      {0.41613156f, 0.42610586f, -0.16495961f, -0.5663873f, 0.30579174f, -0.05115908f,
       -0.33941799f, 0.23364776f, 0.11178309f, 0.09481031f, -0.26424935f, 0.46261835f,
       0.50248802f, 0.26114327f, -0.43736315f, 0.33149987f});

  lstm.SetCellToForgetWeights(
      {0.47485286f, -0.51955009f, -0.24458408f, 0.31544167f});
  lstm.SetCellToOutputWeights(
      {-0.17135078f, 0.82760304f, 0.85573703f, -0.77109635f});

  static float lstm_input[] = {2., 3., 3., 4., 1., 1.};
  static float lstm_golden_output[] = {-0.36444446f, -0.00352185f, 0.12886585f,
                                       -0.05163646f, -0.42312205f, -0.01218222f,
                                       0.24201041f,  -0.08124574f, -0.358325f,
                                       -0.04621704f, 0.21641694f,  -0.06471302f};

  // Resetting cell_state and output_state
  lstm.ResetCellState();
  lstm.ResetOutputState();

  const int input_sequence_size =
      sizeof(lstm_input) / sizeof(float) / (lstm.num_inputs());
  for (int i = 0; i < input_sequence_size; i++) {
    float* batch0_start = lstm_input + i * lstm.num_inputs();
    float* batch0_end = batch0_start + lstm.num_inputs();

    lstm.SetInput(0, batch0_start, batch0_end);

    lstm.Invoke();

    float* golden_start = lstm_golden_output + i * lstm.num_outputs();
    float* golden_end = golden_start + lstm.num_outputs();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start, golden_end);
    EXPECT_THAT(lstm.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

TEST(LSTMOpTest, BlackBoxTestWithPeepholeWithProjectionNoClipping) {
  const int n_batch = 2;
  const int n_input = 5;
  const int n_cell = 20;
  const int n_output = 16;

  LSTMOpModel lstm(n_batch, n_input, n_cell, n_output,
                   /*use_cifg=*/false, /*use_peephole=*/true,
                   /*use_projection_weights=*/true,
                   /*use_projection_bias=*/false,
                   /*cell_clip=*/0.0f, /*proj_clip=*/0.0f,
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

                       {n_cell},  // input_gate_bias tensor
                       {n_cell},  // forget_gate_bias tensor
                       {n_cell},  // cell_bias tensor
                       {n_cell},  // output_gate_bias tensor

                       {n_output, n_cell},  // projection_weight tensor
                       {0},                 // projection_bias tensor
                   });

  lstm.SetInputToInputWeights(
      {0.021393683f,  0.06124551f,    0.046905167f,  -0.014657677f,  -0.03149463f,
       0.09171803f,   0.14647801f,    0.10797193f,   -0.0057968358f, 0.0019193048f,
       -0.2726754f,   0.10154029f,    -0.018539885f, 0.080349885f,   -0.10262385f,
       -0.022599787f, -0.09121155f,   -0.008675967f, -0.045206103f,  -0.0821282f,
       -0.008045952f, 0.015478081f,   0.055217247f,  0.038719587f,   0.044153627f,
       -0.06453243f,  0.05031825f,    -0.046935108f, -0.008164439f,  0.014574226f,
       -0.1671009f,   -0.15519552f,   -0.16819797f,  -0.13971269f,   -0.11953059f,
       0.25005487f,   -0.22790983f,   0.009855087f,  -0.028140958f,  -0.11200698f,
       0.11295408f,   -0.0035217577f, 0.054485075f,  0.05184695f,    0.064711206f,
       0.10989193f,   0.11674786f,    0.03490607f,   0.07727357f,    0.11390585f,
       -0.1863375f,   -0.1034451f,    -0.13945189f,  -0.049401227f,  -0.18767063f,
       0.042483903f,  0.14233552f,    0.13832581f,   0.18350165f,    0.14545603f,
       -0.028545704f, 0.024939531f,   0.050929718f,  0.0076203286f,  -0.0029723682f,
       -0.042484224f, -0.11827596f,   -0.09171104f,  -0.10808628f,   -0.16327988f,
       -0.2273378f,   -0.0993647f,    -0.017155107f, 0.0023917493f,  0.049272764f,
       0.0038534778f, 0.054764505f,   0.089753784f,  0.06947234f,    0.08014476f,
       -0.04544234f,  -0.0497073f,    -0.07135631f,  -0.048929106f,  -0.004042012f,
       -0.009284026f, 0.018042054f,   0.0036860977f, -0.07427302f,   -0.11434604f,
       -0.018995456f, 0.031487543f,   0.012834908f,  0.019977754f,   0.044256654f,
       -0.39292613f,  -0.18519334f,   -0.11651281f,  -0.06809892f,   0.011373677f});

  lstm.SetInputToForgetWeights(
      {-0.0018401089f, -0.004852237f,  0.03698424f,   0.014181704f,   0.028273236f,
       -0.016726194f,  -0.05249759f,   -0.10204261f,  0.00861066f,    -0.040979505f,
       -0.009899187f,  0.01923892f,    -0.028177269f, -0.08535103f,   -0.14585495f,
       0.10662567f,    -0.01909731f,   -0.017883534f, -0.0047269356f, -0.045103323f,
       0.0030784295f,  0.076784775f,   0.07463696f,   0.094531395f,   0.0814421f,
       -0.12257899f,   -0.033945758f,  -0.031303465f, 0.045630626f,   0.06843887f,
       -0.13492945f,   -0.012480007f,  -0.0811829f,   -0.07224499f,   -0.09628791f,
       0.045100946f,   0.0012300825f,  0.013964662f,  0.099372394f,   0.02543059f,
       0.06958324f,    0.034257296f,   0.0482646f,    0.06267997f,    0.052625068f,
       0.12784666f,    0.07077897f,    0.025725935f,  0.04165009f,    0.07241905f,
       0.018668644f,   -0.037377294f,  -0.06277783f,  -0.08833636f,   -0.040120605f,
       -0.011405586f,  -0.007808335f,  -0.010301386f, -0.005102167f,  0.027717464f,
       0.05483423f,    0.11449111f,    0.11289652f,   0.10939839f,    0.13396506f,
       -0.08402166f,   -0.01901462f,   -0.044678304f, -0.07720565f,   0.014350063f,
       -0.11757958f,   -0.0652038f,    -0.08185733f,  -0.076754324f,  -0.092614375f,
       0.10405491f,    0.052960336f,   0.035755895f,  0.035839386f,   -0.012540553f,
       0.036881298f,   0.02913376f,    0.03420159f,   0.05448447f,    -0.054523353f,
       0.02582715f,    0.02327355f,    -0.011857179f, -0.0011980024f, -0.034641717f,
       -0.026125094f,  -0.17582615f,   -0.15923657f,  -0.27486774f,   -0.0006143371f,
       0.0001771948f,  -8.470171e-05f, 0.02651807f,   0.045790765f,   0.06956496f});

  lstm.SetInputToCellWeights(
      {-0.04580283f,   -0.09549462f,   -0.032418985f,  -0.06454633f,
       -0.043528453f,  0.043018587f,   -0.049152344f,  -0.12418144f,
       -0.078985475f,  -0.07596889f,   0.019484362f,   -0.11434962f,
       -0.0074034138f, -0.06314844f,   -0.092981495f,  0.0062155537f,
       -0.025034338f,  -0.0028890965f, 0.048929527f,   0.06235075f,
       0.10665918f,    -0.032036792f,  -0.08505916f,   -0.10843358f,
       -0.13002433f,   -0.036816437f,  -0.02130134f,   -0.016518239f,
       0.0047691227f,  -0.0025825808f, 0.066017866f,   0.029991534f,
       -0.10652836f,   -0.1037554f,    -0.13056071f,   -0.03266643f,
       -0.033702414f,  -0.006473424f,  -0.04611692f,   0.014419339f,
       -0.025174323f,  0.0396852f,     0.081777506f,   0.06157468f,
       0.10210095f,    -0.009658194f,  0.046511717f,   0.03603906f,
       0.0069369148f,  0.015960095f,   -0.06507666f,   0.09551598f,
       0.053568836f,   0.06408714f,    0.12835667f,    -0.008714329f,
       -0.20211966f,   -0.12093674f,   0.029450472f,   0.2849013f,
       -0.029227901f,  0.1164364f,     -0.08560263f,   0.09941786f,
       -0.036999565f,  -0.028842626f,  -0.0033637602f, -0.017012902f,
       -0.09720865f,   -0.11193351f,   -0.029155117f,  -0.017936034f,
       -0.009768936f,  -0.04223324f,   -0.036159635f,  0.06505112f,
       -0.021742892f,  -0.023377212f,  -0.07221364f,   -0.06430552f,
       0.05453865f,    0.091149814f,   0.06387331f,    0.007518393f,
       0.055960953f,   0.069779344f,   0.046411168f,   0.10509911f,
       0.07463894f,    0.0075130584f,  0.012850982f,   0.04555431f,
       0.056955688f,   0.06555285f,    0.050801456f,   -0.009862683f,
       0.00826772f,    -0.026555609f,  -0.0073611983f, -0.0014897042f});

  lstm.SetInputToOutputWeights(
      {-0.0998932f,   -0.07201956f,  -0.052803773f,  -0.15629593f,  -0.15001918f,
       -0.07650751f,  0.02359855f,   -0.075155355f,  -0.08037709f,  -0.15093534f,
       0.029517552f,  -0.04751393f,  0.010350531f,   -0.02664851f,  -0.016839722f,
       -0.023121163f, 0.0077019283f, 0.012851257f,   -0.05040649f,  -0.0129761f,
       -0.021737747f, -0.038305793f, -0.06870586f,   -0.01481247f,  -0.001285394f,
       0.10124236f,   0.083122835f,  0.053313006f,   -0.062235646f, -0.075637154f,
       -0.027833903f, 0.029774971f,  0.1130802f,     0.09218906f,   0.09506135f,
       -0.086665764f, -0.037162706f, -0.038880914f,  -0.035832845f, -0.014481564f,
       -0.09825003f,  -0.12048569f,  -0.097665586f,  -0.05287633f,  -0.0964047f,
       -0.11366429f,  0.035777505f,  0.13568819f,    0.052451383f,  0.050649304f,
       0.05798951f,   -0.021852335f, -0.099848844f,  0.014740475f,  -0.078897946f,
       0.04974699f,   0.014160473f,  0.06973932f,    0.04964942f,   0.033364646f,
       0.08190124f,   0.025535367f,  0.050893165f,   0.048514254f,  0.06945813f,
       -0.078907564f, -0.06707616f,  -0.11844508f,   -0.09986688f,  -0.07509403f,
       0.06263226f,   0.14925587f,   0.20188436f,    0.12098451f,   0.14639415f,
       0.0015017595f, -0.014267382f, -0.03417257f,   0.012711468f,  0.0028300495f,
       -0.024758482f, -0.05098548f,  -0.0821182f,    0.014225672f,  0.021544158f,
       0.08949725f,   0.07505268f,   -0.0020780868f, 0.04908258f,   0.06476295f,
       -0.022907063f, 0.027562456f,  0.040185735f,   0.019567577f,  -0.015598739f,
       -0.049097303f, -0.017121866f, -0.083368234f,  -0.02332002f,  -0.0840956f});

  lstm.SetInputGateBias(
      {0.02234832f,  0.14757581f,   0.18176508f,  0.10380666f,  0.053110216f,
       -0.06928846f, -0.13942584f,  -0.11816189f, 0.19483899f,  0.03652339f,
       -0.10250295f, 0.036714908f,  -0.18426876f, 0.036065217f, 0.21810818f,
       0.02383196f,  -0.043370757f, 0.08690144f,  -0.04444982f, 0.00030581196f});

  lstm.SetForgetGateBias({0.035185695f, -0.042891346f, -0.03032477f, 0.23027696f,
                          0.11098921f,  0.15378423f,   0.09263801f,  0.09790885f,
                          0.09508917f,  0.061199076f,  0.07665568f,  -0.015443159f,
                          -0.03499149f, 0.046190713f,  0.08895977f,  0.10899629f,
                          0.40694186f,  0.06030037f,   0.012413437f, -0.06108739f});

  lstm.SetCellBias({-0.024379363f, 0.0055531194f, 0.23377132f,   0.033463873f,
                    -0.1483596f,   -0.10639995f,  -0.091433935f, 0.058573797f,
                    -0.06809782f,  -0.07889636f,  -0.043246906f, -0.09829136f,
                    -0.4279842f,   0.034901652f,  0.18797937f,   0.0075234566f,
                    0.016178843f,  0.1749513f,    0.13975595f,   0.92058027f});

  lstm.SetOutputGateBias(
      {0.046159424f,  -0.0012809046f, 0.03563469f,   0.12648113f, 0.027195795f,
       0.35373217f,   -0.018957434f,  0.008907322f,  -0.0762701f, 0.12018895f,
       0.04216877f,   0.0022856654f,  0.040952638f,  0.3147856f,  0.08225149f,
       -0.057416286f, -0.14995944f,   -0.008040261f, 0.13208859f, 0.029760877f});

  lstm.SetRecurrentToInputWeights(
      {-0.001374326f,   -0.078856036f,   0.10672688f,    0.029162422f,
       -0.11585556f,    0.02557986f,     -0.13446963f,   -0.035785314f,
       -0.01244275f,    0.025961924f,    -0.02337298f,   -0.044228926f,
       -0.055839065f,   -0.046598054f,   -0.010546039f,  -0.06900766f,
       0.027239809f,    0.022582639f,    -0.013296484f,  -0.05459212f,
       0.08981f,        -0.045407712f,   0.08682226f,    -0.06867011f,
       -0.14390695f,    -0.02916037f,    0.000996957f,   0.091420636f,
       0.14283475f,     -0.07390571f,    -0.06402044f,   0.062524505f,
       -0.093129106f,   0.04860203f,     -0.08364217f,   -0.08119002f,
       0.009352075f,    0.22920375f,     0.0016303885f,  0.11583097f,
       -0.13732095f,    0.012405723f,    -0.07551853f,   0.06343048f,
       0.12162708f,     -0.031923793f,   -0.014335606f,  0.01790974f,
       -0.10650317f,    -0.0724401f,     0.08554849f,    -0.05727212f,
       0.06556731f,     -0.042729504f,   -0.043227166f,  0.011683251f,
       -0.013082158f,   -0.029302018f,   -0.010899579f,  -0.062036745f,
       -0.022509435f,   -0.00964907f,    -0.01567329f,   0.04260106f,
       -0.07787477f,    -0.11576462f,    0.017356863f,   0.048673786f,
       -0.017577527f,   -0.05527947f,    -0.082487635f,  -0.040137455f,
       -0.10820036f,    -0.04666372f,    0.022746278f,   -0.07851417f,
       0.01068115f,     0.032956902f,    0.022433773f,   0.0026891115f,
       0.08944216f,     -0.0685835f,     0.010513544f,   0.07228705f,
       0.02032331f,     -0.059686817f,   -0.0005566496f, -0.086984694f,
       0.040414046f,    -0.1380399f,     0.094208956f,   -0.05722982f,
       0.012092817f,    -0.04989123f,    -0.086576f,     -0.003399834f,
       -0.04696032f,    -0.045747425f,   0.10091314f,    0.048676282f,
       -0.029037097f,   0.031399418f,    -0.0040285117f, 0.047237843f,
       0.09504992f,     0.041799378f,    -0.049185462f,  -0.031518843f,
       -0.10516937f,    0.026374253f,    0.10058866f,    -0.0033195973f,
       -0.041975245f,   0.0073591834f,   0.0033782164f,  -0.004325073f,
       -0.10167381f,    0.042500053f,    -0.01447153f,   0.06464186f,
       -0.017142897f,   0.03312627f,     0.009205989f,   0.024138335f,
       -0.011337001f,   0.035530265f,    -0.010912711f,  0.0706555f,
       -0.005894094f,   0.051841937f,    -0.1401738f,    -0.02351249f,
       0.0365468f,      0.07590991f,     0.08838724f,    0.021681072f,
       -0.10086113f,    0.019608743f,    -0.06195883f,   0.077335775f,
       0.023646897f,    -0.095322326f,   0.02233014f,    0.09756986f,
       -0.048691444f,   -0.009579111f,   0.07595467f,    0.11480546f,
       -0.09801813f,    0.019894179f,    0.08502348f,    0.004032281f,
       0.037211012f,    0.068537936f,    -0.048005626f,  -0.091520436f,
       -0.028379958f,   -0.01556313f,    0.06554592f,    -0.045599163f,
       -0.01672207f,    -0.020169014f,   -0.011877351f,  -0.20212261f,
       0.010889619f,    0.0047078193f,   0.038385306f,   0.08540671f,
       -0.017140968f,   -0.0035865551f,  0.016678626f,   0.005633034f,
       0.015963363f,    0.00871737f,     0.060130805f,   0.028611384f,
       0.10109069f,     -0.015060172f,   -0.07894427f,   0.06401885f,
       0.011584063f,    -0.024466386f,   0.0047652307f,  -0.09041358f,
       0.030737216f,    -0.0046374933f,  0.14215417f,    -0.11823516f,
       0.019899689f,    0.006106124f,    -0.027092824f,  0.0786356f,
       0.05052217f,     -0.058925f,      -0.011402121f,  -0.024987547f,
       -0.0013661642f,  -0.06832946f,    -0.015667673f,  -0.1083353f,
       -0.00096863037f, -0.06988685f,    -0.053350925f,  -0.027275559f,
       -0.033664223f,   -0.07978348f,    -0.025200296f,  -0.017207067f,
       -0.058403496f,   -0.055697463f,   0.005798788f,   0.12965427f,
       -0.062582195f,   0.0013350133f,   -0.10482091f,   0.0379771f,
       0.072521195f,    -0.0029455067f,  -0.13797039f,   -0.03628521f,
       0.013806405f,    -0.017858358f,   -0.01008298f,   -0.07700066f,
       -0.017081132f,   0.019358726f,    0.0027079724f,  0.004635139f,
       0.062634714f,    -0.02338735f,    -0.039547626f,  -0.02050681f,
       0.03385117f,     -0.083611414f,   0.002862572f,   -0.09421313f,
       0.058618143f,    -0.08598433f,    0.00972939f,    0.023867095f,
       -0.053934585f,   -0.023203006f,   0.07452513f,    -0.048767887f,
       -0.07314807f,    -0.056307215f,   -0.10433547f,   -0.06440842f,
       0.04328182f,     0.04389765f,     -0.020006588f,  -0.09076438f,
       -0.11652589f,    -0.021705797f,   0.03345259f,    -0.010329105f,
       -0.025767034f,   0.013057034f,    -0.07316461f,   -0.10145612f,
       0.06358255f,     0.18531723f,     0.07759293f,    0.12006465f,
       0.1305557f,      0.058638252f,    -0.03393652f,   0.09622831f,
       -0.16253184f,    -2.4580743e-06f, 0.079869635f,   -0.070196845f,
       -0.005644518f,   0.06857898f,     -0.12598175f,   -0.035084512f,
       0.03156317f,     -0.12794146f,    -0.031963028f,  0.04692781f,
       0.030070418f,    0.0071660685f,   -0.095516115f,  -0.004643372f,
       0.040170413f,    -0.062104587f,   -0.0037324072f, 0.0554317f,
       0.08184801f,     -0.019164372f,   0.06791302f,    0.034257166f,
       -0.10307039f,    0.021943003f,    0.046745934f,   0.0790918f,
       -0.0265588f,     -0.007824208f,   0.042546265f,   -0.00977924f,
       -0.0002440307f,  -0.017384544f,   -0.017990116f,  0.12252321f,
       -0.014512694f,   -0.08251313f,    0.08861942f,    0.13589665f,
       0.026351685f,    0.012641483f,    0.07466548f,    0.044301085f,
       -0.045414884f,   -0.051112458f,   0.03444247f,    -0.08502782f,
       -0.04106223f,    -0.028126027f,   0.028473156f,   0.10467447f});

  lstm.SetRecurrentToForgetWeights(
      {-0.057784554f,  -0.026057621f,  -0.068447545f,   -0.022581743f,
       0.14811787f,    0.10826372f,    0.09471067f,     0.03987225f,
       -0.0039523416f, 0.00030638507f, 0.053185795f,    0.10572994f,
       0.08414449f,    -0.022036452f,  -0.00066928595f, -0.09203576f,
       0.032950465f,   -0.10985798f,   -0.023809856f,   0.0021431844f,
       -0.02196096f,   -0.00326074f,   0.00058621005f,  -0.074678116f,
       -0.06193199f,   0.055729095f,   0.03736828f,     0.020123724f,
       0.061878487f,   -0.04729229f,   0.034919553f,    -0.07585433f,
       -0.04421272f,   -0.044019096f,  0.085488975f,    0.04058006f,
       -0.06890133f,   -0.030951202f,  -0.024628663f,   -0.07672815f,
       0.034293607f,   0.08556707f,    -0.05293577f,    -0.033561368f,
       -0.04899627f,   0.0241671f,     0.015736353f,    -0.095442444f,
       -0.029564252f,  0.016493602f,   -0.035026584f,   0.022337519f,
       -0.026871363f,  0.004780428f,   0.0077918363f,   -0.03601621f,
       0.016435321f,   -0.03263031f,   -0.09543275f,    -0.047392778f,
       0.013454138f,   0.028934088f,   0.01685226f,     -0.086110644f,
       -0.046250615f,  -0.01847454f,   0.047608484f,    0.07339695f,
       0.034546845f,   -0.04881143f,   0.009128804f,    -0.08802852f,
       0.03761666f,    0.008096139f,   -0.014454086f,   0.014361001f,
       -0.023502491f,  -0.0011840804f, -0.07607001f,    0.001856849f,
       -0.06509276f,   -0.006021153f,  -0.08570962f,    -0.1451793f,
       0.060212336f,   0.055259194f,   0.06974018f,     0.049454916f,
       -0.027794661f,  -0.08077226f,   -0.016179763f,   0.1169753f,
       0.17213494f,    -0.0056326236f, -0.053934924f,   -0.0124349f,
       -0.11520337f,   0.05409887f,    0.088759385f,    0.0019655675f,
       0.0042065294f,  0.03881498f,    0.019844765f,    0.041858196f,
       -0.05695512f,   0.047233116f,   0.038937137f,    -0.06542224f,
       0.014429736f,   -0.09719407f,   0.13908425f,     -0.05379757f,
       0.012321099f,   0.082840554f,   -0.029899208f,   0.044217527f,
       0.059855383f,   0.07711018f,    -0.045319796f,   0.0948846f,
       -0.011724666f,  -0.0033288454f, -0.033542685f,   -0.04764985f,
       -0.13873616f,   0.040668588f,   0.034832682f,    -0.015319203f,
       -0.018715994f,  0.046002675f,   0.0599172f,      -0.043107376f,
       0.0294216f,     -0.002314414f,  -0.022424703f,   0.0030315618f,
       0.0014641669f,  0.0029166266f,  -0.11878115f,    0.013738511f,
       0.12375372f,    -0.0006038222f, 0.029104086f,    0.087442465f,
       0.052958444f,   0.07558703f,    0.04817258f,     0.044462286f,
       -0.015213451f,  -0.08783778f,   -0.0561384f,     -0.003008196f,
       0.047060397f,   -0.002058388f,  0.03429439f,     -0.018839769f,
       0.024734668f,   0.024614193f,   -0.042046934f,   0.09597743f,
       -0.0043254104f, 0.04320769f,    0.0064070094f,   -0.0019131786f,
       -0.02558259f,   -0.022822596f,  -0.023273505f,   -0.02464396f,
       -0.10991725f,   -0.006240552f,  0.0074488563f,   0.024044557f,
       0.04383914f,    -0.046476185f,  0.028658995f,    0.060410924f,
       0.050786525f,   0.009452605f,   -0.0073054377f,  -0.024810238f,
       0.0052906186f,  0.0066939713f,  -0.0020913032f,  0.014515517f,
       0.015898481f,   0.021362653f,   -0.030262267f,   0.016587038f,
       -0.011442813f,  0.041154444f,   -0.007631438f,   -0.03423484f,
       -0.010977775f,  0.036152758f,   0.0066366293f,   0.11915515f,
       0.02318443f,    -0.041350313f,  0.021485701f,    -0.10906167f,
       -0.028218046f,  -0.00954771f,   0.020531068f,    -0.11995105f,
       -0.03672871f,   0.024019798f,   0.014255957f,    -0.05221243f,
       -0.00661567f,   -0.04630967f,   0.033188973f,    0.10107534f,
       -0.014027541f,  0.030796422f,   -0.10270911f,    -0.035999842f,
       0.15443139f,    0.07684145f,    0.036571592f,    -0.035900835f,
       -0.0034699554f, 0.06209149f,    0.015920248f,    -0.031122351f,
       -0.03858649f,   0.01849943f,    0.13872518f,     0.01503974f,
       0.069941424f,   -0.06948533f,   -0.0088794185f,  0.061282158f,
       -0.047401894f,  0.03100163f,    -0.041533746f,   -0.10430945f,
       0.044574402f,   -0.01425562f,   -0.024290353f,   0.034563623f,
       0.05866852f,    0.023947537f,   -0.09445152f,    0.035450947f,
       0.02247216f,    -0.0042998926f, 0.061146557f,    -0.10250651f,
       0.020881841f,   -0.06747029f,   0.10062043f,     -0.0023941975f,
       0.03532124f,    -0.016341697f,  0.09685456f,     -0.016764693f,
       0.051808182f,   0.05875331f,    -0.04536488f,    0.001626336f,
       -0.028892258f,  -0.01048663f,   -0.009793449f,   -0.017093895f,
       0.010987891f,   0.02357273f,    -0.00010856845f, 0.0099760275f,
       -0.001845119f,  -0.03551521f,   0.0018358806f,   0.05763657f,
       -0.01769146f,   0.040995963f,   0.02235177f,     -0.060430344f,
       0.11475477f,    -0.023854522f,  0.10071741f,     0.0686208f,
       -0.014250481f,  0.034261297f,   0.047418304f,    0.08562733f,
       -0.030519066f,  0.0060542435f,  0.014653856f,    -0.038836084f,
       0.04096551f,    0.032249358f,   -0.08355519f,    -0.026823482f,
       0.056386515f,   -0.010401743f,  -0.028396193f,   0.08507674f,
       0.014410365f,   0.020995233f,   0.17040324f,     0.11511526f,
       0.02459721f,    0.0066619175f,  0.025853224f,    -0.023133837f,
       -0.081302024f,  0.017264642f,   -0.009585969f,   0.09491168f,
       -0.051313367f,  0.054532815f,   -0.014298593f,   0.10657464f,
       0.007076659f,   0.10964551f,    0.0409152f,      0.008275321f,
       -0.07283536f,   0.07937492f,    0.04192024f,     -0.1075027f});

  lstm.SetRecurrentToCellWeights(
      {-0.037322544f,   0.018592842f,   0.0056175636f,  -0.06253426f,
       0.055647098f,    -0.05713207f,   -0.05626563f,   0.005559383f,
       0.03375411f,     -0.025757805f,  -0.088049285f,  0.06017052f,
       -0.06570978f,    0.007384076f,   0.035123326f,   -0.07920549f,
       0.053676967f,    0.044480428f,   -0.07663568f,   0.0071805613f,
       0.08089997f,     0.05143358f,    0.038261272f,   0.03339287f,
       -0.027673481f,   0.044746667f,   0.028349208f,   0.020090483f,
       -0.019443132f,   -0.030755889f,  -0.0040000007f, 0.04465846f,
       -0.021585021f,   0.0031670958f,  0.0053199246f,  -0.056117613f,
       -0.10893326f,    0.076739706f,   -0.08509834f,   -0.027997585f,
       0.037871376f,    0.01449768f,    -0.09002357f,   -0.06111149f,
       -0.046195522f,   0.0422062f,     -0.005683705f,  -0.1253618f,
       -0.012925729f,   -0.04890792f,   0.06985068f,    0.037654128f,
       0.03398274f,     -0.004781977f,  0.007032333f,   -0.031787455f,
       0.010868644f,    -0.031489216f,  0.09525667f,    0.013939797f,
       0.0058680447f,   0.0167067f,     0.02668468f,    -0.04797466f,
       -0.048885044f,   -0.12722108f,   0.035304096f,   0.06554885f,
       0.00972396f,     -0.039238118f,  -0.05159735f,   -0.11329045f,
       0.1613692f,      -0.03750952f,   0.06529313f,    -0.071974665f,
       -0.11769596f,    0.015524369f,   -0.0013754242f, -0.12446318f,
       0.02786344f,     -0.014179351f,  0.005264273f,   0.14376344f,
       0.015983658f,    0.03406988f,    -0.06939408f,   0.040699873f,
       0.02111075f,     0.09669095f,    0.041345075f,   -0.08316494f,
       -0.07684199f,    -0.045768797f,  0.032298047f,   -0.041805092f,
       0.0119405f,      0.0061010392f,  0.12652606f,    0.0064572375f,
       -0.024950314f,   0.11574242f,    0.04508852f,    -0.04335324f,
       0.06760663f,     -0.027437469f,  0.07216407f,    0.06977076f,
       -0.05438599f,    0.034033038f,   -0.028602652f,  0.05346137f,
       0.043184172f,    -0.037189785f,  0.10420091f,    0.00882477f,
       -0.054019816f,   -0.074273005f,  -0.030617684f,  -0.0028467078f,
       0.024302477f,    -0.0038869337f, 0.005332455f,   0.0013399826f,
       0.04361412f,     -0.007001822f,  0.09631092f,    -0.06702025f,
       -0.042049985f,   -0.035070654f,  -0.04103342f,   -0.10273396f,
       0.0544271f,      0.037184782f,   -0.13150354f,   -0.0058036847f,
       -0.008264958f,   0.042035464f,   0.05891794f,    0.029673764f,
       0.0063542654f,   0.044788733f,   0.054816857f,   0.062257513f,
       -0.00093483756f, 0.048938446f,   -0.004952862f,  -0.007730018f,
       -0.04043371f,    -0.017094059f,  0.07229206f,    -0.023670016f,
       -0.052195564f,   -0.025616996f,  -0.01520939f,   0.045104615f,
       -0.007376126f,   0.003533447f,   0.006570588f,   0.056037236f,
       0.12436656f,     0.051817212f,   0.028532185f,   -0.08686856f,
       0.11868599f,     0.07663395f,    -0.07323171f,   0.03463402f,
       -0.050708205f,   -0.04458982f,   -0.11590894f,   0.021273347f,
       0.1251325f,      -0.15313013f,   -0.12224372f,   0.17228661f,
       0.023029093f,    0.086124025f,   0.006445803f,   -0.03496501f,
       0.028332196f,    0.04449512f,    -0.042436164f,  -0.026587414f,
       -0.006041347f,   -0.09292539f,   -0.05678812f,   0.03897832f,
       0.09465633f,     0.008115513f,   -0.02171956f,   0.08304309f,
       0.071401566f,    0.019622514f,   0.032163795f,   -0.004167056f,
       0.02295182f,     0.030739572f,   0.056506045f,   0.004612461f,
       0.06524936f,     0.059999723f,   0.046395954f,   -0.0045512207f,
       -0.1335546f,     -0.030136576f,  0.11584653f,    -0.014678886f,
       0.0020118146f,   -0.09688814f,   -0.0790206f,    0.039770417f,
       -0.0329582f,     0.07922767f,    0.029322514f,   0.026405897f,
       0.04207835f,     -0.07073373f,   0.063781224f,   0.0859677f,
       -0.10925287f,    -0.07011058f,   0.048005477f,   0.03438226f,
       -0.09606514f,    -0.006669445f,  -0.043381985f,  0.04240257f,
       -0.06955775f,    -0.06769346f,   0.043903265f,   -0.026784198f,
       -0.017840602f,   0.024307009f,   -0.040079936f,  -0.019946516f,
       0.045318738f,    -0.12233574f,   0.026170589f,   0.0074471775f,
       0.15978073f,     0.10185836f,    0.10298046f,    -0.015476589f,
       -0.039390966f,   -0.072174534f,  0.0739445f,     -0.1211869f,
       -0.0347889f,     -0.07943156f,   0.014809798f,   -0.12412325f,
       -0.0030663363f,  0.039695457f,   0.0647603f,     -0.08291318f,
       -0.018529687f,   -0.004423833f,  0.0037507233f,  0.084633216f,
       -0.01514876f,    -0.056505352f,  -0.012800942f,  -0.06994386f,
       0.012962922f,    -0.031234352f,  0.07029052f,    0.016418684f,
       0.03618972f,     0.055686004f,   -0.08663945f,   -0.017404709f,
       -0.054761406f,   0.029065743f,   0.052404847f,   0.020238016f,
       0.0048197987f,   -0.0214882f,    0.07078733f,    0.013016777f,
       0.06262858f,     0.009184685f,   0.020785125f,   -0.043904778f,
       -0.0270329f,     -0.03299152f,   -0.060088247f,  -0.015162964f,
       -0.001828936f,   0.12642565f,    -0.056757294f,  0.013586685f,
       0.09232601f,     -0.035886683f,  0.06000002f,    0.05229691f,
       -0.052580316f,   -0.082029596f,  -0.010794592f,  0.012947712f,
       -0.036429964f,   -0.085508935f,  -0.13127148f,   -0.017744139f,
       0.031502828f,    0.036232427f,   -0.031581745f,  0.023051167f,
       -0.05325106f,    -0.03421577f,   0.028793324f,   -0.034633752f,
       -0.009881397f,   -0.043551125f,  -0.018609839f,  0.0019097115f,
       -0.008799762f,   0.056595087f,   0.0022273948f,  0.055752404f});

  lstm.SetRecurrentToOutputWeights({
      0.025825322f,   -0.05813119f,  0.09495884f,   -0.045984812f,   -0.01255415f,
      -0.0026479573f, -0.08196161f,  -0.054914974f, -0.0046604523f,  -0.029587349f,
      -0.044576716f,  -0.07480124f,  -0.082868785f, 0.023254942f,    0.027502948f,
      -0.0039728214f, -0.08683098f,  -0.08116779f,  -0.014675607f,   -0.037924774f,
      -0.023314456f,  -0.007401714f, -0.09255757f,  0.029460307f,    -0.08829125f,
      -0.005139627f,  -0.08989442f,  -0.0555066f,   0.13596267f,     -0.025062224f,
      -0.048351806f,  -0.03850004f,  0.07266485f,   -0.022414139f,   0.05940088f,
      0.075114764f,   0.09597592f,   -0.010211725f, -0.0049794707f,  -0.011523867f,
      -0.025980417f,  0.072999895f,  0.11091378f,   -0.081685916f,   0.014416728f,
      0.043229222f,   0.034178585f,  -0.07530371f,  0.035837382f,    -0.085607f,
      -0.007721233f,  -0.03287832f,  -0.043848954f, -0.06404588f,    -0.06632928f,
      -0.073643476f,  0.008214239f,  -0.045984086f, 0.039764922f,    0.03474462f,
      0.060612556f,   -0.080590084f, 0.049127717f,  0.04151091f,     -0.030063879f,
      0.008801774f,   -0.023021035f, -0.019558564f, 0.05158114f,     -0.010947698f,
      -0.011825728f,  0.0075720972f, 0.0699727f,    -0.0039981045f,  0.069350146f,
      0.08799282f,    0.016156472f,  0.035502106f,  0.11695009f,     0.006217345f,
      0.13392477f,    -0.037875112f, 0.025745004f,  0.08940699f,     -0.00924166f,
      0.0046702605f,  -0.036598757f, -0.08811812f,  0.10522024f,     -0.032441203f,
      0.008176899f,   -0.04454919f,  0.07058152f,   0.0067963637f,   0.039206743f,
      0.03259838f,    0.03725492f,   -0.09515802f,  0.013326398f,    -0.052055415f,
      -0.025676316f,  0.03198509f,   -0.015951829f, -0.058556724f,   0.036879618f,
      0.043357447f,   0.028362012f,  -0.05908629f,  0.0059240665f,   -0.04995891f,
      -0.019187413f,  0.0276265f,    -0.01628143f,  0.0025863599f,   0.08800015f,
      0.035250366f,   -0.022165963f, -0.07328642f,  -0.009415526f,   -0.07455109f,
      0.11690406f,    0.0363299f,    0.07411125f,   0.042103454f,    -0.009660886f,
      0.019076364f,   0.018299393f,  -0.046004917f, 0.08891175f,     0.0431396f,
      -0.026327137f,  -0.051502608f, 0.08979574f,   -0.051670972f,   0.04940282f,
      -0.07491107f,   -0.021240504f, 0.022596184f,  -0.034280192f,   0.060163025f,
      -0.058211457f,  -0.051837247f, -0.01349775f,  -0.04639988f,    -0.035936575f,
      -0.011681591f,  0.064818054f,  0.0073146066f, -0.021745546f,   -0.043124277f,
      -0.06471268f,   -0.07053354f,  -0.029321948f, -0.05330136f,    0.016933719f,
      -0.053782392f,  0.13747959f,   -0.1361751f,   -0.11569455f,    0.0033329215f,
      0.05693899f,    -0.053219706f, 0.063698f,     0.07977434f,     -0.07924483f,
      0.06936997f,    0.0034815092f, -0.007305279f, -0.037325785f,   -0.07251102f,
      -0.033633437f,  -0.08677009f,  0.091591336f,  -0.14165086f,    0.021752775f,
      0.019683983f,   0.0011612234f, -0.058154266f, 0.049996935f,    0.0288841f,
      -0.0024567875f, -0.14345716f,  0.010955264f,  -0.10234828f,    0.1183656f,
      -0.0010731248f, -0.023590032f, -0.072285876f, -0.0724771f,     -0.026382286f,
      -0.0014920527f, 0.042667855f,  0.0018776858f, 0.02986552f,     0.009814309f,
      0.0733756f,     0.12289186f,   0.018043943f,  -0.0458958f,     0.049412545f,
      0.033632483f,   0.05495232f,   0.036686596f,  -0.013781798f,   -0.010036754f,
      0.02576849f,    -0.08307328f,  0.010112348f,  0.042521734f,    -0.05869831f,
      -0.071689695f,  0.03876447f,   -0.13275425f,  -0.0352966f,     -0.023077697f,
      0.10285965f,    0.084736146f,  0.15568255f,   -0.00040734606f, 0.027835453f,
      -0.10292561f,   -0.032401145f, 0.10053256f,   -0.026142767f,   -0.08271222f,
      -0.0030240538f, -0.016368777f, 0.1070414f,    0.042672627f,    0.013456989f,
      -0.0437609f,    -0.022309763f, 0.11576483f,   0.04108048f,     0.061026827f,
      -0.0190714f,    -0.0869359f,   0.037901703f,  0.0610107f,      0.07202949f,
      0.01675338f,    0.086139716f,  -0.08795751f,  -0.014898893f,   -0.023771819f,
      -0.01965048f,   0.007955471f,  -0.043740474f, 0.03346837f,     -0.10549954f,
      0.090567775f,   0.042013682f,  -0.03176985f,  0.12569028f,     -0.02421228f,
      -0.029526481f,  0.023851605f,  0.031539805f,  0.05292009f,     -0.02344001f,
      -0.07811758f,   -0.08834428f,  0.10094801f,   0.16594367f,     -0.06861939f,
      -0.021256343f,  -0.041093912f, -0.06669611f,  0.035498552f,    0.021757556f,
      -0.09302526f,   -0.015403468f, -0.06614931f,  -0.051798206f,   -0.013874718f,
      0.03630673f,    0.010412845f,  -0.08077351f,  0.046185967f,    0.0035662893f,
      0.03541868f,    -0.094149634f, -0.034814864f, 0.003128424f,    -0.020674974f,
      -0.03944324f,   -0.008110165f, -0.11113267f,  0.08484226f,     0.043586485f,
      0.040582247f,   0.0968012f,    -0.065249965f, -0.028036479f,   0.0050708856f,
      0.0017462453f,  0.0326779f,    0.041296225f,  0.09164146f,     -0.047743853f,
      -0.015952192f,  -0.034451712f, 0.084197424f,  -0.05347844f,    -0.11768019f,
      0.085926116f,   -0.08251791f,  -0.045081906f, 0.0948852f,      0.068401024f,
      0.024856757f,   0.06978981f,   -0.057309967f, -0.012775832f,   -0.0032452994f,
      0.01977615f,    -0.041040014f, -0.024264973f, 0.063464895f,    0.05431621f,
  });

  lstm.SetCellToInputWeights(
      {0.040369894f, 0.030746894f,  0.24704495f,  0.018586371f,  -0.037586458f,
       -0.15312155f, -0.11812848f,  -0.11465643f, 0.20259799f,   0.11418174f,
       -0.10116027f, -0.011334949f, 0.12411352f,  -0.076769054f, -0.052169047f,
       0.21198851f,  -0.38871562f,  -0.09061183f, -0.09683246f,  -0.21929175f});

  lstm.SetCellToForgetWeights(
      {-0.01998659f,  -0.15568835f,  -0.24248174f,   -0.012770197f, 0.041331276f,
       -0.072311886f, -0.052123554f, -0.0066330447f, -0.043891653f, 0.036225766f,
       -0.047248036f, 0.021479502f,  0.033189066f,   0.11952997f,   -0.020432774f,
       0.64658105f,   -0.06650122f,  -0.03467612f,   0.095340036f,  0.23647355f});

  lstm.SetCellToOutputWeights(
      {0.08286371f,  -0.08261836f, -0.51210177f, 0.002913762f, 0.17764764f,
       -0.5495371f,  -0.08460716f, -0.24552552f, 0.030037103f, 0.04123544f,
       -0.11940523f, 0.007358328f, 0.1890978f,   0.4833202f,   -0.34441817f,
       0.36312827f,  -0.26375428f, 0.1457655f,   -0.19724406f, 0.15548733f});

  lstm.SetProjectionWeights(
      {-0.009802181f,  0.09401916f,    0.0717386f,     -0.13895074f,  0.09641832f,
       0.060420845f,   0.08539281f,    0.054285463f,   0.061395317f,  0.034448683f,
       -0.042991187f,  0.019801661f,   -0.16840284f,   -0.015726732f, -0.23041931f,
       -0.024478018f,  -0.10959692f,   -0.013875541f,  0.18600968f,   -0.061274476f,
       0.0138165f,     -0.08160894f,   -0.07661644f,   0.032372914f,  0.16169067f,
       0.22465782f,    -0.03993472f,   -0.004017731f,  0.08633481f,   -0.28869787f,
       0.08682067f,    0.17240396f,    0.014975425f,   0.056431185f,  0.031037588f,
       0.16702051f,    0.0077946745f,  0.15140012f,    0.29405436f,   0.120285f,
       -0.188994f,     -0.027265169f,  0.043389652f,   -0.022061434f, 0.014777949f,
       -0.20203483f,   0.094781205f,   0.19100232f,    0.13987629f,   -0.036132768f,
       -0.06426278f,   -0.05108664f,   0.13221376f,    0.009441198f,  -0.16715929f,
       0.15859416f,    -0.040437475f,  0.050779544f,   -0.022187516f, 0.012166504f,
       0.027685808f,   -0.07675938f,   -0.0055694645f, -0.09444123f,  0.0046453946f,
       0.050794356f,   0.10770313f,    -0.20790008f,   -0.07149004f,  -0.11425117f,
       0.008225835f,   -0.035802525f,  0.14374903f,    0.15262283f,   0.048710253f,
       0.1847461f,     -0.007487823f,  0.11000021f,    -0.09542012f,  0.22619456f,
       -0.029149994f,  0.08527916f,    0.009043713f,   0.0042746216f, 0.016261552f,
       0.022461696f,   0.12689082f,    -0.043589946f,  -0.12035478f,  -0.08361797f,
       -0.050666027f,  -0.1248618f,    -0.1275799f,    -0.071875185f, 0.07377272f,
       0.09944291f,    -0.18897448f,   -0.1593054f,    -0.06526116f,  -0.040107165f,
       -0.004618631f,  -0.067624845f,  -0.007576253f,  0.10727444f,   0.041546922f,
       -0.20424393f,   0.06907816f,    0.050412357f,   0.00724631f,   0.039827548f,
       0.12449835f,    0.10747581f,    0.13708383f,    0.09134148f,   -0.12617786f,
       -0.06428341f,   0.09956831f,    0.1208086f,     -0.14676677f,  -0.0727722f,
       0.1126304f,     0.010139365f,   0.015571211f,   -0.038128063f, 0.022913318f,
       -0.042050496f,  0.16842307f,    -0.060597885f,  0.10531834f,   -0.06411776f,
       -0.07451711f,   -0.03410368f,   -0.13393489f,   0.06534304f,   0.003620307f,
       0.04490757f,    0.05970546f,    0.05197996f,    0.02839995f,   0.10434969f,
       -0.013699693f,  -0.028353551f,  -0.07260381f,   0.047201227f,  -0.024575593f,
       -0.036445823f,  0.07155557f,    0.009672501f,   -0.02328883f,  0.009533515f,
       -0.03606021f,   -0.07421458f,   -0.028082801f,  -0.2678904f,   -0.13221288f,
       0.18419984f,    -0.13012612f,   -0.014588381f,  -0.035059117f, -0.04824723f,
       0.07830115f,    -0.056184657f,  0.03277091f,    0.025466874f,  0.14494097f,
       -0.12522776f,   -0.098633975f,  -0.10766018f,   -0.08317623f,  0.08594209f,
       0.07749552f,    0.039474737f,   0.1776665f,     -0.07409566f,  -0.0477268f,
       0.29323658f,    0.10801441f,    0.1154011f,     0.013952499f,  0.10739139f,
       0.10708251f,    -0.051456142f,  0.0074137426f,  -0.10430189f,  0.10034707f,
       0.045594677f,   0.0635285f,     -0.0715442f,    -0.089667566f, -0.10811871f,
       0.00026344223f, 0.08298446f,    -0.009525053f,  0.006585689f,  -0.24567553f,
       -0.09450807f,   0.09648481f,    0.026996298f,   -0.06419476f,  -0.04752702f,
       -0.11063944f,   -0.23441927f,   -0.17608605f,   -0.052156363f, 0.067035615f,
       0.19271925f,    -0.0032889997f, -0.043264326f,  0.09663576f,   -0.057112187f,
       -0.10100678f,   0.0628376f,     0.04447668f,    0.017961001f,  -0.10094388f,
       -0.10190601f,   0.18335468f,    0.10494553f,    -0.052095775f, -0.0026118709f,
       0.10539724f,    -0.04383912f,   -0.042349473f,  0.08438151f,   -0.1947263f,
       0.02251204f,    0.11216432f,    -0.10307853f,   0.17351969f,   -0.039091777f,
       0.08066188f,    -0.00561982f,   0.12633002f,    0.11335965f,   -0.0088127935f,
       -0.019777594f,  0.06864014f,    -0.059751723f,  0.016233567f,  -0.06894641f,
       -0.28651384f,   -0.004228674f,  0.019708522f,   -0.16305895f,  -0.07468996f,
       -0.0855457f,    0.099339016f,   -0.07580735f,   -0.13775392f,  0.08434318f,
       0.08330512f,    -0.12131499f,   0.031935584f,   0.09180414f,   -0.08876437f,
       -0.08049874f,   0.008753825f,   0.03498998f,    0.030215185f,  0.03907079f,
       0.089751154f,   0.029194152f,   -0.03337423f,   -0.019092513f, 0.04331237f,
       0.04299654f,    -0.036394123f,  -0.12915532f,   0.09793732f,   0.07512415f,
       -0.11319543f,   -0.032502122f,  0.15661901f,    0.07671967f,   -0.005491124f,
       -0.19379048f,   -0.218606f,     0.21448623f,    0.017840758f,  0.1416943f,
       -0.07051762f,   0.19488361f,    0.02664691f,    -0.18104725f,  -0.09334311f,
       0.15026465f,    -0.15493552f,   -0.057762887f,  -0.11604192f,  -0.262013f,
       -0.01391798f,   0.012185008f,   0.11156489f,    -0.07483202f,  0.06693364f,
       -0.26151478f,   0.046425626f,   0.036540434f,   -0.16435726f,  0.17338543f,
       -0.21401681f,   -0.11385144f,   -0.08283257f,   -0.069031075f, 0.030635102f,
       0.010969227f,   0.11109743f,    0.010919218f,   0.027526086f,  0.13519906f,
       0.01891392f,    -0.046839405f,  -0.040167913f,  0.017953383f,  -0.09700955f,
       0.0061885654f,  -0.07000971f,   0.026893595f,   -0.038844477f, 0.14543656f});

  static float lstm_input[][20] = {
      {// Batch0: 4 (input_sequence_size) * 5 (n_input)
       0.787926f, 0.151646f, 0.071352f, 0.118426f, 0.458058f, 0.596268f, 0.998386f,
       0.568695f, 0.864524f, 0.571277f, 0.073204f, 0.296072f, 0.743333f, 0.069199f,
       0.045348f, 0.867394f, 0.291279f, 0.013714f, 0.482521f, 0.626339f},

      {// Batch1: 4 (input_sequence_size) * 5 (n_input)
       0.295743f, 0.544053f, 0.690064f, 0.858138f, 0.497181f, 0.642421f, 0.524260f,
       0.134799f, 0.003639f, 0.162482f, 0.640394f, 0.930399f, 0.050782f, 0.432485f,
       0.988078f, 0.082922f, 0.563329f, 0.865614f, 0.333232f, 0.259916f}};

  static float lstm_golden_output[][64] = {
      {// Batch0: 4 (input_sequence_size) * 16 (n_output)
       -0.00396806f, 0.029352f,     -0.00279226f, 0.0159977f,   -0.00835576f,
       -0.0211779f,  0.0283512f,    -0.0114597f,  0.00907307f,  -0.0244004f,
       -0.0152191f,  -0.0259063f,   0.00914318f,  0.00415118f,  0.017147f,
       0.0134203f,   -0.0166936f,   0.0381209f,   0.000889694f, 0.0143363f,
       -0.0328911f,  -0.0234288f,   0.0333051f,   -0.012229f,   0.0110322f,
       -0.0457725f,  -0.000832209f, -0.0202817f,  0.0327257f,   0.0121308f,
       0.0155969f,   0.0312091f,    -0.0213783f,  0.0350169f,   0.000324794f,
       0.0276012f,   -0.0263374f,   -0.0371449f,  0.0446149f,   -0.0205474f,
       0.0103729f,   -0.0576349f,   -0.0150052f,  -0.0292043f,  0.0376827f,
       0.0136115f,   0.0243435f,    0.0354492f,   -0.0189322f,  0.0464512f,
       -0.00251373f, 0.0225745f,    -0.0308346f,  -0.0317124f,  0.0460407f,
       -0.0189395f,  0.0149363f,    -0.0530162f,  -0.0150767f,  -0.0340193f,
       0.0286833f,   0.00824207f,   0.0264887f,   0.0305169f},
      {// Batch1: 4 (input_sequence_size) * 16 (n_output)
       -0.013869f,    0.0287268f,   -0.00334693f, 0.00733398f,  -0.0287926f,
       -0.0186926f,   0.0193662f,   -0.0115437f,  0.00422612f,  -0.0345232f,
       0.00223253f,   -0.00957321f, 0.0210624f,   0.013331f,    0.0150954f,
       0.02168f,      -0.0141913f,  0.0322082f,   0.00227024f,  0.0260507f,
       -0.0188721f,   -0.0296489f,  0.0399134f,   -0.0160509f,  0.0116039f,
       -0.0447318f,   -0.0150515f,  -0.0277406f,  0.0316596f,   0.0118233f,
       0.0214762f,    0.0293641f,   -0.0204549f,  0.0450315f,   -0.00117378f,
       0.0167673f,    -0.0375007f,  -0.0238314f,  0.038784f,    -0.0174034f,
       0.0131743f,    -0.0506589f,  -0.0048447f,  -0.0240239f,  0.0325789f,
       0.00790065f,   0.0220157f,   0.0333314f,   -0.0264787f,  0.0387855f,
       -0.000764675f, 0.0217599f,   -0.037537f,   -0.0335206f,  0.0431679f,
       -0.0211424f,   0.010203f,    -0.062785f,   -0.00832363f, -0.025181f,
       0.0412031f,    0.0118723f,   0.0239643f,   0.0394009f}};

  // Resetting cell_state and output_state
  lstm.ResetCellState();
  lstm.ResetOutputState();

  const int input_sequence_size =
      sizeof(lstm_input[0]) / sizeof(float) / (lstm.num_inputs());
  for (int i = 0; i < input_sequence_size; i++) {
    float* batch0_start = lstm_input[0] + i * lstm.num_inputs();
    float* batch0_end = batch0_start + lstm.num_inputs();

    lstm.SetInput(0, batch0_start, batch0_end);

    float* batch1_start = lstm_input[1] + i * lstm.num_inputs();
    float* batch1_end = batch1_start + lstm.num_inputs();
    lstm.SetInput(lstm.num_inputs(), batch1_start, batch1_end);

    lstm.Invoke();

    float* golden_start_batch0 = lstm_golden_output[0] + i * lstm.num_outputs();
    float* golden_end_batch0 = golden_start_batch0 + lstm.num_outputs();
    float* golden_start_batch1 = lstm_golden_output[1] + i * lstm.num_outputs();
    float* golden_end_batch1 = golden_start_batch1 + lstm.num_outputs();
    std::vector<float> expected;
    expected.insert(expected.end(), golden_start_batch0, golden_end_batch0);
    expected.insert(expected.end(), golden_start_batch1, golden_end_batch1);
    EXPECT_THAT(lstm.GetOutput(), ElementsAreArray(ArrayFloatNear(expected)));
  }
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
