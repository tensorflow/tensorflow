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
  void Verify() {
    auto model = tflite::UnPackModel(builder_.GetBufferPointer());
    EXPECT_NE(model, nullptr);
  }

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

  // Resetting cell_state and output_state
  lstm.ResetCellState();
  lstm.ResetOutputState();

  // Verify the model by unpacking it.
  lstm.Verify();
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
