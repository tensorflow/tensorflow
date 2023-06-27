/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
// Shared code between unidirectional LSTM test and bench.

#ifndef TENSORFLOW_LITE_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_TEST_UTIL_H_

#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

class UnidirectionalLSTMOpModel : public SingleOpModel {
 public:
  UnidirectionalLSTMOpModel(int n_batch, int n_input, int n_cell, int n_output,
                            int sequence_length, bool time_major, bool use_cifg,
                            bool use_peephole, bool use_projection_weights,
                            bool use_projection_bias, float cell_clip,
                            float proj_clip,
                            const std::vector<std::vector<int>>& input_shapes,
                            const TensorType& weights_type = TensorType_FLOAT32,
                            bool is_layer_norm = false,
                            bool asymmetric_quantize_inputs = false,
                            bool diagonal_recurrent_weights = false)
      : n_batch_(n_batch),
        n_input_(n_input),
        n_cell_(n_cell),
        n_output_(n_output),
        sequence_length_(sequence_length),
        diagonal_recurrent_weights_(diagonal_recurrent_weights) {
    input_ = AddInput(TensorType_FLOAT32);
    const TensorType recurrent_weight_type =
        diagonal_recurrent_weights_ ? TensorType_FLOAT32 : weights_type;

    if (use_cifg) {
      input_to_input_weights_ = AddNullInput();
    } else {
      input_to_input_weights_ = AddInput(weights_type);
    }

    input_to_forget_weights_ = AddInput(weights_type);
    input_to_cell_weights_ = AddInput(weights_type);
    input_to_output_weights_ = AddInput(weights_type);

    if (use_cifg) {
      recurrent_to_input_weights_ = AddNullInput();
    } else {
      recurrent_to_input_weights_ = AddInput(recurrent_weight_type);
    }

    recurrent_to_forget_weights_ = AddInput(recurrent_weight_type);
    recurrent_to_cell_weights_ = AddInput(recurrent_weight_type);
    recurrent_to_output_weights_ = AddInput(recurrent_weight_type);

    if (use_peephole) {
      if (use_cifg) {
        cell_to_input_weights_ = AddNullInput();
      } else {
        cell_to_input_weights_ = AddInput(weights_type);
      }
      cell_to_forget_weights_ = AddInput(weights_type);
      cell_to_output_weights_ = AddInput(weights_type);
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
    cell_gate_bias_ = AddInput(TensorType_FLOAT32);
    output_gate_bias_ = AddInput(TensorType_FLOAT32);

    if (use_projection_weights) {
      projection_weights_ = AddInput(weights_type);
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
    output_state_ = AddVariableInput(
        TensorData{TensorType_FLOAT32, {n_output_ * n_batch_}});
    cell_state_ =
        AddVariableInput(TensorData{TensorType_FLOAT32, {n_cell_ * n_batch_}});

    // Layer norm weights.
    if (is_layer_norm) {
      if (use_cifg) {
        input_layer_norm_coefficients_ = AddNullInput();
      } else {
        input_layer_norm_coefficients_ =
            AddLayerNormCoeffsTensor(20, input_shapes);
      }
      forget_layer_norm_coefficients_ =
          AddLayerNormCoeffsTensor(21, input_shapes);
      cell_layer_norm_coefficients_ =
          AddLayerNormCoeffsTensor(22, input_shapes);
      output_layer_norm_coefficients_ =
          AddLayerNormCoeffsTensor(23, input_shapes);
    }

    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(
        BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
        BuiltinOptions_UnidirectionalSequenceLSTMOptions,
        CreateUnidirectionalSequenceLSTMOptions(
            builder_, ActivationFunctionType_TANH, cell_clip, proj_clip,
            time_major, asymmetric_quantize_inputs, diagonal_recurrent_weights)
            .Union());
    BuildInterpreter(input_shapes);
  }

  void SetInputToInputWeights(const std::vector<float>& f) {
    PopulateTensor(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(const std::vector<float>& f) {
    PopulateTensor(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(const std::vector<float>& f) {
    PopulateTensor(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(const std::vector<float>& f) {
    PopulateTensor(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(const std::vector<float>& f) {
    PopulateTensor(recurrent_to_input_weights_, f);
  }

  void SetRecurrentToForgetWeights(const std::vector<float>& f) {
    PopulateTensor(recurrent_to_forget_weights_, f);
  }

  void SetRecurrentToCellWeights(const std::vector<float>& f) {
    PopulateTensor(recurrent_to_cell_weights_, f);
  }

  void SetRecurrentToOutputWeights(const std::vector<float>& f) {
    PopulateTensor(recurrent_to_output_weights_, f);
  }

  void SetCellToInputWeights(const std::vector<float>& f) {
    PopulateTensor(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(const std::vector<float>& f) {
    PopulateTensor(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(const std::vector<float>& f) {
    PopulateTensor(cell_to_output_weights_, f);
  }

  void SetInputGateBias(const std::vector<float>& f) {
    PopulateTensor(input_gate_bias_, f);
  }

  void SetForgetGateBias(const std::vector<float>& f) {
    PopulateTensor(forget_gate_bias_, f);
  }

  void SetCellBias(const std::vector<float>& f) {
    PopulateTensor(cell_gate_bias_, f);
  }

  void SetOutputGateBias(const std::vector<float>& f) {
    PopulateTensor(output_gate_bias_, f);
  }

  void SetProjectionWeights(const std::vector<float>& f) {
    PopulateTensor(projection_weights_, f);
  }

  void SetProjectionBias(const std::vector<float>& f) {
    PopulateTensor(projection_bias_, f);
  }

  void SetInputLayerNormCoefficients(std::vector<float> f) {
    PopulateTensor(input_layer_norm_coefficients_, f);
  }

  void SetForgetLayerNormCoefficients(std::vector<float> f) {
    PopulateTensor(forget_layer_norm_coefficients_, f);
  }

  void SetCellLayerNormCoefficients(std::vector<float> f) {
    PopulateTensor(cell_layer_norm_coefficients_, f);
  }

  void SetOutputLayerNormCoefficients(std::vector<float> f) {
    PopulateTensor(output_layer_norm_coefficients_, f);
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
  int sequence_length() { return sequence_length_; }

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

  int input_gate_bias_;
  int forget_gate_bias_;
  int cell_gate_bias_;
  int output_gate_bias_;

  int projection_weights_;
  int projection_bias_;

  int output_state_;
  int cell_state_;

  int input_layer_norm_coefficients_;
  int forget_layer_norm_coefficients_;
  int cell_layer_norm_coefficients_;
  int output_layer_norm_coefficients_;

  int output_;

  int n_batch_;
  int n_input_;
  int n_cell_;
  int n_output_;
  int sequence_length_;
  bool diagonal_recurrent_weights_;

 private:
  int AddLayerNormCoeffsTensor(
      int tensor_index, const std::vector<std::vector<int>>& input_shapes) {
    if (input_shapes[tensor_index][0] != 0) {
      return AddInput(TensorType_FLOAT32);
    } else {
      return AddNullInput();
    }
  }
};

// The hybrid model has quantized weights.
class HybridUnidirectionalLSTMOpModel : public UnidirectionalLSTMOpModel {
 public:
  HybridUnidirectionalLSTMOpModel(
      int n_batch, int n_input, int n_cell, int n_output, int sequence_length,
      bool time_major, bool use_cifg, bool use_peephole,
      bool use_projection_weights, bool use_projection_bias, float cell_clip,
      float proj_clip, const std::vector<std::vector<int>>& input_shapes,
      TensorType tensor_type, bool asymmetric_quantize_inputs,
      bool diagonal_recurrent_weights = false)
      : UnidirectionalLSTMOpModel(
            n_batch, n_input, n_cell, n_output, sequence_length, time_major,
            use_cifg, use_peephole, use_projection_weights, use_projection_bias,
            cell_clip, proj_clip, input_shapes, tensor_type,
            /*is_layer_norm=*/false, asymmetric_quantize_inputs,
            diagonal_recurrent_weights) {
    tensor_type_ = tensor_type;
  }

  void SetWeights(int weights_idx, const std::vector<float>& f) {
    if (tensor_type_ == TensorType_UINT8) {
      SymmetricQuantizeAndPopulate(weights_idx, f);
    } else {
      SignedSymmetricQuantizeAndPopulate(weights_idx, f);
    }
  }

  void SetInputToInputWeights(const std::vector<float>& f) {
    SetWeights(input_to_input_weights_, f);
  }

  void SetInputToForgetWeights(const std::vector<float>& f) {
    SetWeights(input_to_forget_weights_, f);
  }

  void SetInputToCellWeights(const std::vector<float>& f) {
    SetWeights(input_to_cell_weights_, f);
  }

  void SetInputToOutputWeights(const std::vector<float>& f) {
    SetWeights(input_to_output_weights_, f);
  }

  void SetRecurrentToInputWeights(const std::vector<float>& f) {
    if (diagonal_recurrent_weights_) {
      PopulateTensor(recurrent_to_input_weights_, f);
    } else {
      SetWeights(recurrent_to_input_weights_, f);
    }
  }

  void SetRecurrentToForgetWeights(const std::vector<float>& f) {
    if (diagonal_recurrent_weights_) {
      PopulateTensor(recurrent_to_forget_weights_, f);
    } else {
      SetWeights(recurrent_to_forget_weights_, f);
    }
  }

  void SetRecurrentToCellWeights(const std::vector<float>& f) {
    if (diagonal_recurrent_weights_) {
      PopulateTensor(recurrent_to_cell_weights_, f);
    } else {
      SetWeights(recurrent_to_cell_weights_, f);
    }
  }

  void SetRecurrentToOutputWeights(const std::vector<float>& f) {
    if (diagonal_recurrent_weights_) {
      PopulateTensor(recurrent_to_output_weights_, f);
    } else {
      SetWeights(recurrent_to_output_weights_, f);
    }
  }

  void SetCellToInputWeights(const std::vector<float>& f) {
    SetWeights(cell_to_input_weights_, f);
  }

  void SetCellToForgetWeights(const std::vector<float>& f) {
    SetWeights(cell_to_forget_weights_, f);
  }

  void SetCellToOutputWeights(const std::vector<float>& f) {
    SetWeights(cell_to_output_weights_, f);
  }

  void SetProjectionWeights(const std::vector<float>& f) {
    SetWeights(projection_weights_, f);
  }

 protected:
  TensorType tensor_type_;
};

}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_LITE_KERNELS_UNIDIRECTIONAL_SEQUENCE_LSTM_TEST_UTIL_H_
