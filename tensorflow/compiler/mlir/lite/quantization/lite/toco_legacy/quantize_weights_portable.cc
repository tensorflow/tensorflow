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
// clang-format off
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantize_weights.h"
// clang-format on

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/portable_tensor_utils.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace lite {
namespace toco_legacy {
namespace {

using absl::flat_hash_set;
using mlir::lite::toco_legacy::
    CustomOpMap;  // Use this instead of mlir::lite::CustomOpMap because that
                  // uses mlir::lite::CustomOpInfo in
                  // tensorflow/compiler/mlir/lite/quantization/lite/quantize_weights.h,
                  // and we need mlir::lite::toco_legacy::CustomOpInfo, in
                  // tensorflow/compiler/mlir/lite/quantization/lite/optimize/quantize_weights.h
using tflite::BufferT;
using tflite::BuiltinOperator;
using tflite::BuiltinOperator_BATCH_MATMUL;
using tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM;
using tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN;
using tflite::BuiltinOperator_CONV_2D;
using tflite::BuiltinOperator_CUSTOM;
using tflite::BuiltinOperator_DEPTHWISE_CONV_2D;
using tflite::BuiltinOperator_EMBEDDING_LOOKUP;
using tflite::BuiltinOperator_FULLY_CONNECTED;
using tflite::BuiltinOperator_GATHER;
using tflite::BuiltinOperator_LSTM;
using tflite::BuiltinOperator_RNN;
using tflite::BuiltinOperator_SVDF;
using tflite::BuiltinOperator_TRANSPOSE_CONV;
using tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM;
using tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN;
using tflite::FinishModelBuffer;
using tflite::GetBuiltinCode;
using tflite::Model;
using tflite::ModelT;
using tflite::OperatorCodeT;
using tflite::OperatorT;
using tflite::SubGraphT;
using tflite::TensorT;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT8;

struct ConsumerOpInfo {
  OperatorT* op;
  // The index of the op in the operators vector.
  int32_t op_idx;
  // The index of the tensor to quantize in subgraph->tensors.
  int32_t op_input_idx;
};

struct TensorPerChannel {
  TensorT* t;
  bool is_per_channel;
  int channel_dim;
};

// The default minimum number of elements a weights array must have to be
// quantized by this transformation.
const int kWeightsMinNumElementsDefault = 1024;

// Redefined from tensorflow/lite/core/c/common.h as local const int instead of
// discouraged #define macro.
const int kTfLiteOptionalTensor = -1;

// Gets the operators that consume tensor_idx.
std::vector<ConsumerOpInfo> GetTensorConsumers(const ModelT* model,
                                               const SubGraphT* subgraph,
                                               int32_t tensor_idx) {
  // TODO(suharshs): If this proves to be too slow, avoid calling it per tensor,
  // instead doing one sweep for the entire model.
  std::vector<ConsumerOpInfo> consumer_ops;
  for (size_t op_idx = 0; op_idx < subgraph->operators.size(); ++op_idx) {
    OperatorT* op = subgraph->operators[op_idx].get();
    if (op == nullptr) {
      continue;
    }
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == tensor_idx) {
        consumer_ops.push_back(
            {op, static_cast<int32_t>(op_idx), static_cast<int32_t>(i)});
      }
    }
  }
  return consumer_ops;
}

// Gets the list of op->inputs indices of the weights inputs to be quantized for
// the provided op.
std::vector<int32_t> GetWeightInputIndices(const OperatorCodeT* op_code,
                                           const CustomOpMap& custom_op_map) {
  const BuiltinOperator builtin_op_code = GetBuiltinCode(op_code);
  if (builtin_op_code == BuiltinOperator_CUSTOM) {
    const std::string custom_code = op_code->custom_code;
    const auto& custom_op_info = custom_op_map.find(custom_code);
    if (custom_op_info != custom_op_map.end()) {
      return custom_op_info->second.quantizable_input_indices;
    }
  } else if (builtin_op_code == BuiltinOperator_CONV_2D ||
             builtin_op_code == BuiltinOperator_DEPTHWISE_CONV_2D ||
             builtin_op_code == BuiltinOperator_FULLY_CONNECTED ||
             builtin_op_code == BuiltinOperator_BATCH_MATMUL ||
             builtin_op_code == BuiltinOperator_EMBEDDING_LOOKUP ||
             builtin_op_code == BuiltinOperator_TRANSPOSE_CONV) {
    return {1};
  } else if (builtin_op_code == BuiltinOperator_SVDF) {
    // tensorflow/lite/kernels/svdf.cc
    return {1, 2};
  } else if (builtin_op_code == BuiltinOperator_LSTM ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM) {
    // tensorflow/lite/kernels/lstm.cc
    // tensorflow/lite/kernels/unidirectional_sequence_lstm.cc
    return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
  } else if (builtin_op_code == BuiltinOperator_RNN ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    // tensorflow/lite/kernels/basic_rnn.cc
    // tensorflow/lite/kernels/unidirectional_sequence_rnn.cc
    return {1, 2};
  } else if (builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM) {
    // tensorflow/lite/kernels/bidirectional_sequence_lstm.cc
    return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 33, 40, 41, 42, 43, 44, 45, 46, 47};
  } else if (builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN) {
    // tensorflow/lite/kernels/bidirectional_sequence_rnn.cc
    return {1, 2, 4, 5, 6, 8, 9, 10, 11};
  } else if (builtin_op_code == BuiltinOperator_GATHER) {
    // tensorflow/lite/kernels/gather.cc
    return {0};
  }
  return {};
}

// Checks that a specific input can be quantized.
bool IsQuantizedInput(const OperatorCodeT* op_code,
                      const CustomOpMap& custom_op_map, int op_input_idx) {
  const auto quantized_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  return std::find(std::begin(quantized_input_indices),
                   std::end(quantized_input_indices),
                   op_input_idx) != std::end(quantized_input_indices);
}

// Returns true if the operator supports hybrid evaluation.
bool IsHybridEvaluationOp(const OperatorT* op, const OperatorCodeT* op_code,
                          const CustomOpMap& custom_op_map,
                          bool use_updated_hybrid_scheme) {
  const BuiltinOperator builtin_op_code = GetBuiltinCode(op_code);
  // Operations that support hybrid evaluation.
  bool eval_hybrid = false;
  if (builtin_op_code == BuiltinOperator_CUSTOM) {
    const std::string custom_code = op_code->custom_code;
    const auto custom_op_info = custom_op_map.find(custom_code);
    if (custom_op_info == custom_op_map.end()) {
      return {};
    } else {
      return custom_op_info->second.is_hybrid;
    }
  } else if (builtin_op_code == BuiltinOperator_FULLY_CONNECTED ||
             builtin_op_code == BuiltinOperator_BATCH_MATMUL ||
             builtin_op_code == BuiltinOperator_CONV_2D ||
             builtin_op_code == BuiltinOperator_SVDF ||
             builtin_op_code == BuiltinOperator_RNN ||
             builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM ||
             builtin_op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM ||
             builtin_op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    eval_hybrid = true;
  } else if (builtin_op_code == BuiltinOperator_LSTM) {
    const tflite::LSTMOptionsT* options = op->builtin_options.AsLSTMOptions();
    // Only lstm kernel_type full supports hybrid evaluation.
    if (options->kernel_type == tflite::LSTMKernelType_FULL) {
      eval_hybrid = true;
    }
  } else if (builtin_op_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
    eval_hybrid = use_updated_hybrid_scheme;
  }
  return eval_hybrid;
}

// Returns true if all of the op's inputs are quantized.
bool CheckAllOpInputsQuantized(const SubGraphT* subgraph, const OperatorT* op,
                               const OperatorCodeT* op_code,
                               const CustomOpMap& custom_op_map) {
  std::vector<int32_t> op_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  for (const int32_t op_input_idx : op_input_indices) {
    int32_t tensor_idx = op->inputs[op_input_idx];

    if (tensor_idx == -1) {
      // Optional tensor.
      continue;
    }

    TensorT* tensor = subgraph->tensors[tensor_idx].get();

    if (tensor->type != TensorType_INT8) {
      return false;
    }
  }
  return true;
}

// Inserts Tensors for each input tensor of op that should be
// quantized into tensor_map.
absl::Status InsertQuantizableInputTensorsFromOperator(
    const ModelT* model, OperatorT* op, uint64_t weights_min_num_elements,
    const CustomOpMap& custom_op_map,
    absl::flat_hash_map<int32_t, TensorPerChannel>* tensor_map,
    int subgraph_index, bool use_updated_hybrid_scheme) {
  SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();
  const OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
  auto builtin_code = GetBuiltinCode(op_code);

  std::vector<int32_t> op_input_indices =
      GetWeightInputIndices(op_code, custom_op_map);
  for (const int32_t op_input_idx : op_input_indices) {
    int32_t tensor_idx = op->inputs[op_input_idx];
    if (tensor_idx == -1) {
      LOG(INFO) << "Skipping optional tensor input " << op_input_idx
                << " of operation " << EnumNameBuiltinOperator(builtin_code);
      continue;
    }

    TensorT* tensor = subgraph->tensors[tensor_idx].get();
    if (tensor->type != TensorType_FLOAT32) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " that is not type float.";
      continue;
    }

    uint64_t num_elements;
    if (!mlir::lite::toco_legacy::NumElements(*tensor, &num_elements).ok()) {
      return absl::InternalError("Error in quantization_utils NumElements");
    }
    if (num_elements < weights_min_num_elements) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has fewer than " << weights_min_num_elements
                << " elements (" << num_elements << ").";
      continue;
    }

    // Some tensors may have a null buffer vector, indicating an intermediate
    // array.
    if (model->buffers[tensor->buffer]->data.data() == nullptr) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has no allocated buffer.";
      continue;
    }

    if (builtin_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
      tensor_map->insert({tensor_idx,
                          {tensor, /*is_per_channel=*/use_updated_hybrid_scheme,
                           /*dim=*/3}});
    } else if (builtin_code == BuiltinOperator_CONV_2D) {
      tensor_map->insert({tensor_idx,
                          {tensor, /*is_per_channel=*/use_updated_hybrid_scheme,
                           /*dim=*/0}});
    } else {
      switch (builtin_code) {
        case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
          op->builtin_options.AsBidirectionalSequenceLSTMOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
          op->builtin_options.AsBidirectionalSequenceRNNOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_FULLY_CONNECTED:
          op->builtin_options.AsFullyConnectedOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_BATCH_MATMUL:
          op->builtin_options.AsBatchMatMulOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_LSTM:
          op->builtin_options.AsLSTMOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_RNN:
          op->builtin_options.AsRNNOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_SVDF:
          op->builtin_options.AsSVDFOptions()->asymmetric_quantize_inputs =
              use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
          op->builtin_options.AsUnidirectionalSequenceLSTMOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
          op->builtin_options.AsSequenceRNNOptions()
              ->asymmetric_quantize_inputs = use_updated_hybrid_scheme;
          break;
        default:
          break;
      }
      tensor_map->insert({tensor_idx, {tensor, /*is_per_channel=*/false}});
    }
  }

  return absl::OkStatus();
}

// Updates operator code versions for the operators with INT8 inputs.
void UpdateInt8OperatorVersions(ModelT* model, bool use_updated_hybrid_scheme) {
  for (int i = 0, end = model->operator_codes.size(); i < end; ++i) {
    const BuiltinOperator& op_code =
        GetBuiltinCode(model->operator_codes[i].get());
    if (op_code == BuiltinOperator_RNN ||
        op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN ||
        op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM ||
        op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 3 : 2;
    } else if (op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM ||
               op_code == BuiltinOperator_EMBEDDING_LOOKUP) {
      model->operator_codes[i]->version = 3;
    } else if (op_code == BuiltinOperator_LSTM) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 3;
    } else if (op_code == BuiltinOperator_CONV_2D) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 5 : 2;
    } else if (op_code == BuiltinOperator_FULLY_CONNECTED) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 9 : 3;
    } else if (op_code == BuiltinOperator_BATCH_MATMUL) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 1;
    } else if (op_code == BuiltinOperator_SVDF) {
      model->operator_codes[i]->version = use_updated_hybrid_scheme ? 4 : 2;
    } else if (op_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
      model->operator_codes[i]->version = 6;
    }
  }
}

// Returns true if the op in consumer_op_infos can pass through quantization.
bool IsQuantizationPassThroughOps(
    const ModelT* model, const std::vector<ConsumerOpInfo>& consumer_op_infos) {
  if (consumer_op_infos.size() != 1) {
    return false;
  }
  const OperatorT* consumer_op = consumer_op_infos.front().op;
  const BuiltinOperator op_code =
      GetBuiltinCode(model->operator_codes[consumer_op->opcode_index].get());
  return op_code == BuiltinOperator_GATHER ||
         op_code == BuiltinOperator_EMBEDDING_LOOKUP;
}

// Copies quantization parameters from input to output and returns consumers of
// the output tensor as a tuple with values:
// - index of the output tensor
// - pointer to the output tensor
// - vector of consumers ops.
std::tuple<int32_t, TensorT*, std::vector<ConsumerOpInfo>>
PassQuantizationAndGetConsumers(
    const ModelT* model, const SubGraphT* subgraph,
    const std::vector<ConsumerOpInfo>& consumer_op_infos,
    const CustomOpMap& custom_op_map) {
  const OperatorT* op = consumer_op_infos.front().op;
  const OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
  if (op->outputs.size() != 1) {
    LOG(ERROR)
        << "An op that passes quantization has more than one quantized output";
    return std::make_tuple(-1, nullptr, std::vector<ConsumerOpInfo>());
  }
  const int32_t output_tensor_idx = op->outputs.front();
  const auto input_idx = GetWeightInputIndices(op_code, custom_op_map);
  if (input_idx.size() != 1) {
    LOG(ERROR)
        << "An op that passes quantization has more than one quantized input";
    return std::make_tuple(-1, nullptr, std::vector<ConsumerOpInfo>());
  }
  const int32_t input_tensor_idx = op->inputs[input_idx.front()];

  // Propagate quantization params.
  const TensorT* input_tensor = subgraph->tensors[input_tensor_idx].get();
  TensorT* output_tensor = subgraph->tensors[output_tensor_idx].get();
  if (!output_tensor->quantization) {
    output_tensor->quantization =
        std::make_unique<tflite::QuantizationParametersT>();
  }
  *output_tensor->quantization = *input_tensor->quantization;
  output_tensor->type = TensorType_INT8;
  return std::make_tuple(
      output_tensor_idx, output_tensor,
      GetTensorConsumers(model, subgraph, output_tensor_idx));
}

inline bool IsOpDenylisted(const flat_hash_set<BuiltinOperator>& op_denylist,
                           const BuiltinOperator op_code) {
  return op_denylist.find(op_code) != op_denylist.end();
}

absl::Status QuantizeWeightsInt8(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    bool use_hybrid_evaluation, uint64_t weights_min_num_elements,
    const CustomOpMap& custom_op_map, bool use_updated_hybrid_scheme,
    const flat_hash_set<BuiltinOperator>& op_denylist = {}) {
  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());

  for (int subgraph_index = 0, end = model->subgraphs.size();
       subgraph_index < end; ++subgraph_index) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();

    absl::flat_hash_map<int32_t, TensorPerChannel> tensor_map;
    for (int i = 0; i < subgraph->operators.size(); ++i) {
      OperatorT* op = subgraph->operators[i].get();
      if (!InsertQuantizableInputTensorsFromOperator(
               model.get(), op, weights_min_num_elements, custom_op_map,
               &tensor_map, subgraph_index, use_updated_hybrid_scheme)
               .ok()) {
        return absl::InternalError(
            "Failed to insert quantizable input tensors from operator");
      }
    }

    for (std::pair<int32_t, TensorPerChannel> tensor_pair : tensor_map) {
      // Quantize the tensor.
      if (tensor_pair.second.is_per_channel) {
        if (!mlir::lite::toco_legacy::SymmetricQuantizeTensorPerChannel(
                 model.get(), tensor_pair.second.t,
                 tensor_pair.second.channel_dim)
                 .ok()) {
          return absl::InternalError("Failed to quantize tensor per channel");
        }
      } else {
        if (!mlir::lite::toco_legacy::SymmetricQuantizeTensor(
                 model.get(), tensor_pair.second.t)
                 .ok()) {
          return absl::InternalError("Failed to quantize tensor");
        }
      }
    }

    // Examine the tensor consumers to determine which require dequantize ops.
    for (const auto& tensor_pair : tensor_map) {
      int32_t tensor_idx = tensor_pair.first;
      TensorT* tensor = tensor_pair.second.t;
      std::vector<ConsumerOpInfo> consumer_op_infos =
          GetTensorConsumers(model.get(), subgraph, tensor_idx);
      if (IsQuantizationPassThroughOps(model.get(), consumer_op_infos)) {
        std::tie(tensor_idx, tensor, consumer_op_infos) =
            PassQuantizationAndGetConsumers(model.get(), subgraph,
                                            consumer_op_infos, custom_op_map);
        if (tensor_idx < 0) {
          // Error message is already logged by PassQuantizationAndGetConsumers.
          return absl::InternalError(
              "Failed to pass quantization and get consumers");
        }
      }

      std::vector<ConsumerOpInfo> dequant_op_infos;  // Ops that need dequants.
      for (ConsumerOpInfo& consumer_op_info : consumer_op_infos) {
        OperatorT* consumer_op = consumer_op_info.op;
        const OperatorCodeT* consumer_op_code =
            model->operator_codes[consumer_op->opcode_index].get();
        // If the op is a hybrid op and all the required tensors are quantized,
        // we have no further work to do, but for all ops that require
        // dequantization we need to add a Dequantize op.
        bool eval_hybrid =
            use_hybrid_evaluation &&
            !IsOpDenylisted(op_denylist, GetBuiltinCode(consumer_op_code)) &&
            IsHybridEvaluationOp(consumer_op, consumer_op_code, custom_op_map,
                                 use_updated_hybrid_scheme) &&
            CheckAllOpInputsQuantized(subgraph, consumer_op, consumer_op_code,
                                      custom_op_map) &&
            IsQuantizedInput(consumer_op_code, custom_op_map,
                             consumer_op_info.op_input_idx);
        if (!eval_hybrid) {
          dequant_op_infos.push_back(consumer_op_info);
        }
      }

      // Check if this tensor is an output tensor.
      int32_t output_index = -1;
      for (int32_t i = 0; i < subgraph->outputs.size(); ++i) {
        if (subgraph->outputs[i] == tensor_idx) {
          output_index = i;
          break;
        }
      }

      // If no ops require dequant and it is not output, we are done for this
      // tensor.
      if (dequant_op_infos.empty() && output_index < 0) {
        continue;
      }

      // Create a new tensor to be the output of the dequantize op.
      std::unique_ptr<TensorT> dequantize_output;
      const std::string dequant_name = tensor->name + "_dequantize";
      mlir::lite::toco_legacy::MakeTensor(
          dequant_name, tensor->shape, tensor->shape_signature,
          TensorType_FLOAT32, &dequantize_output);
      const int32_t dequantize_output_idx = subgraph->tensors.size();
      subgraph->tensors.push_back(std::move(dequantize_output));

      // Create the Dequantize operation.
      std::unique_ptr<OperatorT> dequantize_op;
      mlir::lite::toco_legacy::MakeDequantizeOperator(
          model.get(), &dequantize_op, tensor_idx, dequantize_output_idx);

      // Update the op_input of all the ops that need the created dequantize
      // operation.
      int32_t min_op_idx = subgraph->operators.size();
      for (ConsumerOpInfo& dequant_op_info : dequant_op_infos) {
        dequant_op_info.op->inputs[dequant_op_info.op_input_idx] =
            dequantize_output_idx;
        min_op_idx = std::min(dequant_op_info.op_idx, min_op_idx);
      }
      // Update output name.
      if (output_index >= 0) {
        subgraph->outputs[output_index] = dequantize_output_idx;
      }

      // Insert the newly created Dequantize operation before the earliest
      // consumer, since TFLite requires operators to be topo-sorted.
      subgraph->operators.insert(subgraph->operators.begin() + min_op_idx,
                                 std::move(dequantize_op));
    }
  }

  // Update the modified operator code versions.
  UpdateInt8OperatorVersions(model.get(), use_updated_hybrid_scheme);

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);

  return absl::OkStatus();
}

absl::Status QuantizeWeightsFloat16(flatbuffers::FlatBufferBuilder* builder,
                                    const Model* input_model) {
  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());

  for (int subgraph_index = 0, end = model->subgraphs.size();
       subgraph_index < end; ++subgraph_index) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_index).get();

    absl::flat_hash_map<int32_t, TensorT*> tensor_map;
    for (int i = 0, sub_end = subgraph->operators.size(); i < sub_end; ++i) {
      OperatorT* op = subgraph->operators[i].get();
      for (auto tensor_idx : op->inputs) {
        // Skip optional tensors.
        if (tensor_idx == kTfLiteOptionalTensor) {
          continue;
        }
        TensorT* tensor = subgraph->tensors[tensor_idx].get();
        BufferT* buffer = model->buffers[tensor->buffer].get();
        if (buffer == nullptr) {
          return absl::InternalError("Buffer is null");
        }
        // Quantize tensors that have data to quantize.
        bool is_constant = !model->buffers[tensor->buffer].get()->data.empty();
        if (tensor->type == TensorType_FLOAT32 && is_constant) {
          tensor_map.insert({tensor_idx, tensor});
        }
      }
    }

    // The hash map ensures that we quantize each tensor exactly once.
    for (std::pair<int32_t, TensorT*> tensor_pair : tensor_map) {
      // Quantize the tensor.
      if (!mlir::lite::toco_legacy::QuantizeTensorFloat16(model.get(),
                                                          tensor_pair.second)
               .ok()) {
        return absl::InternalError("QuantizeTensorFloat16 failed");
      }

      int32_t tensor_idx = tensor_pair.first;
      TensorT* tensor = tensor_pair.second;
      std::vector<ConsumerOpInfo> dequant_op_infos =
          GetTensorConsumers(model.get(), subgraph, tensor_idx);

      // Create a new tensor to be the output of the dequantize op.
      std::unique_ptr<TensorT> dequantize_output;
      const std::string dequant_name = tensor->name + "_dequantize";
      mlir::lite::toco_legacy::MakeTensor(
          dequant_name, tensor->shape, tensor->shape_signature,
          TensorType_FLOAT32, &dequantize_output);
      const int32_t dequantize_output_idx = subgraph->tensors.size();
      subgraph->tensors.push_back(std::move(dequantize_output));

      // Create the Dequantize operation.
      std::unique_ptr<OperatorT> dequantize_op;
      mlir::lite::toco_legacy::MakeDequantizeOperator(
          model.get(), &dequantize_op, tensor_idx, dequantize_output_idx);

      // Update the op_input of all the ops that need the created dequantize
      // operation.
      int32_t min_op_idx = subgraph->operators.size();
      for (ConsumerOpInfo& dequant_op_info : dequant_op_infos) {
        dequant_op_info.op->inputs[dequant_op_info.op_input_idx] =
            dequantize_output_idx;
        min_op_idx = std::min(dequant_op_info.op_idx, min_op_idx);
      }

      // Insert the newly created Dequantize operation before the earliest
      // consumer, since TFLite requires operators to be topo-sorted.
      subgraph->operators.insert(subgraph->operators.begin() + min_op_idx,
                                 std::move(dequantize_op));
    }
  }

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);
  return absl::OkStatus();
}
}  // namespace

namespace internal {
absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             bool use_hybrid_evaluation,
                             QuantizerType quantizer_type) {
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    LOG(ERROR) << "Portable targets cannot use the MLIR quantizer.";
    return absl::InternalError(
        "Portable targets cannot use the MLIR quantizer.");
  }
  // By default we require that only weights with more than
  // kWeightsMinSizeDefault elements are quantized.
  CustomOpMap custom_op_map;
  return QuantizeWeightsInt8(builder, input_model, use_hybrid_evaluation,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}
}  // namespace internal

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             QuantizerType quantizer_type) {
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    LOG(ERROR) << "Portable targets cannot use the MLIR quantizer.";
    return absl::InternalError(
        "Portable targets cannot use the MLIR quantizer.");
  }
  CustomOpMap custom_op_map;
  return QuantizeWeightsInt8(builder, input_model, true,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model, BufferType quant_type,
                             bool use_updated_hybrid_scheme,
                             QuantizerType quantizer_type) {
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    LOG(ERROR) << "Portable targets cannot use the MLIR quantizer.";
    return absl::InternalError(
        "Portable targets cannot use the MLIR quantizer.");
  }
  switch (quant_type) {
    case BufferType::QUANTIZED_INT8: {
      // By default we require that only weights with more than
      // kWeightsMinSizeDefault elements are quantized.
      CustomOpMap custom_op_map;
      return QuantizeWeightsInt8(builder, input_model, true,
                                 kWeightsMinNumElementsDefault, custom_op_map,
                                 use_updated_hybrid_scheme);
    }
    case BufferType::QUANTIZED_FLOAT16:
      return QuantizeWeightsFloat16(builder, input_model);
  }
}

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             QuantizerType quantizer_type) {
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    LOG(ERROR) << "Portable targets cannot use the MLIR quantizer.";
    return absl::InternalError(
        "Portable targets cannot use the MLIR quantizer.");
  }
  return QuantizeWeightsInt8(builder, input_model, true,
                             weights_min_num_elements, custom_op_map,
                             kUseUpdatedHybridSchemeDefault);
}

absl::Status QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements,
                             const CustomOpMap& custom_op_map,
                             bool use_updated_hybrid_scheme,
                             const flat_hash_set<BuiltinOperator>& op_denylist,
                             QuantizerType quantizer_type) {
  if (quantizer_type == QuantizerType::MLIR_QUANTIZER) {
    LOG(ERROR) << "Portable targets cannot use the MLIR quantizer.";
    return absl::InternalError(
        "Portable targets cannot use the MLIR quantizer.");
  }
  return QuantizeWeightsInt8(builder, input_model,
                             /*use_hybrid_evaluation=*/true,
                             weights_min_num_elements, custom_op_map,
                             use_updated_hybrid_scheme, op_denylist);
}

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir
