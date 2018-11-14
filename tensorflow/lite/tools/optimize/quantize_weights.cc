/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantize_weights.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/core/platform/logging.h"

namespace tflite {
namespace optimize {

namespace {

typedef struct {
  TensorT* tensor;
  // The index of the tensor to quantize in subgraph->tensors.
  int32_t tensor_idx;
  // The index of the tensor of the weight tensor to be quantize in op->inputs.
  int32_t op_input_idx;
  // True if the tensor supports hybrid evaluation.
  bool eval_hybrid;
} TensorInfo;

// The default minimum number of elements a weights array must have to be
// quantized by this transformation.
const int kWeightsMinNumElementsDefault = 1024;

// Nudge min and max so that floating point 0 falls exactly on a quantized
// value, returning the nudges scale and zero_point.
//
// Although this code originates from FakeQuantization in quantized training,
// we may deviate from that implementation as we please since we do not fine
// tune the weights with quantized training.
void GetAsymmetricQuantizationParams(
    const float min, const float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params) {
  // Adjust the boundaries to guarantee 0 is included.
  const float quant_min_float = std::min(static_cast<float>(quant_min), 0.0f);
  const float quant_max_float = std::max(static_cast<float>(quant_max), 0.0f);
  const float scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / scale;
  int64_t zero_point;
  if (zero_point_from_min < quant_min_float) {
    zero_point = static_cast<int64_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    zero_point = static_cast<int64_t>(quant_max);
  } else {
    zero_point = static_cast<int64_t>(std::round(zero_point_from_min));
  }
  quantization_params->scale = std::vector<float>(1, scale);
  quantization_params->zero_point = std::vector<int64_t>(1, zero_point);
}

// Returns the number of elements in tensor.
uint64_t NumElements(const TensorT* tensor) {
  if (tensor->shape.empty()) {
    LOG(FATAL) << "Tensor has no shape information.";
  }
  uint64_t num_elements = 1;
  for (const uint64_t dim : tensor->shape) {
    num_elements *= dim;
  }
  return num_elements;
}

uint64_t CountTensorConsumers(const ModelT* model, const SubGraphT* subgraph,
                              int32_t tensor_idx) {
  uint64_t count = 0;
  for (int op_idx = 0; op_idx < subgraph->operators.size(); ++op_idx) {
    const OperatorT* op = subgraph->operators[op_idx].get();
    if (op == nullptr) {
      continue;
    }
    for (int i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == tensor_idx) {
        count++;
      }
    }
  }
  return count;
}

// Gets the list of op->inputs indices of the weights inputs to be quantized for
// the provided op.
std::vector<int32_t> GetWeightInputIndices(const BuiltinOperator& op_code) {
  if (op_code == BuiltinOperator_CONV_2D ||
      op_code == BuiltinOperator_DEPTHWISE_CONV_2D ||
      op_code == BuiltinOperator_FULLY_CONNECTED ||
      op_code == BuiltinOperator_EMBEDDING_LOOKUP) {
    return {1};
  } else if (op_code == BuiltinOperator_SVDF) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/svdf.cc
    return {1, 2};
  } else if (op_code == BuiltinOperator_LSTM ||
             op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/lstm.cc
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/unidirectional_sequence_lstm.cc
    return {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16};
  } else if (op_code == BuiltinOperator_RNN ||
             op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/basic_rnn.cc
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/unidirectional_sequence_rnn.cc
    return {1, 2};
  } else if (op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/bidirectional_sequence_lstm.cc
    return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33};
  } else if (op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN) {
    // https://www.tensorflow.org/code/tensorflow/lite/kernels/bidirectional_sequence_rnn.cc
    return {1, 2, 4, 5};
  }
  return {};
}

// Returns true if the operator supports hybrid evaluation.
bool IsHybridEvaluationOp(const OperatorT* op, const BuiltinOperator& op_code) {
  // Operations that support hybrid evaluation.
  bool eval_hybrid = false;
  if (op_code == BuiltinOperator_FULLY_CONNECTED ||
      op_code == BuiltinOperator_CONV_2D || op_code == BuiltinOperator_SVDF ||
      op_code == BuiltinOperator_EMBEDDING_LOOKUP ||
      op_code == BuiltinOperator_RNN ||
      op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM ||
      op_code == BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN ||
      op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM ||
      op_code == BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN) {
    eval_hybrid = true;
  } else if (op_code == BuiltinOperator_LSTM) {
    const LSTMOptionsT* options = op->builtin_options.AsLSTMOptions();
    // Only lstm kernel_type full supports hybrid evaluation.
    if (options->kernel_type == LSTMKernelType_FULL) {
      eval_hybrid = true;
    }
  }
  return eval_hybrid;
}

// Returns a vector of TensorInfos for each input tensor of op that should be
// quantized.
std::vector<TensorInfo> GetQuantizableTensorsFromOperator(
    const ModelT* model, const OperatorT* op, uint64_t weights_min_num_elements,
    bool use_hybrid_evaluation) {
  SubGraphT* subgraph = model->subgraphs.at(0).get();
  const BuiltinOperator op_code =
      model->operator_codes[op->opcode_index]->builtin_code;

  std::vector<TensorInfo> tensor_infos;

  bool eval_hybrid = use_hybrid_evaluation && IsHybridEvaluationOp(op, op_code);

  std::vector<int32_t> op_input_indices = GetWeightInputIndices(op_code);
  for (const int32_t op_input_idx : op_input_indices) {
    int32_t tensor_idx = op->inputs[op_input_idx];

    if (tensor_idx == -1) {
      LOG(INFO) << "Skipping optional tensor input " << op_input_idx
                << " of operation " << EnumNameBuiltinOperator(op_code);
      continue;
    }

    TensorT* tensor = subgraph->tensors[tensor_idx].get();
    // TODO(suharshs): Support shared weights, i.e. If two tensors share the
    // same weight array, things may break. (i.e. SSD object detection)
    if (CountTensorConsumers(model, subgraph, tensor_idx) != 1) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " that is shared between multiple multiple operations.";
      continue;
    }

    if (tensor->type != TensorType_FLOAT32) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " that is not type float.";
      continue;
    }

    const uint64_t num_elements = NumElements(tensor);
    if (num_elements < weights_min_num_elements) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has fewer than " << weights_min_num_elements
                << " elements (" << num_elements << ").";
      // If one of the weights isn't quantized, then we cannot use the hybrid
      // kernel for this operation, since it expects everything to be quantized.
      eval_hybrid = false;
      continue;
    }

    // Some tensors may have a null buffer vector, indicating an intermediate
    // array.
    if (model->buffers[tensor->buffer]->data.data() == nullptr) {
      LOG(INFO) << "Skipping quantization of tensor " << tensor->name
                << " because it has no allocated buffer.";
      continue;
    }

    TensorInfo tensor_info;
    tensor_info.eval_hybrid = eval_hybrid;
    tensor_info.op_input_idx = op_input_idx;
    tensor_info.tensor_idx = tensor_idx;
    tensor_info.tensor = tensor;

    tensor_infos.push_back(tensor_info);
  }

  return tensor_infos;
}

// Quantizes tensor using asymmetric quantization with the min and max elements
// of the tensor. This is needed to pass to Dequantize operations.
TfLiteStatus AsymmetricQuantizeTensor(ModelT* model, TensorT* tensor) {
  BufferT* buffer = model->buffers[tensor->buffer].get();
  float* float_data = reinterpret_cast<float*>(buffer->data.data());
  const uint64_t num_elements = NumElements(tensor);
  LOG(INFO) << "Quantizing tensor " << tensor->name << " with " << num_elements
            << " elements for float evaluation.";

  // Compute the quantization params.
  float min_value = *std::min_element(float_data, float_data + num_elements);
  float max_value = *std::max_element(float_data, float_data + num_elements);

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  GetAsymmetricQuantizationParams(min_value, max_value, 0, 255,
                                  tensor->quantization.get());

  // Quantize the buffer.
  std::vector<uint8_t> quantized_buffer;
  quantized_buffer.resize(num_elements);
  const double inverse_scale = 1. / tensor->quantization->scale[0];
  for (std::size_t i = 0; i < num_elements; i++) {
    const float src_val = float_data[i];
    double scaled_val;
    if (tensor->quantization->scale[0] == 0) {
      scaled_val = tensor->quantization->zero_point[0];
    } else {
      scaled_val =
          tensor->quantization->zero_point[0] + inverse_scale * src_val;
    }
    uint8_t integer_val = static_cast<uint8_t>(std::round(scaled_val));
    quantized_buffer[i] = integer_val;
  }
  model->buffers[tensor->buffer]->data = quantized_buffer;

  // Update the tensor type.
  tensor->type = TensorType_UINT8;

  return kTfLiteOk;
}

// Quantizes tensor using symmetric quantization with the min and max elements
// of the tensor. This is need for operations with hybrid evaluation
// implemented.
TfLiteStatus SymmetricQuantizeTensor(ModelT* model, TensorT* tensor) {
  BufferT* buffer = model->buffers[tensor->buffer].get();
  float* float_data = reinterpret_cast<float*>(buffer->data.data());
  const uint64_t num_elements = NumElements(tensor);
  LOG(INFO) << "Quantizing tensor " << tensor->name << " with " << num_elements
            << " elements for hybrid evaluation.";

  std::vector<int8_t> quantized_buffer;
  quantized_buffer.resize(num_elements);

  float min_value, max_value, scaling_factor;
  tensor_utils::SymmetricQuantizeFloats(float_data, num_elements,
                                        quantized_buffer.data(), &min_value,
                                        &max_value, &scaling_factor);

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  tensor->quantization->scale = std::vector<float>(1, scaling_factor);
  tensor->quantization->zero_point = std::vector<int64_t>(1, 0);

  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(uint8_buffer,
                                              uint8_buffer + num_elements);

  // Update the tensor type.
  tensor->type = TensorType_UINT8;

  return kTfLiteOk;
}

// Returns the index of the Dequantize op_code.
// If a Dequantize op_code doesn't exist, adds it and returns its index.
int32_t GetOrInsertDequantizeOpCodeIndex(ModelT* model) {
  for (int i = 0; i < model->operator_codes.size(); ++i) {
    if (model->operator_codes[i]->builtin_code == BuiltinOperator_DEQUANTIZE) {
      return i;
    }
  }
  model->operator_codes.push_back(absl::make_unique<OperatorCodeT>());
  int op_code_idx = model->operator_codes.size() - 1;
  model->operator_codes[op_code_idx]->builtin_code = BuiltinOperator_DEQUANTIZE;
  // TODO(suharshs): How should the version be set in this op_code?

  // Return the index of the newly placed OperatorCodeT.
  return op_code_idx;
}

// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output) {
  OperatorT* op_raw = new OperatorT;
  op_raw->opcode_index = GetOrInsertDequantizeOpCodeIndex(model);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}

// Create a new TensorT object.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = shape;

  tensor->reset(tensor_raw);
}

TfLiteStatus QuantizeWeightsInternal(flatbuffers::FlatBufferBuilder* builder,
                                     const Model* input_model,
                                     bool use_hybrid_evaluation,
                                     uint64_t weights_min_num_elements) {
  std::unique_ptr<ModelT> model;
  model.reset(input_model->UnPack());

  // TODO(suharshs): When models support multiple subgraphs, add support.
  if (model->subgraphs.size() != 1) {
    LOG(ERROR) << "Quantize weights tool only supports tflite models with one "
                  "subgraph.";
    return kTfLiteError;
  }

  SubGraphT* subgraph = model->subgraphs.at(0).get();

  std::vector<std::unique_ptr<OperatorT>> new_operators;
  for (int i = 0; i < subgraph->operators.size(); ++i) {
    OperatorT* op = subgraph->operators[i].get();

    std::vector<TensorInfo> tensor_infos = GetQuantizableTensorsFromOperator(
        model.get(), op, weights_min_num_elements, use_hybrid_evaluation);

    for (const TensorInfo& tensor_info : tensor_infos) {
      if (tensor_info.eval_hybrid) {
        // Quantize the tensor.
        TF_LITE_ENSURE_STATUS(
            SymmetricQuantizeTensor(model.get(), tensor_info.tensor));
      } else {
        // Quantize the tensor.
        TF_LITE_ENSURE_STATUS(
            AsymmetricQuantizeTensor(model.get(), tensor_info.tensor));

        // Create a new tensor to be the output of the dequantize op.
        std::unique_ptr<TensorT> dequantize_output;
        MakeTensor(tensor_info.tensor->name + "_dequantize",
                   tensor_info.tensor->shape, &dequantize_output);
        const int32_t dequantize_output_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(dequantize_output));

        // Create the Dequantize operation.
        std::unique_ptr<OperatorT> dequantize_op;
        MakeDequantizeOperator(model.get(), &dequantize_op,
                               tensor_info.tensor_idx, dequantize_output_idx);

        // Update the op_input of tensor_idx to dequantize_output_idx.
        op->inputs[tensor_info.op_input_idx] = dequantize_output_idx;

        // Insert the newly created Dequantize operation.
        new_operators.push_back(std::move(dequantize_op));
      }
    }
    // After (maybe) quantizing inputs, we copy the operator into the new list.
    new_operators.push_back(std::move(subgraph->operators[i]));
  }

  // At this point all unique_ptrs in the original operators are invalid, and
  // we need to replace it with the new_operators vector.
  subgraph->operators = std::move(new_operators);

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model.get());
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

}  // namespace

namespace internal {
TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             bool use_hybrid_evaluation) {
  // By default we require that only weights with more than
  // kWeightsMinSizeDefault elements are quantized.
  return QuantizeWeightsInternal(builder, input_model, use_hybrid_evaluation,
                                 kWeightsMinNumElementsDefault);
}
}  // namespace internal

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model,
                             uint64_t weights_min_num_elements) {
  return QuantizeWeightsInternal(builder, input_model, true,
                                 weights_min_num_elements);
}

TfLiteStatus QuantizeWeights(flatbuffers::FlatBufferBuilder* builder,
                             const Model* input_model) {
  // By default we require that only weights with more than
  // kWeightsMinSizeDefault elements are quantized.
  return QuantizeWeightsInternal(builder, input_model, true,
                                 kWeightsMinNumElementsDefault);
}

}  // namespace optimize
}  // namespace tflite
