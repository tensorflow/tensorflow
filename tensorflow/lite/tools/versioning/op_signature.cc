/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/versioning/op_signature.h"

#include <cstdlib>

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace {

// A BuiltinDataAllocator which just uses malloc()/free().
class MallocDataAllocator : public BuiltinDataAllocator {
 public:
  void* Allocate(size_t size, size_t alignment_hint) override {
    return malloc(size);
  }
  void Deallocate(void* data) override { free(data); }
};

// Get the number of dimensions of a tensor with idx of an operator op.
inline int GetNumDims(const SubGraph* subgraph, const Operator* op, int idx) {
  const flatbuffers::Vector<int32_t>* ret =
      subgraph->tensors()->Get(op->inputs()->Get(idx))->shape();
  if (ret) {
    return ret->size();
  } else {
    return 0;
  }
}

std::vector<OpSignatureTensorSpec> GetOpSignatureTensorSpecs(
    const flatbuffers::Vector<int32_t>* tensors, const SubGraph* subgraph,
    const Model* model) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  if (!tensors) {
    return tensor_specs;
  }
  StderrReporter error_reporter;

  for (int32_t i = 0; i < tensors->Length(); ++i) {
    int32_t tensor_no = tensors->Get(i);

    OpSignatureTensorSpec tensor_spec = {kTfLiteNoType};
    if (tensor_no >= 0) {
      if (subgraph->tensors() && tensor_no < subgraph->tensors()->Length()) {
        auto* fb_tensor = subgraph->tensors()->Get(tensor_no);
        ConvertTensorType(fb_tensor->type(), &tensor_spec.type,
                          &error_reporter);
        auto buffer_idx = fb_tensor->buffer();
        // Check if the tensor is a constant tensor.
        if (buffer_idx != 0 && buffer_idx < model->buffers()->Length()) {
          auto* buffer = model->buffers()->Get(buffer_idx);
          if (buffer->data() && buffer->data()->size() != 0) {
            tensor_spec.is_const = true;
          }
        }
        const flatbuffers::Vector<int32_t>* shape_vec = fb_tensor->shape();
        if (shape_vec) {
          for (int32_t j = 0; j < shape_vec->Length(); ++j) {
            tensor_spec.dims.push_back(shape_vec->Get(j));
          }
        }
        const flatbuffers::Vector<int32_t>* shape_signature_vec =
            fb_tensor->shape_signature();
        tensor_spec.is_shape_dynamic = false;
        if (shape_signature_vec) {
          for (int32_t j = 0; j < shape_signature_vec->Length(); ++j) {
            if (shape_signature_vec->Get(j) == -1) {
              tensor_spec.is_shape_dynamic = true;
              break;
            }
          }
        }
      }
    }
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

std::vector<OpSignatureTensorSpec> GetOpSignatureTensorSpecs(
    TfLiteIntArray* tensors, const TfLiteContext* context,
    const TfLiteNode* tflite_node) {
  std::vector<OpSignatureTensorSpec> tensor_specs;

  for (int32_t i = 0; i < tensors->size; ++i) {
    int32_t tensor_no = tensors->data[i];

    OpSignatureTensorSpec tensor_spec = {kTfLiteNoType};
    if (tensor_no >= 0) {
      const TfLiteTensor* tfl_tensor;
      if (context->tensors != nullptr) {
        tfl_tensor = &context->tensors[tensor_no];
      } else {
        tfl_tensor = context->GetTensor(context, tensor_no);
      }
      if (tfl_tensor != nullptr) {
        tensor_spec.type = tfl_tensor->type;
        tensor_spec.is_const = (tfl_tensor->allocation_type == kTfLiteMmapRo);
        if (tfl_tensor->dims) {
          for (int32_t j = 0; j < tfl_tensor->dims->size; ++j) {
            tensor_spec.dims.push_back(tfl_tensor->dims->data[j]);
          }
        }
        tensor_spec.is_shape_dynamic = HasUnspecifiedDimension(tfl_tensor);
      }
    }
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

}  // namespace

OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph, const Model* model) {
  auto builtin_code = GetBuiltinCode(op_code);
  OpSignature op_sig = {builtin_code};
  std::memset(&op_sig.ext_options, 0, sizeof(op_sig.ext_options));

  if (builtin_code != BuiltinOperator_CUSTOM) {
    StderrReporter error_reporter;
    MallocDataAllocator allocator;
    ParseOpData(op, builtin_code, &error_reporter, &allocator,
                &op_sig.builtin_data);
  } else {
    op_sig.custom_name = op_code->custom_code()->str();
  }

  switch (builtin_code) {
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_channels = filter_tensor->shape()->Get(3);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_channels) {
        op_sig.ext_options.depthwise_conv_2d.is_per_channel_quantized = true;
      }
    } break;

    case BuiltinOperator_FULLY_CONNECTED: {
      const Tensor* weight_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      op_sig.ext_options.fully_connected.sparse_weight =
          (weight_tensor->sparsity() != nullptr);
    } break;

    case BuiltinOperator_MUL: {
      if (op->inputs()->Length() < 2 || op->outputs()->Length() < 1) {
        break;
      }
      const Tensor* input1_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const Tensor* input2_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const Tensor* output_tensor =
          subgraph->tensors()->Get(op->outputs()->Get(0));
      const QuantizationParameters* input1_quant =
          input1_tensor->quantization();
      const QuantizationParameters* input2_qunt = input2_tensor->quantization();
      const QuantizationParameters* output_quant =
          output_tensor->quantization();
      if (input1_quant && input1_quant->scale() &&
          input1_quant->scale()->Length() && input2_qunt &&
          input2_qunt->scale() && input2_qunt->scale()->Length() &&
          output_quant && output_quant->scale() &&
          output_quant->scale()->Length()) {
        op_sig.ext_options.mul.input1_scale = input1_quant->scale()->Get(0);
        op_sig.ext_options.mul.input2_scale = input2_qunt->scale()->Get(0);
        op_sig.ext_options.mul.output_scale = output_quant->scale()->Get(0);
      }
      if (input1_quant || input2_qunt) {
        op_sig.ext_options.mul.input_quantized = true;
      }
    } break;

    case BuiltinOperator_CONV_2D: {
      const Tensor* input_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const Tensor* filter_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(1));
      const QuantizationParameters* filter_quant =
          filter_tensor->quantization();
      int num_filters = filter_tensor->shape()->Get(0);
      if (filter_quant && filter_quant->scale() &&
          filter_quant->scale()->Length() &&
          filter_quant->scale()->Length() == num_filters) {
        op_sig.ext_options.conv_2d.is_per_channel_quantized = true;
      }
      if (input_tensor->shape() && input_tensor->shape()->size()) {
        int num_input_channels = input_tensor->shape()->Get(3);
        int num_filter_input_channels = filter_tensor->shape()->Get(3);
        op_sig.ext_options.conv_2d.is_grouped_convolution =
            num_input_channels != num_filter_input_channels;
      } else {
        op_sig.ext_options.conv_2d.is_grouped_convolution = false;
      }
    } break;

    case BuiltinOperator_STRIDED_SLICE: {
      op_sig.ext_options.strided_slice.num_dims = GetNumDims(subgraph, op, 0);
    } break;

    case BuiltinOperator_ABS: {
      if (subgraph->tensors()->Get(op->inputs()->Get(0))->quantization()) {
        op_sig.ext_options.abs.input_quantized = true;
      }
    } break;

    case BuiltinOperator_DEQUANTIZE: {
      const Tensor* input_tensor =
          subgraph->tensors()->Get(op->inputs()->Get(0));
      const QuantizationParameters* input_quant = input_tensor->quantization();
      if (input_quant && input_quant->scale() &&
          input_quant->scale()->Length() > 1 &&
          input_quant->scale()->Length() ==
              input_tensor->shape()->Get(input_quant->quantized_dimension())) {
        op_sig.ext_options.dequantize.is_per_channel_quantized = true;
      }
    } break;

    case BuiltinOperator_QUANTIZE: {
      const Tensor* output_tensor =
          subgraph->tensors()->Get(op->outputs()->Get(0));
      const QuantizationParameters* output_quant =
          output_tensor->quantization();
      if (output_quant && output_quant->scale() &&
          output_quant->scale()->Length() > 1 &&
          output_quant->scale()->Length() ==
              output_tensor->shape()->Get(
                  output_quant->quantized_dimension())) {
        op_sig.ext_options.quantize.is_per_channel_quantized = true;
      }
    } break;

    case BuiltinOperator_ADD: {
      if (subgraph->tensors()->Get(op->inputs()->Get(0))->quantization()) {
        op_sig.ext_options.add.input_quantized = true;
      }
    } break;

    case BuiltinOperator_EQUAL:
    case BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL: {
      if (subgraph->tensors()->Get(op->inputs()->Get(0))->quantization()) {
        op_sig.ext_options.comparator.input_quantized = true;
      }
    } break;

    default:
      break;
  }

  op_sig.inputs = GetOpSignatureTensorSpecs(op->inputs(), subgraph, model);
  op_sig.outputs = GetOpSignatureTensorSpecs(op->outputs(), subgraph, model);
  op_sig.version = op_code->version();
  return op_sig;
}

OpSignature GetOpSignature(const TfLiteContext* context, const TfLiteNode* node,
                           const TfLiteRegistration* registration) {
  OpSignature op_sig = {
      static_cast<BuiltinOperator>(registration->builtin_code)};
  op_sig.builtin_data = node->builtin_data;
  if (op_sig.op == BuiltinOperator_CUSTOM) {
    op_sig.custom_name = registration->custom_name;
    op_sig.custom_initial_data = node->custom_initial_data;
  }
  std::memset(&op_sig.ext_options, 0, sizeof(op_sig.ext_options));

  op_sig.inputs = GetOpSignatureTensorSpecs(node->inputs, context, node);
  op_sig.outputs = GetOpSignatureTensorSpecs(node->outputs, context, node);
  op_sig.version = registration->version;
  return op_sig;
}

}  // namespace tflite
