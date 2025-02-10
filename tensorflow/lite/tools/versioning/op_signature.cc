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

#include <cstdint>
#include <cstring>
#include <vector>

#include "tensorflow/compiler/mlir/lite/tools/versioning/op_signature.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

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
