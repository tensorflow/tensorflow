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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/tools/optimize/model_utils.cc as part of the
// effort to decouple TFLite from MLIR.

#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

using std::string;
using tflite::BuiltinOperator;
using tflite::BuiltinOperator_DEQUANTIZE;
using tflite::ModelT;
using tflite::OperatorCodeT;
using tflite::OperatorT;
using tflite::TensorT;
using tflite::TensorType;

// LINT.IfChange(GetOrInsertOpCodeIndex)
// Returns the index of the OpCode.
// If a OpCode doesn't exist, adds it and returns its index.
int32_t GetOrInsertOpCodeIndex(ModelT* model, const BuiltinOperator& op_code,
                               int32_t version) {
  for (size_t i = 0; i < model->operator_codes.size(); ++i) {
    if (tflite::GetBuiltinCode(model->operator_codes[i].get()) == op_code) {
      return i;
    }
  }
  model->operator_codes.push_back(std::make_unique<OperatorCodeT>());
  int op_code_idx = model->operator_codes.size() - 1;
  model->operator_codes[op_code_idx]->builtin_code = op_code;
  model->operator_codes[op_code_idx]->deprecated_builtin_code =
      tflite::ConvertBuiltinCodeToDeprecatedBuiltinCode(op_code);
  // Version 2 and onwards supports INT8 inputs.
  model->operator_codes[op_code_idx]->version = version;

  // Return the index of the newly placed OperatorCodeT.
  return op_code_idx;
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.cc:GetOrInsertOpCodeIndex)

// LINT.IfChange(MakeDequantizeOperator)
// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output) {
  OperatorT* op_raw = new OperatorT;
  // Version 2 and onwards supports INT8 inputs.
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_DEQUANTIZE, 2);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.cc:MakeDequantizeOperator)

// LINT.IfChange(MakeTensor)
// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = shape;
  if (!shape_signature.empty()) {
    tensor_raw->shape_signature = shape_signature;
  }
  tensor_raw->type = type;

  tensor->reset(tensor_raw);
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.cc:MakeTensor)

// LINT.IfChange(HasMinMax)
bool HasMinMax(const TensorT* tensor) {
  return tensor->quantization && !tensor->quantization->min.empty() &&
         !tensor->quantization->max.empty();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.cc:HasMinMax)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir
