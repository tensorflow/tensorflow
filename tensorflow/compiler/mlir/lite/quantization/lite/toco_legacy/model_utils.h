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
// third_party/tensorflow/lite/tools/optimize/model_utils.h as part of the
// effort to decouple TFLite from MLIR.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_MODEL_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_MODEL_UTILS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

using std::string;
using tflite::ModelT;
using tflite::OperatorT;
using tflite::TensorT;
using tflite::TensorType;

// LINT.IfChange(MakeDequantizeOperator)
// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.h:MakeDequantizeOperator)

// LINT.IfChange(MakeTensor)
// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.h:MakeTensor)

// LINT.IfChange(HasMinMax)
bool HasMinMax(const TensorT* tensor);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/model_utils.h:HasMinMax)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_MODEL_UTILS_H_
