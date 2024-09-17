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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace utils {

// LINT.IfChange(MakeDequantizeOperator)
// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output);
// LINT.ThenChange(//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.cc:MakeDequantizeOperator)

// Creates a Quantize OperatorT object.
void MakeQuantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                          int32_t input, int32_t output);

// LINT.IfChange(MakeTensor)
// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor);
// LINT.ThenChange(//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.cc:MakeTensor)

// Create a new TensorT object with quantization parameters.
void MakeTensorWithQuantParam(const string& name,
                              const std::vector<int32_t>& shape,
                              const std::vector<int32_t>& shape_signature,
                              const TensorType& type, float scale,
                              int64_t zero_point,
                              std::unique_ptr<TensorT>* tensor);

// Checks if the tensor has scale and zero point populated.
bool QuantizationParametersExist(const TensorT* tensor);

bool HasBuffer(const ModelT* model, const SubGraphT* subgraph,
               int tensor_index);

// LINT.IfChange(HasMinMax)
bool HasMinMax(const TensorT* tensor);
// LINT.ThenChange(//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.cc:HasMinMax)

// Set version of OperatorCode. The version will only be applied for operations
// that have been quantized.
void SetOperatorCodeVersion(ModelT* model);

// Writes model buffer to file.
void WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes);

// Finishes model buffer and returns its builder.
std::unique_ptr<flatbuffers::FlatBufferBuilder> FinishModel(
    const tflite::ModelT* model);

// Reads TensorFlow Lite model from the given path.
std::unique_ptr<tflite::ModelT> CreateMutableModelFromFile(
    const string& model_filepath);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODEL_UTILS_H_
