/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODIFY_MODEL_INTERFACE_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODIFY_MODEL_INTERFACE_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {

// Changes the interface of a quantized model. This method allows the users to
// replace float interface with other types.
// This populates the builder with the new model.
// Currently only int8 and unit8 are supported.
//
// Note: This is a private API, subject to change.
TfLiteStatus ModifyModelInterface(flatbuffers::FlatBufferBuilder* builder,
                                  ModelT* model, const TensorType& input_type,
                                  const TensorType& output_type);

// Same as above but allows input file path and output file path.
//
// Note: This is a private API, subject to change.
TfLiteStatus ModifyModelInterface(const string& input_file,
                                  const string& output_file,
                                  const TensorType& input_type,
                                  const TensorType& output_type);

// Adds uint8 quantize ops for specified inputs and uint8 dequantize ops for
// specified outputs for a float model. The scale and zero point of uint8
// tensors are provided through quant_params.
//   - input_quant_params has a map between tensor name and the
//     <scale and zero_point> pair for inputs.
//   - output_quant_params has a map between tensor name and the
//     <scale and zero_point> pair for inputs.
// For the inputs/output tensors for the model, if its quantization parameters
// are not provided, that tensor is not affected.
//
// Note: This is a private API, subject to change.
TfLiteStatus Uint8QuantizeModelInputsOutputs(
    flatbuffers::FlatBufferBuilder* builder, const Model* input_model,
    const std::unordered_map<string, std::pair<float, int32_t>>&
        input_quant_params,
    const std::unordered_map<string, std::pair<float, int32_t>>&
        output_quant_params);

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODIFY_MODEL_INTERFACE_H_
