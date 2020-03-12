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

}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_MODIFY_MODEL_INTERFACE_H_
