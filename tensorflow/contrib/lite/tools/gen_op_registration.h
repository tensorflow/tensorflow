/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_GEN_OP_REGISTRATION_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_GEN_OP_REGISTRATION_H_

#include "tensorflow/contrib/lite/model.h"

namespace tflite {

// Convert the custom op name to registration name following the convention.
// Example:
//   "custom_op" -> "CUSTOM_OP"
//   "CustomOp" -> "CUSTOM_OP"
// Note "Register_" suffix will be added later in the tool.
string NormalizeCustomOpName(const string& op);

// Read ops from the TFLite model.
// Enum name of builtin ops will be stored, such as "CONV_2D".
// Custom op name will be stored as it is.
void ReadOpsFromModel(const ::tflite::Model* model,
                      std::vector<string>* builtin_ops,
                      std::vector<string>* custom_ops);

}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOOLS_GEN_OP_REGISTRATION_H_
