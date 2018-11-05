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

#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {

TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
TfLiteRegistration* Micro_Register_DEPTHWISE_CONV_2D() {
  return Register_DEPTHWISE_CONV_2D();
}

TfLiteRegistration* Register_FULLY_CONNECTED();
TfLiteRegistration* Micro_Register_FULLY_CONNECTED() {
  return Register_FULLY_CONNECTED();
}

TfLiteRegistration* Register_SOFTMAX();
TfLiteRegistration* Micro_Register_SOFTMAX() { return Register_SOFTMAX(); }

AllOpsResolver::AllOpsResolver() {
  AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D,
             Micro_Register_DEPTHWISE_CONV_2D());
  AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Micro_Register_FULLY_CONNECTED(),
             /* min_version */ 1,
             /* max_version */ 2);
  AddBuiltin(BuiltinOperator_SOFTMAX, Micro_Register_SOFTMAX());
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
