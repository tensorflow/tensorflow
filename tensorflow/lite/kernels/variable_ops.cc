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

#include "tensorflow/lite/kernels/variable_ops.h"

namespace tflite {
namespace ops {
namespace custom {

extern "C" void AddVariableOps(::tflite::MutableOpResolver* resolver) {
  // Add variable op handlers.
  resolver->AddCustom("ReadVariable",
                      tflite::ops::custom::Register_READ_VARIABLE());
  resolver->AddCustom("AssignVariable",
                      tflite::ops::custom::Register_ASSIGN_VARIABLE());
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
