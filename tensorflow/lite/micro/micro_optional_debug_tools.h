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
// Optional debugging functionality. For small sized binaries, these are not
// needed.
#ifndef TENSORFLOW_LITE_MICRO_MICRO_OPTIONAL_DEBUG_TOOLS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_OPTIONAL_DEBUG_TOOLS_H_

#include "tensorflow/lite/micro/micro_interpreter.h"

namespace tflite {
// Prints a dump of what tensors and what nodes are in the interpreter.
class MicroInterpreter;
void PrintInterpreterState(MicroInterpreter* interpreter);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
struct pairTfLiteNodeAndRegistration {
  TfLiteNode node;
  const TfLiteRegistration* registration;
};
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_OPTIONAL_DEBUG_TOOLS_H_
