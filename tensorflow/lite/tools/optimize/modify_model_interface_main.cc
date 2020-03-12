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
#include <string>

#include "tensorflow/lite/tools/optimize/modify_model_interface.h"
//
// Note: This is a private API, subject to change.
int main(int argc, char** argv) {
  if (argc != 5) {
    printf(
        "Wrong number of arguments. Example: modify_model_interface_main "
        "${input} ${output} ${input_interface} ${output_interface}");
    return 1;
  }

  if (!strcmp(argv[3], "uint8") && !strcmp(argv[3], "int8")) {
    printf("Only support uint8 and int8 for input interface");
    return 1;
  }

  if (!strcmp(argv[4], "uint8") && !strcmp(argv[4], "int8")) {
    printf("Only support uint8 and int8 for output interface");
    return 1;
  }

  tflite::TensorType input = tflite::TensorType_INT8;
  tflite::TensorType output = tflite::TensorType_INT8;

  if (!strcmp(argv[3], "uint8")) {
    input = tflite::TensorType_UINT8;
  }
  if (!strcmp(argv[4], "uint8")) {
    output = tflite::TensorType_UINT8;
  }

  tflite::optimize::ModifyModelInterface(argv[1], argv[2], input, output);

  return 0;
}
