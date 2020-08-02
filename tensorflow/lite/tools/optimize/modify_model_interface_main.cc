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

  const std::unordered_map<std::string, tflite::TensorType> supported_types{
      {"uint8", tflite::TensorType_UINT8},
      {"int8", tflite::TensorType_INT8},
      {"int16", tflite::TensorType_INT16}};

  tflite::TensorType input = tflite::TensorType_INT8;
  tflite::TensorType output = tflite::TensorType_INT8;

  try {
    input = supported_types.at(argv[3]);
    output = supported_types.at(argv[4]);
  } catch (const std::out_of_range&) {
    printf("Only supports uint8, int8 and int16 for input and output types");
    return 1;
  }

  tflite::optimize::ModifyModelInterface(argv[1], argv[2], input, output);

  return 0;
}
