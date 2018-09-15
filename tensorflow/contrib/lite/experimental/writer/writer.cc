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
// Just does a read/write loop of tflite file format using the interpreter as
// an intermediate.
//
// Usage:
//   writer <input tflite> <output tflite>

#include <iostream>

#include "tensorflow/contrib/lite/experimental/writer/writer_lib.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s input_file output_file\n", argv[0]);
    return 1;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(argv[1]);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver builtin_op_resolver;
  tflite::InterpreterBuilder(*model, builtin_op_resolver)(&interpreter);
  tflite::InterpreterWriter writer(interpreter.get());
  writer.Write(argv[2]);

  return 0;
}
