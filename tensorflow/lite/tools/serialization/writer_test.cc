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
// Loads the input tflite file into interpreter, serializes it back to a tflite
// buffer, and then verifies that the generated output can be loaded back into
// an interpreter and the model prepared (i.e., AllocateTensors returns ok).
//
// Usage:
//   writer_test <input tflite>

#include <iostream>

#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/serialization/writer_lib.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s input_file\n", argv[0]);
    return 1;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(argv[1]);
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates
      builtin_op_resolver;
  tflite::InterpreterBuilder(*model, builtin_op_resolver)(&interpreter);
  tflite::ModelWriter writer(interpreter.get());
  std::unique_ptr<uint8_t[]> output_buffer;
  size_t output_buffer_size;
  writer.GetBuffer(&output_buffer, &output_buffer_size);

  // Verify the generated model.
  std::unique_ptr<tflite::Interpreter> new_interpreter;
  model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<char*>(output_buffer.get()), output_buffer_size);
  tflite::InterpreterBuilder(*model, builtin_op_resolver)(&new_interpreter);
  if (new_interpreter->AllocateTensors() != kTfLiteOk) {
    fprintf(stderr, "AllocateTensors failed on the round-tripped model.\n");
    return 1;
  }
  return 0;
}
