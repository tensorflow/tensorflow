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
// NOTE: this is an example driver that converts a tflite model to TensorFlow.
// This is an example that will be integrated more tightly into tflite in
// the future.
//
// Usage: bazel run -c opt \
// tensorflow/contrib/lite/nnapi:nnapi_example -- <filename>
//
#include <cstdarg>
#include <cstdio>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"
#include "tensorflow/contrib/lite/testing/parse_testdata.h"

// TODO(aselle): FATAL leaves resources hanging.
void FATAL(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
  exit(1);
}

#define CHECK_TFLITE_SUCCESS(x)                       \
  if (x != kTfLiteOk) {                               \
    FATAL("Aborting since tflite returned failure."); \
  }

void Interpret(const char* filename, const char* examples_filename,
               bool use_nnapi) {
  // TODO(aselle): Resize of input image should go here
  // ...
  // For now I am allocating all tensors. This means I am fixed size.
  // So I am not using the variable size ability yet.
  fprintf(stderr, "example file %s\n", examples_filename);
  std::vector<tflite::testing::Example> examples;
  CHECK_TFLITE_SUCCESS(
      tflite::testing::ParseExamples(examples_filename, &examples));

  for (const tflite::testing::Example& example : examples) {
    auto model = tflite::FlatBufferModel::BuildFromFile(filename);
    if (!model) FATAL("Cannot read file %s\n", filename);
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver builtins;

    CHECK_TFLITE_SUCCESS(
        tflite::InterpreterBuilder(*model, builtins)(&interpreter));

    printf("Use nnapi is set to: %d\n", use_nnapi);
    interpreter->UseNNAPI(use_nnapi);
    CHECK_TFLITE_SUCCESS(
        tflite::testing::FeedExample(interpreter.get(), example));

    {
      TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
      if (float* data =
              interpreter->typed_tensor<float>(interpreter->outputs()[0])) {
        size_t num = tensor->bytes / sizeof(float);
        for (float* p = data; p < data + num; p++) {
          *p = 0;
        }
      }
    }
    interpreter->Invoke();

    CHECK_TFLITE_SUCCESS(
        tflite::testing::CheckOutputs(interpreter.get(), example));

    printf("Result:\n");
    TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
    if (float* data =
            interpreter->typed_tensor<float>(interpreter->outputs()[0])) {
      size_t num = tensor->bytes / sizeof(float);
      for (float* p = data; p < data + num; p++) {
        printf(" %f", *p);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  bool use_nnapi = true;
  if (argc == 4) {
    use_nnapi = strcmp(argv[3], "1") == 0 ? true : false;
  }
  if (argc < 3) {
    fprintf(stderr,
            "Compiled " __DATE__ __TIME__
            "\n"
            "Usage!!!: %s <tflite model> <examples to test> "
            "{ use nn api i.e. 0,1}\n",
            argv[0]);
    return 1;
  }
  Interpret(argv[1], argv[2], use_nnapi);
  return 0;
}
