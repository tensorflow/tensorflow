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

#include <cstdint>
#include <cstdlib>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_benchmark.h"

/*
 * Keyword Spotting Benchmark for performance optimizations. The model used in
 * this benchmark only serves as a reference. The values assigned to the model
 * weights and parameters are not representative of the original model.
 */

namespace {

// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int tensor_arena_size = 73 * 1024;
uint8_t tensor_arena[tensor_arena_size];
// A random number generator seed to generate input values.
constexpr int kRandomSeed = 42;

class KeywordRunner {
 public:
  KeywordRunner()
      : keyword_spotting_model_(
            tflite::GetModel(g_keyword_scrambled_model_data)),
        reporter_(&micro_reporter_),
        interpreter_(keyword_spotting_model_, resolver_, tensor_arena,
                     tensor_arena_size, reporter_) {
    resolver_.AddBuiltin(tflite::BuiltinOperator_SVDF,
                         tflite::ops::micro::Register_SVDF());
    resolver_.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                         tflite::ops::micro::Register_FULLY_CONNECTED());
    resolver_.AddBuiltin(tflite::BuiltinOperator_QUANTIZE,
                         tflite::ops::micro::Register_QUANTIZE());
    resolver_.AddBuiltin(tflite::BuiltinOperator_DEQUANTIZE,
                         tflite::ops::micro::Register_DEQUANTIZE(), 1, 2);
    resolver_.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                         tflite::ops::micro::Register_SOFTMAX());
    interpreter_.AllocateTensors();

    // The pseudo-random number generator is initialized to a constant seed
    std::srand(kRandomSeed);
    TfLiteTensor* input = interpreter_.input(0);
    TFLITE_CHECK_EQ(input->type, kTfLiteInt16);

    // Pre-populate input tensor with random values.
    int input_length = input->bytes / sizeof(int16_t);
    int16_t* input_values = tflite::GetTensorData<int16_t>(input);
    for (int i = 0; i < input_length; i++) {
      // Pre-populate input tensor with a random value based on a constant seed.
      input_values[i] = static_cast<int16_t>(std::rand() % INT16_MAX);
    }
  }

  void RunSingleIteration() {
    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

 private:
  const tflite::Model* keyword_spotting_model_;
  tflite::MicroErrorReporter micro_reporter_;
  tflite::ErrorReporter* reporter_;
  tflite::MicroMutableOpResolver<6> resolver_;
  tflite::MicroInterpreter interpreter_;
};

// NOLINTNEXTLINE
KeywordRunner runner;

void KeywordRunFirstIteration() { runner.RunSingleIteration(); }

void KeywordRunTenIerations() {
  // TODO(b/152644476): Add a way to run more than a single deterministic input.
  for (int i = 0; i < 10; i++) {
    runner.RunSingleIteration();
  }
}

}  //  namespace

TF_LITE_MICRO_BENCHMARKS_BEGIN

TF_LITE_MICRO_BENCHMARK(KeywordRunFirstIteration);

TF_LITE_MICRO_BENCHMARK(KeywordRunTenIerations);

TF_LITE_MICRO_BENCHMARKS_END
