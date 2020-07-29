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

#ifndef TENSORFLOW_LITE_MICRO_BENCHMARKS_MICRO_BENCHMARK_H_
#define TENSORFLOW_LITE_MICRO_BENCHMARKS_MICRO_BENCHMARK_H_

#include <climits>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace micro_benchmark {
extern tflite::ErrorReporter* reporter;
}  // namespace micro_benchmark

#define TF_LITE_MICRO_BENCHMARKS_BEGIN           \
  namespace micro_benchmark {                    \
  tflite::ErrorReporter* reporter;               \
  }                                              \
                                                 \
  int main(int argc, char** argv) {              \
    tflite::MicroErrorReporter error_reporter;   \
    micro_benchmark::reporter = &error_reporter; \
    int32_t start_ticks;                         \
    int32_t duration_ticks;                      \
    int32_t duration_ms;

#define TF_LITE_MICRO_BENCHMARKS_END \
  return 0;                          \
  }

#define TF_LITE_MICRO_BENCHMARK(func)                                         \
  if (tflite::ticks_per_second() == 0) {                                      \
    return 0;                                                                 \
  }                                                                           \
  start_ticks = tflite::GetCurrentTimeTicks();                                \
  func;                                                                       \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;               \
  if (duration_ticks > INT_MAX / 1000) {                                      \
    duration_ms = duration_ticks / (tflite::ticks_per_second() / 1000);       \
  } else {                                                                    \
    duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second();       \
  }                                                                           \
  TF_LITE_REPORT_ERROR(micro_benchmark::reporter, "%s took %d ticks (%d ms)", \
                       #func, duration_ticks, duration_ms);

template <typename inputT>
class MicroBenchmarkRunner {
 public:
  MicroBenchmarkRunner(const uint8_t* model, uint8_t* tensor_arena,
                       int tensor_arena_size, int random_seed)
      : model_(tflite::GetModel(model)),
        reporter_(&micro_reporter_),
        interpreter_(model_, resolver_, tensor_arena, tensor_arena_size,
                     reporter_) {
    interpreter_.AllocateTensors();

    // The pseudo-random number generator is initialized to a constant seed
    std::srand(random_seed);
    TfLiteTensor* input = interpreter_.input(0);

    // Pre-populate input tensor with random values.
    int input_length = input->bytes / sizeof(inputT);
    inputT* input_values = tflite::GetTensorData<inputT>(input);
    for (int i = 0; i < input_length; i++) {
      // Pre-populate input tensor with a random value based on a constant seed.
      input_values[i] = static_cast<inputT>(
          std::rand() % (std::numeric_limits<inputT>::max() -
                         std::numeric_limits<inputT>::min() + 1));
    }
  }

  void RunSingleIterationRandomInput() {
    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

  void RunSingleIterationCustomInput(const inputT* custom_input) {
    // Populate input tensor with an image with no person.
    TfLiteTensor* input = interpreter_.input(0);
    inputT* input_buffer = tflite::GetTensorData<inputT>(input);
    int input_length = input->bytes / sizeof(inputT);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = custom_input[i];
    }

    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

 private:
  const tflite::Model* model_;
  tflite::MicroErrorReporter micro_reporter_;
  tflite::ErrorReporter* reporter_;
  tflite::AllOpsResolver resolver_;
  tflite::MicroInterpreter interpreter_;
};

#endif  // TENSORFLOW_LITE_MICRO_BENCHMARKS_MICRO_BENCHMARK_H_
