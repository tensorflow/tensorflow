/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_benchmark.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int tensor_arena_size = 73 * 1024;
uint8_t tensor_arena[tensor_arena_size];

/*
 * Person Detection benchmark.  Evaluates runtime performance of the visual
 * wakewords person detection model.  This is the same model found in
 * exmaples/person_detection.
 */

namespace {

// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int tensor_arena_size = 73 * 1024;
uint8_t tensor_arena[tensor_arena_size];

class PersonDetectionRunner {
 public:
  PersonDetectionRunner()
      : person_detection_model_(tflite::GetModel(g_person_detect_model_data)),
        reporter_(&micro_reporter_),
        interpreter_(person_detection_model_, resolver_, tensor_arena,
                     tensor_arena_size, reporter_) {
    resolver_.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                         tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    resolver_.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                         tflite::ops::micro::Register_CONV_2D());
    resolver_.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
                         tflite::ops::micro::Register_AVERAGE_POOL_2D());
    interpreter_.AllocateTensors();

    TfLiteTensor* input = interpreter_.input(0);
    TFLITE_CHECK_EQ(input->type, kTfLiteUInt8);
  }

  void RunSingleIterationWithPerson() {
    // Populate input tensor with an image with a person
    TfLiteTensor* input = interpreter_.input(0);
    int8_t* input_buffer = tflite::GetTensorData<int8_t>(input);
    int input_length = tflite::ElementCount(*input->dims);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = g_person_data[i];
    }

    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

  void RunSingleIterationWithoutPerson() {
    // Populate input tensor with an image with no person.
    TfLiteTensor* input = interpreter_.input(0);
    int8_t* input_buffer = tflite::GetTensorData<int8_t>(input);
    int input_length = tflite::ElementCount(*input->dims);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = g_no_person_data[i];
    }

    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(reporter_, "Invoke failed.");
    }
  }

 private:
  const tflite::Model* person_detection_model_;
  tflite::MicroErrorReporter micro_reporter_;
  tflite::ErrorReporter* reporter_;
  tflite::MicroOpResolver<6> resolver_;
  tflite::MicroInterpreter interpreter_;
};

// NOLINTNEXTLINE
PersonDetectionRunner runner;

void PersonDetectionFirstIteration() { runner.RunSingleIterationWithPerson(); }

void PersonDetectionTenIerationsWithPerson() {
  // TODO(b/152644476): Add a way to run more than a single deterministic input.
  for (int i = 0; i < 10; i++) {
    runner.RunSingleIterationWithPerson();
  }
}

void PersonDetectionTenIerationsWithoutPerson() {
  // TODO(b/152644476): Add a way to run more than a single deterministic input.
  for (int i = 0; i < 10; i++) {
    runner.RunSingleIterationWithoutPerson();
  }
}

}  // namespace

TF_LITE_MICRO_BENCHMARKS_BEGIN

TF_LITE_MICRO_BENCHMARK(PersonDetectionFirstIteration);
TF_LITE_MICRO_BENCHMARK(PersonDetectionTenIerationsWithPerson);
TF_LITE_MICRO_BENCHMARK(PersonDetectionTenIerationsWithoutPerson);

TF_LITE_MICRO_BENCHMARKS_END
