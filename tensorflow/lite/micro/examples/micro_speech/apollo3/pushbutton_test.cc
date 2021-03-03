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

/* This file is a modification of the Tensorflow Micro Lite file
 * micro_speech_test.cc */

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/micro_speech/simple_features/model.h"
#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern int16_t captured_data[16000];
uint8_t g_silence_score = 0;
uint8_t g_unknown_score = 0;
uint8_t g_yes_score = 0;
uint8_t g_no_score = 0;

namespace {

TfLiteStatus GenerateSimpleFeatures_1sec(tflite::ErrorReporter* error_reporter,
                                         const int16_t* input,
                                         uint8_t* output) {
  int i;
  for (i = 0; i < 49; i++) {
    GenerateSimpleFeatures(error_reporter, input + i * 320, 480, 43,
                           output + i * 43);
  }
  return kTfLiteOk;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestSimpleFeaturesGenerator) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  uint8_t preprocessed_data[43 * 49];
  TfLiteStatus generate_1sec_status = GenerateSimpleFeatures_1sec(
      error_reporter, captured_data, preprocessed_data);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, generate_1sec_status);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need.
  tflite::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  const int tensor_arena_size = 10 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(49, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(43, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // Copy a spectrogram created from a .wav audio file of someone saying "Yes",
  // into the memory area used for the input.
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = preprocessed_data[i];
  }

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // There are four possible classes in the output, each with a score.
  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;
  const int kYesIndex = 2;
  const int kNoIndex = 3;

  // Make sure that the expected "Yes" score is higher than the other classes.
  g_silence_score = output->data.uint8[kSilenceIndex];
  g_unknown_score = output->data.uint8[kUnknownIndex];
  g_yes_score = output->data.uint8[kYesIndex];
  g_no_score = output->data.uint8[kNoIndex];

  TF_LITE_REPORT_ERROR(error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
