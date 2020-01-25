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

#include "tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h"
#include "tensorflow/lite/micro/examples/magic_wand/ring_micro_features_data.h"
#include "tensorflow/lite/micro/examples/magic_wand/slope_micro_features_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<5> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Create an area of memory to use for input, output, and intermediate arrays.
  // Finding the minimum value for your model may require some trial and error.
  const int tensor_arena_size = 60 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter.AllocateTensors();

  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(128, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(3, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);
  // The input is a 32 bit floating point value
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  // Provide an input value
  const float* ring_features_data = g_ring_micro_f9643d42_nohash_4_data;
  error_reporter->Report("%d", input->bytes);
  for (int i = 0; i < (input->bytes / sizeof(float)); ++i) {
    input->data.f[i] = ring_features_data[i];
  }

  // Run the model on this input and check that it succeeds
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // There are four possible classes in the output, each with a score.
  const int kWingIndex = 0;
  const int kRingIndex = 1;
  const int kSlopeIndex = 2;
  const int kNegativeIndex = 3;

  // Make sure that the expected "Ring" score is higher than the other
  // classes.
  float wing_score = output->data.f[kWingIndex];
  float ring_score = output->data.f[kRingIndex];
  float slope_score = output->data.f[kSlopeIndex];
  float negative_score = output->data.f[kNegativeIndex];
  TF_LITE_MICRO_EXPECT_GT(ring_score, wing_score);
  TF_LITE_MICRO_EXPECT_GT(ring_score, slope_score);
  TF_LITE_MICRO_EXPECT_GT(ring_score, negative_score);

  // Now test with a different input, from a recording of "Slope".
  const float* slope_features_data = g_slope_micro_f2e59fea_nohash_1_data;
  for (int i = 0; i < (input->bytes / sizeof(float)); ++i) {
    input->data.f[i] = slope_features_data[i];
  }

  // Run the model on this "Slope" input.
  invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);

  // Make sure that the expected "Slope" score is higher than the other classes.
  wing_score = output->data.f[kWingIndex];
  ring_score = output->data.f[kRingIndex];
  slope_score = output->data.f[kSlopeIndex];
  negative_score = output->data.f[kNegativeIndex];
  TF_LITE_MICRO_EXPECT_GT(slope_score, wing_score);
  TF_LITE_MICRO_EXPECT_GT(slope_score, ring_score);
  TF_LITE_MICRO_EXPECT_GT(slope_score, negative_score);
}

TF_LITE_MICRO_TESTS_END
