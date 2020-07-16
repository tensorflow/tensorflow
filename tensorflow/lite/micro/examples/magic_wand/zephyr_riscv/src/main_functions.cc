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

#include "tensorflow/lite/micro/examples/magic_wand/main_functions.h"

#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h"
#include "tensorflow/lite/micro/examples/magic_wand/constants.h"
#include "tensorflow/lite/micro/examples/magic_wand/gesture_predictor.h"
#include "tensorflow/lite/micro/examples/magic_wand/magic_wand_model_data.h"
#include "tensorflow/lite/micro/examples/magic_wand/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// #define ZEPHYR_RISCV_PROFILE

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddSoftmax();

  // For fully quantized models
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddReshape();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 128) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Set up failed\n");
  }
}

void loop() {
#ifdef ZEPHYR_RISCV_PROFILE
  unsigned int cycle = 0;
  static unsigned int count = 0;
  static const unsigned int CLEAR = 0;

  // Reset cycle count for ReadAccelerometer()
  __asm__ __volatile__("csrw mcycle, %0" : : "r"(CLEAR));
#endif /* ZEPHYR_RISCV_PROFILE */

  // Attempt to read new data from the accelerometer.
  bool got_data =
      ReadAccelerometer(error_reporter, model_input->data.f, input_length);

#ifdef ZEPHYR_RISCV_PROFILE
  __asm__ __volatile__("csrr %0, mcycle" : "=r"(cycle));
  // Accumulate cycle count till enough data is collected
  count += cycle;
#endif /* ZEPHYR_RISCV_PROFILE */

  // If there was no new data, wait until next time.
  if (!got_data) return;

#ifdef ZEPHYR_RISCV_PROFILE
  TF_LITE_REPORT_ERROR(error_reporter,
                       "[ZEPHYR_RISCV_PROFILE] Before Invoke(): %d cycles\n",
                       count);

  // Reset cycle count for Invoke()
  __asm__ __volatile__("csrw mcycle, %0" : : "r"(CLEAR));
#endif /* ZEPHYR_RISCV_PROFILE */

  // Run inference, and report any error.
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on index: %d\n",
                         begin_index);
    return;
  }

#ifdef ZEPHYR_RISCV_PROFILE
  __asm__ __volatile__("csrr %0, mcycle" : "=r"(cycle));
  TF_LITE_REPORT_ERROR(error_reporter,
                       "[ZEPHYR_RISCV_PROFILE] Invoke(): %d cycles\n",
                       (int)cycle);

  // Reset cycle count for PredictGesture()
  __asm__ __volatile__("csrw mcycle, %0" : : "r"(CLEAR));
#endif /* ZEPHYR_RISCV_PROFILE */

  // Analyze the results to obtain a prediction
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

#ifdef ZEPHYR_RISCV_PROFILE
  __asm__ __volatile__("csrr %0, mcycle" : "=r"(cycle));
  TF_LITE_REPORT_ERROR(error_reporter,
                       "[ZEPHYR_RISCV_PROFILE] PredictGesture(): %d cycles\n",
                       (int)cycle);

  // Reset cycle count for HandleOutput()
  __asm__ __volatile__("csrw mcycle, %0" : : "r"(CLEAR));
#endif /* ZEPHYR_RISCV_PROFILE */

  // Produce an output
  HandleOutput(error_reporter, gesture_index);

#ifdef ZEPHYR_RISCV_PROFILE
  __asm__ __volatile__("csrr %0, mcycle" : "=r"(cycle));
  TF_LITE_REPORT_ERROR(error_reporter,
                       "[ZEPHYR_RISCV_PROFILE] HandleOutput(): %d cycles\n",
                       (int)cycle);
#endif /* ZEPHYR_RISCV_PROFILE */
}
