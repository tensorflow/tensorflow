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

#include "tensorflow/lite/experimental/micro/examples/micro_vision/image_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_vision/person_detect_model_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// TODO(rocky): This is too big for many platforms.  Need to implement a more
// efficient memory manager for intermediate tensors.
const int tensor_arena_size = 291 * 1024;
uint8_t tensor_arena[tensor_arena_size];

int main(int argc, char* argv[]) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;

  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       error_reporter);

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  while (true) {
    // Get image from provider.
    GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
             input->data.uint8);

    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed\n");
    }

    TfLiteTensor* output = interpreter.output(0);

    // Log the person score and no person score.
    uint8_t person_score = output->data.uint8[kPersonIndex];
    uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
    error_reporter->Report(
        "person data.  person score: %d, no person score: %d\n", person_score,
        no_person_score);
  }

  return 0;
}
