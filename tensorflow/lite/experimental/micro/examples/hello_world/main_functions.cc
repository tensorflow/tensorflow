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

#include "tensorflow/lite/experimental/micro/examples/hello_world/main_functions.h"
#include "tensorflow/lite/experimental/micro/examples/hello_world/constants.h"
#include "tensorflow/lite/experimental/micro/examples/hello_world/output_handler.h"
#include "tensorflow/lite/experimental/micro/examples/hello_world/sine_model_data.h"
#include "tensorflow/lite/experimental/micro/tf_micro_simple_api.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int inference_count;

// The name of this function is important for Arduino compatibility.
void setup() {

  tf_micro_simple_setup(g_sine_model_data, tensor_arena, kTensorArenaSize);

  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x_val[1] = {position * kXrange};
  float results[1] = {0};

  tf_micro_simple_invoke(x_val, 1, results, 1);

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  //printf("%f %f", x_val[0], results[0]);
  HandleOutput(x_val[0], results[0]);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
