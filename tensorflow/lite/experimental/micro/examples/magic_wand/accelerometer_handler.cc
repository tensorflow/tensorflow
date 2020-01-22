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

#include "tensorflow/lite/experimental/micro/examples/magic_wand/accelerometer_handler.h"

int begin_index = 0;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  return kTfLiteOk;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length, bool reset_buffer) {
  begin_index += 3;
  // Reset begin_index to simulate behavior of loop buffer
  if (begin_index >= 600) begin_index = 0;
  // Only return true after the function was called 100 times, simulating the
  // desired behavior of a real implementation (which does not return data until
  // a sufficient amount is available)
  if (begin_index > 300) {
    for (int i = 0; i < length; ++i) input[i] = 0;
    return true;
  } else { return false; }
}
