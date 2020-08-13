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

#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h"

#include "hx_drv_tflm.h"

int begin_index = 0;

namespace {
// Ring buffer size
constexpr int ring_buffer_size = 600;
// Ring buffer
float save_data[ring_buffer_size] = {0.0};
// Flag to start detect gesture
bool pending_initial_data = true;
// Available data count in accelerometer FIFO
int available_count = 0;

}  // namespace

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  if (hx_drv_accelerometer_initial() != HX_DRV_LIB_PASS) {
    TF_LITE_REPORT_ERROR(error_reporter, "setup fail");
    return kTfLiteError;
  }

  TF_LITE_REPORT_ERROR(error_reporter, "setup done");

  return kTfLiteOk;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length) {
  // Check how many accelerometer data
  available_count = hx_drv_accelerometer_available_count();

  if (available_count == 0) return false;

  for (int i = 0; i < available_count; i++) {
    float x, y, z;
    hx_drv_accelerometer_receive(&x, &y, &z);

    const float norm_x = -x;
    const float norm_y = y;
    const float norm_z = z;

    // Save data in milli-g unit
    save_data[begin_index++] = norm_x * 1000;
    save_data[begin_index++] = norm_y * 1000;
    save_data[begin_index++] = norm_z * 1000;

    // If reach end of buffer, return to 0 position
    if (begin_index >= ring_buffer_size) begin_index = 0;
  }

  // Check if data enough for prediction
  if (pending_initial_data && begin_index >= 200) {
    pending_initial_data = false;
  }

  // Return if we don't have enough data
  if (pending_initial_data) {
    return false;
  }

  // Copy the requested number of bytes to the provided input tensor
  for (int i = 0; i < length; ++i) {
    int ring_array_index = begin_index + i - length;
    if (ring_array_index < 0) {
      ring_array_index += ring_buffer_size;
    }
    input[i] = save_data[ring_array_index];
  }

  return true;
}
