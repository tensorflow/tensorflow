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

#include "tensorflow/lite/experimental/micro/examples/gesture_recognition/accelerometer_handler.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT
extern "C" {
#include "tf_accelerometer.h"  // NOLINT
#include "tf_adc.h"            // NOLINT
}

// A union representing either int16_t[3] or uint8_t[6],
// storing the most recent data
axis3bit16_t data_raw_acceleration;
// A buffer holding the last 200 sets of 3-channel values
float save_data[600] = {0.0};
// Most recent position in the save_data buffer
int begin_index = 0;
// How many times the most recent gesture has been matched in a row
int continuous_count = 0;
// The result of the last prediction
// 0: wing("W"), 1: ring("O"), 2: slope("angle"), 3: negative
int last_predict = -1;
// True if there is not yet enough data to run inference
bool pending_initial_data = true;
// The number of continuous predictions expected for each gesture, since they
// take varying amounts of time to complete
int should_continuous_count[3] = {15, 12, 10};

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  // Set the clock frequency.
  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

  // Set the default cache configuration
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();

  // Configure the board for low power operation.
  am_bsp_low_power_init();

  // Collecting data at 25Hz.
  int accInitRes = initAccelerometer();

  // Enable the accelerometer's FIFO buffer.
  // Note: LIS2DH12 has a FIFO buffer which holds up to 32 data entries. It
  // accumulates data while the CPU is busy. Old data will be overwritten if
  // it's not fetched in time, so we need to make sure that model inference is
  // faster than 1/25Hz * 32 = 1.28s
  if (lis2dh12_fifo_set(&dev_ctx, 1)) {
    error_reporter->Report("Failed to enable FIFO buffer.");
  }

  if (lis2dh12_fifo_mode_set(&dev_ctx, LIS2DH12_BYPASS_MODE)) {
    error_reporter->Report("Failed to clear FIFO buffer.");
    return 0;
  }

  if (lis2dh12_fifo_mode_set(&dev_ctx, LIS2DH12_DYNAMIC_STREAM_MODE)) {
    error_reporter->Report("Failed to set streaming mode.");
    return 0;
  }

  error_reporter->Report("Magic starts!");

  return kTfLiteOk;
}

// TODO(dansitu): Change return type to TfLiteStatus
bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length) {
  lis2dh12_fifo_src_reg_t status;
  if (lis2dh12_fifo_status_get(&dev_ctx, &status)) {
    error_reporter->Report("Failed to get FIFO status.");
    return false;
  }

  int samples = status.fss;
  if (status.ovrn_fifo) {
    samples++;
  }

  // Skip this round if data is not ready yet
  if (samples == 0) {
    return false;
  }

  // Load data from FIFO buffer
  axis3bit16_t data_raw_acceleration;
  for (int i = 0; i < samples; i++) {
    // Zero out the struct that holds raw accelerometer data
    memset(data_raw_acceleration.u8bit, 0x00, 3 * sizeof(int16_t));
    // If the return value is non-zero, sensor data was successfully read
    if (lis2dh12_acceleration_raw_get(&dev_ctx, data_raw_acceleration.u8bit)) {
      error_reporter->Report("Failed to get raw data.");
    } else {
      // Convert each raw 16-bit value into floating point values representing
      // milli-Gs, a unit of acceleration, and store in the current position of
      // our buffer
      save_data[begin_index++] =
          lis2dh12_from_fs2_hr_to_mg(data_raw_acceleration.i16bit[0]);
      save_data[begin_index++] =
          lis2dh12_from_fs2_hr_to_mg(data_raw_acceleration.i16bit[1]);
      save_data[begin_index++] =
          lis2dh12_from_fs2_hr_to_mg(data_raw_acceleration.i16bit[2]);
      // Start from beginning, imitating loop array.
      if (begin_index >= 600) begin_index = 0;
    }
  }

  // Check if we are ready for prediction or still pending more initial data
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
      ring_array_index += 600;
    }
    input[i] = save_data[ring_array_index];
  }
  return true;
}

// TODO(dansitu): Refactor to allow reuse by other boards and unit testing of
// logic
// Return the result of the last prediction
// 0: wing("W"), 1: ring("O"), 2: slope("angle"), 3: negative
int PredictGesture(float* output) {
  if (pending_initial_data) return 3;
  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < 3; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = 3;
    return 3;
  }
  if (last_predict == this_predict)
    continuous_count += 1;
  else
    continuous_count = 0;
  last_predict = this_predict;
  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < should_continuous_count[this_predict]) {
    return 3;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  memset(save_data, 0, 600 * sizeof(float));
  begin_index = 0;
  last_predict = -1;
  pending_initial_data = true;
  return this_predict;
}
