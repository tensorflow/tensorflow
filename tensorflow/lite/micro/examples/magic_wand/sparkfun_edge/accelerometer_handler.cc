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

#if defined(ARDUINO) && !defined(ARDUINO_SFE_EDGE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_SFE_EDGE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/magic_wand/accelerometer_handler.h"

// These are headers from Ambiq's Apollo3 SDK.
#include <string.h>

#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT
#include "lis2dh12_platform_apollo3.h"

lis2dh12_platform_apollo3_if_t dev_if;  // accelerometer device interface
lis2dh12_ctx_t dev_ctx;                 // accelerometer device control

// A union representing either int16_t[3] or uint8_t[6],
// storing the most recent data
axis3bit16_t data_raw_acceleration;
// A buffer holding the last 200 sets of 3-channel values
float save_data[600] = {0.0};
// Most recent position in the save_data buffer
int begin_index = 0;
// True if there is not yet enough data to run inference
bool pending_initial_data = true;

int initAccelerometer(void) {
  uint32_t retVal32 = 0;
  static uint8_t whoamI = 0;

  am_hal_iom_config_t i2cConfig;
  memset((void*)(&i2cConfig), 0x00, sizeof(am_hal_iom_config_t));
  i2cConfig.eInterfaceMode = AM_HAL_IOM_I2C_MODE;
  i2cConfig.ui32ClockFreq = AM_HAL_IOM_100KHZ;

  // Initialize the IOM.
  retVal32 = am_hal_iom_initialize(
      AM_BSP_ACCELEROMETER_I2C_IOM,
      &(dev_if.iomHandle));  // set the iomHandle of the device interface
  if (retVal32 != AM_HAL_STATUS_SUCCESS) {
    return (int)retVal32;
  }

  retVal32 =
      am_hal_iom_power_ctrl((dev_if.iomHandle), AM_HAL_SYSCTRL_WAKE, false);
  if (retVal32 != AM_HAL_STATUS_SUCCESS) {
    return (int)retVal32;
  }

  retVal32 = am_hal_iom_configure((dev_if.iomHandle), &i2cConfig);
  if (retVal32 != AM_HAL_STATUS_SUCCESS) {
    return (int)retVal32;
  }

  // Configure the IOM pins.
  am_hal_gpio_pinconfig(AM_BSP_ACCELEROMETER_I2C_SDA_PIN,
                        g_AM_BSP_ACCELEROMETER_I2C_SDA_PIN);
  am_hal_gpio_pinconfig(AM_BSP_ACCELEROMETER_I2C_SCL_PIN,
                        g_AM_BSP_ACCELEROMETER_I2C_SDA_PIN);

  // Enable the IOM.
  retVal32 = am_hal_iom_enable((dev_if.iomHandle));
  if (retVal32 != AM_HAL_STATUS_SUCCESS) {
    return (int)retVal32;
  }

  //
  // Apply accelerometer configuration
  lis2dh12_device_id_get(&dev_ctx, &whoamI);
  if (whoamI != LIS2DH12_ID) {
    return AM_HAL_STATUS_FAIL;
  }

  lis2dh12_block_data_update_set(&dev_ctx, PROPERTY_ENABLE);
  lis2dh12_temperature_meas_set(&dev_ctx, LIS2DH12_TEMP_ENABLE);
  lis2dh12_data_rate_set(&dev_ctx, LIS2DH12_ODR_25Hz);
  lis2dh12_full_scale_set(&dev_ctx, LIS2DH12_2g);
  lis2dh12_temperature_meas_set(&dev_ctx, LIS2DH12_TEMP_ENABLE);
  lis2dh12_operating_mode_set(&dev_ctx, LIS2DH12_HR_12bit);

  return (int)AM_HAL_STATUS_SUCCESS;
}

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  // Set the clock frequency.
  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

  // Set the default cache configuration
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();

  // Configure the board for low power operation.
  am_bsp_low_power_init();

  // Initialize the device interface and control structures
  dev_if.iomHandle =
      NULL;  // Gets initialized once iomHandle is known (in initAccel())
  dev_if.addCS = AM_BSP_ACCELEROMETER_I2C_ADDRESS;  // Gets the accelerometer
                                                    // I2C address for the board
  dev_if.useSPI = false;                            // Using I2C

  dev_ctx.write_reg = lis2dh12_write_platform_apollo3;  // write bytes function
  dev_ctx.read_reg = lis2dh12_read_platform_apollo3;    // read bytes function
  dev_ctx.handle = (void*)&dev_if;  // Apollo3-specific interface information

  // Collecting data at 25Hz.
  int accInitRes = initAccelerometer();
  if (accInitRes != (int)AM_HAL_STATUS_SUCCESS) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to initialize the accelerometer. (code %d)",
                         accInitRes);
  }

  // Enable the accelerometer's FIFO buffer.
  // Note: LIS2DH12 has a FIFO buffer which holds up to 32 data entries. It
  // accumulates data while the CPU is busy. Old data will be overwritten if
  // it's not fetched in time, so we need to make sure that model inference is
  // faster than 1/25Hz * 32 = 1.28s
  if (lis2dh12_fifo_set(&dev_ctx, 1)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to enable FIFO buffer.");
  }

  if (lis2dh12_fifo_mode_set(&dev_ctx, LIS2DH12_BYPASS_MODE)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to clear FIFO buffer.");
    return kTfLiteError;
  }

  if (lis2dh12_fifo_mode_set(&dev_ctx, LIS2DH12_DYNAMIC_STREAM_MODE)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to set streaming mode.");
    return kTfLiteError;
  }

  TF_LITE_REPORT_ERROR(error_reporter, "Magic starts!");

  return kTfLiteOk;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length) {
  // Check FIFO buffer for new samples
  lis2dh12_fifo_src_reg_t status;
  if (lis2dh12_fifo_status_get(&dev_ctx, &status)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to get FIFO status.");
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
      TF_LITE_REPORT_ERROR(error_reporter, "Failed to get raw data.");
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

#endif  // ARDUINO_EXCLUDE_CODE
