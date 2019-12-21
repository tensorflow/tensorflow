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

#ifdef ARDUINO_ARDUINO_NANO33BLE

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"

/*
 * The sample requires the following third-party libraries to be installed and
 * configured:
 *
 * Arducam
 * -------
 * 1. Download https://github.com/ArduCAM/Arduino and copy its `ArduCAM`
 *    subdirectory into `Arduino/libraries`. Commit #e216049 has been tested
 *    with this code.
 * 2. Edit `Arduino/libraries/ArduCAM/memorysaver.h` and ensure that
 *    "#define OV2640_MINI_2MP_PLUS" is not commented out. Ensure all other
 *    defines in the same section are commented out.
 *
 * JPEGDecoder
 * -----------
 * 1. Install "JPEGDecoder" 1.8.0 from the Arduino library manager.
 * 2. Edit "Arduino/Libraries/JPEGDecoder/src/User_Config.h" and comment out
 *    "#define LOAD_SD_LIBRARY" and "#define LOAD_SDFAT_LIBRARY".
 */

// Required by Arducam library
#include <SPI.h>
#include <Wire.h>
#include <memorysaver.h>
// Arducam library
#include <ArduCAM.h>
// JPEGDecoder library
#include <JPEGDecoder.h>

// Checks that the Arducam library has been correctly configured
#if !(defined OV2640_MINI_2MP_PLUS)
#error Please select the hardware platform and camera module in the Arduino/libraries/ArduCAM/memorysaver.h
#endif

// The size of our temporary buffer for holding
// JPEG data received from the Arducam module
#define MAX_JPEG_BYTES 4096
// The pin connected to the Arducam Chip Select
#define CS 7

// Camera library instance
ArduCAM myCAM(OV2640, CS);
// Temporary buffer for holding JPEG data from camera
uint8_t jpeg_buffer[MAX_JPEG_BYTES] = {0};
// Length of the JPEG data currently in the buffer
uint32_t jpeg_length = 0;

// Get the camera module ready
TfLiteStatus InitCamera(tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Attempting to start Arducam");
  // Enable the Wire library
  Wire.begin();
  // Configure the CS pin
  pinMode(CS, OUTPUT);
  digitalWrite(CS, HIGH);
  // initialize SPI
  SPI.begin();
  // Reset the CPLD
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);
  // Test whether we can communicate with Arducam via SPI
  myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
  uint8_t test;
  test = myCAM.read_reg(ARDUCHIP_TEST1);
  if (test != 0x55) {
    error_reporter->Report("Can't communicate with Arducam");
    delay(1000);
    return kTfLiteError;
  }
  // Use JPEG capture mode, since it allows us to specify
  // a resolution smaller than the full sensor frame
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  // Specify the smallest possible resolution
  myCAM.OV2640_set_JPEG_size(OV2640_160x120);
  delay(100);
  return kTfLiteOk;
}

// Begin the capture and wait for it to finish
TfLiteStatus PerformCapture(tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Starting capture");
  // Make sure the buffer is emptied before each capture
  myCAM.flush_fifo();
  myCAM.clear_fifo_flag();
  // Start capture
  myCAM.start_capture();
  // Wait for indication that it is done
  while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) {
  }
  error_reporter->Report("Image captured");
  delay(50);
  // Clear the capture done flag
  myCAM.clear_fifo_flag();
  return kTfLiteOk;
}

// Read data from the camera module into a local buffer
TfLiteStatus ReadData(tflite::ErrorReporter* error_reporter) {
  // This represents the total length of the JPEG data
  jpeg_length = myCAM.read_fifo_length();
  error_reporter->Report("Reading %d bytes from Arducam", jpeg_length);
  // Ensure there's not too much data for our buffer
  if (jpeg_length > MAX_JPEG_BYTES) {
    error_reporter->Report("Too many bytes in FIFO buffer (%d)",
                           MAX_JPEG_BYTES);
    return kTfLiteError;
  }
  if (jpeg_length == 0) {
    error_reporter->Report("No data in Arducam FIFO buffer");
    return kTfLiteError;
  }
  myCAM.CS_LOW();
  myCAM.set_fifo_burst();
  for (int index = 0; index < jpeg_length; index++) {
    jpeg_buffer[index] = SPI.transfer(0x00);
  }
  delayMicroseconds(15);
  error_reporter->Report("Finished reading");
  myCAM.CS_HIGH();
  return kTfLiteOk;
}

// Decode the JPEG image, crop it, and convert it to greyscale
TfLiteStatus DecodeAndProcessImage(tflite::ErrorReporter* error_reporter,
                                   int image_width, int image_height,
                                   uint8_t* image_data) {
  error_reporter->Report("Decoding JPEG and converting to greyscale");
  // Parse the JPEG headers. The image will be decoded as a sequence of Minimum
  // Coded Units (MCUs), which are 16x8 blocks of pixels.
  JpegDec.decodeArray(jpeg_buffer, jpeg_length);

  // Crop the image by keeping a certain number of MCUs in each dimension
  const int keep_x_mcus = image_width / JpegDec.MCUWidth;
  const int keep_y_mcus = image_height / JpegDec.MCUHeight;

  // Calculate how many MCUs we will throw away on the x axis
  const int skip_x_mcus = JpegDec.MCUSPerRow - keep_x_mcus;
  // Roughly center the crop by skipping half the throwaway MCUs at the
  // beginning of each row
  const int skip_start_x_mcus = skip_x_mcus / 2;
  // Index where we will start throwing away MCUs after the data
  const int skip_end_x_mcu_index = skip_start_x_mcus + keep_x_mcus;
  // Same approach for the columns
  const int skip_y_mcus = JpegDec.MCUSPerCol - keep_y_mcus;
  const int skip_start_y_mcus = skip_y_mcus / 2;
  const int skip_end_y_mcu_index = skip_start_y_mcus + keep_y_mcus;

  // Pointer to the current pixel
  uint16_t* pImg;
  // Color of the current pixel
  uint16_t color;

  // Loop over the MCUs
  while (JpegDec.read()) {
    // Skip over the initial set of rows
    if (JpegDec.MCUy < skip_start_y_mcus) {
      continue;
    }
    // Skip if we're on a column that we don't want
    if (JpegDec.MCUx < skip_start_x_mcus ||
        JpegDec.MCUx >= skip_end_x_mcu_index) {
      continue;
    }
    // Skip if we've got all the rows we want
    if (JpegDec.MCUy >= skip_end_y_mcu_index) {
      continue;
    }
    // Pointer to the current pixel
    pImg = JpegDec.pImage;

    // The x and y indexes of the current MCU, ignoring the MCUs we skip
    int relative_mcu_x = JpegDec.MCUx - skip_start_x_mcus;
    int relative_mcu_y = JpegDec.MCUy - skip_start_y_mcus;

    // The coordinates of the top left of this MCU when applied to the output
    // image
    int x_origin = relative_mcu_x * JpegDec.MCUWidth;
    int y_origin = relative_mcu_y * JpegDec.MCUHeight;

    // Loop through the MCU's rows and columns
    for (int mcu_row = 0; mcu_row < JpegDec.MCUHeight; mcu_row++) {
      // The y coordinate of this pixel in the output index
      int current_y = y_origin + mcu_row;
      for (int mcu_col = 0; mcu_col < JpegDec.MCUWidth; mcu_col++) {
        // Read the color of the pixel as 16-bit integer
        color = *pImg++;
        // Extract the color values (5 red bits, 6 green, 5 blue)
        uint8_t r, g, b;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        // Convert to grayscale by calculating luminance
        // See https://en.wikipedia.org/wiki/Grayscale for magic numbers
        float gray_value = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);

        // The x coordinate of this pixel in the output image
        int current_x = x_origin + mcu_col;
        // The index of this pixel in our flat output buffer
        int index = (current_y * image_width) + current_x;
        image_data[index] = static_cast<uint8_t>(gray_value);
      }
    }
  }
  error_reporter->Report("Image decoded and processed");
  return kTfLiteOk;
}

// Get an image from the camera module
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, uint8_t* image_data) {
  static bool g_is_camera_initialized = false;
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      error_reporter->Report("InitCamera failed");
      return init_status;
    }
    g_is_camera_initialized = true;
  }

  TfLiteStatus capture_status = PerformCapture(error_reporter);
  if (capture_status != kTfLiteOk) {
    error_reporter->Report("PerformCapture failed");
    return capture_status;
  }

  TfLiteStatus read_data_status = ReadData(error_reporter);
  if (read_data_status != kTfLiteOk) {
    error_reporter->Report("ReadData failed");
    return read_data_status;
  }

  TfLiteStatus decode_status = DecodeAndProcessImage(
      error_reporter, image_width, image_height, image_data);
  if (decode_status != kTfLiteOk) {
    error_reporter->Report("DecodeAndProcessImage failed");
    return decode_status;
  }

  return kTfLiteOk;
}

#endif // ARDUINO_ARDUINO_NANO33BLE



#ifdef ARDUINO_SFE_EDGE

#include "tensorflow/lite/micro/examples/person_detection/image_provider.h"

#include "tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0.h"
#include "tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_RAW8_QVGA_8bits_lsb_5fps.h"
#include "tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_debug.h"
#include "tensorflow/lite/micro/examples/person_detection/himax_driver/HM01B0_optimized.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT

#include "hm01b0_platform.h" // TARGET specific implementation

// #define DEMO_HM01B0_FRAMEBUFFER_DUMP_ENABLE

// Enabling logging increases power consumption by preventing low power mode
// from being enabled.
#define ENABLE_LOGGING

namespace {

//*****************************************************************************
//
// HM01B0 Configuration
//
//*****************************************************************************
static hm01b0_cfg_t s_HM01B0Cfg = {
  // i2c settings
  ui16SlvAddr : HM01B0_DEFAULT_ADDRESS,
  eIOMMode : HM01B0_IOM_MODE,
  ui32IOMModule : HM01B0_IOM_MODULE,
  sIOMCfg : {
    eInterfaceMode : HM01B0_IOM_MODE,
    ui32ClockFreq : HM01B0_I2C_CLOCK_FREQ,
  },
  pIOMHandle : NULL,

  // MCLK settings
  ui32CTimerModule : HM01B0_MCLK_GENERATOR_MOD,
  ui32CTimerSegment : HM01B0_MCLK_GENERATOR_SEG,
  ui32CTimerOutputPin : HM01B0_PIN_MCLK,

  // data interface
  ui8PinSCL : HM01B0_PIN_SCL,
  ui8PinSDA : HM01B0_PIN_SDA,
  ui8PinD0 : HM01B0_PIN_D0,
  ui8PinD1 : HM01B0_PIN_D1,
  ui8PinD2 : HM01B0_PIN_D2,
  ui8PinD3 : HM01B0_PIN_D3,
  ui8PinD4 : HM01B0_PIN_D4,
  ui8PinD5 : HM01B0_PIN_D5,
  ui8PinD6 : HM01B0_PIN_D6,
  ui8PinD7 : HM01B0_PIN_D7,
  ui8PinVSYNC : HM01B0_PIN_VSYNC,
  ui8PinHSYNC : HM01B0_PIN_HSYNC,
  ui8PinPCLK : HM01B0_PIN_PCLK,

  ui8PinTrig : HM01B0_PIN_TRIG,
  ui8PinInt : HM01B0_PIN_INT,
  pfnGpioIsr : NULL,
};

static constexpr int kFramesToInitialize = 4;

bool g_is_camera_initialized = false;

void burst_mode_enable(tflite::ErrorReporter* error_reporter, bool bEnable) {
  am_hal_burst_avail_e eBurstModeAvailable;
  am_hal_burst_mode_e eBurstMode;

  // Check that the Burst Feature is available.
  if (AM_HAL_STATUS_SUCCESS ==
      am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      error_reporter->Report("Apollo3 Burst Mode is Available\n");
    } else {
      error_reporter->Report("Apollo3 Burst Mode is Not Available\n");
      return;
    }
  } else {
    error_reporter->Report("Failed to Initialize for Burst Mode operation\n");
  }

  // Make sure we are in "Normal" mode.
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_disable(&eBurstMode)) {
    if (AM_HAL_NORMAL_MODE == eBurstMode) {
      error_reporter->Report("Apollo3 operating in Normal Mode (48MHz)\n");
    }
  } else {
    error_reporter->Report("Failed to Disable Burst Mode operation\n");
  }

  // Put the MCU into "Burst" mode.
  if (bEnable) {
    if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
      if (AM_HAL_BURST_MODE == eBurstMode) {
        error_reporter->Report("Apollo3 operating in Burst Mode (96MHz)\n");
      }
    } else {
      error_reporter->Report("Failed to Enable Burst Mode operation\n");
    }
  }
}

}  // namespace

TfLiteStatus InitCamera(tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Initializing HM01B0...\n");

  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

  // Set the default cache configuration
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();

  // Configure the board for low power operation. This breaks logging by
  // turning off the itm and uart interfaces.
#ifndef ENABLE_LOGGING
  am_bsp_low_power_init();
#endif

  // Enable interrupts so we can receive messages from the boot host.
  am_hal_interrupt_master_enable();

  burst_mode_enable(error_reporter, true);

  // Turn on the 1.8V regulator for DVDD on the camera.
  am_hal_gpio_pinconfig(AM_BSP_GPIO_CAMERA_HM01B0_DVDDEN, g_AM_HAL_GPIO_OUTPUT_12);
  am_hal_gpio_output_set(AM_BSP_GPIO_CAMERA_HM01B0_DVDDEN);

  // Configure Red LED for debugging.
  am_devices_led_init((am_bsp_psLEDs + AM_BSP_LED_RED));
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);

  hm01b0_power_up(&s_HM01B0Cfg);

  // TODO(njeff): check the delay time to just fit the spec.
  am_util_delay_ms(1);

  hm01b0_mclk_enable(&s_HM01B0Cfg);

  // TODO(njeff): check the delay time to just fit the spec.
  am_util_delay_ms(1);

  if (HM01B0_ERR_OK != hm01b0_init_if(&s_HM01B0Cfg)) {
    return kTfLiteError;
  }

  if (HM01B0_ERR_OK !=
      hm01b0_init_system(&s_HM01B0Cfg, (hm_script_t*)sHM01B0InitScript,
                         sizeof(sHM01B0InitScript) / sizeof(hm_script_t))) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

// Capture single frame.  Frame pointer passed in to reduce memory usage.  This
// allows the input tensor to be used instead of requiring an extra copy.
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int frame_width,
                      int frame_height, int channels, uint8_t* frame) {
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
      return init_status;
    }
    // Drop a few frames until auto exposure is calibrated.
    for (int i = 0; i < kFramesToInitialize; ++i) {
      hm01b0_blocking_read_oneframe_scaled(&s_HM01B0Cfg, frame, frame_width,
                                           frame_height, channels);
    }
    g_is_camera_initialized = true;
  }

  hm01b0_blocking_read_oneframe_scaled(&s_HM01B0Cfg, frame, frame_width,
                                       frame_height, channels);

#ifdef DEMO_HM01B0_FRAMEBUFFER_DUMP_ENABLE
  hm01b0_framebuffer_dump(frame, frame_width * frame_height * channels);
#endif

  return kTfLiteOk;
}

#endif // ARDUINO_SFE_EDGE