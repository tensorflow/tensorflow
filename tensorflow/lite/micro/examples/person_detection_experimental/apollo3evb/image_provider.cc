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

#include "tensorflow/lite/micro/examples/person_detection_experimental/image_provider.h"

#include "tensorflow/lite/micro/examples/person_detection_experimental/himax_driver/HM01B0.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/himax_driver/HM01B0_RAW8_QVGA_8bits_lsb_5fps.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/himax_driver/HM01B0_debug.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/himax_driver/HM01B0_optimized.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/himax_driver/platform_Sparkfun_Edge.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT

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

void boost_mode_enable(tflite::ErrorReporter* error_reporter, bool bEnable) {
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

  boost_mode_enable(error_reporter, true);

  hm01b0_power_up(&s_HM01B0Cfg);

  am_util_delay_ms(1);

  hm01b0_mclk_enable(&s_HM01B0Cfg);

  am_util_delay_ms(1);

  hm01b0_init_if(&s_HM01B0Cfg);

  hm01b0_init_system(&s_HM01B0Cfg, (hm_script_t*)sHM01B0InitScript,
                     sizeof(sHM01B0InitScript) / sizeof(hm_script_t));

  // Put camera into streaming mode - this makes it so that the camera
  // constantly captures images.  It is still OK to read and image since the
  // camera uses a double-buffered input.  This means there is always one valid
  // image to read while the other buffer fills.  Streaming mode allows the
  // camera to perform auto exposure constantly.
  hm01b0_set_mode(&s_HM01B0Cfg, HM01B0_REG_MODE_SELECT_STREAMING, 0);

  return kTfLiteOk;
}

// Capture single frame.  Frame pointer passed in to reduce memory usage.  This
// allows the input tensor to be used instead of requiring an extra copy.
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int frame_width,
                      int frame_height, int channels, uint8_t* frame) {
  if (!g_is_camera_initialized) {
    TfLiteStatus init_status = InitCamera(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    // Drop a few frames until auto exposure is calibrated.
    for (int i = 0; i < kFramesToInitialize; ++i) {
      hm01b0_blocking_read_oneframe_scaled(frame, frame_width, frame_height,
                                           channels);
    }
    g_is_camera_initialized = true;
  }

  hm01b0_blocking_read_oneframe_scaled(frame, frame_width, frame_height,
                                       channels);

#ifdef DEMO_HM01B0_FRAMEBUFFER_DUMP_ENABLE
  // Allow some time to see result of previous inference before dumping image.
  am_util_delay_ms(2000);
  hm01b0_framebuffer_dump(frame, frame_width * frame_height * channels);
#endif

  return kTfLiteOk;
}
