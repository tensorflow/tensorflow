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

#include "tensorflow/lite/micro/examples/magic_wand/output_handler.h"

#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT

void HandleOutput(tflite::ErrorReporter* error_reporter, int kind) {
  // The first time this method runs, set up our LEDs correctly
  static bool is_initialized = false;
  if (!is_initialized) {
    // Setup LED's as outputs
#ifdef AM_BSP_NUM_LEDS
    am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
    am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS, 0x00000000);
#endif
    is_initialized = true;
  }

  // Toggle the yellow LED every time an inference is performed
  am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_LED_YELLOW);

  // Set the LED color and print a symbol (red: wing, blue: ring, green: slope)
  if (kind == 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "WING:\n\r*         *         *\n\r *       * *       "
        "*\n\r  *     *   *     *\n\r   *   *     *   *\n\r    * *       "
        "* *\n\r     *         *\n\r");
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_RED);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_BLUE);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  } else if (kind == 1) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "RING:\n\r          *\n\r       *     *\n\r     *         *\n\r "
        "   *           *\n\r     *         *\n\r       *     *\n\r      "
        "    *\n\r");
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_BLUE);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  } else if (kind == 2) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "SLOPE:\n\r        *\n\r       *\n\r      *\n\r     *\n\r    "
        "*\n\r   *\n\r  *\n\r * * * * * * * *\n\r");
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_BLUE);
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  }
}

#endif  // ARDUINO_EXCLUDE_CODE
