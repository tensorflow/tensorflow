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

#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"

#include "am_bsp.h"  // NOLINT

// This implementation will light up the LEDs on the board in response to
// different commands.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
    // Setup LED's as outputs
#ifdef AM_BSP_NUM_LEDS
    am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
    am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS, 0x00000000);
#endif
    is_initialized = true;
  }

  // Toggle the blue LED every time an inference is performed.
  am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_LED_BLUE);

  // Turn on LEDs corresponding to the detection for the cycle
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
  am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
    if (found_command[0] == 'y') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
    }
    if (found_command[0] == 'n') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_RED);
    }
    if (found_command[0] == 'u') {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_GREEN);
    }
  }
}

#endif  // ARDUINO_EXCLUDE_CODE
