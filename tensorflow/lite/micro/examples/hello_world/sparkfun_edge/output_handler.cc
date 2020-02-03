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

#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"

#include "am_bsp.h"  // NOLINT

/*
This function uses the device's LEDs to visually indicate the current y value.
The y value is in the range -1 <= y <= 1. The LEDs (red, green, blue,
and yellow) are physically lined up in the following order:

                         [ R G B Y ]

The following table represents how we will light the LEDs for different values:

| Range            | LEDs lit    |
| 0.75 <= y <= 1   | [ 0 0 1 1 ] |
| 0 < y < 0.75     | [ 0 0 1 0 ] |
| y = 0            | [ 0 0 0 0 ] |
| -0.75 < y < 0    | [ 0 1 0 0 ] |
| -1 <= y <= 0.75  | [ 1 1 0 0 ] |

*/
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
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

  // Set the LEDs to represent negative values
  if (y_value < 0) {
    // Clear unnecessary LEDs
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_GREEN);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
    // The blue LED is lit for all negative values
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_BLUE);
    // The red LED is lit in only some cases
    if (y_value <= -0.75) {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_RED);
    } else {
      am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
    }
    // Set the LEDs to represent positive values
  } else if (y_value > 0) {
    // Clear unnecessary LEDs
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_RED);
    am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_BLUE);
    // The green LED is lit for all positive values
    am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_GREEN);
    // The yellow LED is lit in only some cases
    if (y_value >= 0.75) {
      am_devices_led_on(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
    } else {
      am_devices_led_off(am_bsp_psLEDs, AM_BSP_LED_YELLOW);
    }
  }
  // Log the current X and Y values
  error_reporter->Report("x_value: %f, y_value: %f\n", x_value, y_value);
}
