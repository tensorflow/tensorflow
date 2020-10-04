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

#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"

#include "am_bsp.h"   // NOLINT
#include "am_util.h"  // NOLINT

int32_t g_PreviousCommandTimestamp = 0;

// This implementation will light up the LEDs on the board in response to
// different commands.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  // Determine time since last command to ignore duplicate detection
  int32_t TimeSinceLastCommand = (current_time - g_PreviousCommandTimestamp);

  // Toggle LED1 every time an inference is performed.
  am_devices_led_toggle(am_bsp_psLEDs, 1);
#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(44, AM_HAL_GPIO_OUTPUT_TOGGLE);  // Slot1 CS pin
#endif

  // Clear LEDs for new inference.
  am_devices_led_off(am_bsp_psLEDs, 2);
  am_devices_led_off(am_bsp_psLEDs, 3);
  am_devices_led_off(am_bsp_psLEDs, 4);
#if USE_MAYA
  am_devices_led_off(am_bsp_psLEDs, 5);
  am_devices_led_off(am_bsp_psLEDs, 6);
  am_devices_led_off(am_bsp_psLEDs, 7);
#endif

  // Ramp LEDs if 'yes' was heard.
  // ARPIT TODO : Fix indication criteria.
  if (is_new_command || ((TimeSinceLastCommand > 1200) && (score > 200))) {
#if USE_DEBUG_GPIO
    // DEBUG : GPIO flag polling.
    am_hal_gpio_state_write(48, AM_HAL_GPIO_OUTPUT_TOGGLE);  // Slot1 PWM pin
#endif

    g_PreviousCommandTimestamp = current_time;
    TF_LITE_REPORT_ERROR(error_reporter, "\nHeard %s (%d) @%dms", found_command,
                         score, current_time);

#if USE_MAYA
    uint32_t delay = 60;
    if (found_command[0] == 'y') {
      am_devices_led_on(am_bsp_psLEDs, 2);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 3);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 4);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 5);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 6);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 7);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 7);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 6);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 5);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 4);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 3);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 2);
    }
#else
#if USE_NO
    if (found_command[0] == 'y') {
      am_devices_led_on(am_bsp_psLEDs, 2);
    }
    if (found_command[0] == 'n') {
      am_devices_led_on(am_bsp_psLEDs, 3);
    }
    if (found_command[0] == 'u') {
      am_devices_led_on(am_bsp_psLEDs, 4);
    }
#else
    uint32_t delay = 60;
    if (found_command[0] == 'y') {
      am_devices_led_on(am_bsp_psLEDs, 2);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 3);
      am_util_delay_ms(delay);
      am_devices_led_on(am_bsp_psLEDs, 4);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 4);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 3);
      am_util_delay_ms(delay);
      am_devices_led_off(am_bsp_psLEDs, 2);
    }
#endif
#endif
  }
}
