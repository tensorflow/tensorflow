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

#include "hx_drv_tflm.h"

static int32_t last_command_time = 0;
static uint32_t loop = 0;
static bool all_on = 0;

void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  loop++;
  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
    if (found_command[0] == 'y') {
      last_command_time = current_time;
      hx_drv_led_off(HX_DRV_LED_RED);
      hx_drv_led_on(HX_DRV_LED_GREEN);
    } else if (found_command[0] == 'n') {
      last_command_time = current_time;
      hx_drv_led_off(HX_DRV_LED_GREEN);
      hx_drv_led_on(HX_DRV_LED_RED);
    }
  }

  if (last_command_time != 0) {
    if (last_command_time < (current_time - 3000)) {
      last_command_time = 0;
      hx_drv_led_off(HX_DRV_LED_GREEN);
      hx_drv_led_off(HX_DRV_LED_RED);
    }
  } else {
    if ((loop % 10) == 0) {
      if (all_on) {
        hx_drv_led_on(HX_DRV_LED_RED);
        hx_drv_led_on(HX_DRV_LED_GREEN);
      } else {
        hx_drv_led_off(HX_DRV_LED_RED);
        hx_drv_led_off(HX_DRV_LED_GREEN);
      }
      all_on = !all_on;
    }
  }
}
