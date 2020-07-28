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

extern "C" {
// These are headers from XMOS toolchain.
#include <xcore/port.h>
// TODO: This should be provided by BSP or build flag
#define LED_PORT    XS1_PORT_4C
}

/*
This function uses the device's LEDs to visually indicate the current y value.
The y value is in the range -1 <= y <= 1. The LEDs are physically lined up in
the following order:

                         [ 3 2 1 0 ]

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
  // The first time this method runs enable the LED port
  static bool is_initialized = false;
  static uint32_t data = 0x0000;
  if (!is_initialized) {
    port_enable( LED_PORT );
    data = 0;
    is_initialized = true;
  }

  if (y_value < 0) {
    data = (y_value < -0.75) ? (0x000C) : (0x0004);
  }
  else if (y_value > 0) {
    data = (y_value > 0.75) ? (0x0003) : (0x0002);
  }
  else {
    data = 0x0000;
  }

  port_out( LED_PORT, data );

  // Log the current X and Y values
  TF_LITE_REPORT_ERROR(error_reporter, "x_value: %f, y_value: %f\n", x_value,
                       y_value);
}
