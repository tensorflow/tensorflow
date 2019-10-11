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

#include "tensorflow/lite/experimental/micro/examples/micro_speech/command_responder.h"

#include "Arduino.h"

// Toggles the LED every inference, and keeps it on for ~2 seconds if a "yes"
// was heard
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    is_initialized = true;
  }
  static int32_t last_yes_time = 0;
  static int count = 0;

  if (is_new_command) {
    error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
    // If we heard a "yes", switch on an LED and store the time.
    if (found_command[0] == 'y') {
      last_yes_time = current_time;
      digitalWrite(LED_BUILTIN, HIGH);
    }
  }

  // If last_yes_time is non-zero but was >3 seconds ago, zero it
  // and switch off the LED.
  if (last_yes_time != 0) {
    if (last_yes_time < (current_time - 3000)) {
      last_yes_time = 0;
      digitalWrite(LED_BUILTIN, LOW);
    }
    // If it is non-zero but <3 seconds ago, do nothing.
    return;
  }

  // Otherwise, toggle the LED every time an inference is performed.
  ++count;
  if (count & 1) {
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(LED_BUILTIN, LOW);
  }
}
