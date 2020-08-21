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

extern "C" {
#include "microspeech_xcore_support.h"
}

static int32_t data = 0x0000;

// The default implementation writes out the name of the recognized command
// to the error console. Real applications will want to take some custom
// action instead, and should implement their own versions of this function.
void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d) @%dms", found_command,
                         score, current_time);
     switch (found_command[0]) {
         case 'y':
           data = 0x0008;
           break;
         case 'n':
           data = 0x0004;
           break;
         case 'u':
           data = 0x0002;
           break;
         case 's':
           data = 0x0001;
           break;
         default:
           data = 0x0000;
           break;
     }
  }
}

extern "C" {
int32_t get_led_status() {
  return data;
}
}
