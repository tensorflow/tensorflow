/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/debug_log.h"

#include <stdio.h>
#include <string.h>

// These are set by openmv py_tf.c code to redirect printing to an error message
// buffer...

char* py_tf_putchar_buffer = NULL;
size_t py_tf_putchar_buffer_len = 0;

extern "C" void DebugLog(const char* s) {
  for (size_t i = 0, j = strlen(s); i < j; i++) {
    if (py_tf_putchar_buffer_len) {
      *py_tf_putchar_buffer++ = s[i];
      py_tf_putchar_buffer_len--;
    } else {
      putchar(s[i]);
    }
  }
}
