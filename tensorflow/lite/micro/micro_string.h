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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_STRING_H_
#define TENSORFLOW_LITE_MICRO_MICRO_STRING_H_

#include <cstdint>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/micro_string.h"

// Implements simple string formatting for numeric types.  Returns the number of
// bytes written to output.
extern "C" {
// Functionally equivalent to vsnprintf, trimmed down for TFLite Micro.
// MicroSnprintf() is implemented using MicroVsnprintf().
int MicroVsnprintf(char* output, int len, const char* format, va_list args);
// Functionally equavalent to snprintf, trimmed down for TFLite Micro.
// For example, MicroSnprintf(buffer, 10, "int %d", 10) will put the string
// "int 10" in the buffer.
// Floating point values are logged in exponent notation (1.XXX*2^N).
int MicroSnprintf(char* output, int len, const char* format, ...);
}

#endif  // TENSORFLOW_LITE_MICRO_MICRO_STRING_H_
