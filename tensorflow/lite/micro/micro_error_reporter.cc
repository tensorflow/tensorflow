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

#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace tflite {
namespace {
void DebugLogPrintf(const char* format, va_list args) {
  const int output_cache_size = 64;
  char output_cache[output_cache_size + 1];
  int output_cache_index = 0;
  const char* current = format;
  while (*current != 0) {
    if (*current == '%') {
      const char next = *(current + 1);
      if ((next == 'd') || (next == 's') || (next == 'f')) {
        current += 1;
        if (output_cache_index > 0) {
          output_cache[output_cache_index] = 0;
          DebugLog(output_cache);
          output_cache_index = 0;
        }
        if (next == 'd') {
          DebugLogInt32(va_arg(args, int));
        } else if (next == 's') {
          DebugLog(va_arg(args, char*));
        } else if (next == 'f') {
          DebugLogFloat(va_arg(args, double));
        }
      }
    } else {
      output_cache[output_cache_index] = *current;
      output_cache_index += 1;
    }
    if (output_cache_index >= output_cache_size) {
      output_cache[output_cache_index] = 0;
      DebugLog(output_cache);
      output_cache_index = 0;
    }
    current += 1;
  }
  if (output_cache_index > 0) {
    output_cache[output_cache_index] = 0;
    DebugLog(output_cache);
    output_cache_index = 0;
  }
  DebugLog("\r\n");
}
}  // namespace

int MicroErrorReporter::Report(const char* format, va_list args) {
  DebugLogPrintf(format, args);
  return 0;
}

}  // namespace tflite
