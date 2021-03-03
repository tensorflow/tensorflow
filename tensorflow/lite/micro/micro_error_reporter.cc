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

#include <cstdarg>
#include <cstdint>
#include <new>

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_string.h"
#endif

namespace {
uint8_t micro_error_reporter_buffer[sizeof(tflite::MicroErrorReporter)];
tflite::MicroErrorReporter* error_reporter_ = nullptr;

void Log(const char* format, va_list args) {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  // Only pulling in the implementation of this function for builds where we
  // expect to make use of it to be extra cautious about not increasing the code
  // size.
  static constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];
  MicroVsnprintf(log_buffer, kMaxLogLen, format, args);
  DebugLog(log_buffer);
  DebugLog("\r\n");
#endif
}

}  // namespace

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  Log(format, args);
  va_end(args);
}
#endif

namespace tflite {
ErrorReporter* GetMicroErrorReporter() {
  if (error_reporter_ == nullptr) {
    error_reporter_ = new (micro_error_reporter_buffer) MicroErrorReporter();
  }
  return error_reporter_;
}

int MicroErrorReporter::Report(const char* format, va_list args) {
  Log(format, args);
  return 0;
}

}  // namespace tflite
