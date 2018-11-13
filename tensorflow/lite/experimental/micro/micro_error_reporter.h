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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ERROR_REPORTER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ERROR_REPORTER_H_

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/experimental/micro/compatibility.h"

#ifdef TF_LITE_MCU_DEBUG_LOG
// These functions should be supplied by the micro target library
extern "C" {
#include <stdint.h>
void DebugLog(const char* s);
void DebugLogInt32(int32_t i);
void DebugLogUInt32(uint32_t i);
void DebugLogHex(uint32_t i);
void DebugLogFloat(float i);
}
#else  // TF_LITE_MCU_DEBUG_LOG
#include <cstdint>
#include <cstdio>
static void inline DebugLog(const char* s) { fprintf(stderr, "%s", s); }
static void inline DebugLogInt32(int32_t i) { fprintf(stderr, "%d", i); }
static void inline DebugLogUInt32(uint32_t i) { fprintf(stderr, "%d", i); }
static void inline DebugLogHex(uint32_t i) { fprintf(stderr, "0x%8x", i); }
static void inline DebugLogFloat(float i) { fprintf(stderr, "%f", i); }
#endif  // TF_LITE_MCU_DEBUG_LOG

namespace tflite {

class MicroErrorReporter : public ErrorReporter {
 public:
  ~MicroErrorReporter() {}
  int Report(const char* format, va_list args) override;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_MICRO_ERROR_REPORTER_H_
