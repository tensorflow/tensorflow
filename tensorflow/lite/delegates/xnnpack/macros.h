/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_MACROS_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_MACROS_H_

#include <cstdio>

#include "tensorflow/lite/minimal_logging.h"

#define XNNPACK_LOG_LIMIT 4048

#define XNNPACK_ABORT_CHECK(TEST, ...)                                   \
  if (!(TEST)) {                                                         \
    char msg[XNNPACK_LOG_LIMIT] = {0};                                   \
    int bytes =                                                          \
        snprintf(msg, XNNPACK_LOG_LIMIT, "%s:%d: ", __FILE__, __LINE__); \
    snprintf(msg + bytes, XNNPACK_LOG_LIMIT - bytes, "" __VA_ARGS__);    \
    TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, msg);                      \
    std::abort();                                                        \
  }

#define XNNPACK_VAR_ARG_HEAD(FIRST, ...) FIRST

#define XNNPACK_RETURN_CHECK(TEST, ...)                                    \
  if (!(TEST)) {                                                           \
    if (sizeof(XNNPACK_VAR_ARG_HEAD("" __VA_ARGS__)) > sizeof("")) {       \
      char msg[XNNPACK_LOG_LIMIT] = {0};                                   \
      int bytes =                                                          \
          snprintf(msg, XNNPACK_LOG_LIMIT, "%s:%d: ", __FILE__, __LINE__); \
      snprintf(msg + bytes, XNNPACK_LOG_LIMIT - bytes, "" __VA_ARGS__);    \
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR, msg);                      \
    }                                                                      \
    return false;                                                          \
  }

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_MACROS_H_
