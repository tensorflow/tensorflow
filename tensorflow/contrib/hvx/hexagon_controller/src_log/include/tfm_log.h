/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GEMM_WRAPPER_LOG_H
#define GEMM_WRAPPER_LOG_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#define TFM_LOG_LEVEL_VERBOSE -2
#define TFM_LOG_LEVEL_DEBUG -1
#define TFM_LOG_LEVEL_INFO 0
#define TFM_LOG_LEVEL_WARNING 1
#define TFM_LOG_LEVEL_ERROR 2
#define TFM_LOG_LEVEL_FATAL 3

static int s_log_level = TFM_LOG_LEVEL_INFO;

static inline bool IsLogOn(int log_level) { return log_level >= s_log_level; }

static inline void SetLogLevel(int log_level) { s_log_level = log_level; }

// Do nothing
static inline void SetExperimentalDebug() {}

#define TFMLOGV(fmt, ...)                       \
  do {                                          \
    if (!IsLogOn(TFM_LOG_LEVEL_VERBOSE)) break; \
    printf(fmt "\n", ##__VA_ARGS__);            \
  } while (0)

#define TFMLOGD(fmt, ...)                     \
  do {                                        \
    if (!IsLogOn(TFM_LOG_LEVEL_DEBUG)) break; \
    printf(fmt "\n", ##__VA_ARGS__);          \
  } while (0)

#define TFMLOGI(fmt, ...)                    \
  do {                                       \
    if (!IsLogOn(TFM_LOG_LEVEL_INFO)) break; \
    printf(fmt "\n", ##__VA_ARGS__);         \
  } while (0)

#define TFMLOGE(fmt, ...)                     \
  do {                                        \
    if (!IsLogOn(TFM_LOG_LEVEL_ERROR)) break; \
    printf(fmt "\n", ##__VA_ARGS__);          \
  } while (0)

static inline void PrintLogHexagon(const char* fmt, va_list ap) {
  char buffer[200];
  const int count = snprintf(buffer, 200, fmt, ap);
  buffer[count] = 0;
  TFMLOGI("%s", buffer);
}

static inline void LogDHexagon(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  PrintLogHexagon(fmt, ap);
  va_end(ap);
}

static inline void DumpNNId(uint32_t nn_id) {
  // TODO(satok): Dump more information
  TFMLOGI("NN Id = %d", nn_id);
}

#endif
