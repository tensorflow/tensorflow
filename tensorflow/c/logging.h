/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_LOGGING_H_
#define TENSORFLOW_C_LOGGING_H_

#include "tensorflow/c/c_api_macros.h"

// --------------------------------------------------------------------------
// C API for tensorflow::Logging.

#ifdef __cplusplus
extern "C" {
#endif

typedef enum TF_LogLevel {
  TF_INFO = 0,
  TF_WARNING = 1,
  TF_ERROR = 2,
  TF_FATAL = 3,
} TF_LogLevel;

TF_CAPI_EXPORT extern void TF_Log(TF_LogLevel level, const char* fmt, ...);
TF_CAPI_EXPORT extern void TF_VLog(int level, const char* fmt, ...);
TF_CAPI_EXPORT extern void TF_DVLog(int level, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_C_LOGGING_H_
