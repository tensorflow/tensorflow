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

// Implementation for the DebugLog() function that prints to the debug logger on an
// generic cortex-m device.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "tensorflow/lite/micro/cortex_m_gcc_generic/debug_log.h"

static void (*DebugLog_callback)(const char* s) = nullptr;

extern "C" void DebugLog_register_callback(void (*cb)(const char* s)) {
  DebugLog_callback = cb;
}

extern "C" void DebugLog(const char* s) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  if (DebugLog_callback) {
	  DebugLog_callback(s);
  }
#endif
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
