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

// Implementation for the DebugLog() function that prints to the debug logger on an
// generic cortex-m4f device.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "tensorflow/lite/micro/cortex-m4f-generic/debug_log.h"

static void (*DebugLog_callback)(const char* s) = nullptr;

extern void DebugLog_register_callback(void (*cb)(const char* s)) {
  DebugLog_callback = cb;
}

extern void DebugLog(const char* s) {
  if (DebugLog_callback) {
	  DebugLog_callback(s);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
