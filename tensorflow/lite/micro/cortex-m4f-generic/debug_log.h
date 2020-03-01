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

#ifndef TENSORFLOW_LITE_MICRO_CORTEX_M4F_GENERIC_DEBUG_LOG_H_
#define TENSORFLOW_LITE_MICRO_CORTEX_M4F_GENERIC_DEBUG_LOG_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// This function is used to register a callback for debug logging.
// It must be called before the first call to DebugLog().
extern void DebugLog_register_callback(void (*cb)(const char* s));

// This function should be implemented by each target platform, and provide a
// way for strings to be output to some text stream. For more information, see
// tensorflow/lite/micro/debug_log.cc.
// Note that before the first call to DebugLog()
// a callback function must be registered by calling DebugLog_register_callback().
extern void DebugLog(const char* s);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_MICRO_CORTEX_M4F_GENERIC_DEBUG_LOG_H_
