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
#ifndef TENSORFLOW_LITE_MICRO_DEBUG_LOG_NUMBERS_H_
#define TENSORFLOW_LITE_MICRO_DEBUG_LOG_NUMBERS_H_

#include <cstdint>

// Output numbers to the debug logging stream.
extern "C" {
void DebugLogInt32(int32_t i);
void DebugLogUInt32(uint32_t i);
void DebugLogHex(uint32_t i);
void DebugLogFloat(float i);
}

#endif  // TENSORFLOW_LITE_MICRO_DEBUG_LOG_NUMBERS_H_
