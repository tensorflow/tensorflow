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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_TIMER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_TIMER_H_

#include <cstdint>

// Returns the time in milliseconds. There's no contract about what time zero
// represents, the accuracy, or the granularity of the result. Subsequent calls
// will generally not return a lower value, but even that's not guaranteed if
// there's an overflow  wraparound.
// The reference implementation of this function just returns a constantly
// incrementing value for each call, since it would need a non-portable platform
// call to access time information. For real applications, you'll need to write
// your own platform-specific implementation.
int32_t TimeInMilliseconds();

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_TIMER_H_
