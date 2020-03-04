/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_STATIC_ALLOC_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_STATIC_ALLOC_H_

// Checks to ensure that the C-style array passed in has a compile-time size of
// at least the number of bytes requested. This doesn't work with raw pointers
// since sizeof() doesn't know their actual length, so only use this to check
// statically-allocated arrays with known sizes.
#define STATIC_ALLOC_ENSURE_ARRAY_SIZE(A, N)                               \
  do {                                                                     \
    if (sizeof(A) < (N)) {                                                 \
      TF_LITE_REPORT_ERROR(error_reporter,                                 \
                           #A " too small (%d bytes, wanted %d) at %s:%d", \
                           sizeof(A), (N), __FILE__, __LINE__);            \
      return 0;                                                            \
    }                                                                      \
  } while (0)

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_STATIC_ALLOC_H_
