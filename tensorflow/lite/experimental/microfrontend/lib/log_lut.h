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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Number of segments in the log lookup table. The table will be kLogSegments+1
// in length (with some padding).
#define kLogSegments 128
#define kLogSegmentsLog2 7

// Scale used by lookup table.
#define kLogScale 65536
#define kLogScaleLog2 16
#define kLogCoeff 45426

extern const uint16_t kLogLut[];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_LUT_H_
