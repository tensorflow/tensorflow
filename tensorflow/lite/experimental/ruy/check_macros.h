/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_

#include "tensorflow/lite/kernels/internal/compatibility.h"

#define RUY_DCHECK(condition) TFLITE_DCHECK(condition)
#define RUY_DCHECK_EQ(x, y) TFLITE_DCHECK_EQ(x, y)
#define RUY_DCHECK_NE(x, y) TFLITE_DCHECK_NE(x, y)
#define RUY_DCHECK_GE(x, y) TFLITE_DCHECK_GE(x, y)
#define RUY_DCHECK_GT(x, y) TFLITE_DCHECK_GT(x, y)
#define RUY_DCHECK_LE(x, y) TFLITE_DCHECK_LE(x, y)
#define RUY_DCHECK_LT(x, y) TFLITE_DCHECK_LT(x, y)
#define RUY_CHECK(condition) TFLITE_CHECK(condition)
#define RUY_CHECK_EQ(x, y) TFLITE_CHECK_EQ(x, y)
#define RUY_CHECK_NE(x, y) TFLITE_CHECK_NE(x, y)
#define RUY_CHECK_GE(x, y) TFLITE_CHECK_GE(x, y)
#define RUY_CHECK_GT(x, y) TFLITE_CHECK_GT(x, y)
#define RUY_CHECK_LE(x, y) TFLITE_CHECK_LE(x, y)
#define RUY_CHECK_LT(x, y) TFLITE_CHECK_LT(x, y)

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_CHECK_MACROS_H_
