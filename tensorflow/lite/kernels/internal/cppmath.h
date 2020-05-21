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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_CPPMATH_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_CPPMATH_H_

#include <cmath>

namespace tflite {

#if defined(TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS) || \
    (defined(__ANDROID__) && !defined(__NDK_MAJOR__)) || defined(ARDUINO)
#define TF_LITE_GLOBAL_STD_PREFIX
#else
#define TF_LITE_GLOBAL_STD_PREFIX std
#endif

#define DECLARE_STD_GLOBAL_SWITCH1(tf_name, std_name) \
  template <class T>                                  \
  inline T tf_name(const T x) {                       \
    return TF_LITE_GLOBAL_STD_PREFIX::std_name(x);    \
  }

DECLARE_STD_GLOBAL_SWITCH1(TfLiteRound, round);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_CPPMATH_H_
