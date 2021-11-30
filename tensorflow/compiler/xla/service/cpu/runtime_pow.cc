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

#include "tensorflow/compiler/xla/service/cpu/runtime_pow.h"

#include "tensorflow/core/platform/macros.h"

template <typename T>
static T Powi(T a, int32_t b) {
  const bool recip = b < 0;
  T r = 1;
  while (true) {
    if (b & 1) r *= a;
    b /= 2;
    if (b == 0) break;
    a *= a;
  }
  return recip ? 1 / r : r;
}

float TF_ATTRIBUTE_WEAK __powisf2(float a, int32_t b) { return Powi(a, b); }

double TF_ATTRIBUTE_WEAK __powidf2(double a, int32_t b) { return Powi(a, b); }
