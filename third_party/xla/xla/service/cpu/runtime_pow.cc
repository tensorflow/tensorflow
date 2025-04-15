/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime_pow.h"

#include <cstdint>

#include "absl/base/attributes.h"

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

float ABSL_ATTRIBUTE_WEAK __powisf2(float a, int32_t b) { return Powi(a, b); }

double ABSL_ATTRIBUTE_WEAK __powidf2(double a, int32_t b) { return Powi(a, b); }
