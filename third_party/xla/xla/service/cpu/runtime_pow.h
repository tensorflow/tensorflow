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

#ifndef XLA_SERVICE_CPU_RUNTIME_POW_H_
#define XLA_SERVICE_CPU_RUNTIME_POW_H_

#include <stdint.h>

// Raises F32 value a to the power of b.
extern "C" float __powisf2(float a, int32_t b);

// Raises F64 value a to the power of b.
extern "C" double __powidf2(double a, int32_t b);

#endif  // XLA_SERVICE_CPU_RUNTIME_POW_H_
