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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_

#include "tensorflow/core/platform/types.h"

// Converts an F32 value to a F16.
extern "C" tensorflow::uint16 __gnu_f2h_ieee(float);

// Converts an F16 value to a F32.
extern "C" float __gnu_h2f_ieee(tensorflow::uint16);

// Converts an F64 value to a F16.
extern "C" tensorflow::uint16 __truncdfhf2(double);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FP16_H_
