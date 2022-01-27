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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_TOPK_H
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_TOPK_H

#include <stdint.h>

extern "C" {

// Calculates `batch_size` topk operations with `input_size` inputs each. The
// outputs are written to `out_values` and `out_indices`.
extern void __xla_cpu_runtime_TopKF32(int64_t batch_size, int64_t input_size,
                                      int64_t k, const float* values,
                                      float* out_values, int32_t* out_indices);
}

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_TOPK_H
