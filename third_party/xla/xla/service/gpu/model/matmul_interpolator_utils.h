/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_
#define XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_

#include <string>

#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Constructs a lowercase key used in interpolation lookup. The representation
// is supposed to loosen some constraints of `HloDotInstruction` serialization
// and has the following format:
//
//   lowercase_str(`lhs`)x{lowercase_str(`rhs`)}->{lowercase_str(`out`)}
//
// For example for LHS=bf16, RHS=f32 and OUT=f32 the key will be:
//
//   bf16xf32->f32
//
std::string MatmulTypeStringRep(PrimitiveType lhs, PrimitiveType rhs,
                                PrimitiveType out);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_
