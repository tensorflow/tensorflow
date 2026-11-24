/* Copyright 2026 The OpenXLA Authors.

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
#ifndef XLA_CODEGEN_INTRINSIC_CPP_EXP2_H_
#define XLA_CODEGEN_INTRINSIC_CPP_EXP2_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

namespace xla::codegen {

// Single precision (float32) exp2 entry points.
float exp2_f32(float x) asm("xla.exp2.f32");
Vec4f exp2_v4f32(Vec4f x) asm("xla.exp2.v4f32");
Vec8f exp2_v8f32(Vec8f x) asm("xla.exp2.v8f32");
Vec16f exp2_v16f32(Vec16f x) asm("xla.exp2.v16f32");

// Double precision (float64) exp2 entry points.
double exp2_f64(double x) asm("xla.exp2.f64");
Vec2d exp2_v2f64(Vec2d x) asm("xla.exp2.v2f64");
Vec4d exp2_v4f64(Vec4d x) asm("xla.exp2.v4f64");
Vec8d exp2_v8f64(Vec8d x) asm("xla.exp2.v8f64");

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#endif  // XLA_CODEGEN_INTRINSIC_CPP_EXP2_H_
