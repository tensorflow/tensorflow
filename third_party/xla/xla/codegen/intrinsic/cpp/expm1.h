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
#ifndef XLA_CODEGEN_INTRINSIC_CPP_EXPM1_H_
#define XLA_CODEGEN_INTRINSIC_CPP_EXPM1_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

// Single precision expm1 entry points.
float expm1_f32(float x) asm("xla.expm1.f32");
Vec4f expm1_v4f32(Vec4f x) asm("xla.expm1.v4f32");
Vec8f expm1_v8f32(Vec8f x) asm("xla.expm1.v8f32");
Vec16f expm1_v16f32(Vec16f x) asm("xla.expm1.v16f32");

// Double precision expm1 entry points.
double expm1_f64(double x) asm("xla.expm1.f64");
Vec4d expm1_v4f64(Vec4d x) asm("xla.expm1.v4f64");
Vec8d expm1_v8f64(Vec8d x) asm("xla.expm1.v8f64");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_EXPM1_H_
