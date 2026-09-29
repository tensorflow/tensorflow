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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_YNNPACK_UNARY_H_
#define XLA_CODEGEN_INTRINSIC_CPP_YNNPACK_UNARY_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

// Single precision log
float xla_log_f32(float x) asm("xla.log.f32");
Vec2f xla_log_v2f32(Vec2f x) asm("xla.log.v2f32");
Vec4f xla_log_v4f32(Vec4f x) asm("xla.log.v4f32");
Vec8f xla_log_v8f32(Vec8f x) asm("xla.log.v8f32");
Vec16f xla_log_v16f32(Vec16f x) asm("xla.log.v16f32");

// Double precision log
double xla_log_f64(double x) asm("xla.log.f64");
Vec2d xla_log_v2f64(Vec2d x) asm("xla.log.v2f64");
Vec4d xla_log_v4f64(Vec4d x) asm("xla.log.v4f64");
Vec8d xla_log_v8f64(Vec8d x) asm("xla.log.v8f64");

// Single precision log1p
float xla_log1p_f32(float x) asm("xla.log1p.f32");
Vec2f xla_log1p_v2f32(Vec2f x) asm("xla.log1p.v2f32");
Vec4f xla_log1p_v4f32(Vec4f x) asm("xla.log1p.v4f32");
Vec8f xla_log1p_v8f32(Vec8f x) asm("xla.log1p.v8f32");
Vec16f xla_log1p_v16f32(Vec16f x) asm("xla.log1p.v16f32");

// Double precision log1p
double xla_log1p_f64(double x) asm("xla.log1p.f64");
Vec2d xla_log1p_v2f64(Vec2d x) asm("xla.log1p.v2f64");
Vec4d xla_log1p_v4f64(Vec4d x) asm("xla.log1p.v4f64");
Vec8d xla_log1p_v8f64(Vec8d x) asm("xla.log1p.v8f64");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_YNNPACK_UNARY_H_
