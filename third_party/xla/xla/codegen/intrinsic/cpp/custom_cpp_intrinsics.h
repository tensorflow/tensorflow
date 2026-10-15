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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_CUSTOM_CPP_INTRINSICS_H_
#define XLA_CODEGEN_INTRINSIC_CPP_CUSTOM_CPP_INTRINSICS_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

float sin_f32(float x) asm("xla.sin.f32");
Vec4f sin_v4f32(Vec4f x) asm("xla.sin.v4f32");
Vec8f sin_v8f32(Vec8f x) asm("xla.sin.v8f32");
Vec16f sin_v16f32(Vec16f x) asm("xla.sin.v16f32");

double sin_f64(double x) asm("xla.sin.f64");
Vec2d sin_v2f64(Vec2d x) asm("xla.sin.v2f64");
Vec4d sin_v4f64(Vec4d x) asm("xla.sin.v4f64");
Vec8d sin_v8f64(Vec8d x) asm("xla.sin.v8f64");

float cos_f32(float x) asm("xla.cos.f32");
Vec4f cos_v4f32(Vec4f x) asm("xla.cos.v4f32");
Vec8f cos_v8f32(Vec8f x) asm("xla.cos.v8f32");
Vec16f cos_v16f32(Vec16f x) asm("xla.cos.v16f32");

double cos_f64(double x) asm("xla.cos.f64");
Vec2d cos_v2f64(Vec2d x) asm("xla.cos.v2f64");
Vec4d cos_v4f64(Vec4d x) asm("xla.cos.v4f64");
Vec8d cos_v8f64(Vec8d x) asm("xla.cos.v8f64");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_CUSTOM_CPP_INTRINSICS_H_
