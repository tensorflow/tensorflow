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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_TRIG_H_
#define XLA_CODEGEN_INTRINSIC_CPP_TRIG_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

namespace xla::codegen {

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
float sinh_f32(float x) asm("xla.sinh.f32");
Vec4f sinh_v4f32(Vec4f x) asm("xla.sinh.v4f32");
Vec8f sinh_v8f32(Vec8f x) asm("xla.sinh.v8f32");
Vec16f sinh_v16f32(Vec16f x) asm("xla.sinh.v16f32");

float cosh_f32(float x) asm("xla.cosh.f32");
Vec4f cosh_v4f32(Vec4f x) asm("xla.cosh.v4f32");
Vec8f cosh_v8f32(Vec8f x) asm("xla.cosh.v8f32");
Vec16f cosh_v16f32(Vec16f x) asm("xla.cosh.v16f32");

float asin_f32(float x) asm("xla.asin.f32");
Vec4f asin_v4f32(Vec4f x) asm("xla.asin.v4f32");
Vec8f asin_v8f32(Vec8f x) asm("xla.asin.v8f32");
Vec16f asin_v16f32(Vec16f x) asm("xla.asin.v16f32");

float acos_f32(float x) asm("xla.acos.f32");
Vec4f acos_v4f32(Vec4f x) asm("xla.acos.v4f32");
Vec8f acos_v8f32(Vec8f x) asm("xla.acos.v8f32");
Vec16f acos_v16f32(Vec16f x) asm("xla.acos.v16f32");
#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#endif  // XLA_CODEGEN_INTRINSIC_CPP_TRIG_H_
