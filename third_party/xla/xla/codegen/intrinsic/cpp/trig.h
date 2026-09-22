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

namespace xla::codegen {

float sinh_f32(float x) asm("xla.sinh.f32");
Vec4f sinh_v4f32(Vec4f x) asm("xla.sinh.v4f32");
Vec8f sinh_v8f32(Vec8f x) asm("xla.sinh.v8f32");
Vec16f sinh_v16f32(Vec16f x) asm("xla.sinh.v16f32");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_TRIG_H_
