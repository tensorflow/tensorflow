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
#ifndef XLA_CODEGEN_INTRINSIC_CPP_EIGEN_UNARY_H_
#define XLA_CODEGEN_INTRINSIC_CPP_EIGEN_UNARY_H_

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

// Single precision
float tanh_f32(float x) asm("xla.tanh.f32");
Vec16f tanh_v16f32(Vec16f x) asm("xla.tanh.v16f32");

// Double precision
double tanh_f64(double x) asm("xla.tanh.f64");
Vec8d tanh_v8f64(Vec8d x) asm("xla.tanh.v8f64");

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_EIGEN_UNARY_H_
