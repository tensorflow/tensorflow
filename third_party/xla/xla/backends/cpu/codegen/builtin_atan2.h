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

#ifndef XLA_BACKENDS_CPU_CODEGEN_BUILTIN_ATAN2_H_
#define XLA_BACKENDS_CPU_CODEGEN_BUILTIN_ATAN2_H_

namespace xla::cpu {

// Computes atan2(y, x), flushing subnormal inputs to signed zero before
// calling libm. XLA:CPU flushes f32 and f64 subnormals, and glibc's atan2 and
// atan2f misbehave on subnormal inputs when the CPU flushes denormals (wrong
// quadrant or NaN), so the inputs are flushed explicitly here. The sign of
// flushed inputs is preserved so quadrant selection still works. This does not
// touch the floating-point environment.
double BuiltinAtan2(double y, double x);
float BuiltinAtan2f(float y, float x);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_BUILTIN_ATAN2_H_
