/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MATH_PASSES
#define XLA_MATH_PASSES

include "mlir/Pass/PassBase.td"

def MathOptimizationPass
    : Pass<"xla-math-optimization", "mlir::func::FuncOp"> {
  let summary = "Optimize operations from the `math` dialect.";

  let description = [{
    This pass performs algebraic simplification and polynomial approximation for
    ops from the Math dialect.
  }];

  let dependentDialects = [
    "mlir::vector::VectorDialect",
    "mlir::x86vector::X86VectorDialect"
  ];

  let constructor = "::xla::CreateMathOptimizationPass()";

  let options = [
   Option<"enable_avx2_", "enable-avx2", "bool", "false",
          "Enable math approximations that emit AVX2 intrinsics.">
  ];
}

#endif  // XLA_MATH_PASSES
