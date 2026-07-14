/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_FUNC_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_FUNC_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir::quant {

// Returns a public `func::FuncOp` in `module_op` whose name matches either
// `main` or `serving_default`. If `func::FuncOps` with both names exist, the
// function with name "main" takes precedence. Returns null if no such a
// function exists.
func::FuncOp FindMainFuncOp(ModuleOp module_op);

}  // namespace mlir::quant

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_FUNC_H_
