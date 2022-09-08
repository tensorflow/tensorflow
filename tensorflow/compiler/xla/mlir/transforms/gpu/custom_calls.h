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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_CUSTOM_CALLS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_CUSTOM_CALLS_H_

#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project

namespace xla {
namespace gpu {

// A helper class to create XLA runtime custom call declarations in the given
// symbol table. This class ensures that for each unique combination of the
// custom call target and function signature we create exaclty one custom call
// funcation declaration.
class CustomCalls {
 public:
  explicit CustomCalls(mlir::SymbolTable sym_table);

  // Returns existing custom call declaration or creates a new one.
  mlir::func::FuncOp GetOrCreate(mlir::ImplicitLocOpBuilder& b,
                                 llvm::StringRef target,
                                 mlir::FunctionType type);

  // Returns existing custom call declaration or creates a new one with a
  // function type constructed from `inputs` and `results`.
  mlir::func::FuncOp GetOrCreate(mlir::ImplicitLocOpBuilder& b,
                                 llvm::StringRef target, mlir::TypeRange inputs,
                                 mlir::TypeRange results);

  // Returns existing custom call declaration or creates a new one with a
  // function type constructed from the `op` operands and results.
  mlir::func::FuncOp GetOrCreate(mlir::ImplicitLocOpBuilder& b,
                                 llvm::StringRef target, mlir::Operation* op);

 private:
  mlir::SymbolTable sym_table_;

  using Key = std::pair<llvm::StringRef, mlir::FunctionType>;
  llvm::DenseMap<Key, mlir::func::FuncOp> custom_calls_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_GPU_CUSTOM_CALLS_H_
