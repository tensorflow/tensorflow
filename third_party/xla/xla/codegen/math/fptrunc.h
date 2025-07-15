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

#ifndef XLA_CODEGEN_MATH_FPTRUNC_H_
#define XLA_CODEGEN_MATH_FPTRUNC_H_

#include <string>

#include "absl/status/statusor.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {

// XLA intrinsic for truncating floating point values (scalars and vectors).
class Intrinsic::FpTrunc {
 public:
  static std::string Name(Type from, Type to);

  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module, Type from,
                                                Type to);

  static mlir::func::FuncOp GetOrInsertDeclaration(mlir::OpBuilder& b,
                                                   mlir::ModuleOp module,
                                                   Type from, Type to);

  static absl::StatusOr<llvm::Function*> CreateDefinition(llvm::Module* module,
                                                          Type from, Type to);
};

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_FPTRUNC_H_
