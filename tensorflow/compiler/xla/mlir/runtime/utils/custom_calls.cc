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

#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT
using mlir::func::FuncOp;

using llvm::ArrayRef;
using llvm::StringRef;

static constexpr const char kCustomCall[] = "rt.custom_call";

CustomCallDeclarations::CustomCallDeclarations(SymbolTable sym_table)
    : sym_table_(sym_table) {}

FuncOp CustomCallDeclarations::GetOrCreate(ImplicitLocOpBuilder& b,
                                           StringRef target,
                                           FunctionType type) {
  // Check if we already have a custom all declaration.
  Key key = {b.getStringAttr(target), type};
  if (auto it = custom_calls_.find(key); it != custom_calls_.end())
    return it->second;

  // Create a new builder not attached to any operation, so that we can later
  // insert created function into the symbol table.
  OpBuilder builder(b.getContext(), b.getListener());

  // Create a custom call declaration.
  NamedAttribute attr(b.getStringAttr(kCustomCall), b.getStringAttr(target));
  auto declaration = builder.create<FuncOp>(b.getLoc(), target, type,
                                            ArrayRef<NamedAttribute>(attr));
  declaration.setPrivate();

  // Add created custom call declaration to the symbol table.
  sym_table_.insert(declaration);
  custom_calls_[key] = declaration;

  return declaration;
}

FuncOp CustomCallDeclarations::GetOrCreate(ImplicitLocOpBuilder& b,
                                           StringRef target, TypeRange inputs,
                                           TypeRange results) {
  auto type = FunctionType::get(b.getContext(), inputs, results);
  return GetOrCreate(b, target, type);
}

FuncOp CustomCallDeclarations::GetOrCreate(ImplicitLocOpBuilder& b,
                                           StringRef target, Operation* op) {
  return GetOrCreate(b, target, op->getOperandTypes(), op->getResultTypes());
}

void AppendCustomCallAttrs(mlir::Operation* op,
                           llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  for (auto& attr : attrs) op->setAttr(attr.getName(), attr.getValue());
}

}  // namespace runtime
}  // namespace xla
