/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tsl/platform/protobuf.h"

namespace mlir {
static mlir::Operation* ExtractOnlyOp(mlir::ModuleOp module) {
  mlir::func::FuncOp fn = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!fn) return nullptr;

  if (!llvm::hasSingleElement(fn)) return nullptr;

  // Here, modules with exactly two operations in the only basic block are
  // supported. The last operation should be a terminator operation and the
  // other operation is the operation of interest.
  auto& block = fn.front();
  if (block.getOperations().size() != 2) return nullptr;
  if (!block.back().hasTrait<OpTrait::IsTerminator>()) return nullptr;

  return &block.front();
}

static LogicalResult MlirToTfNodeDef(ModuleOp module,
                                     llvm::raw_ostream& output) {
  auto* context = module.getContext();

  Operation* op = ExtractOnlyOp(module);
  if (!op) {
    emitError(UnknownLoc::get(context),
              "modules with exactly one op other than terminator in a "
              "'main' function's "
              "only block are supported");
    return failure();
  }

  auto node_def_or = tensorflow::ConvertTFDialectOpToNodeDef(
      op, "node_name", /*ignore_unregistered_attrs=*/false);
  if (!node_def_or.ok()) {
    op->emitError("failed to convert to TF NodeDef:")
        << node_def_or.status().ToString();
    return failure();
  }

  output << tsl::LegacyUnredactedDebugString(*node_def_or.value());
  return success();
}

// Test only translation to convert a simple MLIR module with a single TF
// dialect op to NodeDef.
static TranslateFromMLIRRegistration translate_from_mlir_registration(
    "test-only-mlir-to-tf-nodedef", "test-only-mlir-to-tf-nodedef",
    MlirToTfNodeDef, mlir::RegisterAllTensorFlowDialects);

}  // namespace mlir
