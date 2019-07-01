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

#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"

static mlir::Operation* ExtractOnlyOp(mlir::Module* module) {
  mlir::Function fn = module->getNamedFunction("main");
  if (!fn) return nullptr;

  if (fn.getBlocks().size() != 1) return nullptr;

  // Here, modules with exactly two operations in the only basic block are
  // supported. The last operation should be a terminator operation and the
  // other operation is the operation of interest.
  auto& block = fn.getBlocks().front();
  if (block.getOperations().size() != 2) return nullptr;
  if (!block.back().isKnownTerminator()) return nullptr;

  return &block.front();
}

static bool MlirToTfNodeDef(mlir::Module* module, llvm::StringRef filename) {
  auto* context = module->getContext();

  auto file = mlir::openOutputFile(filename);
  if (!file) {
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "failed to open output file " << filename;
    return true;
  }

  mlir::Operation* op = ExtractOnlyOp(module);
  if (!op) {
    mlir::emitError(mlir::UnknownLoc::get(context),
                    "modules with exactly one op other than terminator in a "
                    "'main' function's "
                    "only block are supported");
    return true;
  }

  auto node_def_or = tensorflow::ConvertTFDialectOpToNodeDef(op, "node_name");
  if (!node_def_or.ok()) {
    op->emitError("failed to convert to TF NodeDef:")
        << node_def_or.status().ToString();
    return true;
  }

  file->os() << node_def_or.ValueOrDie()->DebugString();
  file->keep();
  return false;
}

// Test only translation to convert a simple MLIR module with a single TF
// dialect op to NodeDef.
static mlir::TranslateFromMLIRRegistration registration(
    "test-only-mlir-to-tf-nodedef", MlirToTfNodeDef);
