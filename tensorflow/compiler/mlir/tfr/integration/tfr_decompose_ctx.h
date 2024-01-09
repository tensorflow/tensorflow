/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfr {

extern const char* const kTFRLibEnv;

using tsl::StatusOr;

// An wrapper for all the objects used to decompose a module (graph mode) and
// node_def (eager mode). Note that this class owns the decomposition library.
class TFRDecomposeContext {
 public:
  // The entry function to get a decompose context. All the required passes have
  // been initialized.
  static StatusOr<std::unique_ptr<TFRDecomposeContext>> Get(
      mlir::MLIRContext* mlir_ctx);

  // Constructor of the decompose context. To share the decompose library, the
  // whole decompose TFR function library is loaded.
  explicit TFRDecomposeContext(mlir::ModuleOp tfr_module);

  // Constructs the decompose context from the tfr text module and the mlir
  // context. The tfr text module is added to the mlir context.
  static std::unique_ptr<TFRDecomposeContext> GetFromText(
      StringPiece tfr_raw_text, mlir::MLIRContext* mlir_ctx);

  // Decomposes the op in the NodeDef to a set of primitive ops according to the
  // decompose library in the context. Wrap the decomposed result in a
  // FunctionDef.
  StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                   StringPiece func_name);

  // Runs the decompose passes on the user_module.
  Status DecomposeGraph(mlir::ModuleOp user_module);

  // Erases the tfr_module created.
  void Destroy();

 private:
  mlir::ModuleOp tfr_module_;
  mlir::PassManager pm_;

  GraphExportConfig export_confs_;
};

// Decomposes the NodeDef to a set of primitive ops according to the decompose
// library loaded. Wrap the decomposed result in a FunctionDef.
StatusOr<FunctionDef> ExpandNode(const NodeDef& node_def,
                                 StringPiece func_name);

// Decomposes the ops in the ModuleOp to a set of primitive ops according to
// decompose library in the context.
Status DecomposeGraph(mlir::ModuleOp user_module);

}  // namespace tfr
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_
