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

#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using stream_executor::port::StatusOr;
using NodeAndType = std::pair<StringPiece, DataType>;

// An wrapper for all the objects used to decompose a module (graph mode) and
// node_def (eager mode). Note that this class owns the decomposition library.
class TFRDecomposeContext {
 public:
  // The entry function to get a decompose context. All the required passes have
  // been initialized.
  static std::unique_ptr<TFRDecomposeContext> Get(StringPiece tfr_raw_text,
                                                  mlir::MLIRContext* mlir_ctx);

  // Constructor of the decompose context. To share the decompose library, the
  // whole decompose TFR function library is loaded.
  explicit TFRDecomposeContext(mlir::OwningModuleRef tfr_module);

  // Decompose the op in the NodeDef to a set of primitive ops according to the
  // decompose library in the context. Wrap the decomposed result in a GraphDef.
  StatusOr<std::unique_ptr<GraphDef>> Decompose(const NodeDef& node_def,
                                                absl::Span<NodeAndType> inputs);

  // Decompose the ops in the ModuleOp to a set of primitive ops according to
  // decompose library in the context.
  Status Decompose(mlir::ModuleOp user_module);

  // Release all the owned references.
  Status Destroy();

 private:
  mlir::OwningModuleRef tfr_module_;
  mlir::PassManager pm_;

  FunctionLibraryDefinition flib_def_;
  GraphDebugInfo debug_info_;
  GraphExportConfig export_confs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_INTEGRATION_TFR_DECOMPOSE_CTX_H_
