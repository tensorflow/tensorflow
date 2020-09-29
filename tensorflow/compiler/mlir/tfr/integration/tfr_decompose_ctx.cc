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
#include "tensorflow/compiler/mlir/tfr/integration/tfr_decompose_ctx.h"

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"
#include "tensorflow/compiler/mlir/tfr/passes/passes.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

std::unique_ptr<TFRDecomposeContext> TFRDecomposeContext::Get(
    StringPiece tfr_raw_text, mlir::MLIRContext* mlir_ctx) {
  // Load dialects involved in the conversion
  mlir::DialectRegistry& registry = mlir_ctx->getDialectRegistry();
  // clang-format off
  registry.insert<mlir::StandardOpsDialect,
                  mlir::scf::SCFDialect,
                  mlir::shape::ShapeDialect,
                  mlir::TF::TensorFlowDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::TFR::TFRDialect>();
  // clang-format on

  // Load the TFR functions in a mlir::ModuleOp
  auto memory_buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(tfr_raw_text.data(), tfr_raw_text.size()));
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(memory_buffer), llvm::SMLoc());
  mlir::OwningModuleRef module = mlir::parseSourceFile(source_mgr, mlir_ctx);

  // Create the context
  return absl::make_unique<TFRDecomposeContext>(std::move(module));
}

StatusOr<std::unique_ptr<GraphDef>> TFRDecomposeContext::Decompose(
    const NodeDef& node_def, absl::Span<NodeAndType> inputs) {
  // TODO(fengliuai): implement a cache to return early.

  // Creates a graph from the node def, so it can be imported to MLIR as module.
  GraphImportConfig import_confs;
  Status status;
  Graph graph(OpRegistry::Global());

  // Creates the placeholer nodes, which will be promoted as function arguments.
  // Adds the argument nodes to the importer configs.
  for (const auto& input : inputs) {
    // TODO(fengliuai): how to get shape?
    TensorShape unknown_shape;
    Node* placeholder_node;
    NodeBuilder builder(input.first, "Placeholder");
    builder.Attr("shape", unknown_shape);
    builder.Attr("dtype", input.second);
    TF_RETURN_IF_ERROR(builder.Finalize(&graph, &placeholder_node));
    import_confs.inputs.insert({std::string(input.first), {}});
  }
  // Add the current node and also specify the outputs.
  graph.AddNode(node_def, &status);
  import_confs.outputs.emplace_back(node_def.name());

  TF_ASSIGN_OR_RETURN(
      auto node_module,
      ConvertGraphToMlir(graph, debug_info_, flib_def_, import_confs,
                         tfr_module_->getContext()));
  if (failed(mlir::verify(*node_module))) {
    return errors::Internal(absl::StrCat(
        "Failed to verify the imported NodeDef: ", node_def.DebugString()));
  }

  // Call the decompose passes by using the external symbol table.
  if (failed(pm_.run(*node_module))) {
    return errors::Internal("Failed to run the decompose passes.");
  }

  // Export the result as a GraphDef.
  return ConvertMlirToGraphdef(*node_module, export_confs_);
}

Status TFRDecomposeContext::Decompose(mlir::ModuleOp user_module) {
  // Call the decompose passes by using the external symbol table.
  if (failed(pm_.run(user_module))) {
    return errors::Internal("Failed to run the decompose passes.");
  }
  return Status::OK();
}

Status TFRDecomposeContext::Destroy() {
  tfr_module_.release().erase();
  return Status::OK();
}

// Constructor of the decompose context.
TFRDecomposeContext::TFRDecomposeContext(mlir::OwningModuleRef tfr_module)
    : tfr_module_(std::move(tfr_module)),
      pm_(tfr_module_->getContext()),
      flib_def_(OpRegistry::Global(), FunctionDefLibrary()) {
  mlir::OpPassManager& func_pm = pm_.nest<mlir::FuncOp>();

  // Prepare the imported graph.
  func_pm.addPass(mlir::CreateExecutorDialectToFunctionalConversionPass());

  // Run TFR lowering, inlining and raising to tf.
  func_pm.addPass(mlir::TFR::CreateDecomposeTFOpsPass(tfr_module_.get()));
  func_pm.addPass(mlir::TFR::CreateRaiseToTFOpsPass(
      tfr_module_.get(), /*materialize_derived_attrs=*/true));

  // Prepare to be exported.
  func_pm.addPass(mlir::CreateFunctionalToExecutorDialectConversionPass());
  pm_.addPass(mlir::CreateBreakUpIslandsPass());
}

}  // namespace tensorflow
