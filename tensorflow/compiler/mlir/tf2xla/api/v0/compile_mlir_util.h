/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_MLIR_UTIL_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {

// Lowers MLIR module to XLA HLO inside an XlaComputation. The input module
// should only contain operations in tf dialect. If the input module contains
// operation in the tf_executor dialect, for example, returns an error.
// Exception to this are tf_executor dialect ops that are optimized away through
// canonicalization.
//
// Operations in tf dialect are lowered to XLA HLO through the following steps:
//   . Legalizes control flow operations.
//   . Decomposes compound resource operations so that the only remaining
//     operations on resource variables are resource reads/writes..
//   . Replaces resource reads/writes with function inputs/outputs and
//     eliminates the use of resource variables.
//   . Legalizes the operations to XLA HLO operations.
//   . Canonicalizes the XLA HLO operations.
//
// device_type: XLA JIT device to use for compilation such as "XLA_CPU_JIT",
//   "XLA_GPU_JIT" or "XLA_TPU_JIT".
// use_tuple_args: when this is true, always create a tuple argument for the
//   entry computation.
// prefer_tf2xla: when this is true, prefer tf2xla fallback kernels over MLIR
//   native kernels for legalization to HLO.
// return_tuple: when this is true, always create a tuple result for the
//   entry computation.
// shape_determination_fns: Contains layout preference fn and shape
//   representation fn. The two functions are used to determine argument and
//   result shapes.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
//
// TODO(hinsu): Migrate options to a separate struct.
Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool prefer_tf2xla, bool return_tuple,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns =
        {},
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Creates a MLIR pipeline that lowers MLIR module to MHLO dialect. The input
// module should only contain operations in tf dialect. For example, if the
// input module contains operation in the tf_executor dialect, the pass raises
// an error unless the tf_executor dialect ops are optimized away by
// canonicalization.
//
// The pipeline is used in ConvertMLIRToXlaComputation. And it generally has the
// following pass structure:
// - TensorFlow passes
// - Legalization passes
// - MHLO passes
//
// device_type: XLA JIT device to use for compilation such as "XLA_CPU_JIT",
//   "XLA_GPU_JIT" or "XLA_TPU_JIT".
// prefer_tf2xla: when this is true, prefer tf2xla fallback kernels over MLIR
//   native kernels for legalization to HLO.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
// allow_partial_conversion: when this is true, allow operations that can't be
// legalized.
void CreateConvertMlirToXlaHloPipeline(
    mlir::OpPassManager& pm, llvm::StringRef device_type, bool prefer_tf2xla,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    bool allow_partial_conversion = false);

// Helper struct representing argument tensor or resource handle shapes.
struct TensorOrResourceShape {
  TensorShape shape;
  bool is_resource = false;
};

// Refine MLIR types based on new shape information.
Status RefineShapes(llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                    mlir::ModuleOp module);

// Lower TF to MHLO and insert HLO into the XlaBuilder. xla_params are HLO-level
// inputs to module_op that have already been added to the XlaBuilder. returns
// are the returned XlaOps.
Status BuildHloFromTf(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                      llvm::ArrayRef<xla::XlaOp> xla_params,
                      std::vector<xla::XlaOp>& returns,
                      llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                      llvm::StringRef device_type,
                      llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                          custom_legalization_passes);

// Apply shape, description, and resource information to inputs and outputs
// in the XlaCompilationResult. This should be called after
// compilation_result->computation was set.
Status PopulateResultIOInfo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    bool use_tuple_args, bool use_resource_updates_for_aliases,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result);

// Compiles a MLIR module into XLA HLO, generates all accompanying metadata and
// stores them in CompilationResult.
//
// If analyse_graph is set to true, graph is legalized only if the graph
// analysis for the graph is successful. Otherwise, an error is returned.
Status CompileMlirToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    bool use_return_tuple, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Compiles a serialized MLIR module into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult.
Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Compiles a TensorFlow Graph (already converted to MLIR, imported with
// tf_executor dialect still present) into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult. This will rewrite arguments
// and run the TensorFlow standard pipeline prior to invoking
// `CompileMlirToXlaHlo`.
Status CompileGraphToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args,
    llvm::StringRef device_type, bool use_tuple_args, bool analyse_graph,
    bool use_return_tuple,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Compiles a TensorFlow Graph into XLA HLO, generates all accompanying metadata
// and stores them in CompilationResult.
Status CompileGraphToXlaHlo(
    const Graph& graph, llvm::ArrayRef<XlaArgument> args,
    llvm::ArrayRef<std::string> control_rets, llvm::StringRef device_type,
    bool use_tuple_args, bool analyse_graph,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

// Compiles a Graph from TF to HLO and adds the resulting HLO to the
// XlaBuilder. This function adds HLO to a larger HLO computation, so
// HLO-level inputs are supplied, and HLO-level outputs are produced.
// xla_params is the HLO-level inputs and returns is the HLO-level outputs.
Status BuildHloFromGraph(
    const Graph& graph, xla::XlaBuilder& builder,
    mlir::MLIRContext& mlir_context, llvm::ArrayRef<xla::XlaOp> xla_params,
    std::vector<xla::XlaOp>& returns, llvm::ArrayRef<XlaArgument> args,
    llvm::ArrayRef<std::string> control_rets, llvm::StringRef device_type,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {});

static inline Status CompileToHloGraphAnalysisFailedError() {
  return errors::Internal("disabled after graph analysis");
}

// Register a convenient pipeline for invoking TF/XLA lowering from the command
// line.
void RegisterConvertMlirToXlaHloPipelineWithDefaults();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V0_COMPILE_MLIR_UTIL_H_
