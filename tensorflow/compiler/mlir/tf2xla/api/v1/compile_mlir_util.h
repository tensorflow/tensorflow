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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_MLIR_UTIL_H_

#include <memory>

#include "absl/base/attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/builder/xla_computation.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor_shape.h"

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
// enable_op_fallback: when this is true, prefer tf2xla fallback kernels over
// MLIR
//   native kernels for legalization to HLO.
// return_tuple: when this is true, always create a tuple result for the
//   entry computation.
// shape_determination_fns: Contains layout preference fn and shape
//   representation fn. The two functions are used to determine argument and
//   result shapes.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool enable_op_fallback, bool return_tuple,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns = {},
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {},
    llvm::StringRef module_name = llvm::StringRef());

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
// enable_op_fallback: when this is true, prefer tf2xla fallback kernels over
// MLIR
//   native kernels for legalization to HLO.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
// lower_to_xla_hlo: Temporary parameter to be removed in imminent update. If
//   true, includes legalization and MHLO lowering passes.
// allow_partial_conversion: when this is true, allow operations that can't be
//   legalized.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
void CreateConvertMlirToXlaHloPipeline(
    mlir::OpPassManager& pm, llvm::StringRef device_type,
    bool enable_op_fallback,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    bool lower_to_xla_hlo = true, bool allow_partial_conversion = false);

// Helper struct representing argument tensor or resource handle shapes.
struct TensorOrResourceShape {
  TensorShape shape;
  bool is_resource = false;
};

// Refine MLIR types based on new shape information.
ABSL_DEPRECATED("Not meant to be used directly and should be a util.")
absl::Status RefineShapes(llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                          mlir::ModuleOp module);

// Lower TF to MHLO and insert HLO into the XlaBuilder. xla_params are HLO-level
// inputs to module_op that have already been added to the XlaBuilder. returns
// are the returned XlaOps.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::Status BuildHloFromTf(mlir::ModuleOp module_op, xla::XlaBuilder& builder,
                            llvm::ArrayRef<xla::XlaOp> xla_params,
                            std::vector<xla::XlaOp>& returns,
                            llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
                            llvm::StringRef device_type,
                            llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
                                custom_legalization_passes);

// Apply shape, description, and resource information to inputs and outputs
// in the XlaCompilationResult. This should be called after
// compilation_result->computation was set.
ABSL_DEPRECATED("Not meant to be used directly and should be a util.")
absl::Status PopulateResultIOInfo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    bool use_tuple_args, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result);

// Runs MLIR Bridge on an MLIR module.
//
// If lower_to_xla_hlo is true then compiles down into XLA HLO, generates all
// accompanying metadata and stores them in CompilationResult.
//
// If enable_op_fallback is set to false, graph is legalized only if the graph
// analysis for the graph is successful. Otherwise, an error is returned.
//
// Running the MLIR Bridge performs many transformations on the input module
// which is modified in place.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::Status CompileMlirToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    bool use_return_tuple, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name = llvm::StringRef(),
    bool lower_to_xla_hlo = true);

// Runs MLIR Bridge on a MLIR module.
//
// If lower_to_xla_hlo is true then compiles down into XLA HLO, generates all
// accompanying metadata and stores them in CompilationResult.
//
// If enable_op_fallback is set to false, graph is legalized only if the graph
// analysis for the graph is successful. Otherwise, an error is returned.
//
// On success, returns the serialized MLIR module.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::StatusOr<std::string> CompileMlirToXlaHloAndSerialize(
    mlir::ModuleOp module_op, llvm::ArrayRef<TensorOrResourceShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    bool use_return_tuple, bool use_resource_updates_for_aliases,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes,
    llvm::StringRef module_name = llvm::StringRef(),
    bool lower_to_xla_hlo = true);

// Runs MLIR Bridge on a serialized MLIR module.
//
// If lower_to_xla_hlo is true then compiles down into XLA HLO, generates all
// accompanying metadata and stores them in CompilationResult.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::StatusOr<std::string> CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes = {},
    llvm::StringRef module_name = llvm::StringRef(),
    bool lower_to_xla_hlo = true);

// Compiles a TensorFlow Graph (already converted to MLIR, imported with
// tf_executor dialect still present) into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult. This will rewrite arguments
// and run the TensorFlow standard pipeline prior to invoking
// `CompileMlirToXlaHlo`.
ABSL_DEPRECATED("Use v2/legalize_tf.h::LegalizeMlirToHlo instead.")
absl::Status CompileGraphToXlaHlo(
    mlir::ModuleOp module_op, llvm::ArrayRef<XlaArgument> args,
    llvm::StringRef device_type, bool use_tuple_args, bool enable_op_fallback,
    bool use_return_tuple,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    XlaCompilationResult* compilation_result,
    llvm::MutableArrayRef<std::unique_ptr<mlir::Pass>>
        custom_legalization_passes);

// Compiles a Graph from TF to HLO and adds the resulting HLO to the
// XlaBuilder. This function adds HLO to a larger HLO computation, so
// HLO-level inputs are supplied, and HLO-level outputs are produced.
// xla_params is the HLO-level inputs and returns is the HLO-level outputs.
// If unconditionally_use_output_shapes is true then the unregistered
// attribute _output_shapes is always used to set the output shapes of the ops.
ABSL_DEPRECATED(
    "Use v1/compile_tf_graph.h::CompileTensorflowGraphToHlo instead.")
absl::Status BuildHloFromGraph(
    const Graph& graph, xla::XlaBuilder& builder,
    mlir::MLIRContext& mlir_context, llvm::ArrayRef<xla::XlaOp> xla_params,
    std::vector<xla::XlaOp>& returns, bool unconditionally_use_output_shapes,
    llvm::ArrayRef<XlaArgument> args, llvm::ArrayRef<std::string> control_rets,
    llvm::StringRef device_type, const FunctionLibraryDefinition& flib_def);

static inline absl::Status CompileToHloGraphAnalysisFailedError() {
  return errors::Internal("disabled after graph analysis");
}

// Register a convenient pipeline for invoking TF/XLA lowering from the command
// line.
void RegisterConvertMlirToXlaHloPipelineWithDefaults();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_COMPILE_MLIR_UTIL_H_
