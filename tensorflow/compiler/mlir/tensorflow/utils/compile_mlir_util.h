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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

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
// return_tuple: when this is true, always create a tuple result for the
//   entry computation.
// shape_representation_fn: when this is set, this shape representation function
//   will be used to determine argument and result shapes. Otherwise the
//   original shape will be used as is.
// custom_legalization_passes: passes to run before the default TF legalization
//   passes for backend-specific ops.
Status ConvertMLIRToXlaComputation(
    mlir::ModuleOp module_op, llvm::StringRef device_type,
    xla::XlaComputation* xla_computation, bool use_tuple_args,
    bool return_tuple,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn = nullptr,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes = {});

// Compiles a serialized MLIR module into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult.
Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, llvm::ArrayRef<TensorShape> arg_shapes,
    llvm::StringRef device_type, bool use_tuple_args,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes = {});

// Same as the above but takes input as TensorFlow Graph.
Status CompileGraphToXlaHlo(
    const Graph& graph, llvm::ArrayRef<const XlaCompiler::Argument> args,
    llvm::StringRef device_type, bool use_tuple_args,
    const FunctionLibraryDefinition& flib_def, const GraphDebugInfo& debug_info,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes = {});

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
