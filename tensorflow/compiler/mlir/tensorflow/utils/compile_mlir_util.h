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

#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

// Lowers MLIR module to XLA HLO inside an XlaComputation. The input module
// should only contain operations in tf dialect. If the input module contains
// operation in the tf_executor dialect, for example, returns an error.
//
// use_tuple_args: when this is true, always create a tuple argument for the
//   entry computation.
// return_tuple: when this is true, always create a tuple result for the
//   entry computation.
Status ConvertMLIRToXlaComputation(mlir::ModuleOp module_op,
                                   xla::XlaComputation* xla_computation,
                                   bool use_tuple_args, bool return_tuple);

// Compiles a serialized MLIR module into XLA HLO, generates all accompanying
// metadata and stores them in CompilationResult.
Status CompileSerializedMlirToXlaHlo(
    llvm::StringRef mlir_module_string, absl::Span<TensorShape> arg_shapes,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    XlaCompiler::CompilationResult* compilation_result);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_COMPILE_MLIR_UTIL_H_
