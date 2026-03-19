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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_IMPORT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_IMPORT_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace tflite {
// Converts a TFLite flatbuffer stored in `buffer` to a MLIR module
// The buffer must live for the duration of the function call,
// The caller receives ownership of the module.
// `base_loc` is used for error reporting and debug info.
// If ordered_output_arrays is not empty, then the imported mlir function will
// only return nodes in ordered_output_arrays in the same order. Returns nullptr
// on failure, and more specific errors will be emitted via the context.
// If `use_external_constant` is true, it will create `tfl.external_const`
// instead of `tfl.const`.
// If `experimental_prune_unreachable_nodes_unconditionally` is true, nodes that
// are not ancestors of the output nodes will be pruned.
mlir::OwningOpRef<mlir::ModuleOp> FlatBufferToMlir(
    absl::string_view buffer, mlir::MLIRContext* context,
    mlir::Location base_loc, bool use_external_constant = false,
    const std::vector<std::string>& ordered_input_arrays = {},
    const std::vector<std::string>& ordered_output_arrays = {},
    bool experimental_prune_unreachable_nodes_unconditionally = false,
    bool disable_vhlo_to_stablehlo = false);
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_FLATBUFFER_IMPORT_H_
