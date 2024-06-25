/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_TRANSLATE_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_TRANSLATE_H_

namespace llvm {
class StringRef;
}  // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;
template <typename OpTy>
class OwningOpRef;
}  // namespace mlir

namespace xla {

// Converts a HloModuleProto stored in the file with the given `input_filename`
// into a MLIR module. Creates MLIR entities into the given MLIR `context`.
//
// If `import_all_computation` is set to true, imports all computations
// irrespective if transitively called from entry computation.
//
// If `flatten_computation_args_result` is set to true, flattens all tuple
// arguments and result of every computation when importing them as func ops.
mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations = false,
    bool flatten_computation_args_result = false);

// Converts a HloModule stored in text form for a file with the given
// `input_filename` into a MLIR module. Creates MLIR entities into the given
// MLIR `context`.
//
// If `import_all_computation` is set to true, imports all computations
// irrespective if transitively called from entry computation.
//
// If `flatten_computation_args_result` is set to true, flattens all tuple
// arguments and result of every computation when importing them as func ops.
mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations = false,
    bool flatten_computation_args_result = false);

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_TRANSLATE_H_
