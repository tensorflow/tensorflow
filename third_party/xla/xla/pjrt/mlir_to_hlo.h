/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_MLIR_TO_HLO_H_
#define XLA_PJRT_MLIR_TO_HLO_H_

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/xla_computation.h"
#include "xla/status.h"

namespace xla {

// Converts an mlir::Module to MLIR Bytecode Format, with StableHLO attribute
// downgrades for limited forward/backward compatibility.
StatusOr<std::string> SerializeModule(mlir::ModuleOp);

// Converts an MHLO/CHLO module string to an mlir::Module.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context);

// Converts an CHLO/MHLO module to XLA HLO.
Status MlirToXlaComputation(mlir::ModuleOp module,
                            XlaComputation& xla_computation,
                            bool use_tuple_args, bool return_tuple,
                            bool legalize_sparse_ops = false);

// Converts an MHLO/CHLO module string to an XLA computation.
Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple);

// Downgrades stablehlo ops in the module that are using DenseArrays but need to
// use DenseElements when serialized for backward compatibility. Context: in
// https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
// fields on some ops were changed to use DenseI64ArrayAttr instead of
// I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
// dense elements, not dense arrays, so convert the arrays to elements before
// serializing. The elements need to be converted back to arrays when
// deserializing.
// TODO: b/320507168 - Delete this function.
void DowngradeStablehlo(mlir::ModuleOp);

// Upgrades stablehlo ops in the module that are using DenseElements but should
// be using DenseArrays. Context: in
// https://github.com/google/jax/commit/184e3a88004680dbf34328b05c5fc0d869cc4a93,
// fields on some ops were changed to use DenseI64ArrayAttr instead of
// I64DenseElementsAttr (DenseIntElementsAttr). Some clients still expect
// dense elements, not dense arrays, so when serializing we always convert the
// arrays to elements. The elements need to be converted back to arrays when
// deserializing.
// TODO: b/320507168 - Delete this function.
void UpgradeStablehlo(mlir::ModuleOp);

}  // namespace xla

#endif  // XLA_PJRT_MLIR_TO_HLO_H_
