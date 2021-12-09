/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_MLIR_TO_HLO_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_MLIR_TO_HLO_H_

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

// Converts an MHLO/CHLO module string to an mlir::Module.
StatusOr<mlir::OwningModuleRef> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context);

// Converts an CHLO/MHLO module to XLA HLO.
Status MlirToXlaComputation(mlir::ModuleOp module,
                            XlaComputation& xla_computation,
                            bool use_tuple_args, bool return_tuple);

// Converts an MHLO/CHLO module string to an XLA computation.
Status ParseMlirModuleStringAndConvertToXlaComputation(
    absl::string_view mlir_module_str, XlaComputation& xla_computation,
    bool use_tuple_args, bool return_tuple);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_MLIR_TO_HLO_H_
