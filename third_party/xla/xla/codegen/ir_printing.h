/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_IR_PRINTING_H_
#define XLA_CODEGEN_IR_PRINTING_H_

#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Sets up the pass manager to log the IR to a file if requested by XLA flags.
//
// The file is dumped to the directory specified by --xla_dump_to. If this is
// set to "sponge", the file is dumped to the test's undeclared outputs
// directory.
//
// Dumping can be filtered by --xla_dump_emitter_re, which is matched against
// `pass_manager_name`.
//
// The file is dumped in the MLIR text format. Any diagnostics (e.g. from
// `emitWarning` or `emitError`) are also printed to the file.
//
// This function is a no-op if IR printing is not requested or if the log file
// cannot be created.
void EnableIRPrintingIfRequested(mlir::PassManager& pass_manager,
                                 mlir::MLIRContext* context,
                                 const HloModule& hlo_module,
                                 absl::string_view kernel_name,
                                 absl::string_view pass_manager_name);

}  // namespace xla

#endif  // XLA_CODEGEN_IR_PRINTING_H_
