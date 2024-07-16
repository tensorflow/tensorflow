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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_HLO_MODULE_IMPORTER_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_HLO_MODULE_IMPORTER_H_

#include <unordered_map>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/xla_data.pb.h"

namespace xla {
class HloModule;
class HloModuleProto;
class HloComputation;
class HloInstruction;
class Shape;

// Importer that takes an HloModule and imports it as an MLIR module in the XLA
// dialect. HloModuleImporter does not take ownership.
class HloModuleImporter {
 public:
  explicit HloModuleImporter(mlir::ModuleOp module,
                             bool import_all_computation = false,
                             bool flatten_computation_args_result = false);

  // Import the HloModule into the MLIR Module.
  absl::Status Import(const xla::HloModule& module);

  // Import the HloModuleProto into the MLIR Module.
  absl::Status Import(const xla::HloModuleProto& module);

 private:
  void ImportFrontendAttributes(const xla::HloModule& hlo_module,
                                mlir::ModuleOp module);

  bool import_all_computation_;
  bool flatten_computation_args_result_;
  mlir::SymbolTable symbol_table_;
  mlir::Builder builder_;

  // Map for tracking which MLIR function map to which HLO Computation. This
  // tracks functions as they are imported and provides a quick lookup for
  // functions invoked by control flow related operations (e.g. while, call).
  std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>
      function_map_;
};

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_HLO_MODULE_IMPORTER_H_
