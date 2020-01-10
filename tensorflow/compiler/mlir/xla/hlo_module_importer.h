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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_

#include <unordered_map>

#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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
  explicit HloModuleImporter(mlir::ModuleOp module)
      : module_(module), builder_(module.getContext()) {}

  // Import the HloModule into the MLIR Module.
  Status Import(const xla::HloModule& module);

  // Import the HloModuleProto into the MLIR Module.
  Status Import(const xla::HloModuleProto& module);

 private:
  mlir::ModuleOp module_;
  mlir::Builder builder_;

  // Map for tracking which MLIR function map to which HLO Computation. This
  // tracks functions as they are imported and provides a quick lookup for
  // functions invoked by control flow related operations (e.g. while, call).
  std::unordered_map<xla::HloComputation*, mlir::FuncOp> function_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_HLO_MODULE_IMPORTER_H_
