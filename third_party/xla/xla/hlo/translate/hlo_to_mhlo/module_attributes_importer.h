/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_MODULE_ATTRIBUTES_IMPORTER_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_MODULE_ATTRIBUTES_IMPORTER_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/util.h"

namespace xla {

// Imports the HLO module config into the MLIR module as module attributes
// prefixed with `mhlo.`.
// TODO (b/345755258) Support roundtrip of all HLO module config fields.
void ImportCrossProgramPrefetches(const HloModule& hlo_module,
                                  mlir::ModuleOp module,
                                  bool flatten_computation_args_result,
                                  mlir::Builder builder);

void ImportEntryComputationLayoutAndTiles(const HloModule& hlo_module,
                                          mlir::ModuleOp module,
                                          bool flatten_computation_args_result,
                                          mlir::Builder builder);

void ImportFrontendAttributes(const HloModule& hlo_module,
                              mlir::ModuleOp module, mlir::Builder builder);

void ImportInputOutputAlias(const HloModule& hlo_module, mlir::ModuleOp module,
                            mlir::Builder builder);

void ImportIsDynamic(const HloModule& hlo_module, mlir::ModuleOp module,
                     mlir::Builder builder);

void ImportNumPartitions(const HloModule& hlo_module, mlir::ModuleOp module,
                         mlir::Builder builder);

void ImportNumReplicas(const HloModule& hlo_module, mlir::ModuleOp module,
                       mlir::Builder builder);

void ImportSpmdOutputSharding(const HloModule& hlo_module,
                              mlir::ModuleOp module, mlir::Builder builder);

void ImportSpmdParametersShardings(const HloModule& hlo_module,
                                   mlir::ModuleOp module,
                                   bool flatten_computation_args_result,
                                   mlir::Builder builder);

void ImportUseAutoSpmdPartitioning(const HloModule& hlo_module,
                                   mlir::ModuleOp module,
                                   mlir::Builder builder);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_MODULE_ATTRIBUTES_IMPORTER_H_
