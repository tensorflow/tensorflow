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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_

#include "xla/status.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

namespace xla {
class HloModule;
class HloModuleProto;

// Converts an HLO module proto to a MLIR module in HLO dialect.
// If import_all_computation is set to true, imports all computations
// irrespective if transitively called from entry computation.
Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                           xla::HloModuleProto const* hlo_module,
                           bool import_all_computations = false);

// Converts an HLO module to a MLIR module in HLO dialect.
// If import_all_computation is set to true, imports all computations
// irrespective if transitively called from entry computation.
Status ConvertHloToMlirHlo(mlir::ModuleOp module, xla::HloModule* hlo_module,
                           bool import_all_computations = false);

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_HLO_TO_MLIR_HLO_H_
