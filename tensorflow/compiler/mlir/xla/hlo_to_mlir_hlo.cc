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

#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"

#include "tensorflow/compiler/mlir/xla/hlo_module_importer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                           xla::HloModuleProto* hlo_module_proto,
                           bool import_all_computation) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation)
      .Import(*hlo_module_proto);
}

Status ConvertHloToMlirHlo(mlir::ModuleOp module, xla::HloModule* hlo_module,
                           bool import_all_computation) {
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation).Import(*hlo_module);
}

}  // namespace xla
