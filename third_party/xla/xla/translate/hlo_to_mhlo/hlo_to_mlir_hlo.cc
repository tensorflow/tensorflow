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

#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"

#include "xla/mlir/utils/error_util.h"
#include "xla/status_macros.h"
#include "xla/translate/hlo_to_mhlo/hlo_module_importer.h"

namespace xla {

Status ConvertHloToMlirHlo(mlir::ModuleOp module,
                           xla::HloModuleProto const* hlo_module_proto,
                           bool import_all_computation,
                           bool flatten_computation_args_result) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation,
                           flatten_computation_args_result)
      .Import(*hlo_module_proto);
}

Status ConvertHloToMlirHlo(mlir::ModuleOp module, xla::HloModule* hlo_module,
                           bool import_all_computation,
                           bool flatten_computation_args_result) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  return HloModuleImporter(module, import_all_computation,
                           flatten_computation_args_result)
      .Import(*hlo_module);
}

}  // namespace xla
