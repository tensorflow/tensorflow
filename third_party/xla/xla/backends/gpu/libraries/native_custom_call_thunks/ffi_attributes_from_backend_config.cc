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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/ffi_attributes_from_backend_config.h"

#include <string>

#include "absl/status/statusor.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/attributes.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

absl::StatusOr<xla::ffi::Attributes> FfiAttributesFromBackendConfig(
    const HloCustomCallInstruction& instr, mlir::MLIRContext& mlir_context) {
  absl::StatusOr<GpuBackendConfig> backend_config =
      instr.backend_config<GpuBackendConfig>();
  const std::string& backend_config_str =
      backend_config.ok()
          ? backend_config->custom_call_backend_config().attributes()
          : instr.raw_backend_config_string();
  if (backend_config_str.empty()) {
    return xla::ffi::Attributes::Create(xla::ffi::AttributesMap());
  }

  mlir::Attribute attr =
      mlir::parseAttribute(backend_config_str, &mlir_context);
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  TF_RET_CHECK(dict != nullptr)
      << "Unsupported backend config. Expected a string parsable into a "
         "dictionary attribute.";
  ABSL_ASSIGN_OR_RETURN(xla::ffi::AttributesMap attributes,
                   xla::ffi::BuildAttributesMap(dict));
  return xla::ffi::Attributes::Create(attributes);
}

}  // namespace xla::gpu
