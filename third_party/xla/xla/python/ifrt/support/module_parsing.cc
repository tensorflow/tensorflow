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

#include "xla/python/ifrt/support/module_parsing.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/transforms/built_in_spmd_expansions.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"

namespace xla {
namespace ifrt {
namespace support {

void InitializeMlirDialectRegistry(mlir::DialectRegistry& registry) {
  registry.insert<xla::ifrt::IfrtDialect>();
  registry.insert<xla::ifrt::VifrtDialect>();
  mlir::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::sdy::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  xla::ifrt::AttachBuiltInSpmdExpansions(registry);
}

void RegisterMlirDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  InitializeMlirDialectRegistry(registry);
  context.appendDialectRegistry(registry);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ParseMlirModuleString(
    absl::string_view mlir_module_str, mlir::MLIRContext& context) {
  RegisterMlirDialects(context);
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_module_str, &context);
  if (!module) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to parse IFRT IR module string: %s",
                        diagnostic_handler.ConsumeStatus().message()));
  }
  return module;
}

}  // namespace support
}  // namespace ifrt
}  // namespace xla
