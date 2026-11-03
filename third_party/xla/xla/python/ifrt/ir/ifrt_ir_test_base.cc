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

#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"

namespace xla {
namespace ifrt {
namespace test_util {

IfrtIrTestBase::IfrtIrTestBase() {
  mlir::registerMLIRContextCLOptions();
  support::RegisterMlirDialects(mlir_context_);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
IfrtIrTestBase::LoadFromSource(absl::string_view source) {
  mlir::BaseScopedDiagnosticHandler diagnostic_handler(&mlir_context_);
  mlir::OwningOpRef<mlir::ModuleOp> op_ref =
      mlir::parseSourceString<mlir::ModuleOp>(source, &mlir_context_);
  if (!op_ref) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to parse IFRT IR module string: %s",
                        diagnostic_handler.ConsumeStatus().message()));
  }
  return op_ref;
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
