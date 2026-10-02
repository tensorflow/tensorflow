/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/hlo/hlo_program.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/python/ifrt/mlir/fingerprint_utils.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"

namespace xla::ifrt {

char HloProgram::ID = 0;

absl::StatusOr<std::string> HloProgram::ToBytes() const {
  tsl::StatusScopedDiagnosticHandler diag_handler(mlir_module_->getContext());
  std::string serialized;
  llvm::raw_string_ostream out(serialized);
  mlir::LogicalResult result = mlir::writeBytecodeToFile(
      mlir_module_, out, mlir::BytecodeWriterConfig());
  absl::Status status = diag_handler.consumeStatus();
  if (!status.ok()) {
    tsl::errors::AppendToMessage(
        &status, "Failed while serializing HloProgram into bytes");
    return status;
  }
  TF_RET_CHECK(mlir::succeeded(result));
  return serialized;
}

absl::StatusOr<std::unique_ptr<HloProgram>> HloProgram::FromBytes(
    absl::string_view bytes, std::shared_ptr<mlir::MLIRContext> context) {
  if (context == nullptr) {
    context = std::make_shared<mlir::MLIRContext>(
        mlir::MLIRContext::Threading::DISABLED);
    mlir::DialectRegistry registry;
    xla::RegisterAllHloDialects(registry);
    context->appendDialectRegistry(registry);
  }

  tsl::StatusScopedDiagnosticHandler diag_handler(context.get());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(bytes, context.get());
  absl::Status status = diag_handler.consumeStatus();
  if (!status.ok()) {
    tsl::errors::AppendToMessage(
        &status, "Failed while deserializing HloProgram from bytes");
    return status;
  }
  TF_RET_CHECK(module);

  return std::make_unique<xla::ifrt::HloProgram>(std::move(context),
                                                 std::move(module));
}

absl::StatusOr<uint64_t> HloProgram::Fingerprint() const {
  absl::StatusOr<uint64_t> fingerprint = FingerprintModuleOp(mlir_module_);
  if (!fingerprint.ok()) {
    absl::Status status = fingerprint.status();
    tsl::errors::AppendToMessage(
        &status, "Failed while calculating HloProgram fingerprint");
    return status;
  }
  return *fingerprint;
}

xla::MaybeOwningMlirModule HloProgram::ToMaybeOwningMlirModule() && {
  if (owning_mlir_module_) {
    return xla::MaybeOwningMlirModule(std::move(mlir_context_),
                                      std::move(owning_mlir_module_));
  }
  CHECK(mlir_context_ == nullptr);
  return xla::MaybeOwningMlirModule(std::move(mlir_module_));
}

}  // namespace xla::ifrt
