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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/status_macros.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"

namespace xla::ifrt {

char HloProgram::ID = 0;

absl::StatusOr<std::string> HloProgram::ToBytes() const {
  tsl::StatusScopedDiagnosticHandler diag_handler(mlir_module->getContext());
  std::string serialized;
  llvm::raw_string_ostream out(serialized);
  mlir::LogicalResult result =
      mlir::writeBytecodeToFile(mlir_module, out, mlir::BytecodeWriterConfig());
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
    absl::string_view bytes, std::unique_ptr<mlir::MLIRContext> context) {
  if (context == nullptr) {
    context = std::make_unique<mlir::MLIRContext>(
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

namespace {

// Calculates a HighwayHash fingerprint in a streaming manner.
class HighwayHashStream final : public llvm::raw_ostream {
 public:
  HighwayHashStream() : hash_(kHighwayHashKey), pos_(0) { SetUnbuffered(); }

  // Destructively calculates the fingerprint of the data consumed so far.
  uint64_t fingerprint() && {
    highwayhash::HHResult64 result;
    hash_.Finalize(&result);
    return result;
  }

 private:
  // Arbitrarily chosen, forever-unchanging hash key required by HighwayHash.
  static constexpr highwayhash::HHKey kHighwayHashKey = {
      0x4451e30f87db9609ULL,
      0xca7358a1fd2737f8ULL,
      0x4b2c991fcee4fdeaULL,
      0x0b2658e18326f6baULL,
  };

  void write_impl(const char* Ptr, size_t Size) final {
    hash_.Append(Ptr, Size);
    pos_ += Size;
  }

  uint64_t current_pos() const final { return pos_; }

  highwayhash::HighwayHashCatT<HH_TARGET> hash_;
  uint64_t pos_;
};

}  // namespace

uint64_t HloProgram::Fingerprint() const {
  HighwayHashStream os;
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  mlir_module->print(os, flags);
  return std::move(os).fingerprint();
}

}  // namespace xla::ifrt
