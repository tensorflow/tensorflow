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

#ifndef XLA_PYTHON_IFRT_HLO_HLO_PROGRAM_H_
#define XLA_PYTHON_IFRT_HLO_HLO_PROGRAM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/python/ifrt/program.h"

namespace xla {
namespace ifrt {

class HloProgram : public llvm::RTTIExtends<HloProgram, Program> {
 public:
  HloProgram() = default;

  explicit HloProgram(mlir::ModuleOp module)
      : mlir_module_(module), module_name_(GetModuleName(module)) {}

  explicit HloProgram(mlir::OwningOpRef<mlir::ModuleOp> module)
      : owning_mlir_module_(std::move(module)),
        mlir_module_(*owning_mlir_module_),
        module_name_(GetModuleName(mlir_module_)) {}

  HloProgram(std::shared_ptr<mlir::MLIRContext> context,
             mlir::OwningOpRef<mlir::ModuleOp> module)
      : mlir_context_(std::move(context)),
        owning_mlir_module_(std::move(module)),
        mlir_module_(*owning_mlir_module_),
        module_name_(GetModuleName(mlir_module_)) {}

  mlir::ModuleOp mlir_module() const { return mlir_module_; }

  absl::string_view name() const { return module_name_; }

  // Serializes the HloProgram into bytes such that deserialization via
  // `HloProgram::FromBytes()` results in the exact same program when
  // deserialized at the same binary version.
  //
  // Note: Unlike `HloProgramSerDes`, bytes returned by this method are NOT
  // version compatible and can be deserialized only at the same version. Most
  // users should prefer `Serialize(hlo_program, options)` for this reason.
  absl::StatusOr<std::string> ToBytes() const;

  // Constructs a HloProgram from the given bytes. If the context is not
  // provided, the method creates a new MLIR context just for this program.
  static absl::StatusOr<std::unique_ptr<HloProgram>> FromBytes(
      absl::string_view bytes,
      std::shared_ptr<mlir::MLIRContext> context = nullptr);

  // Returns a fingerprint of the HLO program. Two HLO programs are equivalent
  // if their fingerprints are the same. May ignore debug info.
  absl::StatusOr<uint64_t> Fingerprint() const;

  // Destructively converts this HloProgram into a MaybeOwningMlirModule.
  xla::MaybeOwningMlirModule ToMaybeOwningMlirModule() &&;

  static char ID;  // NOLINT

 private:
  // Returns the name of the module. Returns "unnamed" if the module does not
  // have a symbol name. The method should only be called by the constructors
  // to ensure that MLIR API is only used at construction time.
  static std::string GetModuleName(mlir::ModuleOp module) {
    const std::optional<llvm::StringRef> name = module.getSymName();
    if (name.has_value()) {
      return std::string(*name);
    }

    // Generate a unique and stable computation name to help debugging.
    return absl::StrFormat("unnamed_%x",
                           reinterpret_cast<uintptr_t>(module.getOperation()));
  }

  std::shared_ptr<mlir::MLIRContext> mlir_context_;
  mlir::OwningOpRef<mlir::ModuleOp> owning_mlir_module_;
  mlir::ModuleOp mlir_module_;
  std::string module_name_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_HLO_HLO_PROGRAM_H_
