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
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/python/ifrt/program.h"

namespace xla {
namespace ifrt {

struct HloProgram : llvm::RTTIExtends<HloProgram, Program> {
  HloProgram() = default;

  explicit HloProgram(mlir::ModuleOp module) : mlir_module(module) {}

  HloProgram(std::unique_ptr<mlir::MLIRContext> context,
             mlir::OwningOpRef<mlir::ModuleOp> module)
      : mlir_module(*module),
        mlir_context(std::move(context)),
        owning_mlir_module(std::move(module)) {}

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
      std::unique_ptr<mlir::MLIRContext> context = nullptr);

  // Returns a fingerprint of the HLO program. Two HLO programs are equivalent
  // if their fingerprints are the same. May ignore debug info.
  uint64_t Fingerprint() const;

  mlir::ModuleOp mlir_module;

  static char ID;  // NOLINT

 private:
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> owning_mlir_module;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_HLO_HLO_PROGRAM_H_
