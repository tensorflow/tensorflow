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

#include <memory>
#include <utility>

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

  mlir::ModuleOp mlir_module;

  static char ID;  // NOLINT

 private:
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  mlir::OwningOpRef<mlir::ModuleOp> owning_mlir_module;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_HLO_HLO_PROGRAM_H_
