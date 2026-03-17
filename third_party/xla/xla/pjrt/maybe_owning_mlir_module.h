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

#ifndef XLA_PJRT_MAYBE_OWNING_MLIR_MODULE_H_
#define XLA_PJRT_MAYBE_OWNING_MLIR_MODULE_H_

#include <memory>
#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace xla {

// A wrapper around an MLIR module that can be used to transfer ownership of
// the ModuleOp and MLIR context. This allows the compilation pipeline to
// release the memory used by the Module (and its context) during the
// compilation if it is no longer needed.
class MaybeOwningMlirModule {
 public:
  MaybeOwningMlirModule(std::unique_ptr<mlir::MLIRContext> context,
                        mlir::OwningOpRef<mlir::ModuleOp> module)
      : owned_context_(std::move(context)),
        owned_module_(std::move(module)),
        module_(*owned_module_) {}
  explicit MaybeOwningMlirModule(mlir::OwningOpRef<mlir::ModuleOp> module)
      : owned_module_(std::move(module)), module_(*owned_module_) {}
  explicit MaybeOwningMlirModule(mlir::ModuleOp module) : module_(module) {}

  MaybeOwningMlirModule() = default;
  MaybeOwningMlirModule(const MaybeOwningMlirModule& other) = delete;
  MaybeOwningMlirModule(MaybeOwningMlirModule&& other) noexcept {
    *this = std::move(other);
  }
  MaybeOwningMlirModule& operator=(MaybeOwningMlirModule&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    owned_module_ = std::move(other.owned_module_);
    owned_context_ = std::move(other.owned_context_);
    module_ = other.module_;
    other.module_ = nullptr;
    return *this;
  }

  mlir::ModuleOp mlir_module() { return module_; }

 private:
  std::unique_ptr<mlir::MLIRContext> owned_context_;
  mlir::OwningOpRef<mlir::ModuleOp> owned_module_;
  mlir::ModuleOp module_;
};

}  // namespace xla

#endif  // XLA_PJRT_MAYBE_OWNING_MLIR_MODULE_H_
