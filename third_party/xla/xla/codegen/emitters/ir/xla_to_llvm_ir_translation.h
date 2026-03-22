/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_EMITTERS_IR_XLA_TO_LLVM_IR_TRANSLATION_H_
#define XLA_CODEGEN_EMITTERS_IR_XLA_TO_LLVM_IR_TRANSLATION_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace xla {

// Registers the XLA dialect LLVM IR translation interface in the given
// registry.
void registerXlaDialectTranslation(mlir::DialectRegistry& registry);

// Registers the XLA dialect LLVM IR translation interface in the given
// context.
void registerXlaDialectTranslation(mlir::MLIRContext& context);

}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_IR_XLA_TO_LLVM_IR_TRANSLATION_H_
