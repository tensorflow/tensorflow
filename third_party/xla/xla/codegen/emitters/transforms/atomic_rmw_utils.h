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
#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_

#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/ir/xla_ops.h"

namespace xla {
namespace emitters {

// Describes an atomic read-modify-write as a binary op plus its argument.
struct AtomicModifierParameters {
  mlir::Value modifier;
  mlir::LLVM::AtomicBinOp bin_op;
  // True when `modifier` is subtracted instead of added.
  bool negated = false;
};

// Returns the value combined into the atomic, the operation to apply, and
// whether that value is negated.
std::optional<AtomicModifierParameters> GetAtomicModifierParameters(
    AtomicRMWOp op);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_ATOMIC_RMW_UTILS_H_
