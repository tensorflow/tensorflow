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

#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace xla::cpu {

using EffectsVector = llvm::SmallVectorImpl<
    mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>;

void LoadOp::getEffects(EffectsVector& effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getCallFrameMutable(),
                       mlir::SideEffects::DefaultResource::get());
}

void ExtractWorkgroupIdOp::getEffects(EffectsVector& effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getCallFrameMutable(),
                       mlir::SideEffects::DefaultResource::get());
}

}  // namespace xla::cpu

#define GET_OP_CLASSES
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.cc.inc"
