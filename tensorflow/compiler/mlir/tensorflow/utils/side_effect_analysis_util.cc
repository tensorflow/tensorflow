/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/side_effect_analysis_util.h"

#include <string>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

std::string GetDeviceAttrAsResourceInstanceStr(mlir::Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  // Treat missing device attribute like unspecified (= empty string) attribute.
  // Note that different op instances with the same string (including empty
  // string) are seen as dependent (same resource instance).
  if (!device_attr) return "";
  return device_attr.str();
}

void MarkResourceAsReadAndWrite(
    OpOperand& op_operand,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  if (op_operand.get()
          .getType()
          .cast<TensorType>()
          .getElementType()
          .isa<ResourceType>()) {
    effects.emplace_back(MemoryEffects::Read::get(), &op_operand,
                         ResourceEffects::Variable::get());
    effects.emplace_back(MemoryEffects::Write::get(), &op_operand,
                         ResourceEffects::Variable::get());
  }
}

void MarkResourceAsReadOnly(
    OpOperand& op_operand,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  if (op_operand.get()
          .getType()
          .cast<TensorType>()
          .getElementType()
          .isa<ResourceType>()) {
    effects.emplace_back(MemoryEffects::Read::get(), &op_operand,
                         ResourceEffects::Variable::get());
  }
}

}  // namespace TF
}  // namespace mlir
