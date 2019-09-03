/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/hlo_dialect_emitter.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace mlir_gpu {

namespace {

using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::Builder;
using ::mlir::Identifier;
using ::mlir::Location;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::Type;
using ::mlir::Value;

namespace hlo = ::mlir::xla_hlo;

// TODO(b/137624192) Use tablegen for this.
StatusOr<Value*> InsertMlirOp(
    HloOpcode opcode, OpBuilder func_builder, Location loc, ArrayRef<Type> rets,
    ArrayRef<Value*> args, ArrayRef<std::pair<Identifier, Attribute>> attrs) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return {func_builder.create<hlo::AddOp>(loc, rets, args, attrs)};

    case HloOpcode::kMultiply:
      return {func_builder.create<hlo::MulOp>(loc, rets, args, attrs)};
    case HloOpcode::kSubtract:
      return {func_builder.create<hlo::SubOp>(loc, rets, args, attrs)};
    case HloOpcode::kDivide:
      return {func_builder.create<hlo::DivOp>(loc, rets, args, attrs)};
    case HloOpcode::kAnd:
      return {func_builder.create<hlo::AndOp>(loc, rets, args, attrs)};
    case HloOpcode::kMinimum:
      return {func_builder.create<hlo::MinOp>(loc, rets, args, attrs)};
    case HloOpcode::kMaximum:
      return {func_builder.create<hlo::MaxOp>(loc, rets, args, attrs)};
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Opcode ", HloOpcodeString(opcode), " is not supported."));
  }
}

StatusOr<::mlir::TensorType> ConvertTensorType(const Shape& shape,
                                               Builder builder) {
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());

  switch (shape.element_type()) {
    case PrimitiveType::PRED:
      return builder.getTensorType(array, builder.getI1Type());
    case PrimitiveType::F16:
      return builder.getTensorType(array, builder.getF16Type());
    case PrimitiveType::F32:
      return builder.getTensorType(array, builder.getF32Type());
    case PrimitiveType::F64:
      return builder.getTensorType(array, builder.getF64Type());
    case PrimitiveType::S8:
      return builder.getTensorType(array, builder.getIntegerType(8));
    case PrimitiveType::S16:
      return builder.getTensorType(array, builder.getIntegerType(16));
    case PrimitiveType::S32:
      return builder.getTensorType(array, builder.getIntegerType(32));
    case PrimitiveType::S64:
      return builder.getTensorType(array, builder.getIntegerType(64));
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", PrimitiveType_Name(shape.element_type())));
  }
}

}  // namespace

Status HloDialectEmitter::EmitComputation(const HloComputation& computation) {
  return computation.root_instruction()->Accept(this);
}

Status HloDialectEmitter::DefaultAction(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto resType,
                      ConvertTensorType(instr->shape(), builder_));

  auto attribute =
      builder_.getNamedAttr("name", builder_.getStringAttr(instr->name()));
  llvm::SmallVector<Value*, 4> arguments;
  for (auto operand : instr->operands()) {
    arguments.push_back(instruction_to_values_[operand]);
  }
  TF_ASSIGN_OR_RETURN(
      auto inserted,
      InsertMlirOp(instr->opcode(), builder_, builder_.getUnknownLoc(), resType,
                   arguments, attribute));
  instruction_to_values_[instr] = inserted;
  return Status::OK();
}

Status HloDialectEmitter::HandleParameter(HloInstruction* param) {
  auto argValue = arguments_[param->parameter_number()];
  auto tensorValue =
      builder_.create<::mlir::TensorLoadOp>(builder_.getUnknownLoc(), argValue);
  instruction_to_values_[param] = tensorValue;
  return Status::OK();
}

Status HloDialectEmitter::FinishVisit(HloInstruction* root) {
  auto resultValue = instruction_to_values_[root];
  auto resultMemref = arguments_.back();
  builder_.create<::mlir::TensorStoreOp>(builder_.getUnknownLoc(), resultValue,
                                         resultMemref);
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
