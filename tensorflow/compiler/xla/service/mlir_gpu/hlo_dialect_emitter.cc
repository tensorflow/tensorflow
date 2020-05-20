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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::Identifier;
using ::mlir::Location;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::Value;

namespace hlo = ::mlir::xla_hlo;

// TODO(b/137624192) Use tablegen for this.
StatusOr<Value> InsertMlirOp(HloOpcode opcode, OpBuilder func_builder,
                             Location loc, ArrayRef<Type> rets,
                             ArrayRef<Value> args,
                             ArrayRef<std::pair<Identifier, Attribute>> attrs) {
  switch (opcode) {
    case HloOpcode::kAbs:
      return {func_builder.create<hlo::AbsOp>(loc, rets, args, attrs)};
    case HloOpcode::kAdd:
      return {func_builder.create<hlo::AddOp>(loc, rets, args, attrs)};
    case HloOpcode::kAnd:
      return {func_builder.create<hlo::AndOp>(loc, rets, args, attrs)};
    case HloOpcode::kCeil:
      return {func_builder.create<hlo::CeilOp>(loc, rets, args, attrs)};
    case HloOpcode::kComplex:
      return {func_builder.create<hlo::ComplexOp>(loc, rets, args, attrs)};
    case HloOpcode::kCopy:
      return {func_builder.create<hlo::CopyOp>(loc, rets, args, attrs)};
    case HloOpcode::kCos:
      return {func_builder.create<hlo::CosOp>(loc, rets, args, attrs)};
    case HloOpcode::kDivide:
      return {func_builder.create<hlo::DivOp>(loc, rets, args, attrs)};
    case HloOpcode::kExp:
      return {func_builder.create<hlo::ExpOp>(loc, rets, args, attrs)};
    case HloOpcode::kImag:
      return {func_builder.create<hlo::ImagOp>(loc, rets, args, attrs)};
    case HloOpcode::kLog:
      return {func_builder.create<hlo::LogOp>(loc, rets, args, attrs)};
    case HloOpcode::kMaximum:
      return {func_builder.create<hlo::MaxOp>(loc, rets, args, attrs)};
    case HloOpcode::kMinimum:
      return {func_builder.create<hlo::MinOp>(loc, rets, args, attrs)};
    case HloOpcode::kMultiply:
      return {func_builder.create<hlo::MulOp>(loc, rets, args, attrs)};
    case HloOpcode::kNegate:
      return {func_builder.create<hlo::NegOp>(loc, rets, args, attrs)};
    case HloOpcode::kReal:
      return {func_builder.create<hlo::RealOp>(loc, rets, args, attrs)};
    case HloOpcode::kRemainder:
      return {func_builder.create<hlo::RemOp>(loc, rets, args, attrs)};
    case HloOpcode::kRsqrt:
      return {func_builder.create<hlo::RsqrtOp>(loc, rets, args, attrs)};
    case HloOpcode::kSelect:
      return {func_builder.create<hlo::SelectOp>(loc, rets, args, attrs)};
    case HloOpcode::kSign:
      return {func_builder.create<hlo::SignOp>(loc, rets, args, attrs)};
    case HloOpcode::kSqrt:
      return {func_builder.create<hlo::SqrtOp>(loc, rets, args, attrs)};
    case HloOpcode::kSubtract:
      return {func_builder.create<hlo::SubOp>(loc, rets, args, attrs)};
    case HloOpcode::kTanh:
      return {func_builder.create<hlo::TanhOp>(loc, rets, args, attrs)};
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "HLO Opcode ", HloOpcodeString(opcode), " is not supported."));
  }
}

}  // namespace

mlir::Location HloDialectEmitter::getLocation(
    const HloInstruction* instr) const {
  return emission_context_->getLocation(instr);
}

StatusOr<Value> HloDialectEmitter::EmitComputation(
    const HloComputation& computation) {
  const auto root = computation.root_instruction();
  TF_RETURN_IF_ERROR(root->Accept(this));
  return instruction_to_values_[root];
}

Status HloDialectEmitter::DefaultAction(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto res_type, ConvertTensorShapeToType<RankedTensorType>(
                                         instr->shape(), builder_));
  llvm::SmallVector<Value, 4> arguments;
  arguments.reserve(instr->operand_count());
  for (auto operand : instr->operands()) {
    arguments.push_back(instruction_to_values_[operand]);
  }
  TF_ASSIGN_OR_RETURN(
      auto inserted, InsertMlirOp(instr->opcode(), builder_, getLocation(instr),
                                  res_type, arguments, llvm::None));
  instruction_to_values_[instr] = inserted;
  return Status::OK();
}

Status HloDialectEmitter::HandleBroadcast(HloInstruction* instr) {
  mlir::DenseIntElementsAttr broadcast_dim =
      CreateDenseIntElementsAttrFromVector(instr->dimensions(), builder_);
  TF_ASSIGN_OR_RETURN(Type res_type, ConvertTensorShapeToType<RankedTensorType>(
                                         instr->shape(), builder_));

  instruction_to_values_[instr] = builder_.create<hlo::BroadcastInDimOp>(
      getLocation(instr), llvm::makeArrayRef(res_type),
      instruction_to_values_[instr->operand(0)], broadcast_dim);
  return Status::OK();
}

Status HloDialectEmitter::HandleConcatenate(HloInstruction* instr) {
  int64 concatenate_dim = instr->concatenate_dimension();
  TF_ASSIGN_OR_RETURN(Type res_type, ConvertTensorShapeToType<RankedTensorType>(
                                         instr->shape(), builder_));

  llvm::SmallVector<Value, 4> arguments;
  arguments.reserve(instr->operand_count());
  for (auto operand : instr->operands()) {
    arguments.push_back(instruction_to_values_[operand]);
  }

  instruction_to_values_[instr] = builder_.create<hlo::ConcatenateOp>(
      getLocation(instr), llvm::makeArrayRef(res_type), arguments,
      builder_.getI64IntegerAttr(concatenate_dim));
  return Status::OK();
}

Status HloDialectEmitter::HandleParameter(HloInstruction* instr) {
  auto argValue = arguments_[instr->parameter_number()];
  instruction_to_values_[instr] = argValue;
  return Status::OK();
}

Status HloDialectEmitter::HandleConstant(HloInstruction* instr) {
  auto shape = instr->shape();
  if (!shape.IsArray() || shape.rank() != 0) {
    return Unimplemented("non-scalar constants are not supported yet");
  }
  TF_ASSIGN_OR_RETURN(auto type, ConvertTensorShapeToType<RankedTensorType>(
                                     instr->shape(), builder_));

  TF_ASSIGN_OR_RETURN(auto value, CreateDenseElementsAttrFromLiteral(
                                      instr->literal(), builder_));

  auto const_value =
      builder_.create<hlo::ConstOp>(getLocation(instr), type, value);
  instruction_to_values_[instr] = const_value;
  return Status::OK();
}

Status HloDialectEmitter::HandleReduce(HloInstruction* instr) {
  llvm::SmallVector<Value, 4> operands;
  for (auto operand : instr->operands()) {
    operands.push_back(instruction_to_values_.at(operand));
  }
  const unsigned num_inputs = operands.size() / 2;
  TF_ASSIGN_OR_RETURN(
      const auto return_type,
      ConvertTensorShapeToType<RankedTensorType>(instr->shape(), builder_));
  const auto dimensions_attr =
      CreateDenseIntElementsAttrFromVector(instr->dimensions(), builder_);
  auto reduceOp = builder_.create<hlo::ReduceOp>(
      getLocation(instr), return_type,
      llvm::makeArrayRef(operands).take_front(num_inputs),
      llvm::makeArrayRef(operands).take_back(num_inputs), dimensions_attr);
  {
    auto computation = instr->to_apply();
    auto block = new mlir::Block();
    llvm::SmallVector<Value, 4> arguments;
    arguments.reserve(computation->num_parameters());
    for (auto parameter : computation->parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(auto param_type,
                          ConvertTensorShapeToType<RankedTensorType>(
                              parameter->shape(), builder_));
      arguments.push_back(block->addArgument(param_type));
    }
    reduceOp.body().push_back(block);
    HloDialectEmitter emitter(emission_context_, &reduceOp.body(), arguments);
    TF_ASSIGN_OR_RETURN(auto result, emitter.EmitComputation(*computation));
    OpBuilder body_builder = OpBuilder::atBlockEnd(block);
    body_builder.setInsertionPointToEnd(block);
    body_builder.create<hlo::ReturnOp>(getLocation(instr),
                                       ArrayRef<Value>{result});
  }
  // TODO(b/137624192) Add support for multiple results.
  instruction_to_values_[instr] = reduceOp.getResult(0);
  return Status::OK();
}

Status HloDialectEmitter::HandleCompare(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(Type res_type, ConvertTensorShapeToType<RankedTensorType>(
                                         instr->shape(), builder_));
  auto comparison_direction_attr = builder_.getNamedAttr(
      "comparison_direction",
      builder_.getStringAttr(
          ComparisonDirectionToString(instr->comparison_direction())));
  llvm::SmallVector<Value, 4> arguments;
  arguments.reserve(instr->operand_count());
  for (auto operand : instr->operands()) {
    arguments.push_back(instruction_to_values_[operand]);
  }
  instruction_to_values_[instr] = builder_.create<hlo::CompareOp>(
      getLocation(instr), llvm::makeArrayRef(res_type), arguments,
      comparison_direction_attr);
  return Status::OK();
}

Status HloDialectEmitter::HandleIota(HloInstruction* instr) {
  mlir::IntegerAttr iota_dim = builder_.getI64IntegerAttr(
      static_cast<HloIotaInstruction*>(instr)->iota_dimension());
  TF_ASSIGN_OR_RETURN(Type res_type, ConvertTensorShapeToType<RankedTensorType>(
                                         instr->shape(), builder_));
  instruction_to_values_[instr] =
      builder_.create<hlo::IotaOp>(getLocation(instr), res_type, iota_dim);
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
