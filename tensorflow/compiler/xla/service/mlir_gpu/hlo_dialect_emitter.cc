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
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
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
using ::mlir::ShapedType;
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
    case HloOpcode::kExp:
      return {func_builder.create<hlo::ExpOp>(loc, rets, args, attrs)};
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

StatusOr<Value*> HloDialectEmitter::EmitComputation(
    const HloComputation& computation) {
  const auto root = computation.root_instruction();
  TF_RETURN_IF_ERROR(root->Accept(this));
  return instruction_to_values_[root];
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
  instruction_to_values_[param] = argValue;
  return Status::OK();
}

namespace {

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(const ShapedType& type,
                                                     const Literal& literal) {
  auto data_span = literal.data<CppType>();
  return ::mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data_span.data(), data_span.size()));
}

}  // namespace

Status HloDialectEmitter::HandleConstant(HloInstruction* constant) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorType(constant->shape(), builder_));
  const auto& literal = constant->literal();
  auto element_type = constant->shape().element_type();

  mlir::DenseElementsAttr value;
  switch (element_type) {
    case PrimitiveType::PRED:
      value = CreateDenseAttrFromLiteral<bool>(type, literal);
      break;
    case PrimitiveType::F16:
      value = CreateDenseAttrFromLiteral<float>(type, literal);
      break;
    case PrimitiveType::F32:
      value = CreateDenseAttrFromLiteral<float>(type, literal);
      break;
    case PrimitiveType::F64:
      value = CreateDenseAttrFromLiteral<double>(type, literal);
      break;
    case PrimitiveType::S8:
      value = CreateDenseAttrFromLiteral<int8>(type, literal);
      break;
    case PrimitiveType::S16:
      value = CreateDenseAttrFromLiteral<int16>(type, literal);
      break;
    case PrimitiveType::S32:
      value = CreateDenseAttrFromLiteral<int32>(type, literal);
      break;
    case PrimitiveType::S64:
      value = CreateDenseAttrFromLiteral<int64>(type, literal);
      break;
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }

  auto const_value =
      builder_.create<hlo::ConstOp>(builder_.getUnknownLoc(), type, value);
  instruction_to_values_[constant] = const_value;
  return Status::OK();
}

Status HloDialectEmitter::HandleReduce(HloInstruction* reduce) {
  llvm::SmallVector<Value*, 4> operands;
  for (auto operand : reduce->operands()) {
    operands.push_back(instruction_to_values_.at(operand));
  }
  const unsigned num_inputs = operands.size() / 2;
  TF_ASSIGN_OR_RETURN(const auto return_type,
                      ConvertTensorType(reduce->shape(), builder_));
  const auto& dimensions = reduce->dimensions();
  const auto dimensionsAttr =
      ::mlir::DenseIntElementsAttr::get(
          builder_.getTensorType(dimensions.size(),
                                 builder_.getIntegerType(64)),
          llvm::makeArrayRef(dimensions))
          .cast<::mlir::DenseIntElementsAttr>();
  auto reduceOp = builder_.create<hlo::ReduceOp>(
      builder_.getUnknownLoc(), return_type,
      llvm::makeArrayRef(operands).take_front(num_inputs),
      llvm::makeArrayRef(operands).take_back(num_inputs), dimensionsAttr);
  {
    auto computation = reduce->to_apply();
    auto block = new mlir::Block();
    llvm::SmallVector<Value*, 4> arguments;
    arguments.reserve(computation->num_parameters());
    for (auto parameter : computation->parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(auto param_type,
                          ConvertTensorType(parameter->shape(), builder_));
      arguments.push_back(block->addArgument(param_type));
    }
    reduceOp.body().push_back(block);
    HloDialectEmitter emitter(&reduceOp.body(), arguments);
    TF_ASSIGN_OR_RETURN(auto result, emitter.EmitComputation(*computation));
    OpBuilder body_builder(block);
    body_builder.setInsertionPointToEnd(block);
    body_builder.create<hlo::ReturnOp>(builder_.getUnknownLoc(),
                                       ArrayRef<Value*>{result});
  }
  // TODO(b/137624192) Add support for multiple results.
  instruction_to_values_[reduce] = reduceOp.getResult(0);
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
