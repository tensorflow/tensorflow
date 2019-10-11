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

#include "tensorflow/compiler/xla/service/mlir_gpu/lhlo_dialect_emitter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/hlo_dialect_emitter.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::Builder;
using ::mlir::FuncOp;
using ::mlir::Identifier;
using ::mlir::Location;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::OpBuilder;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::LLVM::LLVMDialect;
using ::xla::gpu::Thunk;
using ::xla::gpu::ThunkEmitter;
using ::xla::gpu::ThunkSequence;

// TODO(b/137624192) Use tablegen for this.
Status InsertMlirOp(HloOpcode opcode, OpBuilder func_builder, Location loc,
                    ArrayRef<Type> rets, ArrayRef<Value*> args,
                    ArrayRef<std::pair<Identifier, Attribute>> attrs) {
  switch (opcode) {
    case HloOpcode::kAdd:
      func_builder.create<::mlir::xla_lhlo::AddOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMultiply:
      func_builder.create<::mlir::xla_lhlo::MulOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSubtract:
      func_builder.create<::mlir::xla_lhlo::SubOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kDivide:
      func_builder.create<::mlir::xla_lhlo::DivOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kAnd:
      func_builder.create<::mlir::xla_lhlo::AndOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMinimum:
      func_builder.create<::mlir::xla_lhlo::MinOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMaximum:
      func_builder.create<::mlir::xla_lhlo::MaxOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kExp:
      func_builder.create<::mlir::xla_lhlo::ExpOp>(loc, rets, args, attrs);
      break;
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Opcode ", HloOpcodeString(opcode), " is not supported."));
  }
  return Status::OK();
}

StatusOr<::mlir::MemRefType> ConvertTensorType(const Shape& shape,
                                               Builder builder) {
  llvm::SmallVector<int64_t, 4> array;
  array.reserve(shape.dimensions_size());
  for (const auto dim : shape.dimensions()) {
    array.push_back(dim);
  }
  switch (shape.element_type()) {
    case PrimitiveType::PRED:
      return builder.getMemRefType(array, builder.getI1Type());
    case PrimitiveType::F16:
      return builder.getMemRefType(array, builder.getF16Type());
    case PrimitiveType::F32:
      return builder.getMemRefType(array, builder.getF32Type());
    case PrimitiveType::F64:
      return builder.getMemRefType(array, builder.getF64Type());
    case PrimitiveType::S8:
      return builder.getMemRefType(array, builder.getIntegerType(8));
    case PrimitiveType::S16:
      return builder.getMemRefType(array, builder.getIntegerType(16));
    case PrimitiveType::S32:
      return builder.getMemRefType(array, builder.getIntegerType(32));
    case PrimitiveType::S64:
      return builder.getMemRefType(array, builder.getIntegerType(64));
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "Unsupported type: ", PrimitiveType_Name(shape.element_type())));
  }
}

StatusOr<Type> ConvertType(const Shape& shape, Builder builder) {
  if (shape.IsTuple()) {
    Type mlir_type;
    llvm::SmallVector<Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype, ConvertType(subtype, builder));
      contents.push_back(mlir_subtype);
    }
    return builder.getTupleType(contents);
  }
  return ConvertTensorType(shape, builder);
}

StatusOr<llvm::SmallVector<Type, 4>> GetInstructionArgTypes(
    const HloInstruction& instruction, Builder builder) {
  llvm::SmallVector<Type, 4> arg_types;
  for (auto operand : instruction.operands()) {
    TF_ASSIGN_OR_RETURN(auto operand_type,
                        ConvertType(operand->shape(), builder));
    arg_types.push_back(operand_type);
  }
  TF_ASSIGN_OR_RETURN(auto operand_type,
                      ConvertType(instruction.shape(), builder));
  arg_types.push_back(operand_type);
  return arg_types;
}

}  // namespace

mlir::Location LhloDialectEmitter::getLocation(
    const HloInstruction* instr) const {
  return emission_context_->getLocation(instr);
}

LhloDialectEmitter::LhloDialectEmitter(
    xla::mlir_gpu::EmissionContext* emission_context,
    const BufferAssignment& assignment, const se::Platform* platform,
    ModuleOp mlir_module)
    : emission_context_(emission_context),
      mlir_module_(mlir_module),
      builder_(mlir_module_.getContext()),
      buffer_assignment_(assignment),
      platform_(platform),
      thunk_sequence_(new ThunkSequence()) {
  LLVMDialect* llvmDialect =
      mlir_module.getContext()->getRegisteredDialect<LLVMDialect>();
  pointer_size_ = llvmDialect->getLLVMModule().getDataLayout().getPointerSize();
}

void LhloDialectEmitter::AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
  thunk_sequence_->push_back(std::move(thunk));
}

StatusOr<BufferAllocation::Slice> LhloDialectEmitter::MaybeGetAllocationSlice(
    const HloInstruction& hlo, const ShapeIndex& index) const {
  return buffer_assignment_.GetUniqueSlice(&hlo, index);
}

int64 LhloDialectEmitter::ByteSizeOf(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, pointer_size_);
}

const se::Platform* LhloDialectEmitter::platform() const { return platform_; }

Status LhloDialectEmitter::EmitComputation(const HloComputation& computation) {
  return computation.root_instruction()->Accept(this);
}

StatusOr<FuncOp> LhloDialectEmitter::CreateFunction(
    const HloInstruction& instr) {
  TF_ASSIGN_OR_RETURN(auto args, GetInstructionArgTypes(instr, builder_));
  auto function_type = builder_.getFunctionType(args, {});
  auto function =
      FuncOp::create(getLocation(&instr), instr.name(), function_type);
  mlir_module_.push_back(function);
  function.addEntryBlock();
  OpBuilder op_builder(function.getBody());
  op_builder.create<::mlir::ReturnOp>(getLocation(&instr));
  instruction_to_mlir_func_[&instr] = function;
  return function;
}

Status LhloDialectEmitter::DefaultAction(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  llvm::SmallVector<Value*, 4> arg_values{function.args_begin(),
                                          function.args_end()};
  llvm::SmallVector<NamedAttribute, 10> attributes{
      builder_.getNamedAttr("name", builder_.getStringAttr(instr->name()))};
  TF_RETURN_IF_ERROR(InsertMlirOp(instr->opcode(), func_builder,
                                  getLocation(instr), ArrayRef<Type>{},
                                  arg_values, attributes));
  return Status::OK();
}

Status LhloDialectEmitter::HandleFusion(HloInstruction* fusion) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*fusion));
  OpBuilder func_builder(function.getBody());
  auto attribute =
      builder_.getNamedAttr("name", builder_.getStringAttr(fusion->name()));

  auto fusion_op = func_builder.create<::mlir::xla_lhlo::FusionOp>(
      getLocation(fusion), attribute);

  // Load the HLO argument tensors from the corresponding buffers. The last
  // argument is for the result, so no need to load it.
  OpBuilder body_builder(fusion_op.region());
  llvm::SmallVector<Value*, 4> arg_values;
  for (int i = 0, e = function.getNumArguments() - 1; i < e; ++i) {
    arg_values.push_back(body_builder.create<::mlir::TensorLoadOp>(
        builder_.getUnknownLoc(), function.getArgument(i)));
  }
  HloDialectEmitter hlo_emitter(body_builder, arg_values);

  TF_ASSIGN_OR_RETURN(
      auto result,
      hlo_emitter.EmitComputation(*fusion->fused_instructions_computation()));

  // Insert the write-back from the HLO computation to the result argument
  // buffer.
  body_builder.setInsertionPoint(fusion_op.region().back().getTerminator());
  Value* result_memref = function.getArgument(function.getNumArguments() - 1);
  body_builder.create<::mlir::TensorStoreOp>(builder_.getUnknownLoc(), result,
                                             result_memref);

  return Status::OK();
}

Status LhloDialectEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return ThunkEmitter(this).HandleCustomCall(custom_call);
}

Status LhloDialectEmitter::HandleParameter(HloInstruction* parameter) {
  return Status::OK();
}

Status LhloDialectEmitter::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
