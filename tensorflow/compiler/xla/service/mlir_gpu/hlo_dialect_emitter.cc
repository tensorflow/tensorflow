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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/lxla_ops.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::LLVM::LLVMDialect;

Status InsertMlirOp(
    HloOpcode opcode, mlir::OpBuilder func_builder, mlir::Location loc,
    mlir::ArrayRef<mlir::Type> rets, mlir::ArrayRef<mlir::Value*> args,
    mlir::ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>> attrs) {
  switch (opcode) {
    case HloOpcode::kAdd:
      func_builder.create<mlir::LXLA::AddOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMultiply:
      func_builder.create<mlir::LXLA::MulOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSubtract:
      func_builder.create<mlir::LXLA::SubOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kDivide:
      func_builder.create<mlir::LXLA::DivOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kAnd:
      func_builder.create<mlir::LXLA::AndOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMinimum:
      func_builder.create<mlir::LXLA::MinOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMaximum:
      func_builder.create<mlir::LXLA::MaxOp>(loc, rets, args, attrs);
      break;
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Opcode: ", opcode, " is not supported."));
  }
  return Status::OK();
}

StatusOr<mlir::MemRefType> ConvertTensorType(const Shape& shape,
                                             mlir::Builder builder) {
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

StatusOr<mlir::Type> ConvertType(const Shape& shape, mlir::Builder builder) {
  if (shape.IsTuple()) {
    mlir::Type mlir_type;
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype, ConvertType(subtype, builder));
      contents.push_back(mlir_subtype);
    }
    return builder.getTupleType(contents);
  }
  return ConvertTensorType(shape, builder);
}

StatusOr<llvm::SmallVector<mlir::Type, 4>> GetInstructionArgTypes(
    const HloInstruction& instruction, mlir::Builder builder) {
  llvm::SmallVector<mlir::Type, 4> arg_types;
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

HloDialectEmitter::HloDialectEmitter(const HloModule& hlo_module,
                                     const BufferAssignment& assignment,
                                     const se::Platform* platform,
                                     ::mlir::ModuleOp mlir_module)
    : mlir_module_(mlir_module),
      builder_(mlir_module_.getContext()),
      buffer_assignment_(assignment),
      platform_(platform),
      thunk_sequence_(new ThunkSequence()) {
  LLVMDialect* llvmDialect =
      mlir_module.getContext()->getRegisteredDialect<LLVMDialect>();
  pointer_size_ = llvmDialect->getLLVMModule().getDataLayout().getPointerSize();
}

void HloDialectEmitter::AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
  thunk_sequence_->push_back(std::move(thunk));
}

StatusOr<BufferAllocation::Slice> HloDialectEmitter::MaybeGetAllocationSlice(
    const HloInstruction& hlo, const ShapeIndex& index) const {
  return buffer_assignment_.GetUniqueSlice(&hlo, index);
}

int64 HloDialectEmitter::ByteSizeOf(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, pointer_size_);
}

const se::Platform* HloDialectEmitter::platform() const { return platform_; }

Status HloDialectEmitter::EmitComputation(const HloComputation& computation) {
  return computation.root_instruction()->Accept(this);
}

StatusOr<mlir::FuncOp> HloDialectEmitter::CreateFunction(
    const HloInstruction& instr) {
  TF_ASSIGN_OR_RETURN(auto args, GetInstructionArgTypes(instr, builder_));
  auto function_type = builder_.getFunctionType(args, {});
  auto function = mlir::FuncOp::create(builder_.getUnknownLoc(), instr.name(),
                                       function_type);
  mlir_module_.push_back(function);
  function.addEntryBlock();
  instruction_to_mlir_func_[&instr] = function;
  return Status::OK();
}

Status HloDialectEmitter::DefaultAction(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  mlir::OpBuilder func_builder(function.getBody());
  llvm::SmallVector<mlir::Value*, 4> arg_values{function.args_begin(),
                                                function.args_end()};
  llvm::SmallVector<mlir::NamedAttribute, 10> attributes{
      builder_.getNamedAttr("name", builder_.getStringAttr(instr->name()))};
  TF_RETURN_IF_ERROR(
      InsertMlirOp(instr->opcode(), func_builder, builder_.getUnknownLoc(),
                   mlir::ArrayRef<mlir::Type>{}, arg_values, attributes));
  return Status::OK();
}

Status HloDialectEmitter::HandleFusion(HloInstruction* fusion) {
  LOG(FATAL) << "Not implemented yet.";
}

Status HloDialectEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return ThunkEmitter(this).HandleCustomCall(custom_call);
}

Status HloDialectEmitter::FinishVisit(HloInstruction* root) {
  LOG(FATAL) << "Not implemented yet.";
}

}  // namespace gpu
}  // namespace xla
