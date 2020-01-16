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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
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
using ::mlir::MemRefType;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::LLVM::LLVMDialect;
using ::xla::gpu::Thunk;
using ::xla::gpu::ThunkEmitter;
using ::xla::gpu::ThunkSequence;

namespace lhlo = ::mlir::xla_lhlo;

// TODO(b/137624192) Use tablegen for this.
Status InsertMlirOp(HloOpcode opcode, OpBuilder func_builder, Location loc,
                    ArrayRef<Type> rets, ArrayRef<Value> args,
                    ArrayRef<std::pair<Identifier, Attribute>> attrs) {
  switch (opcode) {
    case HloOpcode::kAbs:
      func_builder.create<lhlo::AbsOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kAdd:
      func_builder.create<lhlo::AddOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kAnd:
      func_builder.create<lhlo::AndOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kCeil:
      func_builder.create<lhlo::CeilOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kCopy:
      func_builder.create<lhlo::CopyOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kCos:
      func_builder.create<lhlo::CosOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kDivide:
      func_builder.create<lhlo::DivOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kExp:
      func_builder.create<lhlo::ExpOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMaximum:
      func_builder.create<lhlo::MaxOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMinimum:
      func_builder.create<lhlo::MinOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kMultiply:
      func_builder.create<lhlo::MulOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kNegate:
      func_builder.create<lhlo::NegOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kRemainder:
      func_builder.create<lhlo::RemOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSelect:
      func_builder.create<lhlo::SelectOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSign:
      func_builder.create<lhlo::SignOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSubtract:
      func_builder.create<lhlo::SubOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kTanh:
      func_builder.create<lhlo::TanhOp>(loc, rets, args, attrs);
      break;
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "LHLO opcode ", HloOpcodeString(opcode), " is not supported."));
  }
  return Status::OK();
}

StatusOr<llvm::SmallVector<Type, 4>> GetInstructionArgTypes(
    const HloInstruction& instruction, Builder builder) {
  llvm::SmallVector<Type, 4> arg_types;
  for (auto operand : instruction.operands()) {
    TF_ASSIGN_OR_RETURN(auto operand_type, ConvertShapeToType<MemRefType>(
                                               operand->shape(), builder));
    arg_types.push_back(operand_type);
  }
  TF_ASSIGN_OR_RETURN(auto operand_type, ConvertShapeToType<MemRefType>(
                                             instruction.shape(), builder));
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
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  TF_RETURN_IF_ERROR(InsertMlirOp(instr->opcode(), func_builder,
                                  getLocation(instr), ArrayRef<Type>{},
                                  arg_values, llvm::None));
  return Status::OK();
}

Status LhloDialectEmitter::HandleBroadcast(HloInstruction* broadcast) {
  mlir::DenseIntElementsAttr broadcast_dim =
      CreateDenseIntElementsAttrFromVector(broadcast->dimensions(), builder_);

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*broadcast));
  OpBuilder func_builder(function.getBody());
  func_builder.create<lhlo::BroadcastInDimOp>(
      getLocation(broadcast), function.getArgument(0), function.getArgument(1),
      broadcast_dim);
  return Status::OK();
}

Status LhloDialectEmitter::HandleFusion(HloInstruction* fusion) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*fusion));
  OpBuilder func_builder(function.getBody());
  auto fusion_op =
      func_builder.create<lhlo::FusionOp>(getLocation(fusion), llvm::None);

  // Load the HLO argument tensors from the corresponding buffers. The last
  // argument is for the result, so no need to load it.
  OpBuilder body_builder(fusion_op.region());
  llvm::SmallVector<Value, 4> arg_values;
  for (int i = 0, e = function.getNumArguments() - 1; i < e; ++i) {
    arg_values.push_back(body_builder.create<::mlir::TensorLoadOp>(
        getLocation(fusion), function.getArgument(i)));
  }
  HloDialectEmitter hlo_emitter(emission_context_, body_builder, arg_values);

  TF_ASSIGN_OR_RETURN(
      auto result,
      hlo_emitter.EmitComputation(*fusion->fused_instructions_computation()));

  // Insert the write-back from the HLO computation to the result argument
  // buffer.
  body_builder.setInsertionPoint(fusion_op.region().back().getTerminator());
  Value result_memref = function.getArgument(function.getNumArguments() - 1);
  body_builder.create<::mlir::TensorStoreOp>(getLocation(fusion), result,
                                             result_memref);

  return Status::OK();
}

Status LhloDialectEmitter::HandleReduce(HloInstruction* reduce) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*reduce));
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  OpBuilder builder(function.getBody());
  auto loc = getLocation(reduce);
  int input_count = reduce->operand_count() / 3;
  auto inputs = llvm::makeArrayRef(arg_values).slice(input_count);
  auto init_values =
      llvm::makeArrayRef(arg_values).slice(input_count, input_count);
  auto results =
      llvm::makeArrayRef(arg_values).slice(2 * input_count, input_count);
  auto dimensions_attr =
      CreateDenseIntElementsAttrFromVector(reduce->dimensions(), builder_);
  auto reduce_op = builder.create<lhlo::ReduceOp>(loc, inputs, init_values,
                                                  results, dimensions_attr);
  reduce_op.ensureTerminator(reduce_op.body(), builder, getLocation(reduce));

  OpBuilder body_builder(reduce_op.body());
  auto block = body_builder.getInsertionBlock();
  auto to_apply = reduce->to_apply();
  llvm::SmallVector<Value, 4> reduce_arg_values;
  // First map parameters to memrefs on the operation.
  for (auto param : to_apply->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(auto arg_type, ConvertShapeToType<MemRefType>(
                                           param->shape(), builder_));
    auto block_arg = block->addArgument(arg_type);
    reduce_arg_values.push_back(
        body_builder.create<::mlir::TensorLoadOp>(loc, block_arg));
  }
  HloDialectEmitter hlo_emitter(emission_context_, body_builder,
                                reduce_arg_values);

  TF_ASSIGN_OR_RETURN(auto result, hlo_emitter.EmitComputation(*to_apply));

  // Now add a block arg and store for the result.
  body_builder.setInsertionPoint(block->getTerminator());
  TF_ASSIGN_OR_RETURN(auto result_type,
                      ConvertShapeToType<MemRefType>(
                          to_apply->root_instruction()->shape(), builder));
  auto block_arg = block->addArgument(result_type);
  body_builder.create<::mlir::TensorStoreOp>(loc, result, block_arg);

  return Status::OK();
}

Status LhloDialectEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return ThunkEmitter(this).HandleCustomCall(custom_call);
}

Status LhloDialectEmitter::HandleParameter(HloInstruction* parameter) {
  return Status::OK();
}

Status LhloDialectEmitter::HandleCompare(HloInstruction* compare) {
  auto comparison_direction_attr = builder_.getNamedAttr(
      "comparison_direction",
      builder_.getStringAttr(
          ComparisonDirectionToString(compare->comparison_direction())));

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*compare));
  OpBuilder func_builder(function.getBody());
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  func_builder.create<lhlo::CompareOp>(getLocation(compare), llvm::None,
                                       arg_values, comparison_direction_attr);
  return Status::OK();
}

Status LhloDialectEmitter::HandleConstant(HloInstruction* constant) {
  auto shape = constant->shape();
  if (!shape.IsArray() || shape.rank() != 0) {
    return Unimplemented("non-scalar constants are not supported yet");
  }
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*constant));
  OpBuilder func_builder(function.getBody());

  TF_ASSIGN_OR_RETURN(auto value, CreateDenseElementsAttrFromLiteral(
                                      constant->literal(), func_builder));
  func_builder.create<lhlo::ConstOp>(getLocation(constant), value,
                                     *function.args_begin());
  return Status::OK();
}

Status LhloDialectEmitter::HandleIota(HloInstruction* iota) {
  mlir::IntegerAttr iota_dim = builder_.getI64IntegerAttr(
      static_cast<HloIotaInstruction*>(iota)->iota_dimension());

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*iota));
  OpBuilder func_builder(function.getBody());
  func_builder.create<lhlo::IotaOp>(getLocation(iota), iota_dim,
                                    function.getArgument(0));
  return Status::OK();
}

Status LhloDialectEmitter::HandleTuple(HloInstruction* tuple) {
  // For the root node of the entry computation we can elide writing the tuple
  // buffer. We can always figure out the contents of the tuples from buffer
  // assignment because we insert copies to ensure non-ambiguous output buffers.
  // GpuExecutable never reads the tuple buffer.
  if (tuple ==
      tuple->parent()->parent()->entry_computation()->root_instruction()) {
    return Status::OK();
  }
  return Unimplemented("handling of typles not yet implemented");
}

Status LhloDialectEmitter::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
