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

#include <utility>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
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
using ::mlir::DenseIntElementsAttr;
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
    case HloOpcode::kComplex:
      func_builder.create<lhlo::ComplexOp>(loc, rets, args, attrs);
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
    case HloOpcode::kImag:
      func_builder.create<lhlo::ImagOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kLog:
      func_builder.create<lhlo::LogOp>(loc, rets, args, attrs);
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
    case HloOpcode::kReal:
      func_builder.create<lhlo::RealOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kRemainder:
      func_builder.create<lhlo::RemOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kRsqrt:
      func_builder.create<lhlo::RsqrtOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSelect:
      func_builder.create<lhlo::SelectOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSign:
      func_builder.create<lhlo::SignOp>(loc, rets, args, attrs);
      break;
    case HloOpcode::kSqrt:
      func_builder.create<lhlo::SqrtOp>(loc, rets, args, attrs);
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

// Converts HloComputation into a block with HLO dialect ops. The block gets
// memref arguments corresponding to HloComputation arguments and results.
Status SpliceHloComputation(OpBuilder builder, mlir::Location loc,
                            const HloComputation& hlo_computation,
                            xla::mlir_gpu::EmissionContext* emission_context) {
  auto block = builder.getInsertionBlock();
  builder.setInsertionPoint(block->getTerminator());
  llvm::SmallVector<Value, 4> arg_values;
  // First map parameters to memrefs on the operation.
  for (auto param : hlo_computation.parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(
        auto arg_type, ConvertShapeToType<MemRefType>(param->shape(), builder));
    auto block_arg = block->addArgument(arg_type);
    arg_values.push_back(builder.create<::mlir::TensorLoadOp>(loc, block_arg));
  }
  HloDialectEmitter hlo_emitter(emission_context, builder, arg_values);

  TF_ASSIGN_OR_RETURN(auto result,
                      hlo_emitter.EmitComputation(hlo_computation));

  // Now add a block arg and store for the result.
  builder.setInsertionPoint(block->getTerminator());
  TF_ASSIGN_OR_RETURN(
      auto result_type,
      ConvertShapeToType<MemRefType>(
          hlo_computation.root_instruction()->shape(), builder));
  auto block_arg = block->addArgument(result_type);
  builder.create<::mlir::TensorStoreOp>(loc, result, block_arg);

  return Status::OK();
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

Status LhloDialectEmitter::HandleBroadcast(HloInstruction* instr) {
  DenseIntElementsAttr broadcast_dim =
      CreateDenseIntElementsAttrFromVector(instr->dimensions(), builder_);

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  func_builder.create<lhlo::BroadcastInDimOp>(
      getLocation(instr), function.getArgument(0), function.getArgument(1),
      broadcast_dim);
  return Status::OK();
}

Status LhloDialectEmitter::HandleConcatenate(HloInstruction* instr) {
  mlir::IntegerAttr concatenate_dim = builder_.getI64IntegerAttr(
      static_cast<HloConcatenateInstruction*>(instr)->concatenate_dimension());

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  func_builder.create<lhlo::ConcatenateOp>(
      getLocation(instr), function.getArguments().drop_back(),
      function.getArguments().back(), concatenate_dim);
  return Status::OK();
}

Status LhloDialectEmitter::HandleFusion(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  auto fusion_op =
      func_builder.create<lhlo::FusionOp>(getLocation(instr), llvm::None);

  // Load the HLO argument tensors from the corresponding buffers. The last
  // argument is for the result, so no need to load it.
  OpBuilder body_builder(fusion_op.region());
  llvm::SmallVector<Value, 4> arg_values;
  for (int i = 0, e = function.getNumArguments() - 1; i < e; ++i) {
    arg_values.push_back(body_builder.create<::mlir::TensorLoadOp>(
        getLocation(instr), function.getArgument(i)));
  }
  HloDialectEmitter hlo_emitter(emission_context_, body_builder, arg_values);

  TF_ASSIGN_OR_RETURN(
      auto result,
      hlo_emitter.EmitComputation(*instr->fused_instructions_computation()));

  // Insert the write-back from the HLO computation to the result argument
  // buffer.
  body_builder.setInsertionPoint(fusion_op.region().back().getTerminator());
  Value result_memref = function.getArguments().back();
  body_builder.create<::mlir::TensorStoreOp>(getLocation(instr), result,
                                             result_memref);

  return Status::OK();
}

Status LhloDialectEmitter::HandleReduce(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  OpBuilder builder(function.getBody());
  auto loc = getLocation(instr);
  int input_count = instr->operand_count() / 3;
  auto inputs = llvm::makeArrayRef(arg_values).slice(input_count);
  auto init_values =
      llvm::makeArrayRef(arg_values).slice(input_count, input_count);
  auto results =
      llvm::makeArrayRef(arg_values).slice(2 * input_count, input_count);
  auto dimensions_attr =
      CreateDenseIntElementsAttrFromVector(instr->dimensions(), builder_);
  auto reduce_op = builder.create<lhlo::ReduceOp>(loc, inputs, init_values,
                                                  results, dimensions_attr);
  reduce_op.ensureTerminator(reduce_op.body(), builder, getLocation(instr));
  return SpliceHloComputation(OpBuilder{&reduce_op.body()}, loc,
                              *instr->to_apply(), emission_context_);
}

Status LhloDialectEmitter::HandleReduceWindow(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  OpBuilder builder(function.getBody());
  auto loc = getLocation(instr);

  // Collect attribute values.
  llvm::SmallVector<int64, 2> window_dimensions, window_strides, base_dilations,
      window_dilations;
  llvm::SmallVector<int64, 4> padding;
  int64 rank = instr->window().dimensions_size();
  window_dimensions.reserve(rank);
  window_strides.reserve(rank);
  base_dilations.reserve(rank);
  window_dilations.reserve(rank);
  padding.reserve(2 * rank);
  for (const auto& window : instr->window().dimensions()) {
    window_dimensions.push_back(window.size());
    window_strides.push_back(window.stride());
    base_dilations.push_back(window.base_dilation());
    window_dilations.push_back(window.window_dilation());
    padding.push_back(window.padding_low());
    padding.push_back(window.padding_high());
  }

  auto reduce_window_op = builder.create<lhlo::ReduceWindowOp>(
      loc, /*operand=*/arg_values[0], /*init_value=*/arg_values[1],
      /*out=*/arg_values[2],
      CreateDenseIntElementsAttrFromVector(window_dimensions, builder),
      CreateDenseIntElementsAttrFromVector(window_strides, builder),
      CreateDenseIntElementsAttrFromVector(base_dilations, builder),
      CreateDenseIntElementsAttrFromVector(window_dilations, builder),
      CreateDenseIntElementsAttrFromVector(padding, builder, {rank, 2}));
  reduce_window_op.ensureTerminator(reduce_window_op.body(), builder, loc);
  return SpliceHloComputation(OpBuilder{&reduce_window_op.body()}, loc,
                              *instr->to_apply(), emission_context_);
}

Status LhloDialectEmitter::HandleSelectAndScatter(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  OpBuilder builder(function.getBody());
  auto loc = getLocation(instr);

  // Collect attribute values.
  llvm::SmallVector<int64, 2> window_dimensions, window_strides, padding;
  int64 rank = instr->window().dimensions_size();
  window_dimensions.reserve(rank);
  window_strides.reserve(rank);
  padding.reserve(2 * rank);
  for (const auto& window : instr->window().dimensions()) {
    window_dimensions.push_back(window.size());
    window_strides.push_back(window.stride());
    padding.push_back(window.padding_low());
    padding.push_back(window.padding_high());
  }

  auto select_scatter_op = builder.create<lhlo::SelectAndScatterOp>(
      loc, /*operand=*/arg_values[0], /*source=*/arg_values[1],
      /*init_value=*/arg_values[2],
      /*out=*/arg_values[3],
      CreateDenseIntElementsAttrFromVector(window_dimensions, builder),
      CreateDenseIntElementsAttrFromVector(window_strides, builder),
      CreateDenseIntElementsAttrFromVector(padding, builder, {rank, 2}));

  // Convert `select` computation.
  builder.createBlock(&select_scatter_op.select());
  OpBuilder select_builder{&select_scatter_op.select()};
  select_builder.create<lhlo::TerminatorOp>(loc);
  TF_RETURN_IF_ERROR(SpliceHloComputation(select_builder, loc, *instr->select(),
                                          emission_context_));

  // Convert `scatter` computation.
  builder.createBlock(&select_scatter_op.scatter());
  OpBuilder scatter_builder{&select_scatter_op.scatter()};
  scatter_builder.create<lhlo::TerminatorOp>(loc);
  TF_RETURN_IF_ERROR(SpliceHloComputation(
      scatter_builder, loc, *instr->scatter(), emission_context_));

  return Status::OK();
}

Status LhloDialectEmitter::HandleCustomCall(HloInstruction* instr) {
  return ThunkEmitter(this).HandleCustomCall(instr);
}

Status LhloDialectEmitter::HandleParameter(HloInstruction* instr) {
  return Status::OK();
}

Status LhloDialectEmitter::HandleCompare(HloInstruction* instr) {
  auto comparison_direction_attr = builder_.getNamedAttr(
      "comparison_direction",
      builder_.getStringAttr(
          ComparisonDirectionToString(instr->comparison_direction())));

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  llvm::SmallVector<Value, 4> arg_values{function.args_begin(),
                                         function.args_end()};
  func_builder.create<lhlo::CompareOp>(getLocation(instr), llvm::None,
                                       arg_values, comparison_direction_attr);
  return Status::OK();
}

Status LhloDialectEmitter::HandleConstant(HloInstruction* instr) {
  auto shape = instr->shape();
  if (!shape.IsArray() || shape.rank() != 0) {
    return Unimplemented("non-scalar constants are not supported yet");
  }
  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());

  TF_ASSIGN_OR_RETURN(auto value, CreateDenseElementsAttrFromLiteral(
                                      instr->literal(), func_builder));
  func_builder.create<lhlo::ConstOp>(getLocation(instr), value,
                                     *function.args_begin());
  return Status::OK();
}

Status LhloDialectEmitter::HandleIota(HloInstruction* instr) {
  mlir::IntegerAttr iota_dim = builder_.getI64IntegerAttr(
      static_cast<HloIotaInstruction*>(instr)->iota_dimension());

  TF_ASSIGN_OR_RETURN(auto function, CreateFunction(*instr));
  OpBuilder func_builder(function.getBody());
  func_builder.create<lhlo::IotaOp>(getLocation(instr), iota_dim,
                                    function.getArgument(0));
  return Status::OK();
}

Status LhloDialectEmitter::HandleTuple(HloInstruction* instr) {
  // For the root node of the entry computation we can elide writing the tuple
  // buffer. We can always figure out the contents of the tuples from buffer
  // assignment because we insert copies to ensure non-ambiguous output buffers.
  // GpuExecutable never reads the tuple buffer.
  if (instr ==
      instr->parent()->parent()->entry_computation()->root_instruction()) {
    return Status::OK();
  }
  return Unimplemented("handling of typles not yet implemented");
}

Status LhloDialectEmitter::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla
