/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_function.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;
}  // namespace

namespace cpu {

IrEmitter::IrEmitter(
    const HloModule& hlo_module, const BufferAssignment& assignment,
    llvm::Module* llvm_module,
    std::unordered_map<const HloInstruction*, int64> instruction_to_profile_idx,
    std::unordered_map<const HloComputation*, int64> computation_to_profile_idx,
    const TargetMachineFeatures* target_machine_features,
    bool emit_code_for_msan)
    : assignment_(assignment),
      module_(llvm_module),
      arch_type_(llvm::Triple(llvm_module->getTargetTriple()).getArch()),
      b_(llvm_module->getContext()),
      instruction_to_profile_idx_(std::move(instruction_to_profile_idx)),
      computation_to_profile_idx_(std::move(computation_to_profile_idx)),
      alias_analysis_(hlo_module, assignment, &llvm_module->getContext()),
      hlo_module_config_(hlo_module.config()),
      is_top_level_computation_(false),
      target_machine_features_(*target_machine_features),
      emit_code_for_msan_(emit_code_for_msan) {
  b_.setFastMathFlags(llvm_ir::GetCpuFastMathFlags(hlo_module_config_));
  Status s = GatherComputationsByAllocationType(
      &hlo_module, &thread_local_computations_, &global_computations_);
  absl::c_sort(thread_local_computations_);
  absl::c_sort(global_computations_);
  TF_CHECK_OK(s) << "Should have failed buffer assignment.";
}

void IrEmitter::EmitThreadLocalFunctionEpilogue(HloComputation* computation) {
  llvm::Argument* out_parameter = compute_function_->result_arg();
  llvm_ir::IrArray root_value = GetIrArrayFor(computation->root_instruction());
  const Shape& return_shape = computation->root_instruction()->shape();

  if (ShapeUtil::IsScalar(return_shape)) {
    llvm::Value* ret_value =
        Load(root_value.GetBasePointer(), "load_ret_value");
    Store(ret_value,
          BitCast(out_parameter, root_value.GetBasePointer()->getType()));
  } else {
    CHECK(return_shape.IsTuple());

    llvm::Type* tuple_type = llvm_ir::ShapeToIrType(return_shape, module_);
    llvm::Type* tuple_type_lvalue = tuple_type->getPointerTo();
    llvm::Value* tuple_lvalue = BitCast(out_parameter, tuple_type_lvalue);

    for (int i = 0; i < return_shape.tuple_shapes_size(); i++) {
      const Shape& element_shape = return_shape.tuple_shapes(i);
      llvm::Value* destination = llvm_ir::EmitGetTupleElement(
          element_shape,
          /*index=*/i,
          /*alignment=*/MinimumAlignmentForShape(element_shape), tuple_lvalue,
          &b_);

      llvm::Value* source = llvm_ir::EmitGetTupleElement(
          element_shape,
          /*index=*/i,
          /*alignment=*/MinimumAlignmentForShape(element_shape),
          root_value.GetBasePointer(), &b_);

      Store(Load(source), destination);
    }
  }
}

StatusOr<llvm::Function*> IrEmitter::EmitComputation(
    HloComputation* computation, const string& function_name_prefix,
    bool is_top_level_computation,
    absl::Span<HloInstruction* const> instruction_order) {
  string function_name = name_uniquer_.GetUniqueName(function_name_prefix);
  VLOG(2) << "Emitting IR for CPU function [" << function_name_prefix << "]";
  is_top_level_computation_ = is_top_level_computation;
  num_dynamic_loop_bounds_ = 0;
  if (!computation->root_instruction()->outer_dimension_partitions().empty()) {
    num_dynamic_loop_bounds_ =
        computation->root_instruction()->outer_dimension_partitions().size();
  }

  if (computation->root_instruction()->opcode() != HloOpcode::kOutfeed) {
    TF_ASSIGN_OR_RETURN(
        computation_root_allocation_,
        assignment_.GetUniqueTopLevelSlice(computation->root_instruction()));
  }

  for (const HloInstruction* param : computation->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice param_slice,
                        assignment_.GetUniqueTopLevelSlice(param));
    computation_parameter_allocations_[param_slice.allocation()->index()] =
        param->parameter_number();
  }

  InitializeIrFunction(function_name);
  // The rdtscp instruction is x86 specific.  We will fallback to LLVM's generic
  // readcyclecounter if it is unavailable.
  bool use_rdtscp = arch_type_ == llvm::Triple::ArchType::x86 ||
                    arch_type_ == llvm::Triple::ArchType::x86_64;
  profiling_state_ = ProfilingState(use_rdtscp);

  tracing_state_.set_enabled(
      computation->parent()->config().cpu_traceme_enabled());

  TF_RETURN_IF_ERROR(computation->AcceptOrdered(this, instruction_order));
  llvm::Function* ir_function = compute_function_->function();
  InsertOrDie(&emitted_functions_, computation, ir_function);
  // Delete 'compute_function', finalizing 'ir_function' and restoring caller
  // IR insert point.

  // Function epilogue: copying the value over to either the return register,
  // or values pointing from the return register.
  const BufferAllocation* root_allocation =
      computation_root_allocation_.allocation();
  if (root_allocation && root_allocation->is_thread_local()) {
    EmitThreadLocalFunctionEpilogue(computation);
  }

  // Destructor for compute_function_ emits the "ret void" instruction.
  compute_function_.reset();
  computation_root_allocation_ = BufferAllocation::Slice();
  computation_parameter_allocations_.clear();
  return ir_function;
}

void IrEmitter::InitializeIrFunction(const string& function_name) {
  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  llvm::Function::LinkageTypes linkage =
      is_top_level_computation_ ? llvm::GlobalValue::ExternalLinkage
                                : llvm::GlobalValue::InternalLinkage;
  // Create and initialize new IrFunction.
  compute_function_.reset(new IrFunction(function_name, linkage,
                                         hlo_module_config_, module_, &b_,
                                         num_dynamic_loop_bounds_));
}

IrEmitter::~IrEmitter() {}

Status IrEmitter::HandleBitcast(HloInstruction* bitcast) {
  VLOG(2) << "HandleBitcast: " << bitcast->ToString();
  emitted_value_[bitcast] =
      BitCast(GetEmittedValueFor(bitcast->operand(0)),
              IrShapeType(bitcast->shape())->getPointerTo(), IrName(bitcast));
  return Status::OK();
}

llvm::Constant* IrEmitter::EmitGlobalForLiteral(const Literal& literal) {
  llvm::Constant* initializer =
      llvm_ir::ConvertLiteralToIrConstant(literal, module_);
  llvm::GlobalVariable* result_global = new llvm::GlobalVariable(
      /*Module=*/*module_,
      /*Type=*/initializer->getType(),
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/initializer,
      /*Name=*/"");
  result_global->setAlignment(
      llvm::Align(MinimumAlignmentForShape(literal.shape())));
  result_global->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::Global);
  return llvm::ConstantExpr::getBitCast(
      result_global, IrShapeType(literal.shape())->getPointerTo());
}

Status IrEmitter::EmitConstantGlobals() {
  for (const BufferAllocation& allocation : assignment_.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    const Literal& literal = llvm_ir::LiteralForConstantAllocation(allocation);
    llvm::Constant* global_for_const;
    auto it = emitted_literals_.find(&literal);
    if (it != emitted_literals_.end()) {
      global_for_const = it->second;
    } else {
      global_for_const = EmitGlobalForLiteral(literal);
      InsertOrDie(&emitted_literals_, &literal, global_for_const);
    }

    InsertOrDie(&constant_buffer_to_global_, allocation.index(),
                global_for_const);
  }

  return Status::OK();
}

Status IrEmitter::HandleConstant(HloInstruction* constant) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  // IrEmitter::EmitConstantGlobals has already taken care of emitting the body
  // of the constant.
  return EmitTargetAddressForOp(constant);
}

Status IrEmitter::HandleCopy(HloInstruction* copy) {
  if (copy->shape().IsTuple() ||
      (copy->shape().IsArray() &&
       LayoutUtil::Equal(copy->operand(0)->shape().layout(),
                         copy->shape().layout()))) {
    // If the layouts are equal this is just a memcpy. kCopy shallow copies a
    // tuple so just memcpy the top-level buffer for tuples.
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(copy));
    return EmitMemcpy(*(copy->operand(0)), *copy);
  } else if (copy->shape().IsArray()) {
    // Use the elemental emitter for array shapes.
    return DefaultAction(copy);
  }
  return Unimplemented("unsupported operand type %s for copy instruction",
                       PrimitiveType_Name(copy->shape().element_type()));
}

// Calculate the alignment of a buffer allocated for a given primitive type.
int IrEmitter::MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {
  int64 byte_size = ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  DCHECK_GE(byte_size, 0);
  // Largest scalar is a complex128 so we don't need to worry about the
  // int64->int truncation here.
  DCHECK_LE(byte_size, 16);

  // Allocations may be 8-byte aligned if part of a small block.
  return std::min(int64{8}, byte_size);
}

int64 IrEmitter::ByteSizeOf(const Shape& shape) const {
  return llvm_ir::ByteSizeOf(shape, module_->getDataLayout());
}

// Calculate the alignment of a buffer allocated for a given shape.
int IrEmitter::MinimumAlignmentForShape(const Shape& shape) {
  if (ShapeUtil::IsScalar(shape)) {
    return MinimumAlignmentForPrimitiveType(shape.element_type());
  }

  int64 buffer_size = ByteSizeOf(shape);
  DCHECK_GE(buffer_size, 0);
  DCHECK_LE(buffer_size, SIZE_MAX);

  return target_machine_features_.minimum_alignment_for_allocation(buffer_size);
}

void IrEmitter::AttachAlignmentMetadataForLoad(llvm::LoadInst* load,
                                               const Shape& shape) {
  int alignment = MinimumAlignmentForShape(shape);
  if (alignment > 1) {
    llvm_ir::SetAlignmentMetadataForLoad(load, alignment);
  }
}

void IrEmitter::AttachAlignmentMetadataForLoad(llvm::LoadInst* load,
                                               int64 buffer_size) {
  int alignment =
      target_machine_features_.minimum_alignment_for_allocation(buffer_size);
  if (alignment > 1) {
    llvm_ir::SetAlignmentMetadataForLoad(load, alignment);
  }
}

void IrEmitter::AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                                     const Shape& shape) {
  AttachDereferenceableMetadataForLoad(load, ByteSizeOf(shape));
}

void IrEmitter::AttachDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                                     int64 buffer_size) {
  if (buffer_size > 0) {
    llvm_ir::SetDereferenceableMetadataForLoad(load, buffer_size);
  }
}

Status IrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  // A tuple is an array of pointers, one for each operand. Each pointer points
  // to the output buffer of its corresponding operand. A GetTupleElement
  // instruction forwards a pointer to the tuple element buffer at the given
  // index.
  auto operand = get_tuple_element->operand(0);
  const Shape& shape = get_tuple_element->shape();
  emitted_value_[get_tuple_element] = llvm_ir::EmitGetTupleElement(
      shape, get_tuple_element->tuple_index(), MinimumAlignmentForShape(shape),
      GetEmittedValueFor(operand), &b_);
  return Status::OK();
}

Status IrEmitter::HandleSelect(HloInstruction* select) {
  auto pred = select->operand(0);
  TF_RET_CHECK(pred->shape().element_type() == PRED);
  return DefaultAction(select);
}

Status IrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {
  auto pred = tuple_select->operand(0);
  auto on_true = tuple_select->operand(1);
  auto on_false = tuple_select->operand(2);
  TF_RET_CHECK(pred->shape().element_type() == PRED);
  TF_RET_CHECK(ShapeUtil::IsScalar(pred->shape()));
  TF_RET_CHECK(tuple_select->shape().IsTuple());
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(tuple_select));
  llvm_ir::EmitTupleSelect(GetIrArrayFor(tuple_select), GetIrArrayFor(pred),
                           GetEmittedValueFor(on_true),
                           GetEmittedValueFor(on_false), &b_);
  return Status::OK();
}

Status IrEmitter::HandleInfeed(HloInstruction* instruction) {
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(instruction);
  VLOG(2) << "HandleInfeed: " << infeed->ToString();

  // The infeed operation produces a two-element tuple containing data and a
  // token value. HloInfeedInstruction::infeed_shape gives us the data shape.
  const Shape& data_shape = infeed->infeed_shape();
  DCHECK(ShapeUtil::Equal(data_shape,
                          ShapeUtil::GetTupleElementShape(infeed->shape(), 0)));
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(infeed));

  // Write the tuple index table.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data_slice,
                      assignment_.GetUniqueSlice(infeed, {0}));
  llvm::Value* data_address = EmitBufferPointer(data_slice, data_shape);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice token_slice,
                      assignment_.GetUniqueSlice(infeed, {1}));
  llvm::Value* token_address = EmitBufferPointer(
      token_slice, ShapeUtil::GetTupleElementShape(infeed->shape(), 1));
  llvm_ir::EmitTuple(GetIrArrayFor(infeed), {data_address, token_address}, &b_);

  if (data_shape.IsTuple()) {
    TF_RET_CHECK(!ShapeUtil::IsNestedTuple(data_shape));

    // For a tuple, we first copy each of the internal elements to
    // their corresponding target locations. We then construct the
    // tuple outer buffer containing pointers to the internal
    // elements.
    std::vector<llvm::Value*> tuple_element_addresses;
    for (int64 i = 0; i < data_shape.tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice buffer,
                          assignment_.GetUniqueSlice(infeed, {0, i}));

      const Shape& tuple_element_shape =
          ShapeUtil::GetTupleElementShape(data_shape, i);

      // Only the outer tuple buffer's target address is obtained from
      // GetEmittedValueFor, to handle the case when Infeed is the root
      // instruction. Target addresses for internal elements can be obtained
      // from EmitBufferPointer.
      llvm::Value* tuple_element_address =
          EmitBufferPointer(buffer, tuple_element_shape);

      TF_RETURN_IF_ERROR(EmitXfeedTransfer(
          XfeedKind::kInfeed, tuple_element_shape, tuple_element_address));

      tuple_element_addresses.push_back(tuple_element_address);
    }

    llvm_ir::EmitTuple(llvm_ir::IrArray(data_address, data_shape),
                       tuple_element_addresses, &b_);
  } else {
    TF_RETURN_IF_ERROR(
        EmitXfeedTransfer(XfeedKind::kInfeed, data_shape, data_address));
  }

  return Status::OK();
}

Status IrEmitter::EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                                    llvm::Value* program_buffer_address) {
  int64 length = ByteSizeOf(shape);
  if (length <= 0 || length > std::numeric_limits<int32>::max()) {
    return InvalidArgument(
        "xfeed (infeed or outfeed) buffer length %d is outside the valid "
        "size range",
        length);
  }
  int32 length_32 = static_cast<int32>(length);

  int32 shape_length;
  TF_ASSIGN_OR_RETURN(
      llvm::Value * shape_ptr,
      llvm_ir::EncodeSelfDescribingShapeConstant(shape, &shape_length, &b_));

  llvm::Type* int32_type = b_.getInt32Ty();
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::FunctionType* acquire_type = llvm::FunctionType::get(
      i8_ptr_type,
      {/*run_options*/ i8_ptr_type, /*buffer_length*/ int32_type,
       /*shape_ptr*/ i8_ptr_type, /*shape_length*/ int32_type},
      /*isVarArg=*/false);

  llvm::Function* acquire_func;
  if (kind == XfeedKind::kInfeed) {
    acquire_func = llvm::dyn_cast<llvm::Function>(
        module_
            ->getOrInsertFunction(
                runtime::kAcquireInfeedBufferForDequeueSymbolName, acquire_type)
            .getCallee());
  } else {
    acquire_func = llvm::dyn_cast<llvm::Function>(
        module_
            ->getOrInsertFunction(
                runtime::kAcquireOutfeedBufferForPopulationSymbolName,
                acquire_type)
            .getCallee());
  }
  acquire_func->setCallingConv(llvm::CallingConv::C);

  llvm::FunctionType* release_type = llvm::FunctionType::get(
      b_.getVoidTy(),
      {/*run_options*/ i8_ptr_type, /*buffer_length*/ int32_type,
       /*buffer_ptr*/ i8_ptr_type, /*shape_ptr*/ i8_ptr_type,
       /*shape_length*/ int32_type},
      /*isVarArg=*/false);

  llvm::Function* release_func;
  if (kind == XfeedKind::kInfeed) {
    release_func = llvm::dyn_cast<llvm::Function>(
        module_
            ->getOrInsertFunction(
                runtime::kReleaseInfeedBufferAfterDequeueSymbolName,
                release_type)
            .getCallee());
  } else {
    release_func = llvm::dyn_cast<llvm::Function>(
        module_
            ->getOrInsertFunction(
                runtime::kReleaseOutfeedBufferAfterPopulationSymbolName,
                release_type)
            .getCallee());
  }
  release_func->setCallingConv(llvm::CallingConv::C);

  // Implementation note: this call informs the runtime that it wants a buffer
  // of size exactly 'length_32', and the runtime is responsible for
  // check-failing the process if there is a mismatch, versus passing us back a
  // buffer that we might overrun.
  llvm::Value* acquired_pointer = Call(
      acquire_func, {GetExecutableRunOptionsArgument(), b_.getInt32(length_32),
                     shape_ptr, b_.getInt32(shape_length)});

  if (kind == XfeedKind::kInfeed) {
    // Copy to the program buffer address from the acquired buffer.
    MemCpy(program_buffer_address, /*DstAlign=*/llvm::Align(1),
           acquired_pointer,
           /*SrcAlign=*/llvm::Align(1), length_32);
  } else {
    // Outfeed -- copy from the in-program address to the acquired buffer.
    MemCpy(acquired_pointer, /*DstAlign=*/llvm::Align(1),
           program_buffer_address,
           /*SrcAlign=*/llvm::Align(1), length_32);
  }

  Call(release_func, {GetExecutableRunOptionsArgument(), b_.getInt32(length_32),
                      acquired_pointer, shape_ptr, b_.getInt32(shape_length)});

  return Status::OK();
}

Status IrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  // Outfeed produces no useful result, but it does return a token[] that can be
  // threaded through to other side effecting operations to ensure ordering.  In
  // the IR emitter we treat this token as a normal u8[] and thus need to insert
  // an entry for it in emitted_value_.
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(outfeed));

  HloInstruction* operand = outfeed->operands()[0];
  const Shape& operand_shape = operand->shape();

  llvm::Value* value = GetEmittedValueFor(operand);
  if (!operand_shape.IsTuple()) {
    return EmitXfeedTransfer(XfeedKind::kOutfeed, operand_shape, value);
  }

  TF_RET_CHECK(!ShapeUtil::IsNestedTuple(operand_shape));

  for (int64 i = 0; i < operand_shape.tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(operand_shape, i);
    llvm::Value* tuple_element = llvm_ir::EmitGetTupleElement(
        tuple_element_shape, i, MinimumAlignmentForShape(tuple_element_shape),
        value, &b_);
    TF_RETURN_IF_ERROR(EmitXfeedTransfer(XfeedKind::kOutfeed,
                                         tuple_element_shape, tuple_element));
  }

  return Status::OK();
}

Status IrEmitter::HandleSort(HloInstruction* hlo) {
  const HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(sort));
  Shape keys_shape = sort->keys()->shape();
  PrimitiveType keys_type = keys_shape.element_type();
  if (!primitive_util::IsArrayType(keys_type)) {
    return Unimplemented("Element type %s not supported in the Sort op on CPU.",
                         PrimitiveType_Name(keys_type));
  }
  std::vector<llvm::Value*> destination_addresses(sort->operand_count());
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    ShapeIndex shape_index =
        sort->values_count() > 0 ? ShapeIndex({i}) : ShapeIndex({});
    const HloInstruction* operand = sort->operand(i);
    // We assume that the layout of all involved operands and outputs is the
    // same.
    TF_RET_CHECK(
        LayoutUtil::LayoutsInShapesEqual(keys_shape, operand->shape()));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index)));

    // The sort is implemented in-place, therefore we first copy the operand
    // buffer to the output buffer if they are not the same.
    auto destination_buffer = GetAllocationSlice(*sort, shape_index);
    destination_addresses[i] =
        EmitBufferPointer(destination_buffer, operand->shape());
    auto source_address = GetAllocationSlice(*operand);
    if (destination_buffer != source_address) {
      int64 primitive_type_size =
          ShapeUtil::ByteSizeOfPrimitiveType(operand->shape().element_type());
      auto source_buffer = GetEmittedValueFor(operand);
      int64 size = ByteSizeOf(operand->shape());
      MemCpy(destination_addresses[i],
             /*DstAlign=*/llvm::Align(primitive_type_size), source_buffer,
             /*SrcAlign=*/llvm::Align(primitive_type_size), size);
    }
  }

  // Normalize the shape and the dimension to sort.
  Shape normalized_keys_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(keys_shape);
  int64 physical_dimension_to_sort = LayoutUtil::MakeLogicalToPhysical(
      keys_shape.layout())[sort->sort_dimension()];

  int64 sort_dimension_elements =
      normalized_keys_shape.dimensions(physical_dimension_to_sort);
  int64 higher_dimensions = 1;
  for (int64 i = 0; i < physical_dimension_to_sort; ++i) {
    higher_dimensions *= normalized_keys_shape.dimensions(i);
  }
  int64 lower_dimensions = 1;
  for (int64 i = normalized_keys_shape.rank() - 1;
       i > physical_dimension_to_sort; --i) {
    lower_dimensions *= normalized_keys_shape.dimensions(i);
  }

  auto less_than_function = FindOrDie(emitted_functions_, sort->to_apply());
  CHECK(absl::c_binary_search(thread_local_computations_, sort->to_apply()));
  llvm::FunctionType* key_value_sort_type = llvm::FunctionType::get(
      b_.getVoidTy(),
      {b_.getInt64Ty(), b_.getInt64Ty(), b_.getInt64Ty(),
       b_.getInt8PtrTy()->getPointerTo(), b_.getInt32Ty(),
       b_.getInt32Ty()->getPointerTo(), b_.getInt1Ty(), b_.getInt8PtrTy(),
       b_.getInt64Ty()->getPointerTo(), less_than_function->getType()},
      /*isVarArg=*/false);
  auto* key_value_sort_func = llvm::dyn_cast<llvm::Function>(
      module_
          ->getOrInsertFunction(runtime::kKeyValueSortSymbolName,
                                key_value_sort_type)
          .getCallee());
  key_value_sort_func->setCallingConv(llvm::CallingConv::C);
  key_value_sort_func->setDoesNotThrow();
  llvm::Value* values = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      b_.getInt8PtrTy(), b_.getInt32(sort->operand_count()), "cc_values_alloca",
      &b_);
  llvm::Value* sizes = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      b_.getInt32Ty(), b_.getInt32(sort->operand_count()), "cc_sizes_alloca",
      &b_);
  for (int64 i = 0; i < sort->operand_count(); ++i) {
    llvm::Value* value_as_i8ptr =
        PointerCast(destination_addresses[i], b_.getInt8PtrTy());
    llvm::Value* slot_in_values_alloca =
        ConstInBoundsGEP1_32(b_.getInt8PtrTy(), values, i);
    Store(value_as_i8ptr, slot_in_values_alloca);
    llvm::Value* slot_in_sizes_alloca =
        ConstInBoundsGEP1_32(b_.getInt32Ty(), sizes, i);
    llvm::Value* size = b_.getInt32(ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type()));
    Store(size, slot_in_sizes_alloca);
  }

  Call(key_value_sort_func,
       {b_.getInt64(higher_dimensions), b_.getInt64(sort_dimension_elements),
        b_.getInt64(lower_dimensions), values,
        b_.getInt32(sort->operand_count()), sizes,
        b_.getInt1(sort->is_stable()), GetExecutableRunOptionsArgument(),
        GetProfileCountersArgument(), less_than_function});

  if (sort->values_count() > 0) {
    llvm_ir::EmitTuple(GetIrArrayFor(sort), destination_addresses, &b_);
  }
  return Status::OK();
}

Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(tuple));
  std::vector<llvm::Value*> base_ptrs;
  for (auto operand : tuple->operands()) {
    base_ptrs.push_back(GetEmittedValueFor(operand));
  }
  llvm_ir::EmitTuple(GetIrArrayFor(tuple), base_ptrs, &b_);
  return Status::OK();
}

Status IrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {
  // Pseudo code for reduce window:
  //
  //   for (coordinates O in the output)
  //     value = init_value;
  //     for (coordinates W in the window)
  //       for each index i:
  //         input coordinates I_i = O_i * stride_i + W_i - pad_low_i
  //       if I within bounds of input:
  //         value = function(value, input(I));
  //     output(O) = value;
  //
  // This is completely un-optimized and just here to have something
  // that works.
  return DefaultAction(reduce_window);
}

Status IrEmitter::HandleSelectAndScatter(HloInstruction* select_and_scatter) {
  CHECK_EQ(select_and_scatter->operand_count(), 3);
  const auto operand = select_and_scatter->operand(0);
  const auto source = select_and_scatter->operand(1);
  const auto init_value = select_and_scatter->operand(2);
  const Window& window = select_and_scatter->window();
  PrimitiveType operand_element_type = operand->shape().element_type();
  const int64 rank = operand->shape().rank();
  CHECK_EQ(rank, source->shape().rank());
  CHECK_EQ(rank, window.dimensions_size());

  // TODO(b/31410564): Implement dilation for select-and-scatter.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for SelectAndScatter is not implemented on CPU. ");
  }

  // Pseudo code for select-and-scatter:
  //
  // initialized_flag is initially off for every window, and is turned on after
  // the first iteration is completed and the first operand value is selected.
  //
  // output(*) = init_value
  // for (coordinates S in the source) {
  //   initialized_flag = false
  //   for (coordinates W in the window) {
  //     I = S * stride + W - pad_low
  //     if I within bounds of operand:
  //       if !initialized_flag or select(selected_value, operand(I)) == false:
  //         selected_value = operand(I)
  //         selected_index = I
  //         initialized_flag = true
  //   }
  //   output(selected_index) = scatter(output(selected_index), source(S))
  // }
  //

  // Initialize the output array with the given init_value.
  TF_RETURN_IF_ERROR(EmitTargetElementLoop(
      select_and_scatter, /*desc=*/IrName(select_and_scatter, "init"),
      [this, init_value](const llvm_ir::IrArray::Index& target_index) {
        llvm::Value* init_value_addr = GetEmittedValueFor(init_value);
        return Load(init_value_addr);
      }));

  // Create a loop to iterate over the source array to scatter to the output.
  llvm_ir::ForLoopNest source_loops(IrName(select_and_scatter), &b_);
  const llvm_ir::IrArray::Index source_index =
      source_loops.AddLoopsForShape(source->shape(), "source");
  SetToFirstInsertPoint(source_loops.GetInnerLoopBodyBasicBlock(), &b_);

  // Allocate space to keep the currently selected value, its index, and
  // the boolean initialized_flag, which is initially set to false.
  llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(operand_element_type, module_),
      "selected_value_address", &b_,
      MinimumAlignmentForPrimitiveType(operand_element_type));
  llvm::Value* selected_index_address =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          b_.getInt64Ty(), b_.getInt32(rank), "selected_index_address", &b_);
  llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
      b_.getInt1Ty(), "initialized_flag_address", &b_);
  Store(b_.getInt1(false), initialized_flag_address);

  // Create the inner loop to iterate over the window.
  llvm_ir::ForLoopNest window_loops(IrName(select_and_scatter, "window"), &b_);
  std::vector<int64> window_size;
  for (const auto& dim : window.dimensions()) {
    window_size.push_back(dim.size());
  }
  const llvm_ir::IrArray::Index window_index = window_loops.AddLoopsForShape(
      ShapeUtil::MakeShape(operand_element_type, window_size), "window");
  SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(), &b_);

  // Compute the operand index to visit and evaluate the condition whether the
  // operand index is within the bounds. The unsigned comparison includes
  // checking whether the operand index >= 0.
  std::vector<llvm::Value*> operand_multi_index(source_index.size());
  llvm::Value* in_bounds_condition = b_.getTrue();
  for (int64 i = 0; i < rank; ++i) {
    llvm::Value* strided_index =
        NSWMul(source_index[i], b_.getInt64(window.dimensions(i).stride()));
    operand_multi_index[i] =
        NSWSub(NSWAdd(strided_index, window_index[i]),
               b_.getInt64(window.dimensions(i).padding_low()));
    llvm::Value* index_condition =
        ICmpULT(operand_multi_index[i],
                b_.getInt64(ShapeUtil::GetDimension(operand->shape(), i)));
    in_bounds_condition = And(in_bounds_condition, index_condition);
  }
  CHECK(in_bounds_condition != nullptr);

  // Only need to do something if the operand index is within the bounds. First
  // check if the initialized_flag is set.
  llvm_ir::LlvmIfData if_in_bounds =
      llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &b_);
  SetToFirstInsertPoint(if_in_bounds.true_block, &b_);
  llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
      Load(initialized_flag_address), "initialized", &b_);

  // If the initialized_flag is false, initialize the selected value and index
  // with the currently visiting operand.
  SetToFirstInsertPoint(if_initialized.false_block, &b_);
  const auto save_operand_index =
      [&](const llvm_ir::IrArray::Index& operand_index) {
        for (int64 i = 0; i < rank; ++i) {
          llvm::Value* selected_index_address_slot =
              InBoundsGEP(selected_index_address, {b_.getInt32(i)});
          Store(operand_index[i], selected_index_address_slot);
        }
      };
  llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
  llvm_ir::IrArray::Index operand_index(
      operand_multi_index, operand_array.GetShape(), b_.getInt64Ty());
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, &b_);
  Store(operand_data, selected_value_address);
  save_operand_index(operand_index);
  Store(b_.getInt1(true), initialized_flag_address);

  // If the initialized_flag is true, call the `select` function to potentially
  // update the selected value and index with the currently visiting operand.
  SetToFirstInsertPoint(if_initialized.true_block, &b_);
  llvm::Value* operand_address =
      operand_array.EmitArrayElementAddress(operand_index, &b_);
  llvm::Value* operand_element = Load(operand_address);
  llvm::Value* result = EmitScalarReturningThreadLocalCall(
      *select_and_scatter->select(),
      {Load(selected_value_address), operand_element}, "select_function");

  // If the 'select' function returns false, update the selected value and the
  // index to the currently visiting operand.
  llvm::Value* cond = ICmpNE(
      result,
      llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0),
      "boolean_predicate");
  llvm_ir::LlvmIfData if_select_lhs =
      llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &b_);
  SetToFirstInsertPoint(if_select_lhs.false_block, &b_);
  Store(Load(operand_address), selected_value_address);
  save_operand_index(operand_index);

  // After iterating over the window elements, scatter the source element to
  // the selected index of the output. The value we store at the output
  // location is computed by calling the `scatter` function with the source
  // value and the current output value.
  SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(), &b_);
  std::vector<llvm::Value*> selected_multi_index;
  for (int64 i = 0; i < rank; ++i) {
    llvm::Value* selected_index_address_slot =
        InBoundsGEP(selected_index_address, {b_.getInt32(i)});
    selected_multi_index.push_back(Load(selected_index_address_slot));
  }
  llvm_ir::IrArray source_array(GetIrArrayFor(source));
  llvm::Value* source_value =
      source_array.EmitReadArrayElement(source_index, &b_);
  llvm_ir::IrArray output_array(GetIrArrayFor(select_and_scatter));
  llvm_ir::IrArray::Index selected_index(
      selected_multi_index, output_array.GetShape(), source_index.GetType());
  llvm::Value* output_value =
      output_array.EmitReadArrayElement(selected_index, &b_);
  llvm::Value* scatter_value = EmitScalarReturningThreadLocalCall(
      *select_and_scatter->scatter(), {output_value, source_value},
      "scatter_function");
  output_array.EmitWriteArrayElement(selected_index, scatter_value, &b_);

  SetToFirstInsertPoint(source_loops.GetOuterLoopExitBasicBlock(), &b_);
  return Status::OK();
}

Status IrEmitter::HandleDot(HloInstruction* dot) {
  auto lhs = dot->operand(0);
  auto rhs = dot->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*dot, /*operands=*/{lhs, rhs},
      /*supported_types=*/{S32, F16, F32, F64, C64, C128}));
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();

  if (dnums.lhs_contracting_dimensions_size() != 1) {
    // This is disallowed by ShapeInference today.
    return Unimplemented(
        "Dot with multiple contracting dimensions not implemented.");
  }

  llvm_ir::IrArray lhs_array(GetIrArrayFor(lhs));
  llvm_ir::IrArray rhs_array(GetIrArrayFor(rhs));

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(dot));
  llvm_ir::IrArray target_array = GetIrArrayFor(dot);

  VLOG(2) << "HandleDot: ";
  VLOG(2) << "  lhs operand: "
          << llvm_ir::DumpToString(*lhs_array.GetBasePointer());
  VLOG(2) << "  rhs operand: "
          << llvm_ir::DumpToString(*rhs_array.GetBasePointer());
  VLOG(2) << "  target: "
          << llvm_ir::DumpToString(*target_array.GetBasePointer());

  // Dot operation is complicated so we delegate to a helper class.
  return EmitDotOperation(*dot, target_array, lhs_array, rhs_array,
                          /*addend_array=*/nullptr,
                          GetExecutableRunOptionsArgument(), &b_,
                          hlo_module_config_, target_machine_features_);
}

StatusOr<llvm::Value*> IrEmitter::EmitElementalConvolution(
    const HloConvolutionInstruction* convolution,
    const llvm_ir::ElementGenerator& input_generator,
    const llvm_ir::ElementGenerator& kernel_generator,
    const llvm_ir::IrArray::Index& index) {
  const HloInstruction* lhs = convolution->operand(0);
  const HloInstruction* rhs = convolution->operand(1);
  const Window& window = convolution->window();

  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();
  int num_spatial_dims = dnums.output_spatial_dimensions_size();
  std::vector<llvm::Value*> output_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    output_spatial[i] = index[dnums.output_spatial_dimensions(i)];
  }
  llvm::Value* output_feature = index[dnums.output_feature_dimension()];
  llvm::Value* batch = index[dnums.output_batch_dimension()];

  // We will accumulate the products into this sum to calculate the output entry
  // at the given index.
  PrimitiveType lhs_element_type = lhs->shape().element_type();
  llvm::Type* lhs_llvm_type =
      llvm_ir::PrimitiveTypeToIrType(lhs_element_type, module_);
  // Upcast the accumulator to F32 from F16 for increased precision.
  llvm::Type* accumulator_type =
      lhs_element_type == F16 ? b_.getFloatTy() : lhs_llvm_type;
  llvm::Value* sum_address = llvm_ir::EmitAllocaAtFunctionEntry(
      accumulator_type, "convolution_sum_address", &b_,
      MinimumAlignmentForPrimitiveType(lhs_element_type));
  llvm::Value* constant_zero = llvm::Constant::getNullValue(accumulator_type);
  Store(constant_zero, sum_address);

  llvm_ir::ForLoopNest loops(IrName(convolution, "inner"), &b_);
  std::vector<llvm::Value*> kernel_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_spatial[i] =
        loops
            .AddLoop(
                0, rhs->shape().dimensions(dnums.kernel_spatial_dimensions(i)),
                absl::StrCat("k", i))
            ->GetIndVarValue();
  }
  llvm::Value* input_feature =
      loops
          .AddLoop(0, lhs->shape().dimensions(dnums.input_feature_dimension()),
                   "iz")
          ->GetIndVarValue();

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b_);

  // Calculate the spatial index in the input array, taking striding, dilation
  // and padding into account. An index in the padding will be out of the bounds
  // of the array.
  const auto calculate_input_index = [this](llvm::Value* output_index,
                                            llvm::Value* kernel_index,
                                            const WindowDimension& window_dim) {
    llvm::Value* strided_index =
        NSWMul(output_index, b_.getInt64(window_dim.stride()));
    llvm::Value* dilated_kernel_index =
        NSWMul(kernel_index, b_.getInt64(window_dim.window_dilation()));
    return NSWSub(NSWAdd(strided_index, dilated_kernel_index),
                  b_.getInt64(window_dim.padding_low()));
  };
  std::vector<llvm::Value*> input_spatial(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial[i] = calculate_input_index(
        output_spatial[i], kernel_spatial[i], window.dimensions(i));
  }

  // We need to check if 0 <= input dim < bound, as otherwise we are in the
  // padding so that we can skip the computation. That is equivalent to input
  // dim < bound as an *unsigned* comparison, since a negative value will wrap
  // to a large positive value. The input dim is dilated, so we need to dilate
  // the bound as well to match.

  // Also need to check that the input coordinates are not in one of the
  // holes created by base dilation.
  const auto not_in_hole = [&](llvm::Value* input_index, int64 base_dilation) {
    llvm::Value* remainder = SRem(input_index, b_.getInt64(base_dilation));
    return ICmpEQ(remainder, b_.getInt64(0));
  };

  llvm::Value* in_bounds_condition = b_.getInt1(true);
  for (int i = 0; i < num_spatial_dims; ++i) {
    llvm::ConstantInt* input_bound = b_.getInt64(window_util::DilatedBound(
        lhs->shape().dimensions(dnums.input_spatial_dimensions(i)),
        window.dimensions(i).base_dilation()));
    llvm::Value* dim_in_bound = ICmpULT(input_spatial[i], input_bound);
    llvm::Value* dim_not_in_hole =
        not_in_hole(input_spatial[i], window.dimensions(i).base_dilation());
    llvm::Value* dim_ok = And(dim_in_bound, dim_not_in_hole);
    in_bounds_condition = And(in_bounds_condition, dim_ok);
  }

  // Now we need to map the dilated base coordinates back to the actual
  // data indices on the lhs.
  const auto undilate = [&](llvm::Value* input_index, int64 base_dilation) {
    return SDiv(input_index, b_.getInt64(base_dilation));
  };
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_spatial[i] =
        undilate(input_spatial[i], window.dimensions(i).base_dilation());
  }

  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &b_);
  SetToFirstInsertPoint(if_data.true_block, &b_);

  // We are not in the padding, so carry out the computation.
  int num_dims = num_spatial_dims + 2;
  std::vector<llvm::Value*> input_multi_index(num_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    input_multi_index[dnums.input_spatial_dimensions(i)] = input_spatial[i];
  }
  input_multi_index[dnums.input_feature_dimension()] = input_feature;
  input_multi_index[dnums.input_batch_dimension()] = batch;

  std::vector<llvm::Value*> kernel_multi_index(num_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    kernel_multi_index[dnums.kernel_spatial_dimensions(i)] =
        window.dimensions(i).window_reversal()
            ? NSWSub(b_.getInt64(window.dimensions(i).size() - 1),
                     kernel_spatial[i])
            : kernel_spatial[i];
  }

  kernel_multi_index[dnums.kernel_input_feature_dimension()] = input_feature;
  kernel_multi_index[dnums.kernel_output_feature_dimension()] = output_feature;

  llvm_ir::IrArray::Index input_index(input_multi_index, lhs->shape(),
                                      b_.getInt64Ty());
  TF_ASSIGN_OR_RETURN(llvm::Value* const input_value,
                      input_generator(input_index));
  llvm_ir::IrArray::Index kernel_index(kernel_multi_index, rhs->shape(),
                                       b_.getInt64Ty());
  TF_ASSIGN_OR_RETURN(llvm::Value* const kernel_value,
                      kernel_generator(kernel_index));
  llvm::Value* product = FMul(input_value, kernel_value);
  llvm::Value* sum = FAdd(Load(sum_address), FPCast(product, accumulator_type));
  Store(sum, sum_address);

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b_);
  return FPCast(Load(sum_address), lhs_llvm_type);
}

Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*convolution, /*operands=*/{lhs, rhs},
      /*supported_types=*/{F16, F32, F64, C64, C128}));

  // TODO(tonywy): Add PotentiallyImplementedAsMKLConvolution to support
  // different data layouts.
  if (PotentiallyImplementedAsEigenConvolution(*convolution,
                                               target_machine_features_)) {
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();
    const Shape& convolution_shape = convolution->shape();
    // The input, kernel and output agree with respect to layout.
    if (LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(convolution_shape.layout())) {
      // We lower 1D convolutions into calls to the same Eigen function as 2D
      // convolutions, except that we pretend that the 1D convolution is really
      // a 2D convolution with the missing dimension set to 1.  We also adjust
      // the padding, dilation parameters as needed.
      bool one_dim_convolution = lhs_shape.dimensions_size() == 3;
      llvm::Value* lhs_address = GetEmittedValueFor(lhs);
      llvm::Value* rhs_address = GetEmittedValueFor(rhs);
      TF_RETURN_IF_ERROR(EmitTargetAddressForOp(convolution));

      const ConvolutionDimensionNumbers& dnums =
          convolution->convolution_dimension_numbers();

      // Input tensor.
      const Shape& input_shape = convolution->operand(0)->shape();
      int64 input_batch = input_shape.dimensions(dnums.input_batch_dimension());
      int64 input_rows =
          input_shape.dimensions(dnums.input_spatial_dimensions(0));
      int64 input_cols =
          one_dim_convolution
              ? 1
              : input_shape.dimensions(dnums.input_spatial_dimensions(1));
      int64 input_channels =
          input_shape.dimensions(dnums.input_feature_dimension());

      // Kernel tensor.
      const Shape& kernel_shape = convolution->operand(1)->shape();
      int64 kernel_rows =
          kernel_shape.dimensions(dnums.kernel_spatial_dimensions(0));
      int64 kernel_cols =
          one_dim_convolution
              ? 1
              : kernel_shape.dimensions(dnums.kernel_spatial_dimensions(1));
      int64 kernel_channels =
          kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
      int64 kernel_filters =
          kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

      // Output tensor.
      const Shape& convolution_shape = convolution->shape();
      int64 output_rows =
          convolution_shape.dimensions(dnums.output_spatial_dimensions(0));
      int64 output_cols = one_dim_convolution
                              ? 1
                              : convolution_shape.dimensions(
                                    dnums.output_spatial_dimensions(1));

      // Extract the window stride for the convolution.
      const Window& window = convolution->window();
      int64 row_stride = window.dimensions(0).stride();
      int64 col_stride =
          one_dim_convolution ? 1 : window.dimensions(1).stride();

      int64 padding_top = window.dimensions(0).padding_low();
      int64 padding_bottom = window.dimensions(0).padding_high();
      int64 padding_left =
          one_dim_convolution ? 0 : window.dimensions(1).padding_low();
      int64 padding_right =
          one_dim_convolution ? 0 : window.dimensions(1).padding_high();

      int64 lhs_row_dilation = window.dimensions(0).base_dilation();
      int64 lhs_col_dilation =
          one_dim_convolution ? 1 : window.dimensions(1).base_dilation();
      int64 rhs_row_dilation = window.dimensions(0).window_dilation();
      int64 rhs_col_dilation =
          one_dim_convolution ? 1 : window.dimensions(1).window_dilation();

      PrimitiveType primitive_type = lhs->shape().element_type();
      llvm::Type* ir_ptr_type = primitive_type == F16
                                    ? b_.getHalfTy()->getPointerTo()
                                    : b_.getFloatTy()->getPointerTo();
      llvm::Type* int64_type = b_.getInt64Ty();
      llvm::Type* int8_ptr_type = b_.getInt8Ty()->getPointerTo();
      llvm::FunctionType* conv_type = llvm::FunctionType::get(
          b_.getVoidTy(),
          {int8_ptr_type, ir_ptr_type, ir_ptr_type, ir_ptr_type, int64_type,
           int64_type,    int64_type,  int64_type,  int64_type,  int64_type,
           int64_type,    int64_type,  int64_type,  int64_type,  int64_type,
           int64_type,    int64_type,  int64_type,  int64_type,  int64_type,
           int64_type,    int64_type,  int64_type,  int64_type},
          /*isVarArg=*/false);
      bool multi_threaded =
          hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
      bool use_mkl_dnn =
          hlo_module_config_.debug_options().xla_cpu_use_mkl_dnn();

      // TODO(b/78639006) Singlethread MKL conv2d is not implemented due to the
      // potential race condition by setting the omp_num_threads.
      const char* fn_name =
          primitive_type == F16
              ? (multi_threaded
                     ? runtime::kEigenConvF16SymbolName
                     : runtime::kEigenSingleThreadedConvF16SymbolName)
              : (multi_threaded
                     ? (use_mkl_dnn ? runtime::kMKLConvF32SymbolName
                                    : runtime::kEigenConvF32SymbolName)
                     : runtime::kEigenSingleThreadedConvF32SymbolName);
      if (!multi_threaded && use_mkl_dnn) {
        LOG(WARNING) << "Using Eigen instead of MKL-DNN for single-threaded "
                        "conv2d function.";
      }
      llvm::Function* conv_func = llvm::dyn_cast<llvm::Function>(
          module_->getOrInsertFunction(fn_name, conv_type).getCallee());
      conv_func->setCallingConv(llvm::CallingConv::C);
      conv_func->setDoesNotThrow();
      conv_func->setOnlyAccessesArgMemory();
      Call(conv_func, {
                          GetExecutableRunOptionsArgument(),
                          BitCast(GetEmittedValueFor(convolution), ir_ptr_type),
                          BitCast(lhs_address, ir_ptr_type),
                          BitCast(rhs_address, ir_ptr_type),
                          b_.getInt64(input_batch),
                          b_.getInt64(input_rows),
                          b_.getInt64(input_cols),
                          b_.getInt64(input_channels),
                          b_.getInt64(kernel_rows),
                          b_.getInt64(kernel_cols),
                          b_.getInt64(kernel_channels),
                          b_.getInt64(kernel_filters),
                          b_.getInt64(output_rows),
                          b_.getInt64(output_cols),
                          b_.getInt64(row_stride),
                          b_.getInt64(col_stride),
                          b_.getInt64(padding_top),
                          b_.getInt64(padding_bottom),
                          b_.getInt64(padding_left),
                          b_.getInt64(padding_right),
                          b_.getInt64(lhs_row_dilation),
                          b_.getInt64(lhs_col_dilation),
                          b_.getInt64(rhs_row_dilation),
                          b_.getInt64(rhs_col_dilation),
                      });

      return Status::OK();
    }
  }

  // This is a completely un-optimized version of convolution just to
  // have an early version that works. E.g. the input index and
  // padding calculation is not hoisted out of the inner loop.
  //
  // See the description of convolution in the XLA documentation for the pseudo
  // code for convolution.
  return DefaultAction(convolution);
}

Status IrEmitter::HandleFft(HloInstruction* fft) {
  auto operand = fft->operand(0);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*fft, /*operands=*/{operand},
      /*supported_types=*/{F32, C64}));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  VLOG(3) << "operand=" << ShapeUtil::HumanStringWithLayout(operand->shape());
  VLOG(3) << "fft=" << ShapeUtil::HumanStringWithLayout(fft->shape());

  llvm::Value* operand_address = GetEmittedValueFor(operand);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fft));

  const std::vector<int64>& fft_length = fft->fft_length();
  int64 input_batch = 1;
  for (int i = 0; i < fft->shape().dimensions_size() - fft_length.size(); i++) {
    input_batch *= fft->shape().dimensions(i);
  }

  // Args have been computed, make the call.
  llvm::Type* int8_ptr_type = b_.getInt8Ty()->getPointerTo();
  llvm::Type* int32_type = b_.getInt32Ty();
  llvm::Type* int64_type = b_.getInt64Ty();
  llvm::FunctionType* fft_type = llvm::FunctionType::get(
      b_.getVoidTy(),
      {int8_ptr_type, int8_ptr_type, int8_ptr_type, int32_type, int32_type,
       int64_type, int64_type, int64_type, int64_type},
      /*isVarArg=*/false);

  bool multi_threaded_eigen =
      hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
  const char* fn_name = multi_threaded_eigen
                            ? runtime::kEigenFftSymbolName
                            : runtime::kEigenSingleThreadedFftSymbolName;

  llvm::Function* fft_func = llvm::dyn_cast<llvm::Function>(
      module_->getOrInsertFunction(fn_name, fft_type).getCallee());
  fft_func->setCallingConv(llvm::CallingConv::C);
  fft_func->setDoesNotThrow();
  fft_func->setOnlyAccessesInaccessibleMemOrArgMem();
  const int fft_rank = fft_length.size();
  Call(fft_func,
       {GetExecutableRunOptionsArgument(),
        BitCast(GetEmittedValueFor(fft), int8_ptr_type),
        BitCast(operand_address, int8_ptr_type), b_.getInt32(fft->fft_type()),
        b_.getInt32(fft_rank), b_.getInt64(input_batch),
        b_.getInt64(fft_rank > 0 ? fft_length[0] : 0),
        b_.getInt64(fft_rank > 1 ? fft_length[1] : 0),
        b_.getInt64(fft_rank > 2 ? fft_length[2] : 0)});

  return Status::OK();
}

Status IrEmitter::HandleAllReduceSingleReplica(HloInstruction* crs) {
  // When there is a single replica, a cross replica sum is the identity
  // function, and the buffer assignment expects a copy.
  //
  // TODO(b/80100934): We would like to eliminate one-replica CRS nodes entirely
  // in algebraic-simplifier, but currently on some platforms
  // HloModuleConfig::num_replicas changes between when the module is compiled
  // and when it's run.
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(crs));

  // CRS with one operand and one replica is simply the identity function.
  if (crs->operand_count() == 1) {
    return EmitMemcpy(*crs->operand(0), *crs);
  }

  // CRS with multiple operands and one replica produces a (one-deep) tuple.
  std::vector<llvm::Value*> operand_ptrs;
  for (int64 i = 0; i < crs->operand_count(); ++i) {
    llvm::Value* in_ptr = GetEmittedValueFor(crs->operand(i));
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_slice,
                        assignment_.GetUniqueSlice(crs, {i}));

    const Shape& operand_shape = crs->operand(i)->shape();
    CHECK(operand_shape.IsArray())
        << "Operands to all-reduce must be arrays: " << crs->ToString();
    operand_ptrs.push_back(EmitBufferPointer(out_slice, operand_shape));

    // TODO(b/63762267): Be more aggressive about specifying alignment.
    MemCpy(operand_ptrs.back(), /*DstAlign=*/llvm::Align(1), in_ptr,
           /*SrcAlign=*/llvm::Align(1), ShapeUtil::ByteSizeOf(operand_shape));
  }
  llvm_ir::EmitTuple(GetIrArrayFor(crs), operand_ptrs, &b_);
  return Status::OK();
}

Status IrEmitter::HandleAllReduceMultipleReplica(HloInstruction* crs) {
  CHECK_GE(crs->operand_count(), 1);
  PrimitiveType datatype = crs->operand(0)->shape().element_type();
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(crs));

  bool is_datatype_supported = [&] {
    // TODO(cheshire): Fix duplication wrt. cpu_runtime
    switch (datatype) {
      case PRED:
      case S8:
      case U8:
      case S32:
      case U32:
      case S64:
      case U64:
      case F16:
      case F32:
      case F64:
        return true;
      default:
        return false;
    }
  }();

  if (!is_datatype_supported) {
    return Unimplemented("AllReduce for datatype '%s' is not supported",
                         primitive_util::LowercasePrimitiveTypeName(datatype));
  }

  if (!MatchReductionComputation(crs->to_apply()).has_value()) {
    return Unimplemented("AllReduce for computation '%s' is not supported",
                         crs->to_apply()->ToString());
  }

  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::Type* int32_type = b_.getInt32Ty();
  llvm::Type* int64_type = b_.getInt64Ty();
  llvm::FunctionType* all_reduce_func_ty =
      llvm::FunctionType::get(b_.getVoidTy(),
                              {/*run_options=*/i8_ptr_type,
                               /*replica_groups=*/i8_ptr_type,
                               /*replica_groups_size=*/int32_type,
                               /*channel_id_present=*/int32_type,
                               /*op_id=*/int64_type,
                               /*reduction_kind=*/int32_type,
                               /*shape_ptr=*/i8_ptr_type,
                               /*shape_length=*/int32_type,
                               /*num_buffers=*/int32_type,
                               /*input_buffer=*/i8_ptr_type,
                               /*output_buffer=*/i8_ptr_type},
                              /*isVarArg=*/false);

  auto all_reduce_func = llvm::dyn_cast<llvm::Function>(
      module_
          ->getOrInsertFunction(runtime::kAllReduceSymbolName,
                                all_reduce_func_ty)
          .getCallee());
  all_reduce_func->setCallingConv(llvm::CallingConv::C);

  std::string replica_groups = ReplicaGroupsToString(crs->replica_groups());
  int32 replica_groups_size = replica_groups.size();
  llvm::Value* replica_groups_v = b_.CreateGlobalStringPtr(replica_groups);

  bool is_tuple = crs->operand_count() > 1;
  std::vector<llvm::Value*> input_buffer_ptrs;
  std::vector<llvm::Value*> output_buffer_ptrs;
  if (is_tuple) {
    CHECK(crs->shape().IsTuple());

    for (int64 i = 0; i < crs->operand_count(); i++) {
      const HloInstruction* op = crs->operand(i);
      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_slice,
                          assignment_.GetUniqueSlice(crs, {i}));
      const Shape& operand_shape = crs->operand(i)->shape();
      CHECK(operand_shape.IsArray())
          << "Operands to all-reduce must be arrays: " << crs->ToString();
      output_buffer_ptrs.push_back(EmitBufferPointer(out_slice, operand_shape));
      input_buffer_ptrs.push_back(GetEmittedValueFor(op));
    }
  } else {
    Shape shape = crs->operand(0)->shape();
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                        assignment_.GetUniqueSlice(crs->operand(0), {}));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                        assignment_.GetUniqueSlice(crs, {}));
    input_buffer_ptrs.push_back(EmitBufferPointer(input_slice, shape));
    output_buffer_ptrs.push_back(EmitBufferPointer(output_slice, shape));
  }

  llvm::Value* input_buffers =
      EncodeArrayFunctionArguments(input_buffer_ptrs, "input_buffers", &b_);
  llvm::Value* output_buffers =
      EncodeArrayFunctionArguments(output_buffer_ptrs, "output_buffers", &b_);

  int32 shape_length;
  TF_ASSIGN_OR_RETURN(llvm::Value * shape_ptr,
                      llvm_ir::EncodeSelfDescribingShapeConstant(
                          crs->shape(), &shape_length, &b_));

  Call(all_reduce_func,
       {/*run_options=*/GetExecutableRunOptionsArgument(),
        /*replica_groups=*/replica_groups_v,
        /*replica_groups_size=*/b_.getInt32(replica_groups_size),

        /*channel_id_present=*/
        b_.getInt32(static_cast<int32>(crs->channel_id().has_value())),
        /*op_id=*/
        b_.getInt64(crs->channel_id().has_value()
                        ? *crs->channel_id()
                        : crs->GetModule()->unique_id()),
        /*reduction_kind=*/
        b_.getInt32(
            static_cast<int32>(*MatchReductionComputation(crs->to_apply()))),
        /*shape_ptr=*/shape_ptr,
        /*shape_length=*/b_.getInt32(shape_length),
        /*num_buffers=*/b_.getInt32(crs->operand_count()),
        /*input_buffers=*/b_.CreateBitCast(input_buffers, i8_ptr_type),
        /*output_buffers=*/b_.CreateBitCast(output_buffers, i8_ptr_type)});

  return Status::OK();
}

Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  if (hlo_module_config_.replica_count() == 1) {
    return HandleAllReduceSingleReplica(crs);
  }
  return HandleAllReduceMultipleReplica(crs);
}

Status IrEmitter::HandleCollectivePermute(HloInstruction* crs) {
  auto* instr = Cast<HloCollectivePermuteInstruction>(crs);
  std::string source_target_pairs = absl::StrJoin(
      instr->source_target_pairs(), ",", absl::PairFormatter("="));
  llvm::Value* source_target_pairs_v =
      b_.CreateGlobalStringPtr(source_target_pairs);

  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::Type* int32_type = b_.getInt32Ty();
  llvm::Type* int64_type = b_.getInt64Ty();
  llvm::FunctionType* collective_permute_func_ty =
      llvm::FunctionType::get(b_.getVoidTy(),
                              {
                                  /*run_options=*/i8_ptr_type,
                                  /*channel_id_present=*/int32_type,
                                  /*op_id=*/int64_type,
                                  /*byte_size=*/int32_type,
                                  /*input_buffer=*/i8_ptr_type,
                                  /*output_buffer=*/i8_ptr_type,
                                  /*source_target_pairs=*/i8_ptr_type,
                                  /*source_target_pairs_size=*/int32_type,
                              },
                              /*isVarArg=*/false);

  auto collective_permute_func = llvm::dyn_cast<llvm::Function>(
      module_
          ->getOrInsertFunction(runtime::kCollectivePermuteSymbolName,
                                collective_permute_func_ty)
          .getCallee());
  collective_permute_func->setCallingConv(llvm::CallingConv::C);

  Shape shape = crs->operand(0)->shape();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                      assignment_.GetUniqueSlice(crs->operand(0), {}));
  llvm::Value* input_buffer = EmitBufferPointer(input_slice, shape);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(crs, {}));
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, shape);

  Call(collective_permute_func,
       {/*run_options=*/GetExecutableRunOptionsArgument(),
        /*channel_id_present=*/
        b_.getInt32(static_cast<int32>(crs->channel_id().has_value())),
        /*op_id=*/
        b_.getInt64(crs->channel_id().has_value()
                        ? *crs->channel_id()
                        : crs->GetModule()->unique_id()),
        /*byte_size=*/b_.getInt32(ShapeUtil::ByteSizeOf(shape)),
        /*input_buffer=*/b_.CreateBitCast(input_buffer, i8_ptr_type),
        /*output_buffer=*/b_.CreateBitCast(output_buffer, i8_ptr_type),
        /*source_target_pairs=*/source_target_pairs_v,
        /*source_target_pairs_size=*/b_.getInt32(source_target_pairs.size())});

  return Status::OK();
}

Status IrEmitter::HandleReplicaId(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::FunctionType* replica_id_function_ty =
      llvm::FunctionType::get(b_.getVoidTy(),
                              {/*run_options=*/i8_ptr_type,
                               /*output_buffer=*/i8_ptr_type},
                              /*isVarArg=*/false);
  auto* replica_id_func = llvm::dyn_cast<llvm::Function>(
      module_
          ->getOrInsertFunction(runtime::kReplicaIdSymbolName,
                                replica_id_function_ty)
          .getCallee());
  replica_id_func->setCallingConv(llvm::CallingConv::C);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(hlo, {}));
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, hlo->shape());
  Call(replica_id_func,
       {/*run_options=*/GetExecutableRunOptionsArgument(),
        /*output_buffer=*/b_.CreateBitCast(output_buffer, i8_ptr_type)});

  return Status::OK();
}

Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  VLOG(2) << "HandleParameter: " << parameter->ToString();
  return EmitTargetAddressForOp(parameter);
}

// Returns true if the relative order of the unreduced dimensions stays the same
// through the reduce operation.
static bool ReductionPreservesLayout(const HloInstruction& reduce) {
  DCHECK_EQ(reduce.opcode(), HloOpcode::kReduce);

  // Maps dimensions that were not reduced from their dimension numbers in the
  // source shape to their dimensions numbers in the destination shape.
  //
  // So if we reduce f32[A,B,C,D] on dimensions 1 and 2, this map contains
  // [0->0, 3->1].
  absl::flat_hash_map<int64, int64> unreduced_dim_map;

  absl::flat_hash_set<int64> reduced_dims(reduce.dimensions().begin(),
                                          reduce.dimensions().end());

  const Shape& operand_shape = reduce.operand(0)->shape();
  const Shape& result_shape = reduce.shape();

  int64 delta = 0;
  for (int64 i = 0; i < operand_shape.dimensions_size(); i++) {
    if (reduced_dims.contains(i)) {
      delta++;
    } else {
      InsertOrDie(&unreduced_dim_map, i, i - delta);
    }
  }

  // Iterate dimensions minor to major and check that the corresponding
  // dimensions in the source and target shapes are equivalent.
  int64 result_dim_idx = 0;
  for (int64 operand_dim_idx = 0;
       operand_dim_idx < operand_shape.dimensions_size(); operand_dim_idx++) {
    int64 operand_dim = operand_shape.layout().minor_to_major(operand_dim_idx);
    if (!reduced_dims.contains(operand_dim)) {
      if (FindOrDie(unreduced_dim_map, operand_dim) !=
          result_shape.layout().minor_to_major(result_dim_idx++)) {
        return false;
      }
    }
  }

  CHECK_EQ(result_dim_idx, result_shape.dimensions_size());

  return true;
}

IrEmitter::ReductionGenerator IrEmitter::MatchReductionGenerator(
    HloComputation* function, string* failure_reason) const {
  CHECK_EQ(function->num_parameters(), 2);

  auto root_instruction = function->root_instruction();
  CHECK(ShapeUtil::IsScalar(root_instruction->shape()));

  if (root_instruction->operand_count() != 2) {
    *failure_reason = "root instruction is not a binary operation";
    return nullptr;
  }

  const Shape& root_shape = root_instruction->shape();
  if (ShapeUtil::ElementIsComplex(root_shape)) {
    // TODO(b/65408531): Complex add could by done via bitcast to <float x [2N]>
    // Complex multiply would be more challenging. We could perhaps use a
    // strided load to get all reals in a vector, all images in a vector, or use
    // CreateShuffleVector on a bitcast to float x [2N].
    *failure_reason = "complex values not supported";
    return nullptr;
  }
  bool root_is_floating_point = ShapeUtil::ElementIsFloating(root_shape);
  bool root_is_integral = ShapeUtil::ElementIsIntegral(root_shape);
  bool root_is_signed = ShapeUtil::ElementIsSigned(root_shape);

  auto lhs = root_instruction->operand(0);
  auto rhs = root_instruction->operand(1);

  auto param_0 = function->parameter_instruction(0);
  auto param_1 = function->parameter_instruction(1);
  if (!(lhs == param_0 && rhs == param_1) &&
      !(rhs == param_0 && lhs == param_1)) {
    *failure_reason =
        "root instruction is not a binary operation on the incoming arguments";
    return nullptr;
  }

  CHECK(ShapeUtil::IsScalar(lhs->shape()) && ShapeUtil::IsScalar(rhs->shape()));

  // This is visually similar to ElementalIrEmitter, though conceptually we're
  // doing something different here.  ElementalIrEmitter emits scalar operations
  // while these emit scalar or vector operations depending on the type of the
  // operands. See CreateShardedVectorType for the actual types in use here.
  switch (root_instruction->opcode()) {
    default:
      *failure_reason = "did not recognize root instruction opcode";
      return nullptr;

    case HloOpcode::kAdd:
      return [root_is_integral](llvm::IRBuilder<>* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateAdd(lhs, rhs)
                                : b->CreateFAdd(lhs, rhs);
      };

    case HloOpcode::kMultiply:
      return [root_is_integral](llvm::IRBuilder<>* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateMul(lhs, rhs)
                                : b->CreateFMul(lhs, rhs);
      };

    case HloOpcode::kAnd:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateAnd(lhs, rhs);
      };

    case HloOpcode::kOr:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateOr(lhs, rhs);
      };

    case HloOpcode::kXor:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateXor(lhs, rhs);
      };

    case HloOpcode::kMaximum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
        if (root_is_floating_point) {
          return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::maxnum,
                                              {lhs, rhs}, {lhs->getType()}, b);
        }

        return b->CreateSelect(
            b->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SGE
                                         : llvm::ICmpInst::ICMP_UGE,
                          lhs, rhs),
            lhs, rhs);
      };

    case HloOpcode::kMinimum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
        if (root_is_floating_point) {
          return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::minnum,
                                              {lhs, rhs}, {lhs->getType()}, b);
        }

        return b->CreateSelect(
            b->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SLE
                                         : llvm::ICmpInst::ICMP_ULE,
                          lhs, rhs),
            lhs, rhs);
      };
  }
}

IrEmitter::ShardedVectorType IrEmitter::CreateShardedVectorType(
    PrimitiveType element_type, unsigned element_count) {
  int vector_register_size_in_elements =
      target_machine_features_.vector_register_byte_size(
          *compute_function_->function()) /
      ShapeUtil::ByteSizeOfPrimitiveType(element_type);

  ShardedVectorType sharded_vector_type;
  llvm::Type* element_ir_type =
      llvm_ir::PrimitiveTypeToIrType(element_type, module_);

  for (int i = 0, e = 1 + tensorflow::Log2Ceiling(element_count); i < e; i++) {
    // For every power of two present in element_count, we generate one or more
    // vector or scalar types.
    const unsigned current_size_fragment = 1u << i;
    if (!(element_count & current_size_fragment)) {
      // Power of two not present in element_count.
      continue;
    }

    if (current_size_fragment == 1) {
      // Single element, use a scalar type.
      sharded_vector_type.push_back(element_ir_type);
      continue;
    }

    // Lower "current_size_fragment" number of elements using (as few as
    // possible) vector registers.

    if (current_size_fragment >= vector_register_size_in_elements) {
      auto vector_type = llvm::VectorType::get(
          element_ir_type, vector_register_size_in_elements);
      sharded_vector_type.insert(
          sharded_vector_type.end(),
          current_size_fragment / vector_register_size_in_elements,
          vector_type);

      // Both current_size_fragment and vector_register_size_in_elements are
      // powers of two.
      CHECK_EQ(current_size_fragment % vector_register_size_in_elements, 0);
      continue;
    }

    // For now we assume that vector_register_size_in_elements and lower powers
    // of two are all legal vector sizes (or at least can be lowered easily by
    // LLVM).
    sharded_vector_type.push_back(
        llvm::VectorType::get(element_ir_type, current_size_fragment));
  }
  return sharded_vector_type;
}

StatusOr<IrEmitter::ShardedVector>
IrEmitter::EmitInnerLoopForVectorizedReduction(
    const ReductionGenerator& reduction_generator,
    const llvm_ir::IrArray::Index& output_index,
    const ShardedVectorType& accumulator_type, HloInstruction* init_value,
    HloInstruction* arg, absl::Span<const int64> dimensions,
    unsigned element_alignment) {
  ShardedVector accumulator;
  accumulator.reserve(accumulator_type.size());
  for (auto accumulator_shard_type : accumulator_type) {
    accumulator.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        accumulator_shard_type, "accumulator", &b_, 0));
  }

  llvm::Value* init_value_ssa = Load(GetEmittedValueFor(init_value));

  for (llvm::Value* accumulator_shard : accumulator) {
    llvm::Value* initial_value;
    auto shard_type = accumulator_shard->getType()->getPointerElementType();
    if (auto vector_type = llvm::dyn_cast<llvm::VectorType>(shard_type)) {
      initial_value =
          VectorSplat(vector_type->getNumElements(), init_value_ssa);
    } else {
      initial_value = init_value_ssa;
    }

    AlignedStore(initial_value, accumulator_shard, element_alignment);
  }

  llvm_ir::ForLoopNest reduction_loop_nest(IrName(arg, "vectorized_inner"),
                                           &b_);
  std::vector<llvm::Value*> input_multi_index =
      reduction_loop_nest.AddLoopsForShapeOnDimensions(arg->shape(), dimensions,
                                                       "reduction_dim");

  SetToFirstInsertPoint(reduction_loop_nest.GetInnerLoopBodyBasicBlock(), &b_);

  llvm_ir::IrArray arg_array(GetIrArrayFor(arg));
  llvm_ir::IrArray::Index::const_iterator it = output_index.begin();

  for (auto& i : input_multi_index) {
    if (i == nullptr) {
      i = *it++;
    }
  }
  CHECK(output_index.end() == it);
  llvm_ir::IrArray::Index input_index(input_multi_index, arg->shape(),
                                      b_.getInt64Ty());

  llvm::Value* input_address = BitCast(
      arg_array.EmitArrayElementAddress(input_index, &b_), b_.getInt8PtrTy());

  for (int i = 0; i < accumulator.size(); i++) {
    auto input_address_typed =
        BitCast(input_address, accumulator[i]->getType());
    auto current_accumulator_value =
        AlignedLoad(accumulator[i], element_alignment);
    auto addend = AlignedLoad(input_address_typed, element_alignment);
    arg_array.AnnotateLoadStoreInstructionWithMetadata(addend);

    auto reduced_result =
        reduction_generator(&b_, current_accumulator_value, addend);
    AlignedStore(reduced_result, accumulator[i], element_alignment);

    if (i != (accumulator.size() - 1)) {
      input_address = ConstInBoundsGEP1_32(reduced_result->getType(),
                                           input_address_typed, 1);
    }
  }

  SetToFirstInsertPoint(reduction_loop_nest.GetOuterLoopExitBasicBlock(), &b_);

  ShardedVector result_ssa;
  result_ssa.reserve(accumulator.size());
  for (auto accumulator_shard : accumulator) {
    result_ssa.push_back(AlignedLoad(accumulator_shard, element_alignment));
  }
  return result_ssa;
}

void IrEmitter::EmitShardedVectorStore(
    llvm::Value* store_address, const std::vector<llvm::Value*>& value_to_store,
    const int alignment, const llvm_ir::IrArray& containing_array) {
  for (int i = 0; i < value_to_store.size(); i++) {
    auto store_address_typed =
        BitCast(store_address,
                llvm::PointerType::getUnqual(value_to_store[i]->getType()));

    auto store_instruction =
        AlignedStore(value_to_store[i], store_address_typed, alignment);
    containing_array.AnnotateLoadStoreInstructionWithMetadata(
        store_instruction);

    if (i != (value_to_store.size() - 1)) {
      store_address = ConstInBoundsGEP1_32(value_to_store[i]->getType(),
                                           store_address_typed, 1);
    }
  }
}

StatusOr<bool> IrEmitter::EmitVectorizedReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    absl::Span<const int64> dimensions, HloComputation* function,
    string* failure_reason) {
  if (!reduce->shape().IsArray()) {
    *failure_reason = "vectorization of variadic reduce not implemented";
    return false;
  }

  if (!ReductionPreservesLayout(*reduce)) {
    return false;
  }

  ReductionGenerator reduction_generator =
      MatchReductionGenerator(function, failure_reason);
  if (!reduction_generator) {
    return false;
  }

  int vector_register_size_in_elements =
      target_machine_features_.vector_register_byte_size(
          *compute_function_->function()) /
      ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type());
  if (vector_register_size_in_elements == 0) {
    // Either we don't know the vector register width for the target or the
    // vector register is smaller than the size of the primitive type.
    return false;
  }

  int vectorization_factor_in_bytes =
      target_machine_features_.vectorization_factor_in_bytes();

  // We try to process vectorization_factor elements at the same time.
  const int vectorization_factor =
      vectorization_factor_in_bytes /
      ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type());

  bool is_reduction_over_minor_dimension = absl::c_linear_search(
      dimensions, LayoutUtil::Minor(arg->shape().layout(), 0));

  unsigned element_alignment = tensorflow::MathUtil::GCD<unsigned>(
      ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type()),
      MinimumAlignmentForPrimitiveType(reduce->shape().element_type()));

  if (is_reduction_over_minor_dimension) {
    // TODO(sanjoy): Implement vectorized reduction over the minor dimension.
    *failure_reason = "reduction over minor dimension not implemented";
    return false;
  }

  CHECK(!reduce->shape().IsTuple());
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(reduce));

  // We know we're not reducing over the most minor dimension, which means we
  // can lower the reduction loop as:
  //
  //  1. We're reducing over dimensions R0, R1.
  //  2. D0 is the most minor dimension.
  //  3. VS is the vectorization stride (we want to reduce this many elements at
  //     once)
  //
  //  for (d1 in D1) {
  //    for (d0 in D0 with stride VS) {
  //      vector_acc = init
  //      for (r1 in R1) {
  //        for (r0 in R0) {
  //          vector_acc = elementwise_reduce(vector_acc, input[d1, d0, r1, r0]
  //        }
  //      }
  //      output[d1, d0] = vector_acc
  //    }
  //  }

  llvm_ir::ForLoopNest loop_nest(IrName(reduce), &b_);
  std::vector<llvm::Value*> array_multi_index(
      reduce->shape().dimensions_size());
  for (int i = LayoutUtil::MinorToMajor(reduce->shape()).size() - 1; i > 0;
       --i) {
    int64 dimension = LayoutUtil::Minor(reduce->shape().layout(), i);
    int64 start_index = 0;
    int64 end_index = reduce->shape().dimensions(dimension);
    std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
        start_index, end_index, absl::StrFormat("dim.%d", dimension));
    array_multi_index[dimension] = loop->GetIndVarValue();
  }

  int64 innermost_dimension = LayoutUtil::Minor(reduce->shape().layout(), 0);
  int64 innermost_dimension_size =
      reduce->shape().dimensions(innermost_dimension);

  if (llvm::BasicBlock* innermost_body_bb =
          loop_nest.GetInnerLoopBodyBasicBlock()) {
    SetToFirstInsertPoint(innermost_body_bb, &b_);
  }

  auto outermost_loop_exit_block = loop_nest.GetOuterLoopExitBasicBlock();

  if (innermost_dimension_size >= vectorization_factor) {
    int64 start_index = 0;
    int64 end_index = (innermost_dimension_size / vectorization_factor) *
                      vectorization_factor;
    std::unique_ptr<llvm_ir::ForLoop> loop =
        loop_nest.AddLoop(start_index, end_index, vectorization_factor,
                          absl::StrFormat("dim.%d", innermost_dimension));
    array_multi_index[innermost_dimension] = loop->GetIndVarValue();

    SetToFirstInsertPoint(loop->GetBodyBasicBlock(), &b_);

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(), vectorization_factor);
    llvm_ir::IrArray::Index array_index(array_multi_index, reduce->shape(),
                                        b_.getInt64Ty());
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, &b_);
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);

    if (auto exit_terminator = loop->GetExitBasicBlock()->getTerminator()) {
      CHECK_GT(LayoutUtil::MinorToMajor(reduce->shape()).size(), 1);
      b_.SetInsertPoint(exit_terminator);
    } else {
      CHECK_EQ(LayoutUtil::MinorToMajor(reduce->shape()).size(), 1);
      b_.SetInsertPoint(loop->GetExitBasicBlock());
    }
  }

  // Since we increment the stride for the inner dimension by more than 1, we
  // may need to peel out an "epilogue" iteration to get the remaining elements
  // in the following case:
  if (innermost_dimension_size % vectorization_factor) {
    // TODO(b/63775531): Consider using a scalar loop here to save on code size.
    array_multi_index[innermost_dimension] =
        b_.getInt64(innermost_dimension_size -
                    (innermost_dimension_size % vectorization_factor));

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(),
        innermost_dimension_size % vectorization_factor);
    llvm_ir::IrArray::Index array_index(array_multi_index, reduce->shape(),
                                        b_.getInt64Ty());
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, &b_);
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);
  }

  if (outermost_loop_exit_block) {
    b_.SetInsertPoint(outermost_loop_exit_block);
  }

  return true;
}

Status IrEmitter::HandleReduce(HloInstruction* reduce) {
  auto arg = reduce->mutable_operand(0);
  auto init_value = reduce->mutable_operand(1);
  absl::Span<const int64> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (!options::VectorizedReduceDisabled(hlo_module_config_)) {
    string vectorization_failure_reason;
    TF_ASSIGN_OR_RETURN(
        bool vectorization_successful,
        EmitVectorizedReduce(reduce, arg, init_value, dimensions, function,
                             &vectorization_failure_reason));
    if (vectorization_successful) {
      VLOG(1) << "Successfully vectorized reduction " << reduce->ToString()
              << "\n";
      return Status::OK();
    } else {
      VLOG(1) << "Could not vectorize reduction " << reduce->ToString() << ": "
              << vectorization_failure_reason;
    }
  }

  return DefaultAction(reduce);
}

Status IrEmitter::HandleAllToAll(HloInstruction*) {
  return Unimplemented("AllToAll is not implemented on CPU.");
}

Status IrEmitter::HandleSend(HloInstruction* send) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Send is not implemented on CPU.");
}

Status IrEmitter::HandleSendDone(HloInstruction* send_done) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Send-done is not implemented on CPU.");
}

Status IrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter is not implemented on CPUs.");
}

Status IrEmitter::HandleSlice(HloInstruction* slice) {
  VLOG(2) << "HandleSlice: " << slice->ToString();
  auto operand = slice->operand(0);
  // The code below emits a sequential loop nest. For the parallel backend, use
  // ParallelLoopEmitter which respects dynamic loop bounds.
  if (ShouldEmitParallelLoopFor(*slice)) {
    return DefaultAction(slice);
  }

  // The code below assumes the layouts are equal.
  if (!LayoutUtil::Equal(operand->shape().layout(), slice->shape().layout())) {
    return DefaultAction(slice);
  }

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(slice));

  if (ShapeUtil::IsZeroElementArray(slice->shape())) {
    return Status::OK();
  }

  const Layout& layout = operand->shape().layout();
  const int64 num_dims = operand->shape().dimensions_size();

  // The slice lowering finds maximal contiguous blocks of memory that can be
  // copied from the source to the target. This is done by looking at the
  // source/target layout in minor to major order and do the following:
  //
  // * Find an initial segment of dimensions along which the slice uses the
  //   whole dimension. These are the "inner" dimensions and can be folded into
  //   the memcpy.
  //
  // * Of the remaining dimensions decide which ones require loops.
  //
  // * Implement the memcpy within the innermost loop.

  absl::flat_hash_set<int64> inner_dims;
  for (int64 dim : LayoutUtil::MinorToMajor(layout)) {
    if (operand->shape().dimensions(dim) != slice->shape().dimensions(dim)) {
      break;
    }
    inner_dims.insert(dim);
  }

  const bool is_trivial_copy = (inner_dims.size() == num_dims);
  if (is_trivial_copy) {
    if (ShapeUtil::IsEffectiveScalar(slice->shape())) {
      return DefaultAction(slice);
    } else {
      return EmitMemcpy(*slice, *operand);
    }
  }

  // The memcpy will copy elements that are logically this shape (allowed to be
  // scalar).
  const Shape logical_element_shape = ShapeUtil::FilterDimensions(
      [&inner_dims](int64 dim) { return inner_dims.contains(dim); },
      operand->shape());

  const int64 primitive_elements_per_logical_element =
      ShapeUtil::ElementsIn(logical_element_shape);

  // memcpy_dim is the innermost (in terms of layout) dimension for which the
  // slice does *not* just copy all the elements along the dimension.
  const int64 memcpy_dim = LayoutUtil::Minor(layout, inner_dims.size());

  const bool memcpy_is_contiguous = slice->slice_strides(memcpy_dim) == 1;
  // The number of logical elements that can be copied in a single call
  // to memcpy. We can only copy 1 element at a time if there is a non-trivial
  // stride.
  const int64 memcpy_logical_elements =
      memcpy_is_contiguous
          ? slice->slice_limits(memcpy_dim) - slice->slice_starts(memcpy_dim)
          : 1;

  // Determine the dimensions that get lowered as loops.
  std::vector<int64> outer_dims;
  for (int64 i = 0; i < num_dims - inner_dims.size() - 1; ++i) {
    outer_dims.push_back(LayoutUtil::Major(layout, i));
  }

  // Is the slice along the memcpy dimension contiguous? If not, then memcpy_dim
  // needs to be wrapped around a loop as well.
  if (!memcpy_is_contiguous) {
    outer_dims.push_back(memcpy_dim);
  }

  llvm_ir::IrArray target_array = GetIrArrayFor(slice);

  const int64 num_outer_loops = outer_dims.size();
  llvm_ir::ForLoopNest loops(IrName(slice), &b_);
  std::vector<llvm::Value*> target_multi_index =
      loops.AddLoopsForShapeOnDimensions(slice->shape(), outer_dims, "slice");

  // Only the indices for the outer dimensions have been initialized in
  // target_index. The rest of the indices should get initialized to 0, since
  // for the rest of the dimensions the copy writes to the full dimension.
  std::replace(target_multi_index.begin(), target_multi_index.end(),
               static_cast<llvm::Value*>(nullptr),
               static_cast<llvm::Value*>(b_.getInt64(0)));
  llvm_ir::IrArray::Index target_index(target_multi_index, slice->shape(),
                                       b_.getInt64Ty());

  if (num_outer_loops > 0) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b_);
  }

  llvm_ir::IrArray source_array = GetIrArrayFor(operand);
  const llvm_ir::IrArray::Index source_index = target_index.SourceIndexOfSlice(
      /*operand_shape=*/operand->shape(), /*starts=*/slice->slice_starts(),
      /*strides=*/slice->slice_strides(), /*builder=*/&b_);

  llvm::Value* memcpy_dest =
      target_array.EmitArrayElementAddress(target_index, &b_, "slice.dest");
  llvm::Value* memcpy_source =
      source_array.EmitArrayElementAddress(source_index, &b_, "slice.source");

  const int64 memcpy_elements =
      primitive_elements_per_logical_element * memcpy_logical_elements;

  EmitTransferElements(memcpy_dest, memcpy_source, memcpy_elements,
                       slice->shape().element_type(), target_array,
                       source_array);

  if (VLOG_IS_ON(2)) {
    const int64 memcpy_bytes =
        ShapeUtil::ByteSizeOf(logical_element_shape) * memcpy_elements;
    VLOG(2) << "  emitted copy of " << memcpy_bytes << " bytes inside "
            << num_outer_loops << " loops";
  }

  if (num_outer_loops > 0) {
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b_);
  }

  return Status::OK();
}

Status IrEmitter::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_slice));
    return EmitMemcpy(*dynamic_slice->operand(0), *dynamic_slice);
  }
  return DefaultAction(dynamic_slice);
}

Status IrEmitter::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  auto update = dynamic_update_slice->operand(1);
  if (ShapeUtil::IsScalar(dynamic_update_slice->shape())) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_update_slice));
    return EmitMemcpy(*update, *dynamic_update_slice);
  } else if (llvm_ir::CanUpdateDynamicSliceInPlace(dynamic_update_slice,
                                                   assignment_)) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_update_slice));
    auto operands = GetIrArraysForOperandsOf(dynamic_update_slice);
    return llvm_ir::EmitDynamicUpdateSliceInPlace(
        operands, GetIrArrayFor(dynamic_update_slice),
        IrName(dynamic_update_slice, "in_place"), &b_);
  }
  return DefaultAction(dynamic_update_slice);
}

Status IrEmitter::HandleRecv(HloInstruction* recv) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Recv is not implemented on CPU.");
}

Status IrEmitter::HandleRecvDone(HloInstruction* recv_done) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Recv-done is not implemented on CPU.");
}

Status IrEmitter::HandlePad(HloInstruction* pad) {
  // CPU backend does not properly handle negative padding but this is ok
  // because negative padding should be removed by the algebraic simplifier.
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      return InternalErrorStrCat(
          "Encountered negative padding in IrEmitter on CPU. "
          "This should have been eliminated at the HLO level. ",
          pad->ToString());
    }
  }

  // First, fill in the padding value to all output elements.
  TF_RETURN_IF_ERROR(EmitTargetElementLoop(
      pad, "initialize",
      [this, pad](const llvm_ir::IrArray::Index& target_index) {
        const HloInstruction* padding_value = pad->operand(1);
        llvm::Value* padding_value_addr = GetEmittedValueFor(padding_value);
        return Load(padding_value_addr);
      }));

  // Create a loop to iterate over the operand elements and update the output
  // locations where the operand elements should be stored.
  llvm_ir::ForLoopNest loops(IrName(pad, "assign"), &b_);
  const HloInstruction* operand = pad->operand(0);
  const llvm_ir::IrArray::Index operand_index =
      loops.AddLoopsForShape(operand->shape(), "operand");

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b_);

  // Load an element from the operand.
  llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, &b_);

  // Compute the output index the operand element should be assigned to.
  // output_index := edge_padding_low + operand_index * (interior_padding + 1)
  const PaddingConfig& padding_config = pad->padding_config();
  std::vector<llvm::Value*> output_multi_index;
  for (size_t i = 0; i < operand_index.size(); ++i) {
    llvm::Value* offset =
        Mul(operand_index[i],
            b_.getInt64(padding_config.dimensions(i).interior_padding() + 1));
    llvm::Value* index = Add(
        offset, b_.getInt64(padding_config.dimensions(i).edge_padding_low()));
    output_multi_index.push_back(index);
  }

  // Store the operand element to the computed output location.
  llvm_ir::IrArray output_array(GetIrArrayFor(pad));
  llvm_ir::IrArray::Index output_index(
      output_multi_index, output_array.GetShape(), operand_index.GetType());
  output_array.EmitWriteArrayElement(output_index, operand_data, &b_);

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b_);
  return Status::OK();
}

Status IrEmitter::HandleFusion(HloInstruction* fusion) {
  auto* root = fusion->fused_expression_root();
  if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(fusion, assignment_)) {
    VLOG(3) << "HandleFusion FusedDynamicUpdateSliceInPlace";
    CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
    // Delegate to common implementation of fused in-place dynamic-update-slice.
    return llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
        fusion, GetGeneratorForOperandIrArrays(fusion), GetIrArrayFor(fusion),
        &elemental_emitter, &b_);
  } else if (fusion->IsLoopFusion()) {
    VLOG(3) << "HandleFusion kLoop";
    CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
    auto operands = GetIrArraysForOperandsOf(fusion);
    FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(fusion),
                                 &elemental_emitter);
    TF_RETURN_IF_ERROR(fusion->fused_expression_root()->Accept(&fused_emitter));

    return EmitTargetElementLoop(fusion, fused_emitter.GetRootGenerator());
  } else if (fusion->IsOutputFusion()) {
    VLOG(3) << "HandleFusion kOutput";
    int64 dot_op_index = root->operand(0)->opcode() == HloOpcode::kDot ? 0 : 1;
    const HloInstruction* dot = root->operand(dot_op_index);
    CHECK_EQ(dot->opcode(), HloOpcode::kDot)
        << dot->ToString() << "  "
        << fusion->fused_instructions_computation()->ToString();

    int64 dot_lhs_param_number = dot->operand(0)->parameter_number();
    int64 dot_rhs_param_number = dot->operand(1)->parameter_number();
    int64 addend_param_number =
        root->operand(1 - dot_op_index)->parameter_number();

    Shape target_shape = fusion->shape();
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
    llvm_ir::IrArray target_array = GetIrArrayFor(fusion);

    llvm_ir::IrArray lhs_array(
        GetIrArrayFor(fusion->operand(dot_lhs_param_number)));
    llvm_ir::IrArray rhs_array(
        GetIrArrayFor(fusion->operand(dot_rhs_param_number)));
    llvm_ir::IrArray addend_array(
        GetIrArrayFor(fusion->operand(addend_param_number)));

    TF_RETURN_IF_ERROR(
        EmitDotOperation(*dot, target_array, lhs_array, rhs_array,
                         &addend_array, GetExecutableRunOptionsArgument(), &b_,
                         hlo_module_config_, target_machine_features_));
    return Status::OK();
  } else {
    return Unimplemented("Fusion kind not implemented on CPU");
  }
}

Status IrEmitter::HandleCall(HloInstruction* call) {
  HloComputation* computation = call->to_apply();
  llvm::Function* call_ir_function = FindOrDie(emitted_functions_, computation);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(call));

  if (!computation->root_instruction()->outer_dimension_partitions().empty()) {
    // ParallelTaskAssignment assigned partitions, emit call to
    // ParallelForkJoin.
    std::vector<llvm::Value*> call_args = GetArrayFunctionCallArguments(
        {}, &b_, computation->name(),
        /*return_value_buffer=*/emitted_value_[call],
        /*exec_run_options_arg=*/GetExecutableRunOptionsArgument(),
        /*buffer_table_arg=*/GetBufferTableArgument(),
        /*profile_counters_arg=*/GetProfileCountersArgument());

    HloInstruction* root = computation->root_instruction();
    TF_RETURN_IF_ERROR(EmitCallToParallelForkJoin(
        call_args, root->shape(), root->outer_dimension_partitions(), &b_,
        call_ir_function, computation->name()));
  } else {
    EmitGlobalCall(*computation, computation->name());
  }

  return Status::OK();
}

Status IrEmitter::HandleSliceToDynamic(HloInstruction* hlo) {
  // TODO(jackcao): Generalize this to generic llvm emitter.
  TF_RET_CHECK(hlo->shape().rank() == 1);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  for (int64 i = 1; i < hlo->operand_count(); ++i) {
    const int64 dim_index = i - 1;
    llvm::Value* source_buffer = GetEmittedValueFor(hlo->operand(i));
    llvm::LoadInst* dim_size = b_.CreateLoad(source_buffer, "dim_size");
    llvm::Value* dest_buffer = GetEmittedValueFor(hlo);
    llvm::Value* raw_buffer =
        b_.CreateBitCast(dest_buffer, b_.getInt8Ty()->getPointerTo());

    int32 raw_data_size =
        ShapeUtil::ByteSizeOf(ShapeUtil::MakeStaticShape(hlo->shape()));
    llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), raw_buffer, raw_data_size + dim_index * sizeof(int32));
    b_.CreateStore(dim_size,
                   b_.CreateBitCast(metadata, b_.getInt32Ty()->getPointerTo()));
  }

  return EmitTargetElementLoop(hlo,
                               [=](const llvm_ir::IrArray::Index& dest_index) {
                                 // TODO(jackcao): Properly linearize dest_index
                                 // and delinearize to source index.
                                 return GetIrArrayFor(hlo->operand(0))
                                     .EmitReadArrayElement(dest_index, &b_);
                               });
}

Status IrEmitter::HandlePadToStatic(HloInstruction* hlo) {
  // TODO(jackcao): Generalize this to generic llvm emitter.
  TF_RET_CHECK(hlo->operand(0)->shape().rank() == 1);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data_slice,
                      assignment_.GetUniqueSlice(hlo, {0}));
  const Shape& data_shape = ShapeUtil::GetSubshape(hlo->shape(), {0});
  llvm::Value* data_address = EmitBufferPointer(data_slice, data_shape);
  llvm_ir::IrArray data_array(data_address, data_shape);
  TF_RETURN_IF_ERROR(llvm_ir::LoopEmitter(
                         [=](const llvm_ir::IrArray::Index& dest_index) {
                           // TODO(jackcao): Properly linearize dest_index and
                           // delinearize to source index.
                           return GetIrArrayFor(hlo->operand(0))
                               .EmitReadArrayElement(dest_index, &b_);
                         },
                         llvm_ir::IrArray(data_address, data_shape), &b_)
                         .EmitLoop(IrName(hlo)));
  std::vector<llvm::Value*> tuple_operand_ptrs;
  tuple_operand_ptrs.push_back(data_array.GetBasePointer());

  // PadToStatic has a dynamic tensor as input and variadic size of outputs:
  // (static_tensor, dynamic_dim_0, dynamic_dim_1, ... )
  // Dynamic dimension sizes starts from output index 1.
  for (int64 i = 1; i < hlo->shape().tuple_shapes_size(); ++i) {
    // Read from the metadata section of the dynamic input (operand 0).
    const Shape& dim_shape = ShapeUtil::GetSubshape(hlo->shape(), {i});
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dim_size_slice,
                        assignment_.GetUniqueSlice(hlo, {i}));
    llvm::Value* dest_dim_size_address =
        EmitBufferPointer(dim_size_slice, data_shape);
    const int64 dim_index = i - 1;
    llvm::Value* source_buffer = GetEmittedValueFor(hlo->operand(0));
    llvm::Value* raw_buffer =
        b_.CreateBitCast(source_buffer, b_.getInt8Ty()->getPointerTo());
    int32 raw_data_size = ShapeUtil::ByteSizeOf(
        ShapeUtil::MakeStaticShape(hlo->operand(0)->shape()));
    llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), raw_buffer, raw_data_size + dim_index * sizeof(int32));
    llvm::Value* dim_size = b_.CreateLoad(
        b_.CreateBitCast(metadata, b_.getInt32Ty()->getPointerTo()));
    b_.CreateStore(dim_size, b_.CreateBitCast(dest_dim_size_address,
                                              b_.getInt32Ty()->getPointerTo()));
    tuple_operand_ptrs.push_back(dest_dim_size_address);
  }

  // Emit static tensor and dynamic sizes as one tuple.
  llvm_ir::EmitTuple(GetIrArrayFor(hlo), tuple_operand_ptrs, &b_);
  return Status::OK();
}

Status IrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  if (custom_call->custom_call_target() == "PadToStatic") {
    return HandlePadToStatic(custom_call);
  }
  if (custom_call->custom_call_target() == "SliceToDynamic") {
    return HandleSliceToDynamic(custom_call);
  }
  absl::Span<HloInstruction* const> operands(custom_call->operands());
  llvm::Type* i8_ptr_type = b_.getInt8PtrTy();
  llvm::AllocaInst* operands_alloca =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          i8_ptr_type, b_.getInt32(operands.size()), "cc_operands_alloca", &b_);
  for (size_t i = 0; i < operands.size(); ++i) {
    const HloInstruction* operand = operands[i];
    llvm::Value* operand_as_i8ptr =
        PointerCast(GetEmittedValueFor(operand), i8_ptr_type);
    llvm::Value* slot_in_operands_alloca =
        InBoundsGEP(operands_alloca, {b_.getInt64(i)});
    Store(operand_as_i8ptr, slot_in_operands_alloca);
  }
  if (emit_code_for_msan_) {
    // Mark the alloca as initialized for msan. The buffer gets read by the
    // custom callee, which might be msan-instrumented.
    // TODO(b/66051036): Run the msan instrumentation pass instead.
    const llvm::DataLayout& dl = module_->getDataLayout();
    llvm::Type* intptr_type = b_.getIntPtrTy(dl);
    auto* msan_unpoison_ir_function = llvm::cast<llvm::Function>(
        module_
            ->getOrInsertFunction(
                "__msan_unpoison",
                llvm::FunctionType::get(
                    /*Result=*/b_.getVoidTy(),
                    /*Params=*/{i8_ptr_type, intptr_type}, /*isVarArg=*/false))
            .getCallee());
    Call(msan_unpoison_ir_function,
         {PointerCast(operands_alloca, i8_ptr_type),
          llvm::ConstantInt::get(
              intptr_type, *operands_alloca->getAllocationSizeInBits(dl) / 8)});
  }
  auto* custom_call_ir_function = llvm::dyn_cast<llvm::Function>(
      module_
          ->getOrInsertFunction(
              custom_call->custom_call_target(),
              llvm::FunctionType::get(
                  /*Result=*/b_.getVoidTy(),
                  /*Params=*/{i8_ptr_type, operands_alloca->getType()},
                  /*isVarArg=*/false))
          .getCallee());

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));
  // Write the tuple table if the output is a tuple.
  if (custom_call->shape().IsTuple()) {
    std::vector<llvm::Value*> base_ptrs;
    for (int i = 0; i < ShapeUtil::TupleElementCount(custom_call->shape());
         ++i) {
      const Shape& elem_shape =
          ShapeUtil::GetTupleElementShape(custom_call->shape(), i);
      TF_RET_CHECK(!elem_shape.IsTuple()) << "Nested tuples not implemented";
      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                          assignment_.GetUniqueSlice(custom_call, {i}));
      llvm::Value* addr = EmitBufferPointer(slice, elem_shape);
      base_ptrs.push_back(addr);
    }
    llvm_ir::EmitTuple(GetIrArrayFor(custom_call), base_ptrs, &b_);
  }
  auto* output_address_arg =
      PointerCast(GetEmittedValueFor(custom_call), i8_ptr_type);

  Call(custom_call_ir_function, {output_address_arg, operands_alloca});

  return Status::OK();
}

Status IrEmitter::HandleWhile(HloInstruction* xla_while) {
  // Precondition: Condition computation must return a scalar bool.
  HloComputation* condition = xla_while->while_condition();
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool; got: "
      << ShapeUtil::HumanString(condition->root_instruction()->shape());
  // Check that all while-related buffers share an allocation slice.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      xla_while->shape(),
      [this, &xla_while](const Shape& /*subshape*/,
                         const ShapeIndex& index) -> Status {
        auto check = [this](const HloInstruction* a, const HloInstruction* b,
                            const ShapeIndex& index) {
          const BufferAllocation::Slice slice_a =
              assignment_.GetUniqueSlice(a, index).ConsumeValueOrDie();
          const BufferAllocation::Slice slice_b =
              assignment_.GetUniqueSlice(b, index).ConsumeValueOrDie();
          if (slice_a != slice_b) {
            return InternalError(
                "instruction %s %s does not share slice with "
                "instruction %s %s",
                a->ToString(), slice_a.ToString(), b->ToString(),
                slice_b.ToString());
          }
          return Status::OK();
        };
        TF_RETURN_IF_ERROR(check(xla_while, xla_while->operand(0), index));
        TF_RETURN_IF_ERROR(check(
            xla_while, xla_while->while_condition()->parameter_instruction(0),
            index));
        TF_RETURN_IF_ERROR(
            check(xla_while, xla_while->while_body()->parameter_instruction(0),
                  index));
        TF_RETURN_IF_ERROR(check(
            xla_while, xla_while->while_body()->root_instruction(), index));
        return Status::OK();
      }));

  // Set emitted value to that of 'init' with which it shares an allocation.
  const HloInstruction* init = xla_while->operand(0);
  emitted_value_[xla_while] = GetEmittedValueFor(init);

  // Generating:
  //   while (Condition(while_result)) {
  //     // CopyInsertion pass inserts copies which enable 'while_result' to
  //     // be passed back in as 'Body' parameter.
  //     while_result = Body(while_result);  // Insert
  //   }

  // Terminates the current block with a branch to a while header.
  llvm::BasicBlock* header_bb = llvm::BasicBlock::Create(
      module_->getContext(), IrName(xla_while, "header"),
      compute_function_->function());
  Br(header_bb);
  b_.SetInsertPoint(header_bb);

  // Calls the condition function to determine whether to proceed with the
  // body.  It must return a bool, so use the scalar call form.
  EmitGlobalCall(*xla_while->while_condition(), IrName(xla_while, "cond"));
  llvm::Value* while_predicate = ICmpNE(
      Load(GetBufferForGlobalCallReturnValue(*xla_while->while_condition())),
      llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0));

  // Branches to the body or to the while exit depending on the condition.
  llvm::BasicBlock* body_bb =
      llvm::BasicBlock::Create(module_->getContext(), IrName(xla_while, "body"),
                               compute_function_->function());
  llvm::BasicBlock* exit_bb = llvm::BasicBlock::Create(
      module_->getContext(), IrName(xla_while, "exit"));
  CondBr(while_predicate, body_bb, exit_bb);

  // Calls the body function from the body block.
  b_.SetInsertPoint(body_bb);

  // Calls the body function.
  EmitGlobalCall(*xla_while->while_body(), IrName(xla_while, "body"));

  // Finishes with a branch back to the header.
  Br(header_bb);

  // Adds the exit block to the function and sets the insert point there.
  compute_function_->function()->getBasicBlockList().push_back(exit_bb);
  b_.SetInsertPoint(exit_bb);

  return Status::OK();
}

StatusOr<bool> IrEmitter::EmitFastConcatenate(
    HloInstruction* concatenate, absl::Span<HloInstruction* const> operands,
    string* failure_reason) {
  if (ShouldEmitParallelLoopFor(*concatenate)) {
    *failure_reason =
        "cannot generate memcpy-based concat for the parallel CPU backend";
    return false;
  }

  const Shape& output_shape = concatenate->shape();
  for (auto* op : operands) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      *failure_reason = "operand has mismatching layouts";
      return false;
    }
  }

  // We split the dimensions into three categories: the dimension over which we
  // are concatenating (concat_dim), the dimensions that are minor to it
  // (inner_dims) and the dimensions that are major to it (outer_dims).

  int64 concat_dim = concatenate->dimensions(0);
  const Layout& output_layout = output_shape.layout();
  auto output_min2maj = LayoutUtil::MinorToMajor(output_layout);
  auto concat_dim_layout_itr = absl::c_find(output_min2maj, concat_dim);

  std::vector<int64> inner_dims(output_min2maj.begin(), concat_dim_layout_itr);
  std::vector<int64> outer_dims(std::next(concat_dim_layout_itr),
                                output_min2maj.end());

  llvm::Type* i8_ptr_type = b_.getInt8PtrTy();

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(concatenate));
  llvm_ir::IrArray target_array = GetIrArrayFor(concatenate);

  llvm_ir::ForLoopNest loops(IrName(concatenate), &b_);
  std::vector<llvm::Value*> target_multi_index =
      loops.AddLoopsForShapeOnDimensions(output_shape, outer_dims, "concat");
  std::replace(target_multi_index.begin(), target_multi_index.end(),
               static_cast<llvm::Value*>(nullptr),
               static_cast<llvm::Value*>(b_.getInt64(0)));
  llvm_ir::IrArray::Index target_index(target_multi_index, output_shape,
                                       b_.getInt64Ty());

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b_);
  }

  PrimitiveType primitive_type = output_shape.element_type();
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);

  // Contiguous subregions from each operand to the concatenate contribute to a
  // contiguous subregion in the target buffer starting at target_region_begin.
  llvm::Value* target_region_begin = BitCast(
      target_array.EmitArrayElementAddress(target_index, &b_, "target_region"),
      i8_ptr_type);
  int64 byte_offset_into_target_region = 0;

  int64 inner_dims_product =
      std::accumulate(inner_dims.begin(), inner_dims.end(), 1l,
                      [&](int64 product, int64 inner_dim) {
                        return product * output_shape.dimensions(inner_dim);
                      });

  // For each operand, emit a memcpy from the operand to the target of size
  // equal to the product of inner dimensions.
  for (HloInstruction* operand : operands) {
    const Shape& input_shape = operand->shape();
    llvm_ir::IrArray source_array = GetIrArrayFor(operand);
    llvm_ir::IrArray::Index source_index(target_multi_index, operand->shape(),
                                         b_.getInt64Ty());
    llvm::Value* copy_source_address = BitCast(
        source_array.EmitArrayElementAddress(source_index, &b_, "src_addr"),
        i8_ptr_type);

    llvm::Value* copy_target_address =
        GEP(target_region_begin, b_.getInt64(byte_offset_into_target_region));

    EmitTransferElements(
        copy_target_address, copy_source_address,
        inner_dims_product * input_shape.dimensions(concat_dim), primitive_type,
        target_array, source_array);

    byte_offset_into_target_region += inner_dims_product *
                                      input_shape.dimensions(concat_dim) *
                                      primitive_type_size;
  }

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b_);
  }

  return true;
}

void IrEmitter::EmitTransferElements(llvm::Value* target, llvm::Value* source,
                                     int64 element_count,
                                     PrimitiveType primitive_type,
                                     const llvm_ir::IrArray& target_array,
                                     const llvm_ir::IrArray& source_array) {
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  unsigned element_alignment = tensorflow::MathUtil::GCD<unsigned>(
      primitive_type_size, MinimumAlignmentForPrimitiveType(primitive_type));
  llvm::Type* primitive_ptr_type = llvm::PointerType::getUnqual(
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module_));

  if (element_count == 1) {
    auto* load_instruction =
        AlignedLoad(BitCast(source, primitive_ptr_type), element_alignment);
    source_array.AnnotateLoadStoreInstructionWithMetadata(load_instruction);
    auto* store_instruction =
        AlignedStore(load_instruction, BitCast(target, primitive_ptr_type),
                     element_alignment);
    target_array.AnnotateLoadStoreInstructionWithMetadata(store_instruction);
  } else {
    auto* memcpy_instruction =
        MemCpy(target, /*DstAlign=*/llvm::Align(element_alignment), source,
               /*SrcAlign=*/llvm::Align(element_alignment),
               element_count * primitive_type_size);

    // The memcpy does the load and the store internally.  The aliasing related
    // metadata has to reflect that.
    std::map<int, llvm::MDNode*> merged_metadata =
        llvm_ir::MergeMetadata(&module_->getContext(), source_array.metadata(),
                               target_array.metadata());
    for (const auto& kind_md_pair : merged_metadata) {
      memcpy_instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
    }
  }
}

Status IrEmitter::HandleConcatenate(HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  string failure_reason;
  TF_ASSIGN_OR_RETURN(
      bool successful,
      EmitFastConcatenate(concatenate, operands, &failure_reason));
  if (successful) {
    VLOG(1) << "Emitted fast concatenate for " << concatenate->ToString();
    return Status::OK();
  }

  VLOG(1) << "Could not emit fast concatenate for " << concatenate->ToString()
          << ": " << failure_reason;

  return DefaultAction(concatenate);
}

Status IrEmitter::HandleConditional(HloInstruction* conditional) {
  auto branch_index = conditional->operand(0);
  int num_branches = conditional->branch_count();
  TF_RET_CHECK(ShapeUtil::IsScalar(branch_index->shape()) &&
               (branch_index->shape().element_type() == PRED ||
                branch_index->shape().element_type() == S32))
      << "Branch index on a conditional must be scalar bool or int32; got: "
      << ShapeUtil::HumanString(branch_index->shape());

  for (int b = 0; b < num_branches; ++b) {
    HloComputation* br_computation = conditional->branch_computation(b);
    TF_RET_CHECK(ShapeUtil::Equal(conditional->shape(),
                                  br_computation->root_instruction()->shape()))
        << "Shape of conditional should be same as the shape of the " << b
        << "th branch computation; got: "
        << ShapeUtil::HumanString(conditional->shape()) << " and "
        << ShapeUtil::HumanString(br_computation->root_instruction()->shape());
  }

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(conditional));

  if (branch_index->shape().element_type() == PRED) {
    // Emit an if-else to LLVM:
    //   if (pred)
    //     cond_result = true_computation(true_operand)
    //   else
    //     cond_result = false_computation(false_operand)
    llvm::LoadInst* pred_value = Load(
        GetIrArrayFor(branch_index).GetBasePointer(), "load_predicate_value");
    llvm::Value* pred_cond =
        ICmpNE(pred_value,
               llvm::ConstantInt::get(
                   llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0),
               "boolean_predicate");
    llvm_ir::LlvmIfData if_data =
        llvm_ir::EmitIfThenElse(pred_cond, "conditional", &b_);

    SetToFirstInsertPoint(if_data.true_block, &b_);
    EmitGlobalCall(*conditional->branch_computation(0),
                   IrName(conditional, "_true"));

    SetToFirstInsertPoint(if_data.false_block, &b_);
    EmitGlobalCall(*conditional->branch_computation(1),
                   IrName(conditional, "_false"));

    SetToFirstInsertPoint(if_data.after_block, &b_);
    return Status::OK();
  }
  // We emit a switch statement to LLVM:
  // switch (branch_index) {
  //   default:
  //     result = branch_computations[num_branches-1](operands[num_branches-1]);
  //     break;
  //   case 0:
  //     result = branch_computations[0](operands[0]); break;
  //   case 1:
  //     result = branch_computations[1](operands[1]); break;
  //   ...
  //   case [[num_branches-2]]:
  //     result = branch_computations[num_branches-2](operands[num_branches-2]);
  //     break;
  // }
  llvm::LoadInst* branch_index_value = Load(
      GetIrArrayFor(branch_index).GetBasePointer(), "load_branch_index_value");

  auto case_block = b_.GetInsertBlock();
  llvm::BasicBlock* after_block;
  // Add a terminator to the case block, if necessary.
  if (case_block->getTerminator() == nullptr) {
    after_block = llvm_ir::CreateBasicBlock(nullptr, "case-after", &b_);
    b_.SetInsertPoint(case_block);
    b_.CreateBr(after_block);
  } else {
    after_block =
        case_block->splitBasicBlock(b_.GetInsertPoint(), "case-after");
  }
  // Our basic block should now end with an unconditional branch.  Remove it;
  // we're going to replace it with a switch based branch.
  case_block->getTerminator()->eraseFromParent();

  // Lower the default branch computation.
  auto default_block = llvm_ir::CreateBasicBlock(nullptr, "case-default", &b_);
  b_.SetInsertPoint(default_block);
  EmitGlobalCall(*conditional->branch_computation(num_branches - 1),
                 IrName(conditional, "_default"));
  b_.CreateBr(after_block);

  // Prepare the switch (branch_index) { ... } instruction.
  b_.SetInsertPoint(case_block);
  llvm::SwitchInst* case_inst =
      b_.CreateSwitch(branch_index_value, default_block, num_branches - 1);
  // Lower each branch's computation.
  for (int b = 0; b < num_branches - 1; ++b) {  // last branch is default
    // Lower the case b: { ... ; break; } computation.
    auto branch_block =
        llvm_ir::CreateBasicBlock(nullptr, absl::StrCat("case-branch", b), &b_);
    b_.SetInsertPoint(branch_block);
    EmitGlobalCall(*conditional->branch_computation(b),
                   IrName(conditional, absl::StrCat("_branch", b)));
    b_.CreateBr(after_block);
    case_inst->addCase(b_.getInt32(b), branch_block);
  }

  SetToFirstInsertPoint(after_block, &b_);
  return Status::OK();
}

Status IrEmitter::HandleAfterAll(HloInstruction* after_all) {
  TF_RET_CHECK(ByteSizeOf(after_all->shape()) == 0);
  // No code to generate, but we need to emit an address for book-keeping.
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(after_all));
  return Status::OK();
}

Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  // AddDedendency just forwards its zero-th operand.
  emitted_value_[add_dependency] =
      GetEmittedValueFor(add_dependency->operand(0));
  return Status::OK();
}

Status IrEmitter::HandleRng(HloInstruction* rng) {
  return Unimplemented("Rng should be expanded for CPU.");
}

Status IrEmitter::HandleRngGetAndUpdateState(HloInstruction* rng_state) {
  VLOG(2) << "RngGetAndUpdateState: " << rng_state->ToString();
  llvm::Value* old_state = llvm_ir::RngGetAndUpdateState(
      Cast<HloRngGetAndUpdateStateInstruction>(rng_state)->delta(), module_,
      &b_);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(rng_state));
  llvm::Value* address = GetEmittedValueFor(rng_state);

  // The buffer has an array type while the value has a i128. Cast the
  // buffer to i128 type to store the value.
  address = BitCast(address, llvm::PointerType::get(
                                 old_state->getType()->getScalarType(),
                                 address->getType()->getPointerAddressSpace()));
  llvm::StoreInst* store = Store(old_state, address);
  store->setAlignment(llvm::Align(IrEmitter::MinimumAlignmentForPrimitiveType(
      rng_state->shape().element_type())));

  return Status::OK();
}

Status IrEmitter::FinishVisit(HloInstruction* root) {
  // When this method is called, we should have already emitted an IR value for
  // the root (return) op. The IR value holds the address of the buffer holding
  // the value. If the root is a constant or parameter, we perform a memcpy from
  // this buffer to the retval buffer of the computation. Otherwise, there's
  // nothing to do since the result was already written directly into the output
  // buffer.
  VLOG(2) << "FinishVisit root: " << root->ToString();
  if (root->opcode() == HloOpcode::kOutfeed) {
    VLOG(2) << "  outfeed with value: "
            << llvm_ir::DumpToString(*GetEmittedValueFor(root->operand(0)));
  } else {
    VLOG(2) << "  value: " << llvm_ir::DumpToString(*GetEmittedValueFor(root));
  }

  auto record_complete_computation = [&](llvm::Value* prof_counter) {
    if (prof_counter) {
      profiling_state_.RecordCompleteComputation(&b_, prof_counter);
    }
  };

  // For the entry computation this increment is cumulative of embedded
  // computations since it includes cycles spent in computations invoked by
  // While, Call etc.
  record_complete_computation(GetProfileCounterFor(*root->parent()));
  return Status::OK();
}

template <typename T>
llvm::Value* IrEmitter::GetProfileCounterCommon(
    const T& hlo,
    const std::unordered_map<const T*, int64>& profile_index_map) {
  auto it = profile_index_map.find(&hlo);
  if (it == profile_index_map.end()) {
    return nullptr;
  }

  int64 prof_counter_idx = it->second;
  string counter_name = IrName("prof_counter", hlo.name());
  return GEP(GetProfileCountersArgument(), b_.getInt64(prof_counter_idx),
             counter_name);
}

llvm::Value* IrEmitter::GetProfileCounterFor(
    const HloInstruction& instruction) {
  return GetProfileCounterCommon<HloInstruction>(instruction,
                                                 instruction_to_profile_idx_);
}

llvm::Value* IrEmitter::GetProfileCounterFor(
    const HloComputation& computation) {
  return GetProfileCounterCommon<HloComputation>(computation,
                                                 computation_to_profile_idx_);
}

void IrEmitter::ProfilingState::UpdateProfileCounter(llvm::IRBuilder<>* b,
                                                     llvm::Value* prof_counter,
                                                     llvm::Value* cycle_end,
                                                     llvm::Value* cycle_start) {
  auto* cycle_diff = b->CreateSub(cycle_end, cycle_start);
  llvm::LoadInst* old_cycle_count =
      b->CreateLoad(prof_counter, "old_cycle_count");
  auto* new_cycle_count =
      b->CreateAdd(cycle_diff, old_cycle_count, "new_cycle_count");
  b->CreateStore(new_cycle_count, prof_counter);
}

llvm::Value* IrEmitter::ProfilingState::ReadCycleCounter(llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  if (!use_rdtscp_) {
    llvm::Function* func_llvm_readcyclecounter =
        llvm::Intrinsic::getDeclaration(module,
                                        llvm::Intrinsic::readcyclecounter);
    return b->CreateCall(func_llvm_readcyclecounter);
  }
  llvm::Function* func_llvm_x86_rdtscp =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::x86_rdtscp);
  llvm::Value* rdtscp_call = b->CreateCall(func_llvm_x86_rdtscp);
  return b->CreateExtractValue(rdtscp_call, {0});
}

void IrEmitter::ProfilingState::RecordCycleStart(llvm::IRBuilder<>* b,
                                                 HloInstruction* hlo) {
  auto* cycle_start = ReadCycleCounter(b);
  cycle_start->setName(IrName(hlo, "cycle_start"));
  cycle_starts_[hlo] = cycle_start;
  if (first_read_cycle_start_ == nullptr) {
    first_read_cycle_start_ = cycle_start;
  }
}

void IrEmitter::ProfilingState::RecordCycleDelta(llvm::IRBuilder<>* b,
                                                 HloInstruction* hlo,
                                                 llvm::Value* prof_counter) {
  auto* cycle_end = ReadCycleCounter(b);
  cycle_end->setName(IrName(hlo, "cycle_end"));
  auto* cycle_start = cycle_starts_[hlo];
  UpdateProfileCounter(b, prof_counter, cycle_end, cycle_start);
  last_read_cycle_end_ = cycle_end;
}

void IrEmitter::ProfilingState::RecordCompleteComputation(
    llvm::IRBuilder<>* b, llvm::Value* prof_counter) {
  if (last_read_cycle_end_ && first_read_cycle_start_) {
    UpdateProfileCounter(b, prof_counter, last_read_cycle_end_,
                         first_read_cycle_start_);
  }
}

void IrEmitter::TracingState::EmitTracingStart(llvm::IRBuilder<>* b,
                                               HloInstruction* hlo,
                                               llvm::Value* run_options) {
  if (!enabled_) {
    return;
  }

  llvm::Type* int8_ptr_type = b->getInt8Ty()->getPointerTo();
  llvm::Type* void_ptr_type =
      int8_ptr_type;  // LLVM does not have a void*, we use an int8* instead.
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(b->getInt64Ty(), {void_ptr_type, int8_ptr_type},
                              /*isVarArg=*/false);

  llvm::Function* function = b->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();
  const char* fn_name = runtime::kTracingStartSymbolName;
  llvm::FunctionCallee trace_func =
      module->getOrInsertFunction(fn_name, fn_type);
  if (auto* fn = llvm::dyn_cast<llvm::Function>(trace_func.getCallee())) {
    fn->setCallingConv(llvm::CallingConv::C);
    fn->setDoesNotThrow();
    fn->setOnlyAccessesArgMemory();
  }
  auto* hlo_name = b->CreateGlobalStringPtr(hlo->name());
  auto* activity_id =
      b->CreateCall(trace_func, {b->CreateBitCast(run_options, void_ptr_type),
                                 b->CreateBitCast(hlo_name, int8_ptr_type)});
  activity_id->setName(IrName(hlo, "activity_id"));
  activity_ids_[hlo] = activity_id;
}

void IrEmitter::TracingState::EmitTracingEnd(llvm::IRBuilder<>* b,
                                             HloInstruction* hlo,
                                             llvm::Value* run_options) {
  if (!enabled_) {
    return;
  }

  llvm::Type* void_ptr_type =
      b->getInt8Ty()->getPointerTo();  // LLVM does not have a void*, we use an
                                       // int8* instead.
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(b->getVoidTy(), {void_ptr_type, b->getInt64Ty()},
                              /*isVarArg=*/false);

  llvm::Function* function = b->GetInsertBlock()->getParent();
  llvm::Module* module = function->getParent();
  const char* fn_name = runtime::kTracingEndSymbolName;
  llvm::FunctionCallee trace_func =
      module->getOrInsertFunction(fn_name, fn_type);
  if (auto* fn = llvm::dyn_cast<llvm::Function>(trace_func.getCallee())) {
    fn->setCallingConv(llvm::CallingConv::C);
    fn->setDoesNotThrow();
    fn->setOnlyAccessesArgMemory();
  }
  auto* activity_id = activity_ids_.at(hlo);
  b->CreateCall(trace_func,
                {b->CreateBitCast(run_options, void_ptr_type), activity_id});
}

Status IrEmitter::Preprocess(HloInstruction* hlo) {
  VLOG(3) << "Visiting: " << hlo->ToString();
  if (instruction_to_profile_idx_.count(hlo)) {
    // Only trace the same HLOs that the profiler does.
    tracing_state_.EmitTracingStart(&b_, hlo,
                                    GetExecutableRunOptionsArgument());
    profiling_state_.RecordCycleStart(&b_, hlo);
  }
  return Status::OK();
}

Status IrEmitter::Postprocess(HloInstruction* hlo) {
  if (auto* prof_counter = GetProfileCounterFor(*hlo)) {
    profiling_state_.RecordCycleDelta(&b_, hlo, prof_counter);
  }
  // Only trace the same HLOs that the profiler does.
  if (instruction_to_profile_idx_.count(hlo)) {
    tracing_state_.EmitTracingEnd(&b_, hlo, GetExecutableRunOptionsArgument());
  }
  return Status::OK();
}

llvm_ir::IrArray IrEmitter::GetIrArrayFor(const HloInstruction* hlo) {
  llvm::Value* value_for_op = GetEmittedValueFor(hlo);

  llvm_ir::IrArray array(value_for_op, hlo->shape());
  AddAliasingInformationToIrArray(*hlo, &array);
  return array;
}

std::vector<llvm_ir::IrArray> IrEmitter::GetIrArraysForOperandsOf(
    const HloInstruction* hlo) {
  std::vector<llvm_ir::IrArray> arrays;
  std::transform(
      hlo->operands().begin(), hlo->operands().end(),
      std::back_inserter(arrays),
      [&](const HloInstruction* operand) { return GetIrArrayFor(operand); });
  return arrays;
}

llvm::Value* IrEmitter::GetEmittedValueFor(const HloInstruction* hlo) {
  auto it = emitted_value_.find(hlo);
  if (it == emitted_value_.end()) {
    LOG(FATAL) << "could not find emitted value for: " << hlo->ToString();
  }
  return it->second;
}

llvm::Type* IrEmitter::IrShapeType(const Shape& shape) {
  return llvm_ir::ShapeToIrType(shape, module_);
}

llvm::Value* IrEmitter::GetProfileCountersArgument() {
  return compute_function_->profile_counters_arg();
}

llvm::Value* IrEmitter::GetBufferTableArgument() {
  return compute_function_->buffer_table_arg();
}

llvm::Value* IrEmitter::GetExecutableRunOptionsArgument() {
  return compute_function_->exec_run_options_arg();
}

llvm::Value* IrEmitter::EmitThreadLocalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {
  const BufferAllocation& allocation = *slice.allocation();
  llvm::Value* tempbuf_address = [&]() -> llvm::Value* {
    auto param_it =
        computation_parameter_allocations_.find(slice.allocation()->index());
    if (param_it != computation_parameter_allocations_.end()) {
      int64 param_number = param_it->second;
      // We have to access the parameter at offset param_number in the params
      // array. The code generated here is equivalent to this C code:
      //
      //   i8* param_address_untyped = params[param_number];
      //   Param* param_address_typed = (Param*)param_address_untyped;
      //
      // Where Param is the actual element type of the underlying buffer (for
      // example, float for an XLA F32 element type).
      llvm::Value* params = compute_function_->parameters_arg();
      llvm::Value* param_address_offset =
          llvm_ir::EmitBufferIndexingGEP(params, param_number, &b_);
      llvm::LoadInst* param_address_untyped = Load(param_address_offset);

      if (!target_shape.IsOpaque()) {
        AttachAlignmentMetadataForLoad(param_address_untyped, target_shape);
        AttachDereferenceableMetadataForLoad(param_address_untyped,
                                             target_shape);
      }
      return param_address_untyped;
    }

    // Thread-local allocations should only be assigned a single buffer.
    const auto& assigned_buffers = allocation.assigned_buffers();
    CHECK_EQ(1, assigned_buffers.size());
    const Shape& shape = assigned_buffers.begin()->first->shape();

    std::pair<llvm::Function*, BufferAllocation::Slice> key = {
        compute_function_->function(), slice};
    auto buf_it = thread_local_buffers_.find(key);
    if (buf_it == thread_local_buffers_.end()) {
      llvm::Value* buffer = llvm_ir::EmitAllocaAtFunctionEntry(
          IrShapeType(shape), absl::StrCat("thread_local", slice.ToString()),
          &b_, MinimumAlignmentForShape(target_shape));
      auto it_inserted_pair = thread_local_buffers_.insert({key, buffer});
      CHECK(it_inserted_pair.second);
      buf_it = it_inserted_pair.first;
    }
    return buf_it->second;
  }();
  return BitCast(tempbuf_address, IrShapeType(target_shape)->getPointerTo());
}

llvm::Value* IrEmitter::EmitGlobalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {
  const BufferAllocation& allocation = *slice.allocation();
  llvm::Value* tempbuf_address_ptr = llvm_ir::EmitBufferIndexingGEP(
      GetBufferTableArgument(), slice.index(), &b_);
  llvm::LoadInst* tempbuf_address_base = Load(tempbuf_address_ptr);
  if (hlo_module_config_.debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    tempbuf_address_base->setMetadata(
        llvm::LLVMContext::MD_invariant_load,
        llvm::MDNode::get(tempbuf_address_base->getContext(), /*MDs=*/{}));
  }
  AttachAlignmentMetadataForLoad(tempbuf_address_base, allocation.size());
  AttachDereferenceableMetadataForLoad(tempbuf_address_base, allocation.size());

  llvm::Value* tempbuf_address_untyped = tempbuf_address_base;
  if (slice.offset() > 0) {
    // Adjust the address to account for the slice offset.
    tempbuf_address_untyped =
        InBoundsGEP(tempbuf_address_base, b_.getInt64(slice.offset()));
  }
  return BitCast(tempbuf_address_untyped,
                 IrShapeType(target_shape)->getPointerTo());
}

llvm::Value* IrEmitter::EmitBufferPointer(const BufferAllocation::Slice& slice,
                                          const Shape& target_shape) {
  if (slice.allocation()->is_thread_local()) {
    return EmitThreadLocalBufferPointer(slice, target_shape);
  } else if (slice.allocation()->is_constant()) {
    return BitCast(
        FindOrDie(constant_buffer_to_global_, slice.allocation()->index()),
        IrShapeType(target_shape)->getPointerTo());
  } else {
    return EmitGlobalBufferPointer(slice, target_shape);
  }
}

Status IrEmitter::EmitTargetAddressForOp(const HloInstruction* op) {
  const Shape& target_shape = op->shape();
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                      assignment_.GetUniqueTopLevelSlice(op));
  llvm::Value* addr = EmitBufferPointer(slice, target_shape);
  addr->setName(IrName(op));
  emitted_value_[op] = addr;
  return Status::OK();
}

Status IrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op,
    const llvm_ir::ElementGenerator& element_generator) {
  return EmitTargetElementLoop(target_op, /*desc=*/"", element_generator);
}

Status IrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op, absl::string_view desc,
    const llvm_ir::ElementGenerator& element_generator) {
  VLOG(2) << "EmitTargetElementLoop: " << target_op->ToString();

  const Shape& target_shape = target_op->shape();
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(target_op));
  llvm_ir::IrArray target_array = GetIrArrayFor(target_op);

  if (target_shape.IsTuple() && (target_op->opcode() == HloOpcode::kFusion ||
                                 target_op->opcode() == HloOpcode::kReduce)) {
    // For multiple outputs fusion, we need to emit each operand and the root.
    TF_RET_CHECK(num_dynamic_loop_bounds_ == 0);
    std::vector<llvm_ir::IrArray> output_arrays;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(target_shape); ++i) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          assignment_.GetUniqueSlice(target_op, {i}));
      const Shape& element_shape = ShapeUtil::GetSubshape(target_shape, {i});
      llvm::Value* op_target_address = EmitBufferPointer(slice, element_shape);
      output_arrays.push_back(
          llvm_ir::IrArray(op_target_address, element_shape));
    }
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, output_arrays, &b_)
            .EmitLoop(IrName(target_op)));

    std::vector<llvm::Value*> tuple_operand_ptrs;
    for (int64 i = 0; i < output_arrays.size(); ++i) {
      tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
    }
    llvm_ir::EmitTuple(target_array, tuple_operand_ptrs, &b_);

  } else {
    if (ShouldEmitParallelLoopFor(*target_op)) {
      // Emit code to read dynamic loop bounds from compute function argument.
      std::vector<std::pair<llvm::Value*, llvm::Value*>> dynamic_loop_bounds =
          compute_function_->GetDynamicLoopBounds();
      // Emit parallel loop with dynamic loop bounds for most-major dimensions.
      TF_RETURN_IF_ERROR(ParallelLoopEmitter(element_generator, target_array,
                                             &dynamic_loop_bounds, &b_)
                             .EmitLoop(IrName(target_op)));
    } else {
      TF_RETURN_IF_ERROR(
          llvm_ir::LoopEmitter(element_generator, target_array, &b_)
              .EmitLoop(IrName(target_op)));
    }
  }
  return Status::OK();
}

Status IrEmitter::EmitMemcpy(const HloInstruction& source,
                             const HloInstruction& destination) {
  llvm::Value* source_value = GetEmittedValueFor(&source);
  llvm::Value* destination_value = GetEmittedValueFor(&destination);
  int64 source_size = ByteSizeOf(source.shape());
  // TODO(b/63762267): Be more aggressive about specifying alignment.
  MemCpy(destination_value, /*DstAlign=*/llvm::Align(1), source_value,
         /*SrcAlign=*/llvm::Align(1), source_size);
  return Status::OK();
}

Status IrEmitter::ElementTypesSameAndSupported(
    const HloInstruction& instruction,
    absl::Span<const HloInstruction* const> operands,
    absl::Span<const PrimitiveType> supported_types) {
  for (auto operand : operands) {
    TF_RET_CHECK(
        ShapeUtil::SameElementType(operands[0]->shape(), operand->shape()));
  }

  TF_RET_CHECK(!operands.empty());
  PrimitiveType primitive_type = operands[0]->shape().element_type();
  if (!absl::c_linear_search(supported_types, primitive_type)) {
    return Unimplemented("unsupported operand type %s in op %s",
                         PrimitiveType_Name(primitive_type),
                         HloOpcodeString(instruction.opcode()));
  }
  return Status::OK();
}

Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArrayFor(operand).EmitReadArrayElement(index, &b_);
    };
  }
  CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
  return EmitTargetElementLoop(
      hlo, elemental_emitter.MakeElementGenerator(hlo, operand_to_generator));
}

llvm::Value* IrEmitter::EmitScalarReturningThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name) {
  std::vector<llvm::Value*> return_value =
      EmitThreadLocalCall(callee, parameters, name);
  CHECK_EQ(return_value.size(), 1);
  return return_value[0];
}

std::vector<llvm::Value*> IrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name) {
  CHECK(absl::c_binary_search(thread_local_computations_, &callee));
  const Shape& return_shape = callee.root_instruction()->shape();
  bool is_scalar_return = ShapeUtil::IsScalar(return_shape);
  bool is_tuple_of_scalars_return =
      return_shape.IsTuple() &&
      absl::c_all_of(return_shape.tuple_shapes(), [&](const Shape& shape) {
        return ShapeUtil::IsScalar(shape);
      });
  CHECK(is_scalar_return || is_tuple_of_scalars_return);

  std::vector<llvm::Value*> parameter_addrs;
  for (llvm::Value* parameter : parameters) {
    CHECK(!parameter->getType()->isPointerTy());
    llvm::Value* parameter_addr = llvm_ir::EmitAllocaAtFunctionEntry(
        parameter->getType(), "arg_addr", &b_);
    Store(parameter, parameter_addr);
    parameter_addrs.push_back(parameter_addr);
  }

  llvm::Type* return_value_buffer_type =
      llvm_ir::ShapeToIrType(return_shape, module_);
  std::string retval_alloca_name = absl::StrCat(name, "_return_value_addr");
  int retval_alignment =
      is_scalar_return
          ? MinimumAlignmentForPrimitiveType(return_shape.element_type())
          : 0;
  llvm::Value* return_value_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      return_value_buffer_type, retval_alloca_name, &b_, retval_alignment);

  std::vector<llvm::Value*> allocas_for_returned_scalars;
  if (is_scalar_return) {
    allocas_for_returned_scalars.push_back(return_value_buffer);
  } else {
    constexpr int max_tuple_size = 1000;
    CHECK_LT(return_shape.tuple_shapes_size(), max_tuple_size)
        << "Multivalue function can not return more than 1000 elements to avoid"
        << " stack smashing";
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, &b_);
    llvm_ir::IrArray tuple_array(return_value_buffer, return_shape);

    EmitTuple(tuple_array, allocas_for_returned_scalars, &b_);
  }

  Call(FindOrDie(emitted_functions_, &callee),
       GetArrayFunctionCallArguments(
           parameter_addrs, &b_, name,
           /*return_value_buffer=*/return_value_buffer,
           /*exec_run_options_arg=*/GetExecutableRunOptionsArgument(),
           /*buffer_table_arg=*/
           llvm::Constant::getNullValue(b_.getInt8PtrTy()->getPointerTo()),
           /*profile_counters_arg=*/GetProfileCountersArgument()));

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    returned_scalars.push_back(Load(addr));
  }
  return returned_scalars;
}

void IrEmitter::EmitGlobalCall(const HloComputation& callee,
                               absl::string_view name) {
  CHECK(absl::c_binary_search(global_computations_, &callee));

  Call(FindOrDie(emitted_functions_, &callee),
       GetArrayFunctionCallArguments(
           /*parameter_addresses=*/{}, &b_, name,
           /*return_value_buffer=*/
           llvm::Constant::getNullValue(b_.getInt8PtrTy()),
           /*exec_run_options_arg=*/GetExecutableRunOptionsArgument(),
           /*buffer_table_arg=*/GetBufferTableArgument(),
           /*profile_counters_arg=*/GetProfileCountersArgument()));
}

llvm::Value* IrEmitter::GetBufferForGlobalCallReturnValue(
    const HloComputation& callee) {
  const HloInstruction* root_inst = callee.root_instruction();
  if (root_inst->opcode() == HloOpcode::kOutfeed) {
    return llvm::Constant::getNullValue(b_.getInt8PtrTy());
  }

  const BufferAllocation::Slice root_buffer =
      assignment_.GetUniqueTopLevelSlice(root_inst).ValueOrDie();
  return EmitBufferPointer(root_buffer, root_inst->shape());
}

}  // namespace cpu
}  // namespace xla
