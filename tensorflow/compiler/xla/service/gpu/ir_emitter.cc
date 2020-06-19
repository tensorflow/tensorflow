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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"

#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"

// Convenient function to cast the provided llvm::Value* using IRBuilder
// to default address space. This is useful in particular for generating
// IR for AMDGPU target, as its kernel variables are in address space 5
// instead of the default address space.
static llvm::Value* AddrCastToDefault(llvm::Value* arg, llvm::IRBuilder<>& b) {
  llvm::Type* arg_type = arg->getType();
  CHECK(arg_type->isPointerTy());
  if (arg_type->getPointerAddressSpace() != 0) {
    llvm::Type* generic_arg_type =
        arg_type->getPointerElementType()->getPointerTo(0);
    llvm::Value* addrspacecast_arg =
        b.CreateAddrSpaceCast(arg, generic_arg_type);
    return addrspacecast_arg;
  }
  return arg;
}

namespace xla {

using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace gpu {

IrEmitter::IrEmitter(const HloModuleConfig& hlo_module_config,
                     IrEmitterContext* ir_emitter_context, bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext()),
      bindings_(ir_emitter_context->hlo_module(),
                &ir_emitter_context->buffer_assignment(), &b_, module_,
                is_nested),
      hlo_module_config_(hlo_module_config) {
}

Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArray(*operand, *hlo).EmitReadArrayElement(index, &b_);
    };
  }
  return EmitTargetElementLoop(
      *hlo, GpuElementalIrEmitter(hlo_module_config_, module_, &b_,
                                  GetNestedComputer())
                .MakeElementGenerator(hlo, operand_to_generator));
}

Status IrEmitter::HandleConstant(HloInstruction* constant) {
  return Status::OK();
}

Status IrEmitter::HandleBitcast(HloInstruction* bitcast) {
  VLOG(2) << "HandleBitcast: " << bitcast->ToString();
  const HloInstruction* operand = bitcast->operand(0);
  // Bitcast is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*bitcast, GetBasePointer(*operand));
  }
  return Status::OK();
}

Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  VLOG(2) << "HandleAddDependency: " << add_dependency->ToString();
  const HloInstruction* operand = add_dependency->operand(0);
  // Add_Dependency is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  }
  return Status::OK();
}

Status IrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->operand(0);
  CHECK(bindings_.BoundToIrValue(*operand));
  bindings_.BindHloToIrValue(
      *get_tuple_element,
      llvm_ir::EmitGetTupleElement(
          get_tuple_element->shape(), get_tuple_element->tuple_index(),
          // TODO(b/26344050): tighten the alignment here
          // based on the real element type.
          /*alignment=*/1, GetBasePointer(*operand), &b_));
  return Status::OK();
}

Status IrEmitter::HandleSend(HloInstruction*) {
  return Unimplemented("Send is not implemented on GPU");
}

Status IrEmitter::HandleSendDone(HloInstruction*) {
  return Unimplemented("Send-Done is not implemented on GPU");
}

Status IrEmitter::HandleRecv(HloInstruction*) {
  return Unimplemented("Recv is not implemented on GPU");
}

Status IrEmitter::HandleRecvDone(HloInstruction*) {
  return Unimplemented("Recv-done is not implemented on GPU");
}

Status IrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter is not implemented on GPUs.");
}

Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  std::vector<llvm::Value*> base_ptrs;
  for (const HloInstruction* operand : tuple->operands()) {
    base_ptrs.push_back(GetBasePointer(*operand));
  }
  llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
  return Status::OK();
}

Status IrEmitter::EmitCallToNestedComputation(
    const HloComputation& nested_computation,
    absl::Span<llvm::Value* const> operands, llvm::Value* output) {
  TF_RET_CHECK(nested_computation.num_parameters() > 0);
  llvm::Function*& emitted_function =
      computation_to_ir_function_[&nested_computation];
  if (emitted_function == nullptr) {
    IrEmitterNested ir_emitter_nested(hlo_module_config_, nested_computation,
                                      ir_emitter_context_);
    TF_RETURN_IF_ERROR(ir_emitter_nested.CodegenNestedComputation());
    emitted_function = ir_emitter_nested.GetEmittedFunction();
  }

  // Operands are in default address space for non-AMDGPU target.
  // However for AMDGPU target, addrspacecast alloca variables from
  // addrspace 5 to addrspace 0 is needed.
  std::vector<llvm::Value*> arguments;
  absl::c_transform(
      operands, std::back_inserter(arguments),
      [this](llvm::Value* arg) { return AddrCastToDefault(arg, b_); });

  llvm::Value* casted_output = AddrCastToDefault(output, b_);
  arguments.push_back(casted_output);

  // It is not required to do address space cast because TempBufferBase
  // is always in addrspace 0.
  arguments.push_back(bindings_.GetTempBufferBase());
  Call(emitted_function, arguments);

  return Status::OK();
}

bool IrEmitter::MaybeEmitDirectAtomicOperation(
    const HloComputation& computation, llvm::Value* output_address,
    llvm::Value* source_address) {
  CHECK_EQ(2, computation.num_parameters());

  HloOpcode root_opcode = computation.root_instruction()->opcode();
  PrimitiveType element_type =
      computation.root_instruction()->shape().element_type();
  bool is_atomic_integral = element_type == S32 || element_type == U32 ||
                            element_type == S64 || element_type == U64;
  llvm::Value* source = Load(source_address, "source");

  // Just passing along RHS -> atomic store.
  if (computation.instruction_count() == 2 &&
      root_opcode == HloOpcode::kParameter &&
      (element_type == F32 || is_atomic_integral) &&
      computation.root_instruction()->parameter_number() == 1) {
    llvm::StoreInst* store = Store(source, output_address);
    store->setAtomic(llvm::AtomicOrdering::Unordered);
    // Derive a minimum alignment from the type. The optimizer can increase it
    // later.
    store->setAlignment(
        llvm::Align(ShapeUtil::ByteSizeOfPrimitiveType(element_type)));
    return true;
  }

  if (computation.instruction_count() != 3) {
    // We special-case only computations with one computing instruction for now.
    // Such computation has exactly three instructions given it has two
    // parameters.
    return false;
  }

  if (root_opcode == HloOpcode::kAdd) {
    llvm::Triple target_triple = llvm::Triple(module_->getTargetTriple());
    // NVPTX supports atomicAdd on F32 and integer types.
    if (target_triple.isNVPTX()) {
      // "atom.add.f64 requires sm_60 or higher."
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
      int cc_major = 0, cc_minor = 0;
      ir_emitter_context_->device_description().cuda_compute_capability(
          &cc_major, &cc_minor);

      bool f64_atomic_add_supported = cc_major >= 6;

      bool atomic_add_supported =
          element_type == F32 ||
          (f64_atomic_add_supported && element_type == F64);
      if (atomic_add_supported) {
        AtomicRMW(llvm::AtomicRMWInst::FAdd, output_address, source,
                  llvm::AtomicOrdering::SequentiallyConsistent);
        return true;
      }
    }

    if (is_atomic_integral) {
      // integral + integral
      AtomicRMW(llvm::AtomicRMWInst::Add, output_address, source,
                llvm::AtomicOrdering::SequentiallyConsistent);
      return true;
    }
  }

  // NVPTX supports atomicMax and atomicMin only on integer types.
  if (root_opcode == HloOpcode::kMaximum && is_atomic_integral) {
    // max(integral, integral)
    auto opcode = primitive_util::IsSignedIntegralType(element_type)
                      ? llvm::AtomicRMWInst::Max
                      : llvm::AtomicRMWInst::UMax;
    AtomicRMW(opcode, output_address, source,
              llvm::AtomicOrdering::SequentiallyConsistent);
    return true;
  }

  if (root_opcode == HloOpcode::kMinimum && is_atomic_integral) {
    // min(integral, integral)
    auto opcode = primitive_util::IsSignedIntegralType(element_type)
                      ? llvm::AtomicRMWInst::Min
                      : llvm::AtomicRMWInst::UMin;
    AtomicRMW(opcode, output_address, source,
              llvm::AtomicOrdering::SequentiallyConsistent);
    return true;
  }

  return false;
}

// Implements atomic binary operations using atomic compare-and-swap
// (atomicCAS) as follows:
//   1. Reads the value from the memory pointed to by output_address and
//     records it as old_output.
//   2. Uses old_output as one of the source operand to perform the binary
//     operation and stores the result in new_output.
//   3. Calls atomicCAS which implements compare-and-swap as an atomic
//     operation. In particular, atomicCAS reads the value from the memory
//     pointed to by output_address, and compares the value with old_output. If
//     the two values equal, new_output is written to the same memory location
//     and true is returned to indicate that the atomic operation succeeds.
//     Otherwise, the new value read from the memory is returned. In this case,
//     the new value is copied to old_output, and steps 2. and 3. are repeated
//     until atomicCAS succeeds.
//
// On Nvidia GPUs, atomicCAS can only operate on 32 bit and 64 bit integers. If
// the element type of the binary operation is 32 bits or 64 bits, the integer
// type of the same size is used for the atomicCAS operation. On the other hand,
// if the element type is smaller than 32 bits, int32 is used for the atomicCAS
// operation. In this case, atomicCAS reads and writes 32 bit values from
// the memory, which is larger than the memory size required by the original
// atomic binary operation. We mask off the last two bits of the output_address
// and use the result as an address to read the 32 bit values from the memory.
// This can avoid out of bound memory accesses if tensor buffers are 4 byte
// aligned and have a size of 4N, an assumption that the runtime can guarantee.
//
// The pseudo code is shown below. Variables *_address are pointers to a memory
// region with a size equal to the size of the atomicCAS operation, with the
// exception that new_output_address is a pointer to a memory region with a size
// equal to the element size of the binary operation.
//
//   element_size = sizeof(element_type);
//   atomic_size = max(32, element_size);
//   cas_new_output_address = alloca(atomic_size);
//   cas_old_output_address = alloca(atomic_size);
//   if (atomic_size != element_size) {
//     atomic_address = output_address & ((int64)(-4));
//     new_output_address = cas_new_output_address + (output_address & 3);
//   } else {
//     atomic_address = output_address;
//     new_output_address = cas_new_output_address;
//   }
//
//   *cas_old_output_address = *atomic_address;
//   do {
//     *cas_new_output_address = *cas_old_output_address;
//     *new_output_address = operation(*new_output_address, *source_address);
//     (*cas_old_output_address, success) =
//       atomicCAS(atomic_address, *cas_old_output_address,
//       *cas_new_output_address);
//   } while (!success);
//
Status IrEmitter::EmitAtomicOperationUsingCAS(const HloComputation& computation,
                                              llvm::Value* output_address,
                                              llvm::Value* source_address) {
  llvm::PointerType* output_address_type =
      llvm::dyn_cast<llvm::PointerType>(output_address->getType());
  CHECK_NE(output_address_type, nullptr);

  // element_type is the data type for the binary operation.
  llvm::Type* element_type = output_address_type->getPointerElementType();
  int element_size = llvm_ir::GetSizeInBits(element_type);

  int atomic_size = (element_size < 32) ? 32 : element_size;
  llvm::Type* atomic_type = b_.getIntNTy(atomic_size);
  llvm::Type* atomic_address_type =
      atomic_type->getPointerTo(output_address_type->getPointerAddressSpace());

  // cas_old_output_address and cas_new_output_address point to the scratch
  // memory where we store the old and new values for the repeated atomicCAS
  // operations.
  llvm::Value* cas_old_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_old_output_address", &b_);
  llvm::Value* cas_new_output_address = llvm_ir::EmitAllocaAtFunctionEntry(
      atomic_type, "cas_new_output_address", &b_);

  // Emit preparation code to the preheader.
  llvm::BasicBlock* loop_preheader_bb = b_.GetInsertBlock();

  llvm::Value* atomic_memory_address;
  // binop_output_address points to the scratch memory that stores the
  // result of the binary operation.
  llvm::Value* binop_output_address;
  if (element_size < 32) {
    // Assume the element size is an integer number of bytes.
    CHECK_EQ((element_size % sizeof(char)), 0);
    llvm::Type* address_int_type =
        module_->getDataLayout().getIntPtrType(output_address_type);
    atomic_memory_address = PtrToInt(output_address, address_int_type);
    llvm::Value* mask = llvm::ConstantInt::get(address_int_type, 3);
    llvm::Value* offset = And(atomic_memory_address, mask);
    mask = llvm::ConstantInt::get(address_int_type, -4);
    atomic_memory_address = And(atomic_memory_address, mask);
    atomic_memory_address =
        IntToPtr(atomic_memory_address, atomic_address_type);
    binop_output_address =
        Add(PtrToInt(cas_new_output_address, address_int_type), offset);
    binop_output_address = IntToPtr(
        binop_output_address,
        llvm::PointerType::get(
            element_type,
            cas_new_output_address->getType()->getPointerAddressSpace()));
  } else {
    atomic_memory_address = b_.CreatePointerBitCastOrAddrSpaceCast(
        output_address, atomic_address_type);
    binop_output_address = b_.CreatePointerBitCastOrAddrSpaceCast(
        cas_new_output_address,
        llvm::PointerType::get(
            element_type,
            cas_new_output_address->getType()->getPointerAddressSpace()));
  }

  // Use the value from the memory that atomicCAS operates on to initialize
  // cas_old_output.
  llvm::Value* cas_old_output = Load(atomic_memory_address, "cas_old_output");
  Store(cas_old_output, cas_old_output_address);

  llvm::BasicBlock* loop_exit_bb = loop_preheader_bb->splitBasicBlock(
      b_.GetInsertPoint(), "atomic_op_loop_exit");
  llvm::BasicBlock* loop_body_bb = llvm::BasicBlock::Create(
      b_.getContext(), "atomic_op_loop_body", b_.GetInsertBlock()->getParent());
  b_.SetInsertPoint(loop_body_bb);
  // Change preheader's successor from loop_exit_bb to loop_body_bb.
  loop_preheader_bb->getTerminator()->setSuccessor(0, loop_body_bb);

  // Emit the body of the loop that repeatedly invokes atomicCAS.
  //
  // Use cas_old_output to initialize cas_new_output.
  cas_old_output = Load(cas_old_output_address, "cas_old_output");
  Store(cas_old_output, cas_new_output_address);
  // Emits code to calculate new_output = operation(old_output, source);
  TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
      computation, {binop_output_address, source_address},
      binop_output_address));

  llvm::Value* cas_new_output = Load(cas_new_output_address, "cas_new_output");

  // Emit code to perform the atomicCAS operation
  // (cas_old_output, success) = atomicCAS(memory_address, cas_old_output,
  //                                       cas_new_output);
  llvm::Value* ret_value =
      AtomicCmpXchg(atomic_memory_address, cas_old_output, cas_new_output,
                    llvm::AtomicOrdering::SequentiallyConsistent,
                    llvm::AtomicOrdering::SequentiallyConsistent);

  // Extract the memory value returned from atomicCAS and store it as
  // cas_old_output.
  Store(ExtractValue(ret_value, 0, "cas_old_output"), cas_old_output_address);
  // Extract the success bit returned from atomicCAS and generate a
  // conditional branch on the success bit.
  CondBr(ExtractValue(ret_value, 1, "success"), loop_exit_bb, loop_body_bb);

  // Set the insertion point to the exit basic block so that the caller of
  // this method can continue emitting code to the right place.
  SetToFirstInsertPoint(loop_exit_bb, &b_);
  return Status::OK();
}

Status IrEmitter::EmitAtomicOperationForNestedComputation(
    const HloComputation& computation, llvm::Value* output_address,
    llvm::Value* source_address) {
  if (computation.num_parameters() != 2) {
    // TODO(b/30258929): We only accept binary computations so far.
    return Unimplemented(
        "We only support atomic functions with exactly two parameters, but "
        "computation %s has %d.",
        computation.name(), computation.num_parameters());
  }

  if (MaybeEmitDirectAtomicOperation(computation, output_address,
                                     source_address)) {
    return Status::OK();
  }

  return EmitAtomicOperationUsingCAS(computation, output_address,
                                     source_address);
}

Status IrEmitter::HandleSelect(HloInstruction* select) {
  auto pred = select->operand(0);
  TF_RET_CHECK(pred->shape().element_type() == PRED);
  // We must not call the subclass `DefaultAction` method, lest its
  // `HandleSelect` call `IrEmitter::HandleSelect` and its `DefaultAction`
  // assume no handler has already been called.
  return IrEmitter::DefaultAction(select);
}

Status IrEmitter::HandleTupleSelect(HloInstruction* tuple_select) {
  auto pred = tuple_select->operand(0);
  auto on_true = tuple_select->operand(1);
  auto on_false = tuple_select->operand(2);
  TF_RET_CHECK(pred->shape().element_type() == PRED);
  TF_RET_CHECK(ShapeUtil::IsScalar(pred->shape()));
  TF_RET_CHECK(tuple_select->shape().IsTuple());
  llvm_ir::EmitTupleSelect(GetIrArray(*tuple_select, *tuple_select),
                           GetIrArray(*pred, *tuple_select),
                           GetBasePointer(*on_true), GetBasePointer(*on_false),
                           &b_);
  return Status::OK();
}

namespace {
llvm::Value* Real(llvm::Value* x, llvm::IRBuilder<>* b) {
  return b->CreateExtractValue(x, {0});
}

llvm::Value* Imag(llvm::Value* x, llvm::IRBuilder<>* b) {
  return b->CreateExtractValue(x, {1});
}

std::pair<llvm::Value*, llvm::Value*> MultiplyComplex(llvm::Value* lhs_value,
                                                      llvm::Value* rhs_value,
                                                      llvm::IRBuilder<>* b) {
  llvm::Value* lhs_real = Real(lhs_value, b);
  llvm::Value* lhs_imag = Imag(lhs_value, b);
  llvm::Value* rhs_real = Real(rhs_value, b);
  llvm::Value* rhs_imag = Imag(rhs_value, b);
  llvm::Value* real_result1 = b->CreateFMul(lhs_real, rhs_real);
  llvm::Value* real_result2 = b->CreateFMul(lhs_imag, rhs_imag);
  llvm::Value* real_result = b->CreateFSub(real_result1, real_result2);
  llvm::Value* imag_result1 = b->CreateFMul(lhs_real, rhs_imag);
  llvm::Value* imag_result2 = b->CreateFMul(lhs_imag, rhs_real);
  llvm::Value* imag_result = b->CreateFAdd(imag_result1, imag_result2);
  return {real_result, imag_result};
}
}  // namespace

Status IrEmitter::HandleDot(HloInstruction* dot) {
  auto lhs_instruction = dot->operand(0);
  auto rhs_instruction = dot->operand(1);
  const llvm_ir::IrArray& target_array = GetIrArray(*dot, *dot);
  const llvm_ir::IrArray& lhs_array = GetIrArray(*lhs_instruction, *dot);
  const llvm_ir::IrArray& rhs_array = GetIrArray(*rhs_instruction, *dot);

  const Shape& lhs_shape = lhs_instruction->shape();
  const Shape& rhs_shape = rhs_instruction->shape();
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  CHECK_EQ(dnums.lhs_batch_dimensions_size(),
           dnums.rhs_batch_dimensions_size());

  // TODO(b/110211620): Convert to use i32 index_type when it is possible.
  llvm::Type* index_type = b_.getInt64Ty();
  llvm_ir::IrArray::Index element_index(index_type);
  if (ShapeUtil::IsScalar(lhs_shape) && ShapeUtil::IsScalar(rhs_shape)) {
    // If the operands are scalar, don't emit any loops.
    llvm::Value* lhs_value =
        lhs_array.EmitReadArrayElement(/*index=*/element_index, &b_);
    llvm::Value* rhs_value =
        rhs_array.EmitReadArrayElement(/*index=*/element_index, &b_);
    llvm::Value* result;
    if (ShapeUtil::ElementIsComplex(lhs_shape)) {
      auto value = MultiplyComplex(lhs_value, rhs_value, &b_);
      result = llvm::ConstantAggregateZero::get(lhs_array.GetElementLlvmType());
      result = InsertValue(result, value.first, {0});
      result = InsertValue(result, value.second, {1});
    } else if (ShapeUtil::ElementIsFloating(lhs_shape)) {
      result = FMul(lhs_value, rhs_value);
    } else {
      TF_RET_CHECK(ShapeUtil::ElementIsIntegral(lhs_shape));
      result = Mul(lhs_value, rhs_value);
    }
    target_array.EmitWriteArrayElement(/*index=*/element_index, result, &b_);
    return Status::OK();
  }

  // "Scalar dot non-scalar" or "non-scalar dot scalar" is invalid. See
  // the semantics of Dot in the XLA documentation for details.
  TF_RET_CHECK(!ShapeUtil::IsScalar(lhs_shape) &&
               !ShapeUtil::IsScalar(rhs_shape));

  const int64 lhs_reduction_dimension = dnums.lhs_contracting_dimensions(0);
  const int64 rhs_reduction_dimension = dnums.rhs_contracting_dimensions(0);

  // Check that the batch dims don't cover the reduction dimensions.
  for (int64 batch_dim : dnums.lhs_batch_dimensions()) {
    CHECK_NE(lhs_reduction_dimension, batch_dim);
    CHECK_NE(rhs_reduction_dimension, batch_dim);
  }

  // Verify the reduction dimension in the two operands are the same size.
  TF_RET_CHECK(lhs_shape.dimensions(lhs_reduction_dimension) ==
               rhs_shape.dimensions(rhs_reduction_dimension))
      << "lhs_shape.dimensions(" << lhs_reduction_dimension
      << ") = " << lhs_shape.dimensions(lhs_reduction_dimension)
      << ", and rhs_shape.dimensions(" << rhs_reduction_dimension
      << ") = " << rhs_shape.dimensions(rhs_reduction_dimension);

  // Create loop nests which loop through the LHS operand dimensions and the RHS
  // operand dimensions. The reduction dimension of the LHS and RHS are handled
  // in a separate innermost loop which performs the sum of products.
  llvm_ir::ForLoopNest loop_nest(IrName(dot), &b_);
  std::vector<llvm::Value*> lhs_multi_index =
      loop_nest.EmitOperandArrayLoopNest(
          lhs_array, /*dimension_to_skip=*/lhs_reduction_dimension, "lhs");
  std::vector<llvm::Value*> rhs_multi_index =
      loop_nest.EmitOperandArrayLoopNest(
          rhs_array, /*dimension_to_skip=*/rhs_reduction_dimension, "rhs");

  // We don't have to iterate over the batch dimensions in both arrays, simplify
  // the loop nest of the rhs.
  for (int i = 0; i != dnums.lhs_batch_dimensions_size(); ++i) {
    DCHECK(absl::c_linear_search(dnums.lhs_batch_dimensions(), i));
    rhs_multi_index[i] = lhs_multi_index[i];
  }

  // Create the reduction loop which does the sum of products reduction.
  std::unique_ptr<llvm_ir::ForLoop> reduction_loop = loop_nest.AddLoop(
      /*start_index=*/0,
      /*end_index=*/lhs_shape.dimensions(lhs_reduction_dimension),
      /*suffix=*/"reduction");

  // The final entry in the rhs and lhs indexes is the indvar of the reduction
  // loop.
  lhs_multi_index[lhs_reduction_dimension] = reduction_loop->GetIndVarValue();
  rhs_multi_index[rhs_reduction_dimension] = reduction_loop->GetIndVarValue();

  // For computing the sum of products we alloca a single location to store the
  // dot product result as we accumulate it within the reduction loop. After the
  // reduction loop we load the result and store into the output array.
  llvm::Type* accum_type = target_array.GetElementLlvmType();
  llvm::Value* accum_address = llvm_ir::EmitAllocaAtFunctionEntry(
      accum_type,       // The pointee type of the alloca instruction.
      "accum_address",  // The name of the alloca instruction.
      &b_);

  // Initialize the accumulator in the preheader to zero.
  new llvm::StoreInst(
      llvm::Constant::getNullValue(lhs_array.GetElementLlvmType()),  // init 0
      accum_address,  // The address.
      reduction_loop->GetPreheaderBasicBlock()
          ->getTerminator());  // The instruction this store is inserted before.

  // Emit the body of the reduction loop:
  //   accum = *accum_address
  //   updated_accum = accum + lhs_element * rhs_element
  //   *accum_address = updated_accum
  TF_RET_CHECK(!reduction_loop->GetBodyBasicBlock()->empty());
  b_.SetInsertPoint(
      &*reduction_loop->GetBodyBasicBlock()->getFirstInsertionPt());
  llvm_ir::IrArray::Index lhs_index(lhs_multi_index, lhs_array.GetShape(),
                                    b_.getInt64Ty());
  llvm::Value* lhs_element = lhs_array.EmitReadArrayElement(lhs_index, &b_);
  llvm_ir::IrArray::Index rhs_index(rhs_multi_index, rhs_array.GetShape(),
                                    b_.getInt64Ty());
  llvm::Value* rhs_element = rhs_array.EmitReadArrayElement(rhs_index, &b_);
  llvm::Value* accum = Load(accum_address);
  llvm::Value* updated_accum;
  if (ShapeUtil::ElementIsComplex(lhs_shape)) {
    auto value = MultiplyComplex(lhs_element, rhs_element, &b_);
    llvm::Value* accum_real = Real(accum, &b_);
    llvm::Value* real_sum = FAdd(accum_real, value.first);
    updated_accum = InsertValue(accum, real_sum, {0});
    llvm::Value* accum_imag = Imag(accum, &b_);
    llvm::Value* imag_sum = FAdd(accum_imag, value.second);
    updated_accum = InsertValue(updated_accum, imag_sum, {1});
  } else if (ShapeUtil::ElementIsFloating(lhs_shape)) {
    llvm::Value* product = FMul(lhs_element, rhs_element);
    updated_accum = FAdd(accum, product);
  } else {
    TF_RET_CHECK(ShapeUtil::ElementIsIntegral(lhs_shape));
    llvm::Value* product = Mul(lhs_element, rhs_element);
    updated_accum = Add(accum, product);
  }
  Store(updated_accum, accum_address);

  // After the reduction loop exits, store the accumulator into the target
  // address. The index into the target address is the concatenation of the rhs
  // and lhs indexes with the reduction dimensions removed. The terms from the
  // rhs index are the lower dimensions in the index so we add them first.
  std::vector<llvm::Value*> target_multi_index;
  for (size_t dimension = 0; dimension < lhs_index.size(); ++dimension) {
    if (dimension != lhs_reduction_dimension) {
      target_multi_index.push_back(lhs_index[dimension]);
    }
  }
  // Skip over the batch dimensions to not have them in the index twice.
  for (size_t dimension = dnums.lhs_batch_dimensions_size();
       dimension < rhs_index.size(); ++dimension) {
    if (dimension != rhs_reduction_dimension) {
      target_multi_index.push_back(rhs_index[dimension]);
    }
  }
  SetToFirstInsertPoint(reduction_loop->GetExitBasicBlock(), &b_);
  llvm_ir::IrArray::Index target_index(target_multi_index,
                                       target_array.GetShape(), index_type);
  target_array.EmitWriteArrayElement(
      target_index,
      Load(accum_address),  // The value written to the target array.
      &b_);

  // Set the IR builder insert point to the exit basic block of the outer most
  // loop. This ensures later instructions are inserted after this loop nest.
  b_.SetInsertPoint(loop_nest.GetOuterLoopExitBasicBlock());

  return Status::OK();
}

Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  if (ShapeUtil::IsZeroElementArray(convolution->shape())) {
    // Emit no code for an empty output.
    return Status::OK();
  }
  // TODO(b/31409998): Support convolution with dilation.
  return Unimplemented(
      "Hit a case for convolution that is not implemented on GPU.");
}

Status IrEmitter::HandleFft(HloInstruction* fft) {
  if (ShapeUtil::IsZeroElementArray(fft->shape())) {
    // Emit no code for an empty output.
    return Status::OK();
  }
  return Unimplemented("Hit a case for fft that is not implemented on GPU.");
}

Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented(
      "AllReduce cannot be nested inside of fusion, map, etc.");
}

Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  return Status::OK();
}

Status IrEmitter::HandleFusion(HloInstruction* fusion) {
  // kFusion for library calls should be handled by
  // IrEmitterUnnested::HandleFusion.
  CHECK_EQ(HloInstruction::FusionKind::kLoop, fusion->fusion_kind());
  GpuElementalIrEmitter elemental_emitter(hlo_module_config_, module_, &b_,
                                          GetNestedComputer());
  FusedIrEmitter fused_emitter(GetGeneratorForOperandIrArrays(fusion),
                               &elemental_emitter);
  TF_RETURN_IF_ERROR(fusion->fused_expression_root()->Accept(&fused_emitter));

  return EmitTargetElementLoop(*fusion, fused_emitter.GetRootGenerator());
}

Status IrEmitter::HandleCall(HloInstruction* call) {
  std::vector<llvm::Value*> operand_addresses;
  for (HloInstruction* operand : call->operands()) {
    operand_addresses.push_back(GetBasePointer(*operand));
  }
  return EmitCallToNestedComputation(*call->to_apply(), operand_addresses,
                                     GetBasePointer(*call));
}

Status IrEmitter::HandleCustomCall(HloInstruction*) {
  return Unimplemented("custom-call");
}

Status IrEmitter::HandleInfeed(HloInstruction*) {
  // TODO(b/30467474): Implement infeed on GPU.
  return Unimplemented("Infeed is not supported on GPU.");
}

Status IrEmitter::HandleOutfeed(HloInstruction*) {
  // TODO(b/34359662): Implement outfeed on GPU.
  return Unimplemented("Outfeed is not supported on GPU.");
}

Status IrEmitter::HandleBatchNormInference(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormInference directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter or to a cudnn CustomCall using "
      "CudnnBatchNormRewriter.");
}

Status IrEmitter::HandleBatchNormTraining(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormTraining directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter or to a cudnn CustomCall using "
      "CudnnBatchNormRewriter.");
}

Status IrEmitter::HandleBatchNormGrad(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormGrad directly.  It should "
      "be lowered before IR emission to HLO-soup (using BatchNormRewriter) or "
      "to a cudnn CustomCall using CudnnBatchNormRewriter.");
}

StatusOr<std::vector<llvm::Value*>> IrEmitter::ComputeNestedElement(
    const HloComputation& computation,
    absl::Span<llvm::Value* const> parameter_elements) {
  const Shape& return_shape = computation.root_instruction()->shape();
  llvm::Value* return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::ShapeToIrType(return_shape, module_), "return_buffer", &b_);
  std::vector<llvm::Value*> parameter_buffers;
  for (llvm::Value* parameter_element : parameter_elements) {
    parameter_buffers.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        parameter_element->getType(), "parameter_buffer", &b_));
    Store(parameter_element, parameter_buffers.back());
  }

  std::vector<llvm::Value*> allocas_for_returned_scalars;
  if (!return_shape.IsTuple()) {
    allocas_for_returned_scalars.push_back(return_buffer);
  } else {
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, &b_);
    llvm_ir::IrArray tuple_array(return_buffer, return_shape);

    EmitTuple(tuple_array, allocas_for_returned_scalars, &b_);
  }

  TF_RETURN_IF_ERROR(EmitCallToNestedComputation(computation, parameter_buffers,
                                                 return_buffer));

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    returned_scalars.push_back(Load(addr));
  }
  return returned_scalars;
}

std::vector<llvm_ir::IrArray> IrEmitter::ConstructIrArrayForOutputs(
    const HloInstruction& hlo) {
  std::vector<llvm_ir::IrArray> output_arrays;
  if (hlo.shape().IsTuple()) {
    int64 num_outputs = ShapeUtil::TupleElementCount(hlo.shape());
    output_arrays.reserve(num_outputs);
    for (int64 i = 0; i < num_outputs; ++i) {
      output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
    }
  } else {
    output_arrays.push_back(GetIrArray(hlo, hlo));
  }
  return output_arrays;
}

}  // namespace gpu
}  // namespace xla
