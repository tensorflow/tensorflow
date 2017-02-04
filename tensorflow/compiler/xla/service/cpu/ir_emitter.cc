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

#include "tensorflow/core/platform/logging.h"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Constants.h"
#include "external/llvm/include/llvm/IR/GlobalVariable.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "external/llvm/include/llvm/IR/Intrinsics.h"
#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_runtime_flags.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

using llvm_ir::SetToFirstInsertPoint;

namespace cpu {

IrEmitter::IrEmitter(
    const HloModule& hlo_module, const HloModuleConfig& hlo_module_config,
    const BufferAssignment& assignment, llvm::Module* llvm_module,
    const std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx)
    : assignment_(assignment),
      module_(llvm_module),
      arch_type_(llvm::Triple(llvm_module->getTargetTriple()).getArch()),
      ir_builder_(llvm_module->getContext()),
      hlo_to_profile_idx_(hlo_to_profile_idx),
      alias_analysis_(hlo_module, assignment, &llvm_module->getContext()),
      hlo_module_config_(hlo_module_config) {
  ir_builder_.setFastMathFlags(llvm_ir::GetFastMathFlags(hlo_module_config));
}

StatusOr<llvm::Function*> IrEmitter::EmitComputation(
    HloComputation* computation, const string& function_name_prefix,
    bool is_entry_computation,
    std::vector<const HloInstruction*>* instruction_order) {
  string function_name = name_uniquer_.GetUniqueName(function_name_prefix);
  VLOG(2) << "Emitting IR for CPU function [" << function_name_prefix << "]";
  InitializeIrFunction(function_name, is_entry_computation);
  // The rdtscp instruction is x86 specific.  We will fallback to LLVM's generic
  // readcyclecounter if it is unavailable.
  bool use_rdtscp = arch_type_ == llvm::Triple::ArchType::x86 ||
                    arch_type_ == llvm::Triple::ArchType::x86_64;
  profiling_state_ = ProfilingState(is_entry_computation, use_rdtscp,
                                    GetProfileCountersArgument());
  if (instruction_order != nullptr) {
    TF_RETURN_IF_ERROR(computation->root_instruction()->AcceptOrdered(
        this, *instruction_order));
  } else {
    TF_RETURN_IF_ERROR(computation->root_instruction()->Accept(this));
  }
  InsertOrDie(&emitted_functions_, computation, compute_function_);

  return compute_function_;
}

static llvm::Argument* GetArg(llvm::Function* f, int idx) {
  llvm::Function::arg_iterator arg_iter = f->arg_begin();
  std::advance(arg_iter, idx);
  return &*arg_iter;
}

void IrEmitter::InitializeIrFunction(const string& function_name,
                                     bool is_entry_computation) {
  // The function signature is:
  //   void function(i8* retval, i8* run_options, i8** params, i8** temps,
  //                 i64* prof_counters)
  //
  // retval: points to the returned value.
  // params: address of an array with pointers to parameters.
  // temps: address of an array with pointers to temporary buffers.
  //
  // Therefore, the generated function's signature (FunctionType) is statically
  // determined - parameter unpacking is done in code generated into the
  // function, rather than by a prologue dictated by the platform ABI.
  //
  //                      /--------------\
  //   retval ----------> | return value |
  //                      \--------------/
  //
  //                      /-------------------------------\
  //   run_options -----> | xla::ExecutableRunOptions |
  //                      \-------------------------------/
  //
  //                     /---------------------------------------------\
  //   params -------->  |  param 0  |  param 1  | ..... |  param N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | param 0 |  | param 1 |         | param N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                     /---------------------------------------------\
  //   temps --------->  |  temp  0  |  temp  1  | ..... |  temp  N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | temp  0 |  | temp  1 |         | temp  N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                     /---------------------------------------------\
  //   prof counters ->  | counter 0 | counter 1 | ..... | counter N-1 |
  //  (elided for aot)   \---------------------------------------------/

  // Even though the type of params and temps is void** in the host's view, in
  // LLVM IR this is represented by i8*, similarly to void*. It's up to the code
  // to use GEPs to unravel the indirection layers.
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::Type* i8_ptr_ptr_type = i8_ptr_type->getPointerTo();
  llvm::Type* i64_ptr_type = llvm::Type::getInt64PtrTy(module_->getContext());
  std::vector<llvm::Type*> compute_function_params(
      {i8_ptr_type, i8_ptr_type, i8_ptr_ptr_type, i8_ptr_ptr_type});
  if (hlo_to_profile_idx_) {
    compute_function_params.push_back(i64_ptr_type);
  }
  llvm::FunctionType* compute_function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(module_->getContext()),
      /*Params=*/compute_function_params,
      /*isVarArg=*/false);

  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  llvm::Function::LinkageTypes linkage =
      is_entry_computation ? llvm::GlobalValue::ExternalLinkage
                           : llvm::GlobalValue::InternalLinkage;
  compute_function_ = llvm::Function::Create(/*Ty=*/compute_function_type,
                                             /*Linkage=*/linkage,
                                             /*Name=*/function_name.c_str(),
                                             /*Module=*/module_);
  compute_function_->setCallingConv(llvm::CallingConv::C);

  // Set meaningful names for the function's arguments: useful for debugging.
  llvm::Function::arg_iterator arg_iter = compute_function_->arg_begin();
  arg_iter->setName("retval");
  (++arg_iter)->setName("run_options");
  (++arg_iter)->setName("params");
  (++arg_iter)->setName("temps");
  if (hlo_to_profile_idx_) {
    (++arg_iter)->setName("prof_counters");
  }

  // We know a-priori that the function arguments are guaranteed to point to
  // disjoint objects.
  llvm::Argument* retval = GetResultArgument();
  for (llvm::Argument& argument : compute_function_->args()) {
    // However, the return buffer aliases the temporaries and thus cannot be
    // marked noalias.
    if (&argument == retval) {
      continue;
    }
    compute_function_->setDoesNotAlias(argument.getArgNo() + 1);
  }

  ir_builder_.SetInsertPoint(llvm::BasicBlock::Create(
      /*Context=*/module_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/compute_function_));
}

IrEmitter::~IrEmitter() {}

Status IrEmitter::HandleBitcast(HloInstruction* bitcast) {
  VLOG(2) << "HandleBitcast: " << bitcast->ToString();
  emitted_value_[bitcast] = ir_builder_.CreateBitCast(
      GetEmittedValueFor(bitcast->operand(0)),
      IrShapeType(bitcast->shape())->getPointerTo(), bitcast->name().c_str());
  return Status::OK();
}

Status IrEmitter::HandleConstant(HloInstruction* constant,
                                 const Literal& literal) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  llvm::Constant* initializer =
      llvm_ir::ConvertLiteralToIrConstant(literal, &ir_builder_);
  llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
      /*Module=*/*module_,
      /*Type=*/initializer->getType(),
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/initializer,
      /*Name=*/"");
  emitted_value_[constant] = global_for_const;
  VLOG(2) << "  emitted value: " << llvm_ir::DumpToString(*global_for_const);
  VLOG(2) << "  its type: "
          << llvm_ir::DumpToString(*global_for_const->getType());
  return Status::OK();
}

Status IrEmitter::HandleCopy(HloInstruction* copy, HloInstruction* operand) {
  if (ShapeUtil::IsTuple(copy->shape())) {
    // kCopy shallow copies a tuple so just memcpy the top-level buffer.
    TF_ASSIGN_OR_RETURN(llvm::Value * copy_value, EmitTargetAddressForOp(copy));
    emitted_value_[copy] = copy_value;
    return EmitMemcpy(*operand, *copy);
  } else {
    // Use the elemental emitter for non-tuple shapes.
    return DefaultAction(copy);
  }
}

// Calculate the alignment of a buffer with a particular size.
int IrEmitter::MinimumAlignmentForBufferSize(int64 buffer_size) {
  // GLibc returns a pointer with alignment 8 on 32-bit platforms and 16 on
  // 64-bit platforms.  TCMalloc returns a pointer with alignment 8 for
  // allocations smaller than 16 bytes and at least alignment 16 for allocations
  // greater than or equal to 16 bytes.  N.B. We could improve on this lower
  // bound by explicitly allocating the memory with posix_memalign.  This is
  // complicated by our desire to allow parameter buffers created by clients to
  // be consumed directly by the JIT.
  if (buffer_size == 0) {
    // No need to align empty buffers.
    return 1;
  }
  int pointer_size = module_->getDataLayout().getPointerSize();
  int buffer_alignment = buffer_size >= 16 ? 2 * pointer_size : 8;
  DCHECK_GT(buffer_alignment, 0);

  return buffer_alignment;
}

// Calculate the alignment of a buffer allocated for a given primitive type.
int IrEmitter::MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {
  int64 buffer_size = ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  DCHECK_GE(buffer_size, 0);
  DCHECK_LE(buffer_size, SIZE_MAX);

  return MinimumAlignmentForBufferSize(buffer_size);
}

int64 IrEmitter::ByteSizeOf(const Shape& shape) const {
  return llvm_ir::ByteSizeOf(shape, module_->getDataLayout());
}

// Calculate the alignment of a buffer allocated for a given shape.
int IrEmitter::MinimumAlignmentForShape(const Shape& shape) {
  int64 buffer_size = ByteSizeOf(shape);
  DCHECK_GE(buffer_size, 0);
  DCHECK_LE(buffer_size, SIZE_MAX);

  return MinimumAlignmentForBufferSize(buffer_size);
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
  int alignment = MinimumAlignmentForBufferSize(buffer_size);
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

Status IrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element,
                                        HloInstruction* operand) {
  // A tuple is an array of pointers, one for each operand. Each pointer points
  // to the output buffer of its corresponding operand. A GetTupleElement
  // instruction forwards a pointer to the tuple element buffer at the given
  // index.
  const Shape& shape = get_tuple_element->shape();
  emitted_value_[get_tuple_element] = llvm_ir::EmitGetTupleElement(
      shape, get_tuple_element->tuple_index(), MinimumAlignmentForShape(shape),
      GetEmittedValueFor(operand), &ir_builder_);
  return Status::OK();
}

Status IrEmitter::HandleSelect(HloInstruction* select, HloInstruction* pred,
                               HloInstruction* on_true,
                               HloInstruction* on_false) {
  TF_RET_CHECK(pred->shape().element_type() == PRED);

  if (ShapeUtil::IsTuple(select->shape())) {
    TF_ASSIGN_OR_RETURN(llvm::Value * output_address,
                        EmitTargetAddressForOp(select));
    llvm_ir::EmitTupleSelect(llvm_ir::IrArray(output_address, select->shape()),
                             GetIrArrayForOp(pred), GetEmittedValueFor(on_true),
                             GetEmittedValueFor(on_false), &ir_builder_);
    emitted_value_[select] = output_address;
    return Status::OK();
  }

  return DefaultAction(select);
}

Status IrEmitter::HandleInfeed(HloInstruction* infeed) {
  VLOG(2) << "HandleInfeed: " << infeed->ToString();

  // The signature of the acquire infeed buffer function is:
  //
  //   (void*)(int32 length);
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::Type* int32_type = ir_builder_.getInt32Ty();
  llvm::FunctionType* acquire_type =
      llvm::FunctionType::get(i8_ptr_type, {int32_type},
                              /*isVarArg=*/false);

  llvm::Function* acquire_func =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          runtime::kAcquireInfeedBufferForDequeueSymbolName, acquire_type));
  acquire_func->setCallingConv(llvm::CallingConv::C);

  // The signature of the release infeed buffer function is:
  //
  //   (void)(int32 length, void* buffer);
  llvm::FunctionType* release_type = llvm::FunctionType::get(
      ir_builder_.getVoidTy(), {int32_type, i8_ptr_type},
      /*isVarArg=*/false);

  llvm::Function* release_func =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          runtime::kReleaseInfeedBufferAfterDequeueSymbolName, release_type));
  release_func->setCallingConv(llvm::CallingConv::C);

  const Shape& shape = infeed->shape();
  int64 length = ByteSizeOf(shape);
  if (length > std::numeric_limits<int32>::max()) {
    return InvalidArgument("infeed buffer length %lld is too large", length);
  }
  int32 length_32 = static_cast<int32>(length);

  llvm::Value* acquired_pointer =
      ir_builder_.CreateCall(acquire_func, {ir_builder_.getInt32(length_32)});

  TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                      EmitTargetAddressForOp(infeed));

  ir_builder_.CreateMemCpy(target_address, acquired_pointer, length_32, 1);

  ir_builder_.CreateCall(release_func,
                         {ir_builder_.getInt32(length_32), acquired_pointer});

  emitted_value_[infeed] = target_address;

  return Status::OK();
}

Status IrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  // TODO(b/34359662): Implement outfeed on CPU.
  return Unimplemented("Outfeed is not supported on CPU (b/34359662).");
}

Status IrEmitter::HandleSort(HloInstruction* sort, HloInstruction* operand) {
  // TODO(b/26783907): Implement sort on CPU.
  return Unimplemented("Sort is not supported on GPU (b/26783907).");
}

Status IrEmitter::HandleTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                      EmitTargetAddressForOp(tuple));
  std::vector<llvm::Value*> base_ptrs;
  for (auto operand : operands) {
    base_ptrs.push_back(GetEmittedValueFor(operand));
  }
  llvm_ir::EmitTuple(llvm_ir::IrArray(target_address, tuple->shape()),
                     base_ptrs, &ir_builder_);
  emitted_value_[tuple] = target_address;
  return Status::OK();
}

Status IrEmitter::HandleMap(
    HloInstruction* map, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* function,
    tensorflow::gtl::ArraySlice<HloInstruction*> /*static_operands*/) {
  // The called computation should have been emitted previously.
  llvm::Function* mapped_ir_function = FindOrDie(emitted_functions_, function);

  return EmitTargetElementLoop(map, [this, map, operands, mapped_ir_function](
                                        const llvm_ir::IrArray::Index& index) {
    std::vector<llvm::Value*> parameter_addresses;
    for (const HloInstruction* operand : operands) {
      const llvm_ir::IrArray& array = GetIrArrayForOp(operand);
      parameter_addresses.push_back(
          array.EmitArrayElementAddress(index, &ir_builder_));
    }
    return EmitElementFunctionCall(mapped_ir_function, map->shape(),
                                   parameter_addresses, "map_function");
  });
}

Status IrEmitter::HandleReduceWindow(HloInstruction* reduce_window,
                                     HloInstruction* operand,
                                     const Window& window,
                                     HloComputation* function) {
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*reduce_window, /*operands=*/{operand},
      /*supported_types=*/{F32}));

  // TODO(b/31410564): Implement dilation for reduce-window.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for reduce-window not implemented on CPU. See b/31410564.");
  }

  // The called computation should have been emitted previously.
  llvm::Function* reducer_function = FindOrDie(emitted_functions_, function);

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
  return EmitTargetElementLoop(
      reduce_window, [this, reduce_window, operand, window,
                      reducer_function](const llvm_ir::IrArray::Index& index) {
        // We fold inputs into the accumulator and initialize it to
        // the initial value on the reduce_window.
        PrimitiveType operand_element_type = operand->shape().element_type();
        llvm::Value* accumulator_address = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(operand_element_type, &ir_builder_),
            "reduce_window_accumulator_address", &ir_builder_,
            MinimumAlignmentForPrimitiveType(operand_element_type));
        ir_builder_.CreateStore(ir_builder_.CreateLoad(GetEmittedValueFor(
                                    reduce_window->operand(1))),
                                accumulator_address);

        llvm_ir::ForLoopNest loops(&ir_builder_);
        std::vector<int64> window_size;
        for (const auto& dim : window.dimensions()) {
          window_size.push_back(dim.size());
        }
        const llvm_ir::IrArray::Index window_index = loops.AddLoopsForShape(
            ShapeUtil::MakeShape(operand_element_type, window_size), "window");
        CHECK_EQ(window_index.size(), index.size());

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

        llvm_ir::IrArray::Index input_index(index.size());
        llvm::Value* in_bounds_condition = nullptr;
        for (int64 i = 0; i < index.size(); ++i) {
          llvm::Value* strided_index = ir_builder_.CreateNSWMul(
              index[i], ir_builder_.getInt64(window.dimensions(i).stride()));
          input_index[i] = ir_builder_.CreateNSWSub(
              ir_builder_.CreateNSWAdd(strided_index, window_index[i]),
              ir_builder_.getInt64(window.dimensions(i).padding_low()));

          // We need to check if 0 <= input_index[i] < bound, as
          // otherwise we are in the padding so that we can skip the
          // computation. That is equivalent to input_index[i] < bound
          // as an *unsigned* comparison, since a negative value will
          // wrap to a large positive value.
          llvm::Value* index_condition = ir_builder_.CreateICmpULT(
              input_index[i], ir_builder_.getInt64(ShapeUtil::GetDimension(
                                  operand->shape(), i)));
          if (in_bounds_condition == nullptr) {
            in_bounds_condition = index_condition;
          } else {
            in_bounds_condition =
                ir_builder_.CreateAnd(in_bounds_condition, index_condition);
          }
        }
        CHECK(in_bounds_condition != nullptr);

        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            in_bounds_condition, "in-bounds", &ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, &ir_builder_);

        // We are not in the padding, so carry out the computation.
        llvm_ir::IrArray input_array(GetIrArrayForOp(operand));
        llvm::Value* input_value_address =
            input_array.EmitArrayElementAddress(input_index, &ir_builder_);
        llvm::Value* result = EmitElementFunctionCall(
            reducer_function, reduce_window->shape(),
            {accumulator_address, input_value_address}, "reducer_function");
        ir_builder_.CreateStore(result, accumulator_address);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
        return ir_builder_.CreateLoad(accumulator_address);
      });
}

Status IrEmitter::HandleSelectAndScatter(HloInstruction* select_and_scatter) {
  CHECK_EQ(select_and_scatter->operand_count(), 3);
  const auto operand = select_and_scatter->operand(0);
  const auto source = select_and_scatter->operand(1);
  const auto init_value = select_and_scatter->operand(2);
  const Window& window = select_and_scatter->window();
  PrimitiveType operand_element_type = operand->shape().element_type();
  const int64 rank = ShapeUtil::Rank(operand->shape());
  CHECK_EQ(rank, ShapeUtil::Rank(source->shape()));
  CHECK_EQ(rank, window.dimensions_size());

  // TODO(b/31410564): Implement dilation for select-and-scatter.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for select-and-scatter not implemented on CPU. "
        "See b/31410564.");
  }

  // The select and scatter computations should have been emitted previously.
  llvm::Function* select_function =
      FindOrDie(emitted_functions_, select_and_scatter->select());
  llvm::Function* scatter_function =
      FindOrDie(emitted_functions_, select_and_scatter->scatter());

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
      select_and_scatter,
      [this, init_value](const llvm_ir::IrArray::Index& target_index) {
        llvm::Value* init_value_addr = GetEmittedValueFor(init_value);
        return ir_builder_.CreateLoad(init_value_addr);
      }));

  // Create a loop to iterate over the source array to scatter to the output.
  llvm_ir::ForLoopNest source_loops(&ir_builder_);
  const llvm_ir::IrArray::Index source_index =
      source_loops.AddLoopsForShape(source->shape(), "source");
  SetToFirstInsertPoint(source_loops.GetInnerLoopBodyBasicBlock(),
                        &ir_builder_);

  // Allocate space to keep the currently selected value, its index, and
  // the boolean initialized_flag, which is initially set to false.
  llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(operand_element_type, &ir_builder_),
      "selected_value_address", &ir_builder_,
      MinimumAlignmentForPrimitiveType(operand_element_type));
  llvm::Value* selected_index_address =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          ir_builder_.getInt64Ty(), ir_builder_.getInt32(rank),
          "selected_index_address", &ir_builder_);
  llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
      ir_builder_.getInt1Ty(), "initialized_flag_address", &ir_builder_);
  ir_builder_.CreateStore(ir_builder_.getInt1(false), initialized_flag_address);

  // Create the inner loop to iterate over the window.
  llvm_ir::ForLoopNest window_loops(&ir_builder_);
  std::vector<int64> window_size;
  for (const auto& dim : window.dimensions()) {
    window_size.push_back(dim.size());
  }
  const llvm_ir::IrArray::Index window_index = window_loops.AddLoopsForShape(
      ShapeUtil::MakeShape(operand_element_type, window_size), "window");
  SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                        &ir_builder_);

  // Compute the operand index to visit and evaluate the condition whether the
  // operand index is within the bounds. The unsigned comparison includes
  // checking whether the operand index >= 0.
  llvm_ir::IrArray::Index operand_index(source_index.size());
  llvm::Value* in_bounds_condition = ir_builder_.getInt1(true);
  for (int64 i = 0; i < rank; ++i) {
    llvm::Value* strided_index = ir_builder_.CreateNSWMul(
        source_index[i], ir_builder_.getInt64(window.dimensions(i).stride()));
    operand_index[i] = ir_builder_.CreateNSWSub(
        ir_builder_.CreateNSWAdd(strided_index, window_index[i]),
        ir_builder_.getInt64(window.dimensions(i).padding_low()));
    llvm::Value* index_condition = ir_builder_.CreateICmpULT(
        operand_index[i],
        ir_builder_.getInt64(ShapeUtil::GetDimension(operand->shape(), i)));
    in_bounds_condition =
        ir_builder_.CreateAnd(in_bounds_condition, index_condition);
  }
  CHECK(in_bounds_condition != nullptr);

  // Only need to do something if the operand index is within the bounds. First
  // check if the initialized_flag is set.
  llvm_ir::LlvmIfData if_in_bounds =
      llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &ir_builder_);
  SetToFirstInsertPoint(if_in_bounds.true_block, &ir_builder_);
  llvm_ir::LlvmIfData if_initialized =
      llvm_ir::EmitIfThenElse(ir_builder_.CreateLoad(initialized_flag_address),
                              "initialized", &ir_builder_);

  // If the initialized_flag is false, initialize the selected value and index
  // with the currently visiting operand.
  SetToFirstInsertPoint(if_initialized.false_block, &ir_builder_);
  const auto save_operand_index = [&](
      const llvm_ir::IrArray::Index& operand_index) {
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot = ir_builder_.CreateInBoundsGEP(
          selected_index_address, {ir_builder_.getInt32(i)});
      ir_builder_.CreateStore(operand_index[i], selected_index_address_slot);
    }
  };
  llvm_ir::IrArray operand_array(GetIrArrayForOp(operand));
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, &ir_builder_);
  ir_builder_.CreateStore(operand_data, selected_value_address);
  save_operand_index(operand_index);
  ir_builder_.CreateStore(ir_builder_.getInt1(true), initialized_flag_address);

  // If the initialized_flag is true, call the `select` function to potentially
  // update the selected value and index with the currently visiting operand.
  SetToFirstInsertPoint(if_initialized.true_block, &ir_builder_);
  const Shape output_shape = ShapeUtil::MakeShape(PRED, {});
  llvm::Value* operand_address =
      operand_array.EmitArrayElementAddress(operand_index, &ir_builder_);
  llvm::Value* result = EmitElementFunctionCall(
      select_function, output_shape, {selected_value_address, operand_address},
      "select_function");

  // If the 'select' function returns false, update the selected value and the
  // index to the currently visiting operand.
  llvm::Value* cond = ir_builder_.CreateICmpNE(
      result, llvm::ConstantInt::get(
                  llvm_ir::PrimitiveTypeToIrType(PRED, &ir_builder_), 0),
      "boolean_predicate");
  llvm_ir::LlvmIfData if_select_lhs =
      llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &ir_builder_);
  SetToFirstInsertPoint(if_select_lhs.false_block, &ir_builder_);
  ir_builder_.CreateStore(ir_builder_.CreateLoad(operand_address),
                          selected_value_address);
  save_operand_index(operand_index);

  // After iterating over the window elements, scatter the source element to
  // the selected index of the output. The value we store at the output
  // location is computed by calling the `scatter` function with the source
  // value and the current output value.
  SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                        &ir_builder_);
  llvm_ir::IrArray::Index selected_index;
  for (int64 i = 0; i < rank; ++i) {
    llvm::Value* selected_index_address_slot = ir_builder_.CreateInBoundsGEP(
        selected_index_address, {ir_builder_.getInt32(i)});
    selected_index.push_back(
        ir_builder_.CreateLoad(selected_index_address_slot));
  }
  llvm_ir::IrArray source_array(GetIrArrayForOp(source));
  llvm::Value* source_value_address =
      source_array.EmitArrayElementAddress(source_index, &ir_builder_);
  llvm_ir::IrArray output_array(GetIrArrayForOp(select_and_scatter));
  llvm::Value* output_value_address =
      output_array.EmitArrayElementAddress(selected_index, &ir_builder_);
  llvm::Value* scatter_value = EmitElementFunctionCall(
      scatter_function, source->shape(),
      {output_value_address, source_value_address}, "scatter_function");
  output_array.EmitWriteArrayElement(selected_index, scatter_value,
                                     &ir_builder_);

  SetToFirstInsertPoint(source_loops.GetOuterLoopExitBasicBlock(),
                        &ir_builder_);
  return Status::OK();
}

Status IrEmitter::HandleDot(HloInstruction* dot, HloInstruction* lhs,
                            HloInstruction* rhs) {
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*dot, /*operands=*/{lhs, rhs},
      /*supported_types=*/{F32, F64}));

  llvm_ir::IrArray lhs_array(GetIrArrayForOp(lhs));
  llvm_ir::IrArray rhs_array(GetIrArrayForOp(rhs));

  Shape target_shape = dot->shape();
  TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                      EmitTargetAddressForOp(dot));
  llvm_ir::IrArray target_array(target_address, target_shape);
  AddAliasingInformationToIrArray(*dot, &target_array);

  VLOG(2) << "HandleDot: ";
  VLOG(2) << "  lhs operand: "
          << llvm_ir::DumpToString(*lhs_array.GetBasePointer());
  VLOG(2) << "  rhs operand: "
          << llvm_ir::DumpToString(*rhs_array.GetBasePointer());
  VLOG(2) << "  target: "
          << llvm_ir::DumpToString(*target_array.GetBasePointer());

  // Dot operation is complicated so we delegate to a helper class.
  TF_RETURN_IF_ERROR(DotOpEmitter::EmitDotOperation(
      *dot, /*transpose_lhs=*/false, /*transpose_rhs=*/false, target_array,
      lhs_array, rhs_array, GetExecutableRunOptionsArgument(), &ir_builder_));

  emitted_value_[dot] = target_address;
  return Status::OK();
}

Status IrEmitter::HandleConvolution(HloInstruction* convolution,
                                    HloInstruction* lhs, HloInstruction* rhs,
                                    const Window& window) {
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*convolution, /*operands=*/{lhs, rhs},
      /*supported_types=*/{F32}));

  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (PotentiallyImplementedAsEigenConvolution(*convolution)) {
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();
    const Shape& convolution_shape = convolution->shape();
    // The input, kernel and output agree with respect to layout.
    if (LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout()) &&
        LayoutUtil::IsMonotonicWithDim0Major(convolution_shape.layout())) {
      llvm::Value* lhs_address = GetEmittedValueFor(lhs);
      llvm::Value* rhs_address = GetEmittedValueFor(rhs);
      TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                          EmitTargetAddressForOp(convolution));

      const ConvolutionDimensionNumbers& dnums =
          convolution->convolution_dimension_numbers();

      // Input tensor.
      const Shape& input_shape = convolution->operand(0)->shape();
      int64 input_batch = input_shape.dimensions(dnums.batch_dimension());
      int64 input_rows = input_shape.dimensions(dnums.spatial_dimensions(0));
      int64 input_cols = input_shape.dimensions(dnums.spatial_dimensions(1));
      int64 input_channels = input_shape.dimensions(dnums.feature_dimension());

      // Kernel tensor.
      const Shape& kernel_shape = convolution->operand(1)->shape();
      int64 kernel_rows =
          kernel_shape.dimensions(dnums.kernel_spatial_dimensions(0));
      int64 kernel_cols =
          kernel_shape.dimensions(dnums.kernel_spatial_dimensions(1));
      int64 kernel_channels =
          kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
      int64 kernel_filters =
          kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

      // Output tensor.
      const Shape& convolution_shape = convolution->shape();
      int64 output_rows =
          convolution_shape.dimensions(dnums.spatial_dimensions(0));
      int64 output_cols =
          convolution_shape.dimensions(dnums.spatial_dimensions(1));

      // Extract the window stride for the convolution.
      const Window& window = convolution->window();
      int64 row_stride = window.dimensions(0).stride();
      int64 col_stride = window.dimensions(1).stride();

      int64 padding_top = window.dimensions(0).padding_low();
      int64 padding_bottom = window.dimensions(0).padding_high();
      int64 padding_left = window.dimensions(1).padding_low();
      int64 padding_right = window.dimensions(1).padding_high();

      int64 lhs_row_dilation = window.dimensions(0).base_dilation();
      int64 lhs_col_dilation = window.dimensions(1).base_dilation();
      int64 rhs_row_dilation = window.dimensions(0).window_dilation();
      int64 rhs_col_dilation = window.dimensions(1).window_dilation();

      // Args have been computed, make the call.
      llvm::Type* float_ptr_type = ir_builder_.getFloatTy()->getPointerTo();
      llvm::Type* int64_type = ir_builder_.getInt64Ty();
      llvm::Type* int8_ptr_type = ir_builder_.getInt8Ty()->getPointerTo();
      llvm::FunctionType* conv_type = llvm::FunctionType::get(
          ir_builder_.getVoidTy(),
          {int8_ptr_type, float_ptr_type, float_ptr_type, float_ptr_type,
           int64_type,    int64_type,     int64_type,     int64_type,
           int64_type,    int64_type,     int64_type,     int64_type,
           int64_type,    int64_type,     int64_type,     int64_type,
           int64_type,    int64_type,     int64_type,     int64_type,
           int64_type,    int64_type,     int64_type,     int64_type},
          /*isVarArg=*/false);
      legacy_flags::CpuRuntimeFlags* flags = legacy_flags::GetCpuRuntimeFlags();
      const char* fn_name =
          (flags->xla_cpu_multi_thread_eigen
               ? runtime::kEigenConvF32SymbolName
               : runtime::kEigenSingleThreadedConvF32SymbolName);
      llvm::Function* conv_func = llvm::cast<llvm::Function>(
          module_->getOrInsertFunction(fn_name, conv_type));
      conv_func->setCallingConv(llvm::CallingConv::C);
      conv_func->setDoesNotThrow();
      conv_func->setOnlyAccessesArgMemory();
      ir_builder_.CreateCall(
          conv_func,
          {
              GetExecutableRunOptionsArgument(),
              ir_builder_.CreateBitCast(target_address, float_ptr_type),
              ir_builder_.CreateBitCast(lhs_address, float_ptr_type),
              ir_builder_.CreateBitCast(rhs_address, float_ptr_type),
              ir_builder_.getInt64(input_batch),
              ir_builder_.getInt64(input_rows),
              ir_builder_.getInt64(input_cols),
              ir_builder_.getInt64(input_channels),
              ir_builder_.getInt64(kernel_rows),
              ir_builder_.getInt64(kernel_cols),
              ir_builder_.getInt64(kernel_channels),
              ir_builder_.getInt64(kernel_filters),
              ir_builder_.getInt64(output_rows),
              ir_builder_.getInt64(output_cols),
              ir_builder_.getInt64(row_stride),
              ir_builder_.getInt64(col_stride),
              ir_builder_.getInt64(padding_top),
              ir_builder_.getInt64(padding_bottom),
              ir_builder_.getInt64(padding_left),
              ir_builder_.getInt64(padding_right),
              ir_builder_.getInt64(lhs_row_dilation),
              ir_builder_.getInt64(lhs_col_dilation),
              ir_builder_.getInt64(rhs_row_dilation),
              ir_builder_.getInt64(rhs_col_dilation),
          });
      emitted_value_[convolution] = target_address;

      return Status::OK();
    }
  }

  // This is a completely un-optimized version of convolution just to
  // have an early version that works. E.g. the input index and
  // padding calculation is not hoisted out of the inner loop.
  //
  // See the description of convolution in the XLA documentation for the pseudo
  // code for convolution.
  return EmitTargetElementLoop(
      convolution, [this, convolution, lhs, rhs, window,
                    dnums](const llvm_ir::IrArray::Index& index) {
        int num_spatial_dims = dnums.spatial_dimensions_size();
        std::vector<llvm::Value*> output_spatial(num_spatial_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          output_spatial[i] = index[dnums.spatial_dimensions(i)];
        }
        llvm::Value* output_feature = index[dnums.feature_dimension()];
        llvm::Value* batch = index[dnums.batch_dimension()];

        // We will accumulate the products into this sum to calculate
        // the output entry at the given index.
        PrimitiveType lhs_element_type = lhs->shape().element_type();
        llvm::Value* sum_address = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(lhs_element_type, &ir_builder_),
            "convolution_sum_address", &ir_builder_,
            MinimumAlignmentForPrimitiveType(lhs_element_type));
        ir_builder_.CreateStore(
            llvm::ConstantFP::get(ir_builder_.getFloatTy(), 0.0), sum_address);

        llvm_ir::ForLoopNest loops(&ir_builder_);
        std::vector<llvm::Value*> kernel_spatial(num_spatial_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          kernel_spatial[i] =
              loops
                  .AddLoop(0, rhs->shape().dimensions(
                                  dnums.kernel_spatial_dimensions(i)),
                           tensorflow::strings::StrCat("k", i))
                  ->GetIndVarValue();
        }
        llvm::Value* input_feature =
            loops
                .AddLoop(0, lhs->shape().dimensions(dnums.feature_dimension()),
                         "iz")
                ->GetIndVarValue();

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

        // Calculate the spatial index in the input array, taking striding,
        // dilation and padding into account. An index in the padding will be
        // out of the bounds of the array.
        const auto calculate_input_index = [this](
            llvm::Value* output_index, llvm::Value* kernel_index,
            const WindowDimension& window_dim) {
          llvm::Value* strided_index = ir_builder_.CreateNSWMul(
              output_index, ir_builder_.getInt64(window_dim.stride()));
          llvm::Value* dilated_kernel_index = ir_builder_.CreateNSWMul(
              kernel_index, ir_builder_.getInt64(window_dim.window_dilation()));
          return ir_builder_.CreateNSWSub(
              ir_builder_.CreateNSWAdd(strided_index, dilated_kernel_index),
              ir_builder_.getInt64(window_dim.padding_low()));
        };
        std::vector<llvm::Value*> input_spatial(num_spatial_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          input_spatial[i] = calculate_input_index(
              output_spatial[i], kernel_spatial[i], window.dimensions(i));
        }

        // We need to check if 0 <= input dim < bound, as otherwise we are in
        // the padding so that we can skip the computation. That is equivalent
        // to input dim < bound as an *unsigned* comparison, since a negative
        // value will wrap to a large positive value. The input dim is dilated,
        // so we need to dilate the bound as well to match.

        // Also need to check that the input coordinates are not in one of the
        // holes created by base dilation.
        const auto not_in_hole = [&](llvm::Value* input_index,
                                     int64 base_dilation) {
          llvm::Value* remainder = ir_builder_.CreateSRem(
              input_index, ir_builder_.getInt64(base_dilation));
          return ir_builder_.CreateICmpEQ(remainder, ir_builder_.getInt64(0));
        };

        llvm::Value* in_bounds_condition = nullptr;
        for (int i = 0; i < num_spatial_dims; ++i) {
          llvm::ConstantInt* input_bound =
              ir_builder_.getInt64(window_util::DilatedBound(
                  lhs->shape().dimensions(dnums.spatial_dimensions(i)),
                  window.dimensions(i).base_dilation()));
          llvm::Value* dim_in_bound =
              ir_builder_.CreateICmpULT(input_spatial[i], input_bound);
          llvm::Value* dim_not_in_hole = not_in_hole(
              input_spatial[i], window.dimensions(i).base_dilation());
          llvm::Value* dim_ok =
              ir_builder_.CreateAnd(dim_in_bound, dim_not_in_hole);
          in_bounds_condition =
              in_bounds_condition
                  ? ir_builder_.CreateAnd(in_bounds_condition, dim_ok)
                  : dim_ok;
        }

        // Now we need to map the dilated base coordinates back to the actual
        // data indices on the lhs.
        const auto undilate = [&](llvm::Value* input_index,
                                  int64 base_dilation) {
          return ir_builder_.CreateSDiv(input_index,
                                        ir_builder_.getInt64(base_dilation));
        };
        for (int i = 0; i < num_spatial_dims; ++i) {
          input_spatial[i] =
              undilate(input_spatial[i], window.dimensions(i).base_dilation());
        }

        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            in_bounds_condition, "in-bounds", &ir_builder_);
        SetToFirstInsertPoint(if_data.true_block, &ir_builder_);

        // We are not in the padding, so carry out the computation.
        int num_dims = num_spatial_dims + 2;
        llvm_ir::IrArray::Index input_index(num_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          input_index[dnums.spatial_dimensions(i)] = input_spatial[i];
        }
        input_index[dnums.feature_dimension()] = input_feature;
        input_index[dnums.batch_dimension()] = batch;

        llvm_ir::IrArray kernel_array(GetIrArrayForOp(rhs));
        llvm_ir::IrArray::Index kernel_index(num_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          kernel_index[dnums.kernel_spatial_dimensions(i)] = kernel_spatial[i];
        }
        kernel_index[dnums.kernel_input_feature_dimension()] = input_feature;
        kernel_index[dnums.kernel_output_feature_dimension()] = output_feature;

        llvm_ir::IrArray input_array(GetIrArrayForOp(lhs));
        llvm::Value* product = ir_builder_.CreateFMul(
            input_array.EmitReadArrayElement(input_index, &ir_builder_),
            kernel_array.EmitReadArrayElement(kernel_index, &ir_builder_));
        llvm::Value* sum = ir_builder_.CreateFAdd(
            ir_builder_.CreateLoad(sum_address), product);
        ir_builder_.CreateStore(sum, sum_address);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
        return ir_builder_.CreateLoad(sum_address);
      });
}

Status IrEmitter::HandleCrossReplicaSum(HloInstruction* crs) {
  // TODO(b/33011107): Support cross replica sum on CPU.
  return Unimplemented(
      "Cross replica sum not implemented on CPU. See b/33011107.");
}

Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  VLOG(2) << "HandleParameter: " << parameter->ToString();
  auto param_number = parameter->parameter_number();
  auto param_shape = parameter->shape();

  // We have to access the parameter at offset param_number in the params
  // array. The code generated here is equivalent to this C code:
  //
  //   i8* param_address_untyped = params[param_number];
  //   Param* param_address_typed = (Param*)param_address_untyped;
  //
  // Where Param is the actual element type of the underlying buffer (for
  // example, float for an XLA F32 element type).
  llvm::Argument* params = GetArg(compute_function_, 2);
  llvm::Value* param_address_offset =
      llvm_ir::EmitBufferIndexingGEP(params, param_number, &ir_builder_);
  llvm::LoadInst* param_address_untyped =
      ir_builder_.CreateLoad(param_address_offset);
  llvm::Value* param_address_typed = ir_builder_.CreateBitCast(
      param_address_untyped, IrShapeType(param_shape)->getPointerTo());
  emitted_value_[parameter] = param_address_typed;

  // Parameters of different types may not alias one another.
  llvm_ir::SetTbaaForInstruction(param_address_untyped, param_shape,
                                 /*is_pointer_to=*/true);
  if (!ShapeUtil::IsOpaque(param_shape)) {
    AttachAlignmentMetadataForLoad(param_address_untyped, param_shape);
    AttachDereferenceableMetadataForLoad(param_address_untyped, param_shape);
  }

  VLOG(2) << "  emitted value: " << llvm_ir::DumpToString(*param_address_typed);
  return Status::OK();
}

Status IrEmitter::HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                               HloInstruction* init_value,
                               tensorflow::gtl::ArraySlice<int64> dimensions,
                               HloComputation* function) {
  // The called computation should have been emitted previously.
  llvm::Function* reducer_function = FindOrDie(emitted_functions_, function);
  return EmitTargetElementLoop(
      reduce, [this, reduce, arg, init_value, dimensions,
               reducer_function](const llvm_ir::IrArray::Index& index) {
        // Initialize an accumulator with init_value.
        PrimitiveType accumulator_type = reduce->shape().element_type();
        llvm::AllocaInst* accumulator_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(accumulator_type, &ir_builder_),
            "accumulator", &ir_builder_,
            MinimumAlignmentForPrimitiveType(accumulator_type));
        llvm::Value* init_value_addr = GetEmittedValueFor(init_value);
        llvm::Value* load_init_value = ir_builder_.CreateLoad(init_value_addr);
        ir_builder_.CreateStore(load_init_value, accumulator_addr);

        // The enclosing loops go over all the target elements. Now we have to
        // compute the actual target element. For this, we build a new loop nest
        // to iterate over all the reduction dimensions in the argument.
        // AddLoopsForShapeOnDimensions will return an Index where induction
        // Value*s are placed for each dimension in dimensions, and all the rest
        // are nullptrs.
        llvm_ir::ForLoopNest loops(&ir_builder_);
        const llvm_ir::IrArray::Index reduced_dims_index =
            loops.AddLoopsForShapeOnDimensions(arg->shape(), dimensions,
                                               "reduction_dim");

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

        // Build a full index for the input argument, using reduced_dims_index
        // as the base. In reduced_dims_index only the reduction dimensions are
        // filled in. We fill in the rest of the dimensions with induction
        // Value*s taken from 'index' which iterates over the target array.
        // See the high-level description in the XLA documentation for details.
        llvm_ir::IrArray arg_array(GetIrArrayForOp(arg));
        llvm_ir::IrArray::Index input_index = reduced_dims_index;
        llvm_ir::IrArray::Index::const_iterator it = index.begin();

        for (int64 i = 0; i < input_index.size(); ++i) {
          if (input_index[i] == nullptr) {
            input_index[i] = *it++;
          }
        }
        CHECK(index.end() == it);

        // Apply the reduction function to the loaded value.
        llvm::Value* input_address =
            arg_array.EmitArrayElementAddress(input_index, &ir_builder_);
        llvm::Value* result = EmitElementFunctionCall(
            reducer_function, reduce->shape(),
            {accumulator_addr, input_address}, "reduce_function");
        ir_builder_.CreateStore(result, accumulator_addr);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
        return ir_builder_.CreateLoad(accumulator_addr);
      });
}

Status IrEmitter::HandleSend(HloInstruction* send) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Send is not implemented on CPU. See b/33942983.");
}

Status IrEmitter::HandleRecv(HloInstruction* recv) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Recv is not implemented on CPU. See b/33942983.");
}

Status IrEmitter::HandlePad(HloInstruction* pad) {
  // CPU backend does not properly handle negative padding but this is ok
  // because negative padding should be removed by the algebraic simplifier.
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      return Unimplemented(
          "Negative padding not supported in the CPU backend (b/34628603); "
          "this should have been eliminated at the HLO level: %s",
          pad->padding_config().ShortDebugString().c_str());
    }
  }

  // First, fill in the padding value to all output elements.
  TF_RETURN_IF_ERROR(EmitTargetElementLoop(
      pad, [this, pad](const llvm_ir::IrArray::Index& target_index) {
        const HloInstruction* padding_value = pad->operand(1);
        llvm::Value* padding_value_addr = GetEmittedValueFor(padding_value);
        return ir_builder_.CreateLoad(padding_value_addr);
      }));

  // Create a loop to iterate over the operand elements and update the output
  // locations where the operand elements should be stored.
  llvm_ir::ForLoopNest loops(&ir_builder_);
  const HloInstruction* operand = pad->operand(0);
  const llvm_ir::IrArray::Index operand_index =
      loops.AddLoopsForShape(operand->shape(), "operand");

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

  // Load an element from the operand.
  llvm_ir::IrArray operand_array(GetIrArrayForOp(operand));
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, &ir_builder_);

  // Compute the output index the operand element should be assigned to.
  // output_index := edge_padding_low + operand_index * (interior_padding + 1)
  const PaddingConfig& padding_config = pad->padding_config();
  llvm_ir::IrArray::Index output_index;
  for (int64 i = 0; i < operand_index.size(); ++i) {
    llvm::Value* offset = ir_builder_.CreateMul(
        operand_index[i],
        ir_builder_.getInt64(padding_config.dimensions(i).interior_padding() +
                             1));
    llvm::Value* index = ir_builder_.CreateAdd(
        offset,
        ir_builder_.getInt64(padding_config.dimensions(i).edge_padding_low()));
    output_index.push_back(index);
  }

  // Store the operand element to the computed output location.
  llvm_ir::IrArray output_array(GetIrArrayForOp(pad));
  output_array.EmitWriteArrayElement(output_index, operand_data, &ir_builder_);

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
  return Status::OK();
}

// If `hlo` is a Transpose, returns its operand; otherwise returns `hlo` itself.
static const HloInstruction* StripTranspose(const HloInstruction& hlo) {
  if (hlo.IsRank2Transpose()) {
    return hlo.operand(0);
  }
  return &hlo;
}

Status IrEmitter::HandleFusion(HloInstruction* fusion) {
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kTransposeDot) {
    const HloInstruction* dot = fusion->fused_expression_root();
    DCHECK(dot->opcode() == HloOpcode::kDot);
    const HloInstruction* lhs_parameter = StripTranspose(*dot->operand(0));
    const HloInstruction* rhs_parameter = StripTranspose(*dot->operand(1));
    DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
           rhs_parameter->opcode() == HloOpcode::kParameter);
    const HloInstruction* lhs =
        fusion->operand(lhs_parameter->parameter_number());
    const HloInstruction* rhs =
        fusion->operand(rhs_parameter->parameter_number());

    TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
        /*instruction=*/*dot, /*operands=*/{lhs, rhs},
        /*supported_types=*/{F32}));

    llvm_ir::IrArray lhs_array(GetIrArrayForOp(lhs));
    llvm_ir::IrArray rhs_array(GetIrArrayForOp(rhs));

    Shape target_shape = fusion->shape();
    TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                        EmitTargetAddressForOp(fusion));
    llvm_ir::IrArray target_array(target_address, target_shape);
    AddAliasingInformationToIrArray(*fusion, &target_array);

    VLOG(2) << "HandleFusion kTransposeDot: ";
    VLOG(2) << "  lhs operand: "
            << llvm_ir::DumpToString(*lhs_array.GetBasePointer());
    VLOG(2) << "  rhs operand: "
            << llvm_ir::DumpToString(*rhs_array.GetBasePointer());
    VLOG(2) << "  target: "
            << llvm_ir::DumpToString(*target_array.GetBasePointer());

    // Dot operation is complicated so we delegate to a helper class.
    TF_RETURN_IF_ERROR(DotOpEmitter::EmitDotOperation(
        *dot, dot->operand(0)->IsRank2Transpose(),
        dot->operand(1)->IsRank2Transpose(), target_array, lhs_array, rhs_array,
        GetExecutableRunOptionsArgument(), &ir_builder_));

    emitted_value_[fusion] = target_address;
    return Status::OK();
  } else if (fusion->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    std::vector<llvm_ir::IrArray> parameter_arrays;
    for (HloInstruction* operand : fusion->operands()) {
      parameter_arrays.push_back(GetIrArrayForOp(operand));
    }
    CpuElementalIrEmitter elemental_emitter(hlo_module_config_, &ir_builder_,
                                            module_);
    FusedIrEmitter fused_emitter(parameter_arrays, &elemental_emitter);
    TF_RETURN_IF_ERROR(fusion->fused_expression_root()->Accept(&fused_emitter));

    return EmitTargetElementLoop(fusion, fused_emitter.GetRootGenerator());
  } else {
    return Unimplemented("Fusion kind not implemented on CPU");
  }
}

Status IrEmitter::HandleCall(
    HloInstruction* call, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* computation) {
  llvm::Function* call_ir_function = FindOrDie(emitted_functions_, computation);

  std::vector<llvm::Value*> parameter_addresses;
  for (HloInstruction* operand : operands) {
    parameter_addresses.push_back(GetEmittedValueFor(operand));
  }

  TF_ASSIGN_OR_RETURN(llvm::Value * output_address,
                      EmitTargetAddressForOp(call));

  EmitArrayFunctionCallInto(call_ir_function, parameter_addresses,
                            output_address, computation->name());

  emitted_value_[call] = output_address;
  return Status::OK();
}

Status IrEmitter::HandleCustomCall(
    HloInstruction* custom_call,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target) {
  llvm::Type* i8_ptr_type = ir_builder_.getInt8PtrTy();
  llvm::AllocaInst* operands_alloca =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          i8_ptr_type, ir_builder_.getInt32(operands.size()),
          "cc_operands_alloca", &ir_builder_);
  for (int i = 0; i < operands.size(); ++i) {
    const HloInstruction* operand = operands[i];
    llvm::Value* operand_as_i8ptr =
        ir_builder_.CreatePointerCast(GetEmittedValueFor(operand), i8_ptr_type);
    llvm::Value* slot_in_operands_alloca = ir_builder_.CreateInBoundsGEP(
        operands_alloca, {ir_builder_.getInt32(i)});
    ir_builder_.CreateStore(operand_as_i8ptr, slot_in_operands_alloca);
  }
  auto* custom_call_ir_function =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          llvm_ir::AsStringRef(custom_call_target),
          llvm::FunctionType::get(
              /*Result=*/ir_builder_.getVoidTy(),
              /*Params=*/{i8_ptr_type, operands_alloca->getType()},
              /*isVarArg=*/false)));

  TF_ASSIGN_OR_RETURN(llvm::Value * output_address,
                      EmitTargetAddressForOp(custom_call));
  auto* output_address_arg =
      ir_builder_.CreatePointerCast(output_address, i8_ptr_type);

  ir_builder_.CreateCall(custom_call_ir_function,
                         {output_address_arg, operands_alloca});

  emitted_value_[custom_call] = output_address;
  return Status::OK();
}

Status IrEmitter::HandleWhile(HloInstruction* xla_while, HloInstruction* init,
                              HloComputation* condition, HloComputation* body) {
  // Precondition: Condition computation must return a scalar bool.
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool";
  // Check that all while-related buffers share an allocation.
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshape(
      xla_while->shape(),
      [this, &xla_while](const Shape& /*subshape*/,
                         const ShapeIndex& index) -> Status {
        auto check = [this](const HloInstruction* a, const HloInstruction* b,
                            const ShapeIndex& index) {
          BufferAllocation::Index index_a =
              assignment_.GetUniqueAllocation(a, index)
                  .ConsumeValueOrDie()
                  ->index();
          BufferAllocation::Index index_b =
              assignment_.GetUniqueAllocation(b, index)
                  .ConsumeValueOrDie()
                  ->index();
          if (index_a != index_b) {
            return InternalError(
                "instruction %s does not share allocation with "
                "instruction %s ",
                a->ToString().c_str(), b->ToString().c_str());
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
  emitted_value_[xla_while] = GetEmittedValueFor(init);

  // The called computation should have been emitted previously.
  llvm::Function* condition_ir_function =
      FindOrDie(emitted_functions_, condition);
  llvm::Function* body_ir_function = FindOrDie(emitted_functions_, body);

  // Generating:
  //   while (Condition(while_result)) {
  //     // CopyInsertion pass inserts copies which enable 'while_result' to
  //     // be passed back in as 'Body' parameter.
  //     while_result = Body(while_result);  // Insert
  //   }

  // Terminates the current block with a branch to a while header.
  llvm::BasicBlock* header_bb = llvm::BasicBlock::Create(
      module_->getContext(), "while_header", compute_function_);
  ir_builder_.CreateBr(header_bb);
  ir_builder_.SetInsertPoint(header_bb);

  // Calls the condition function to determine whether to proceed with the
  // body.  It must return a bool, so use the scalar call form.
  llvm::Value* while_result = GetEmittedValueFor(xla_while);
  llvm::Value* while_condition = EmitElementFunctionCall(
      condition_ir_function, condition->root_instruction()->shape(),
      {while_result}, "condition_function");
  llvm::Value* while_predicate = ir_builder_.CreateICmpNE(
      while_condition,
      llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(PRED, &ir_builder_),
                             0));

  // Branches to the body or to the while exit depending on the condition.
  llvm::BasicBlock* body_bb = llvm::BasicBlock::Create(
      module_->getContext(), "while_body", compute_function_);
  llvm::BasicBlock* exit_bb =
      llvm::BasicBlock::Create(module_->getContext(), "while__exit");
  ir_builder_.CreateCondBr(while_predicate, body_bb, exit_bb);

  // Calls the body function from the body block.
  ir_builder_.SetInsertPoint(body_bb);

  // Calls the body function.
  EmitArrayFunctionCallInto(body_ir_function, {while_result}, while_result,
                            "while_body");
  // Finishes with a branch back to the header.
  ir_builder_.CreateBr(header_bb);

  // Adds the exit block to the function and sets the insert point there.
  compute_function_->getBasicBlockList().push_back(exit_bb);
  ir_builder_.SetInsertPoint(exit_bb);

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
  llvm::Value* root_value = GetEmittedValueFor(root);
  VLOG(2) << "  value: " << llvm_ir::DumpToString(*root_value);

  if (auto* prof_counter = GetProfileCounterFor(/*hlo=*/nullptr)) {
    profiling_state_.RecordCompleteComputation(&ir_builder_, prof_counter);
  }

  ir_builder_.CreateRetVoid();
  return Status::OK();
}

llvm::Value* IrEmitter::GetProfileCounterFor(const HloInstruction* hlo) {
  string counter_name;
  size_t prof_counter_idx;
  if (!hlo_to_profile_idx_) {
    return nullptr;
  }
  if (hlo) {
    auto it = hlo_to_profile_idx_->find(hlo);
    if (it == hlo_to_profile_idx_->end()) {
      return nullptr;
    }

    prof_counter_idx = it->second;
    uintptr_t hlo_address = reinterpret_cast<uintptr_t>(hlo);
    counter_name = tensorflow::strings::StrCat(
        "prof_counter_0x",
        tensorflow::strings::Hex(
            hlo_address, tensorflow::strings::PadSpec(sizeof(hlo_address))));
  } else {
    prof_counter_idx = hlo_to_profile_idx_->size();
    counter_name = "prof_counter_computation";
  }
  return ir_builder_.CreateGEP(GetProfileCountersArgument(),
                               ir_builder_.getInt64(prof_counter_idx),
                               llvm_ir::AsStringRef(counter_name));
}

void IrEmitter::ProfilingState::UpdateProfileCounter(
    llvm::IRBuilder<>* ir_builder, llvm::Value* prof_counter,
    llvm::Value* cycle_end, llvm::Value* cycle_start) {
  auto* cycle_diff = ir_builder->CreateSub(cycle_end, cycle_start);
  llvm::LoadInst* old_cycle_count =
      ir_builder->CreateLoad(prof_counter, "old_cycle_count");
  auto* new_cycle_count =
      ir_builder->CreateAdd(cycle_diff, old_cycle_count, "new_cycle_count");
  ir_builder->CreateStore(new_cycle_count, prof_counter);
}

llvm::Value* IrEmitter::ProfilingState::ReadCycleCounter(
    llvm::IRBuilder<>* ir_builder) {
  llvm::Module* module = ir_builder->GetInsertBlock()->getModule();
  if (use_rdtscp_) {
    llvm::Function* func_llvm_readcyclecounter =
        llvm::Intrinsic::getDeclaration(module,
                                        llvm::Intrinsic::readcyclecounter);
    return ir_builder->CreateCall(func_llvm_readcyclecounter);
  }
  llvm::Function* func_llvm_x86_rdtscp =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::x86_rdtscp);
  if (!aux_i8ptr_) {
    llvm::AllocaInst* rdtscp_aux = llvm_ir::EmitAllocaAtFunctionEntry(
        ir_builder->getInt32Ty(), "rdtscp_aux", ir_builder);
    aux_i8ptr_ =
        ir_builder->CreateBitCast(rdtscp_aux, ir_builder->getInt8PtrTy());
  }
  llvm::ConstantInt* alloca_size = ir_builder->getInt64(4);
  llvm::Function* func_llvm_lifetime_start =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::lifetime_start);
  ir_builder->CreateCall(func_llvm_lifetime_start, {alloca_size, aux_i8ptr_});
  llvm::Value* rdtscp_call =
      ir_builder->CreateCall(func_llvm_x86_rdtscp, aux_i8ptr_);
  llvm::Function* func_llvm_lifetime_end =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::lifetime_end);
  ir_builder->CreateCall(func_llvm_lifetime_end, {alloca_size, aux_i8ptr_});
  return rdtscp_call;
}

void IrEmitter::ProfilingState::RecordCycleStart(llvm::IRBuilder<>* ir_builder,
                                                 HloInstruction* hlo) {
  auto* cycle_start = ReadCycleCounter(ir_builder);
  cycle_starts_[hlo] = cycle_start;
  if (first_read_cycle_start_ == nullptr) {
    first_read_cycle_start_ = cycle_start;
  }
}

void IrEmitter::ProfilingState::RecordCycleDelta(llvm::IRBuilder<>* ir_builder,
                                                 HloInstruction* hlo,
                                                 llvm::Value* prof_counter) {
  auto* cycle_end = ReadCycleCounter(ir_builder);
  auto* cycle_start = cycle_starts_[hlo];
  UpdateProfileCounter(ir_builder, prof_counter, cycle_end, cycle_start);
  last_read_cycle_end_ = cycle_end;
}

void IrEmitter::ProfilingState::RecordCompleteComputation(
    llvm::IRBuilder<>* ir_builder, llvm::Value* prof_counter) {
  if (is_entry_computation_ && last_read_cycle_end_ &&
      first_read_cycle_start_) {
    UpdateProfileCounter(ir_builder, prof_counter, last_read_cycle_end_,
                         first_read_cycle_start_);
  }
}

Status IrEmitter::Preprocess(HloInstruction* hlo) {
  if (hlo_to_profile_idx_ && hlo_to_profile_idx_->count(hlo)) {
    profiling_state_.RecordCycleStart(&ir_builder_, hlo);
  }
  return Status::OK();
}

Status IrEmitter::Postprocess(HloInstruction* hlo) {
  if (auto* prof_counter = GetProfileCounterFor(hlo)) {
    profiling_state_.RecordCycleDelta(&ir_builder_, hlo, prof_counter);
  }
  return Status::OK();
}

llvm_ir::IrArray IrEmitter::GetIrArrayForOp(const HloInstruction* hlo) {
  llvm::Value* value_for_op = GetEmittedValueFor(hlo);

  llvm_ir::IrArray array(value_for_op, hlo->shape());
  AddAliasingInformationToIrArray(*hlo, &array);
  return array;
}

llvm::Value* IrEmitter::GetEmittedValueFor(const HloInstruction* hlo) {
  auto it = emitted_value_.find(hlo);
  if (it == emitted_value_.end()) {
    LOG(FATAL) << "could not find emitted value for: " << hlo->ToString();
  }
  return it->second;
}

llvm::Type* IrEmitter::IrShapeType(const Shape& shape) {
  return llvm_ir::ShapeToIrType(shape, &ir_builder_);
}

llvm::Argument* IrEmitter::GetResultArgument() {
  return GetArg(compute_function_, 0);
}

llvm::Argument* IrEmitter::GetProfileCountersArgument() {
  return hlo_to_profile_idx_ ? GetArg(compute_function_, 4) : nullptr;
}

llvm::Value* IrEmitter::GetTempBuffersArgument() {
  return GetArg(compute_function_, 3);
}

llvm::Value* IrEmitter::GetExecutableRunOptionsArgument() {
  return GetArg(compute_function_, 1);
}

llvm::Value* IrEmitter::EmitTempBufferPointer(
    BufferAllocation::Index temp_buf_index, const Shape& target_shape) {
  llvm::Type* element_type = IrShapeType(target_shape);
  // The alignment and number of bytes within the temporary buffer is determined
  // by the maximal shape as determined by buffer assignment.
  const BufferAllocation& allocation =
      assignment_.GetAllocation(temp_buf_index);
  if (allocation.is_thread_local()) {
    // Thread-local allocations should only be assigned a single buffer.
    CHECK_EQ(1, allocation.assigned_buffers().size());
    const Shape& shape = allocation.assigned_buffers()[0]->shape();

    llvm::AllocaInst*& tempbuf_address = thread_local_buffers_[{
        ir_builder_.GetInsertBlock()->getParent(), temp_buf_index}];
    if (tempbuf_address == nullptr) {
      tempbuf_address = llvm_ir::EmitAllocaAtFunctionEntry(
          IrShapeType(shape),
          tensorflow::strings::StrCat("thread_local", temp_buf_index),
          &ir_builder_, MinimumAlignmentForShape(target_shape));
    }
    return ir_builder_.CreateBitCast(tempbuf_address,
                                     element_type->getPointerTo());
  }

  llvm::Value* tempbuf_address_offset = llvm_ir::EmitBufferIndexingGEP(
      GetTempBuffersArgument(), temp_buf_index, &ir_builder_);
  llvm::LoadInst* tempbuf_address_untyped =
      ir_builder_.CreateLoad(tempbuf_address_offset);
  //  Loading the address of a buffer is invariant of the point at which the
  //  load is executed in the program because we never reassign buffers.
  tempbuf_address_untyped->setMetadata(
      llvm::LLVMContext::MD_invariant_load,
      llvm::MDNode::get(tempbuf_address_untyped->getContext(), /*MDs=*/{}));
  llvm_ir::SetTbaaForInstruction(tempbuf_address_untyped, target_shape,
                                 /*is_pointer_to=*/true);

  AttachAlignmentMetadataForLoad(tempbuf_address_untyped, allocation.size());
  AttachDereferenceableMetadataForLoad(tempbuf_address_untyped,
                                       allocation.size());
  return ir_builder_.CreateBitCast(tempbuf_address_untyped,
                                   element_type->getPointerTo());
}

// Emits a function call returning a single array element.  Allocates space
// for a single element_type value, and loads it after call.
llvm::Value* IrEmitter::EmitElementFunctionCall(
    llvm::Function* function, const Shape& return_shape,
    tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
    tensorflow::StringPiece name) {
  llvm::Value* return_value_buffer = EmitArrayFunctionCall(
      function, return_shape, 1, parameter_addresses, name);
  return ir_builder_.CreateLoad(
      return_value_buffer,
      llvm_ir::AsStringRef(tensorflow::strings::StrCat(name, "_return_value")));
}

// Emits a core function call based on the following pseudo-code.
//
//   char** parameter_addresses_buffer =
//       allocate buffer with a pointer for each parameter to the function
//   for each parameter index, i.e. for i = 0, ..., #parameters:
//     parameter_addresses_buffer[i] = parameter_addresses[i]
//   call function(return_value_buffer,
//                 parameter_addresses_buffer,
//                 temps)
//   return return_value_buffer  -- address of the return value.
void IrEmitter::EmitArrayFunctionCallInto(
    llvm::Function* function,
    tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
    llvm::Value* return_value_buffer, tensorflow::StringPiece name) {
  llvm::Value* parameter_addresses_buffer =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          ir_builder_.getInt8PtrTy(),
          ir_builder_.getInt32(parameter_addresses.size()),
          tensorflow::strings::StrCat(name, "_parameter_addresses"),
          &ir_builder_);
  for (int i = 0; i < parameter_addresses.size(); ++i) {
    llvm::Value* parameter_as_i8ptr = ir_builder_.CreateBitCast(
        parameter_addresses[i], ir_builder_.getInt8PtrTy(),
        llvm_ir::AsStringRef(tensorflow::strings::StrCat(name, "_parameter_", i,
                                                         "_address_as_i8ptr")));
    llvm::Value* slot_in_param_adresses = ir_builder_.CreateInBoundsGEP(
        parameter_addresses_buffer, {ir_builder_.getInt32(i)});
    ir_builder_.CreateStore(parameter_as_i8ptr, slot_in_param_adresses);
  }

  const auto to_int8_ptr = [this](llvm::Value* ptr) {
    return ir_builder_.CreatePointerCast(ptr, ir_builder_.getInt8PtrTy());
  };
  std::vector<llvm::Value*> arguments{
      to_int8_ptr(return_value_buffer),
      to_int8_ptr(GetExecutableRunOptionsArgument()),
      parameter_addresses_buffer, GetTempBuffersArgument()};
  if (auto* profile_counters = GetProfileCountersArgument()) {
    arguments.push_back(profile_counters);
  }
  ir_builder_.CreateCall(function, arguments);
}

llvm::Value* IrEmitter::EmitArrayFunctionCall(
    llvm::Function* function, const Shape& return_shape, int64 element_count,
    tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
    tensorflow::StringPiece name) {
  llvm::Value* elements =
      llvm::ConstantInt::get(ir_builder_.getInt64Ty(), element_count);
  PrimitiveType return_type = return_shape.element_type();
  llvm::Value* return_value_buffer =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          llvm_ir::PrimitiveTypeToIrType(return_type, &ir_builder_), elements,
          tensorflow::strings::StrCat(name, "_return_value_address"),
          &ir_builder_, MinimumAlignmentForPrimitiveType(return_type));
  EmitArrayFunctionCallInto(function, parameter_addresses, return_value_buffer,
                            name);
  return return_value_buffer;
}

StatusOr<llvm::Value*> IrEmitter::EmitTargetAddressForOp(
    const HloInstruction* op) {
  const Shape& target_shape = op->shape();
  if (op == op->parent()->root_instruction()) {
    // For the root node, we write directly to the output buffer of the
    // function.
    llvm::Argument* retval = GetResultArgument();
    if (!ShapeUtil::HasZeroElements(target_shape)) {
      llvm::AttrBuilder attr_builder;
      attr_builder.addAlignmentAttr(MinimumAlignmentForShape(target_shape));
      attr_builder.addDereferenceableAttr(ByteSizeOf(target_shape));
      retval->addAttr(llvm::AttributeSet::get(
          retval->getContext(), retval->getArgNo() + 1, attr_builder));
    }
    return ir_builder_.CreateBitCast(retval,
                                     IrShapeType(target_shape)->getPointerTo());
  }

  // For other nodes, we need the temporary buffer allocated for this node to
  // write the result into.
  TF_ASSIGN_OR_RETURN(const BufferAllocation* allocation,
                      assignment_.GetUniqueTopLevelAllocation(op));
  return EmitTempBufferPointer(allocation->index(), target_shape);
}

Status IrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op,
    const llvm_ir::ElementGenerator& element_generator) {
  VLOG(2) << "EmitTargetElementLoop: " << target_op->ToString();

  // target_address will hold the address of the target buffer we will write the
  // result of the computation into.
  const Shape& target_shape = target_op->shape();
  TF_ASSIGN_OR_RETURN(llvm::Value * target_address,
                      EmitTargetAddressForOp(target_op));
  VLOG(2) << "  target address: " << llvm_ir::DumpToString(*target_address);
  llvm_ir::IrArray target_array(target_address, target_shape);
  AddAliasingInformationToIrArray(*target_op, &target_array);

  TF_RETURN_IF_ERROR(
      llvm_ir::LoopEmitter(element_generator, target_array, &ir_builder_)
          .EmitLoop());
  emitted_value_[target_op] = target_address;
  return Status::OK();
}

Status IrEmitter::EmitMemcpy(const HloInstruction& source,
                             const HloInstruction& destination) {
  llvm::Value* source_value = GetEmittedValueFor(&source);
  llvm::Value* destination_value = GetEmittedValueFor(&destination);
  int64 source_size = ByteSizeOf(source.shape());
  ir_builder_.CreateMemCpy(destination_value, source_value, source_size, 1);
  return Status::OK();
}

Status IrEmitter::ElementTypesSameAndSupported(
    const HloInstruction& instruction,
    tensorflow::gtl::ArraySlice<const HloInstruction*> operands,
    tensorflow::gtl::ArraySlice<PrimitiveType> supported_types) {
  for (auto operand : operands) {
    TF_RET_CHECK(
        ShapeUtil::SameElementType(operands[0]->shape(), operand->shape()));
  }

  TF_RET_CHECK(!operands.empty());
  PrimitiveType primitive_type = operands[0]->shape().element_type();
  if (std::find(supported_types.begin(), supported_types.end(),
                primitive_type) == supported_types.end()) {
    return Unimplemented("unsupported operand type %s in op %s",
                         PrimitiveType_Name(primitive_type).c_str(),
                         HloOpcodeString(instruction.opcode()).c_str());
  }
  return Status::OK();
}

Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArrayForOp(operand).EmitReadArrayElement(index, &ir_builder_);
    };
  }
  CpuElementalIrEmitter elemental_emitter(hlo_module_config_, &ir_builder_,
                                          module_);
  return EmitTargetElementLoop(
      hlo, elemental_emitter.MakeElementGenerator(hlo, operand_to_generator));
}

}  // namespace cpu
}  // namespace xla
