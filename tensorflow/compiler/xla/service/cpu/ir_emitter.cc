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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {

namespace {
using llvm_ir::AsStringRef;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;
}  // namespace

namespace cpu {

IrEmitter::IrEmitter(
    const HloModule& hlo_module, const BufferAssignment& assignment,
    llvm::Module* llvm_module,
    const std::unordered_map<const HloInstruction*, size_t>* hlo_to_profile_idx,
    llvm::TargetMachine* target_machine,
    ExternalConstantPool* external_constant_pool)
    : assignment_(assignment),
      module_(llvm_module),
      arch_type_(llvm::Triple(llvm_module->getTargetTriple()).getArch()),
      ir_builder_(llvm_module->getContext()),
      hlo_to_profile_idx_(hlo_to_profile_idx),
      alias_analysis_(hlo_module, assignment, &llvm_module->getContext()),
      hlo_module_config_(hlo_module.config()),
      parallel_cpu_backend_(
          options::CpuParallelBackendRequested(hlo_module_config_)),
      is_top_level_computation_(false),
      target_machine_features_(target_machine),
      external_constant_pool_(external_constant_pool) {
  ir_builder_.setFastMathFlags(llvm_ir::GetFastMathFlags(
      /*fast_math_enabled=*/hlo_module_config_.debug_options()
          .xla_enable_fast_math()));
}

StatusOr<llvm::Function*> IrEmitter::EmitComputation(
    HloComputation* computation, const string& function_name_prefix,
    bool is_top_level_computation,
    std::vector<const HloInstruction*>* instruction_order) {
  string function_name = name_uniquer_.GetUniqueName(function_name_prefix);
  VLOG(2) << "Emitting IR for CPU function [" << function_name_prefix
          << "]; ordered? " << (instruction_order != nullptr);
  is_top_level_computation_ = is_top_level_computation;
  num_dynamic_loop_bounds_ = 0;
  if (!computation->root_instruction()->outer_dimension_partitions().empty()) {
    num_dynamic_loop_bounds_ =
        computation->root_instruction()->outer_dimension_partitions().size();
  }

  InitializeIrFunction(function_name);
  // The rdtscp instruction is x86 specific.  We will fallback to LLVM's generic
  // readcyclecounter if it is unavailable.
  bool use_rdtscp = arch_type_ == llvm::Triple::ArchType::x86 ||
                    arch_type_ == llvm::Triple::ArchType::x86_64;
  profiling_state_ = ProfilingState(is_top_level_computation_, use_rdtscp,
                                    GetProfileCountersArgument());
  if (instruction_order == nullptr) {
    TF_RETURN_IF_ERROR(computation->Accept(this));
  } else {
    TF_RETURN_IF_ERROR(computation->AcceptOrdered(this, *instruction_order));
  }
  InsertOrDie(&emitted_functions_, computation, compute_function_);

  return compute_function_;
}

static llvm::Argument* GetArg(llvm::Function* f, int idx) {
  llvm::Function::arg_iterator arg_iter = f->arg_begin();
  std::advance(arg_iter, idx);
  return &*arg_iter;
}

void IrEmitter::InitializeIrFunction(const string& function_name) {
  // The function signature is:
  //   void function(i8* retval, i8* run_options, i8** params, i8** temps,
  //                 i64* dynamic_loop_bounds, i64* prof_counters)
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
  //                        /--------------------------------------------\
  // dynamic loop bounds -> | outer_dim0_start | outer_dim0_limit | .....|
  //  (elided for aot)      \--------------------------------------------/
  //
  //                     /---------------------------------------------\
  //   prof counters ->  | counter 0 | counter 1 | ..... | counter N-1 |
  //  (elided for aot)   \---------------------------------------------/

  // Even though the type of params and temps is void** in the host's view, in
  // LLVM IR this is represented by i8*, similarly to void*. It's up to the code
  // to use GEPs to unravel the indirection layers.
  llvm::FunctionType* compute_function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(module_->getContext()),
      /*Params=*/GetComputeFunctionParams(),
      /*isVarArg=*/false);

  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  llvm::Function::LinkageTypes linkage =
      is_top_level_computation_ ? llvm::GlobalValue::ExternalLinkage
                                : llvm::GlobalValue::InternalLinkage;
  compute_function_ =
      llvm::Function::Create(/*Ty=*/compute_function_type,
                             /*Linkage=*/linkage,
                             /*Name=*/AsStringRef(function_name),
                             /*Module=*/module_);
  compute_function_->setCallingConv(llvm::CallingConv::C);

  // Set meaningful names for the function's arguments: useful for debugging.
  llvm::Function::arg_iterator arg_iter = compute_function_->arg_begin();
  arg_iter->setName("retval");
  (++arg_iter)->setName("run_options");
  (++arg_iter)->setName("params");
  (++arg_iter)->setName("temps");
  if (num_dynamic_loop_bounds_ > 0) {
    (++arg_iter)->setName("dynamic_loop_bounds");
  }
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
    compute_function_->addAttribute(argument.getArgNo() + 1,
                                    llvm::Attribute::NoAlias);
  }

  // Add the optize attribute to the function if optimizing for size. This
  // controls internal behavior of some optimization passes (e.g. loop
  // unrolling).
  if (options::OptimizeForSizeRequested(hlo_module_config_)) {
    compute_function_->addFnAttr(llvm::Attribute::OptimizeForSize);
  }

  if (hlo_module_config_.debug_options().xla_enable_fast_math()) {
    compute_function_->addFnAttr("unsafe-fp-math", "true");
    compute_function_->addFnAttr("no-infs-fp-math", "true");
    compute_function_->addFnAttr("no-nans-fp-math", "true");
    compute_function_->addFnAttr("no-signed-zeros-fp-math", "true");
  }

  ir_builder_.SetInsertPoint(llvm::BasicBlock::Create(
      /*Context=*/module_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/compute_function_));
}

IrEmitter::~IrEmitter() {}

Status IrEmitter::HandleBitcast(HloInstruction* bitcast) {
  VLOG(2) << "HandleBitcast: " << bitcast->ToString();
  emitted_value_[bitcast] =
      ir_builder_.CreateBitCast(GetEmittedValueFor(bitcast->operand(0)),
                                IrShapeType(bitcast->shape())->getPointerTo(),
                                AsStringRef(IrName(bitcast)));
  return Status::OK();
}

Status IrEmitter::HandleConstant(HloInstruction* constant) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  const Literal& literal = constant->literal();
  llvm::GlobalVariable* global_for_const;

  // We avoid creating large constants in the LLVM IR since LLVM is not
  // efficient for large constant arrays.  We still emit "small enough" constant
  // arrays into the Ir, in the off chance the LLVM optimizer can do something
  // interesting with it.
  const int kMaxInternalConstantSizeInBytes = 128;
  if (external_constant_pool_ &&
      ByteSizeOf(literal.shape()) >= kMaxInternalConstantSizeInBytes) {
    string global_name = tensorflow::strings::StrCat(
        "constant_global_", external_global_constant_counter_++);
    global_for_const = new llvm::GlobalVariable(
        /*Module=*/*module_,
        /*Type=*/IrShapeType(literal.shape()),
        /*isConstant=*/true,
        /*Linkage=*/llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/nullptr,
        /*Name=*/AsStringRef(global_name));
    global_for_const->setAlignment(MinimumAlignmentForShape(literal.shape()));
    external_constant_pool_->Insert(global_name, literal,
                                    MinimumAlignmentForShape(literal.shape()));
  } else {
    llvm::Constant* initializer =
        llvm_ir::ConvertLiteralToIrConstant(literal, module_);
    global_for_const = new llvm::GlobalVariable(
        /*Module=*/*module_,
        /*Type=*/initializer->getType(),
        /*isConstant=*/true,
        /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
        /*Initializer=*/initializer,
        /*Name=*/"");
    global_for_const->setAlignment(MinimumAlignmentForShape(literal.shape()));
  }
  emitted_value_[constant] = global_for_const;
  VLOG(2) << "  emitted value: " << llvm_ir::DumpToString(*global_for_const);
  VLOG(2) << "  its type: "
          << llvm_ir::DumpToString(*global_for_const->getType());
  return Status::OK();
}

Status IrEmitter::HandleCopy(HloInstruction* copy) {
  if (ShapeUtil::IsTuple(copy->shape())) {
    // kCopy shallow copies a tuple so just memcpy the top-level buffer.
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(copy));
    return EmitMemcpy(*(copy->operand(0)), *copy);
  } else {
    // Use the elemental emitter for non-tuple shapes.
    return DefaultAction(copy);
  }
}

// Calculate the alignment of a buffer with a particular size.
int IrEmitter::MinimumAlignmentForBufferSize(int64 buffer_size) {
  // GLibc returns a pointer with alignment 8 on 32-bit platforms and 16 on
  // 64-bit platforms.  TCMalloc returns a pointer with alignment 8 for
  // allocations smaller than kMallocAlignmentThreshold bytes and at least
  // alignment 16 for allocations greater than or equal to
  // kMallocAlignmentThreshold bytes.  N.B. We could improve on this lower bound
  // by explicitly allocating the memory with posix_memalign.  This is
  // complicated by our desire to allow parameter buffers created by clients to
  // be consumed directly by the JIT.
  if (buffer_size == 0) {
    // No need to align empty buffers.
    return 1;
  }

  const int64 kMallocAlignmentThreshold = 512;

  int pointer_size = module_->getDataLayout().getPointerSize();
  int buffer_alignment = buffer_size >= kMallocAlignmentThreshold
                             ? 2 * pointer_size
                             : pointer_size;
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

Status IrEmitter::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  // A tuple is an array of pointers, one for each operand. Each pointer points
  // to the output buffer of its corresponding operand. A GetTupleElement
  // instruction forwards a pointer to the tuple element buffer at the given
  // index.
  auto operand = get_tuple_element->operand(0);
  const Shape& shape = get_tuple_element->shape();
  emitted_value_[get_tuple_element] = llvm_ir::EmitGetTupleElement(
      shape, get_tuple_element->tuple_index(), MinimumAlignmentForShape(shape),
      GetEmittedValueFor(operand), &ir_builder_, module_);
  return Status::OK();
}

Status IrEmitter::HandleSelect(HloInstruction* select) {
  auto pred = select->operand(0);
  auto on_true = select->operand(1);
  auto on_false = select->operand(2);
  TF_RET_CHECK(pred->shape().element_type() == PRED);

  if (ShapeUtil::IsTuple(select->shape())) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(select));
    llvm_ir::EmitTupleSelect(
        GetIrArrayFor(select), GetIrArrayFor(pred), GetEmittedValueFor(on_true),
        GetEmittedValueFor(on_false), &ir_builder_, module_);
    return Status::OK();
  }

  return DefaultAction(select);
}

Status IrEmitter::HandleInfeed(HloInstruction* infeed) {
  VLOG(2) << "HandleInfeed: " << infeed->ToString();

  const Shape& shape = infeed->shape();

  // The infeed operation produces data (dequeued from the infeed queue) at this
  // address, which has been provided by buffer assignment.
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(infeed));
  llvm_ir::IrArray infeed_array = GetIrArrayFor(infeed);

  if (ShapeUtil::IsTuple(shape)) {
    TF_RET_CHECK(!ShapeUtil::IsNestedTuple(shape));

    // For a tuple, we first copy each of the internal elements to
    // their corresponding target locations. We then construct the
    // tuple outer buffer containing pointers to the internal
    // elements.
    std::vector<llvm::Value*> tuple_element_addresses;
    for (int64 i = 0; i < shape.tuple_shapes_size(); ++i) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice buffer,
                          assignment_.GetUniqueSlice(infeed, {i}));

      const Shape& tuple_element_shape =
          ShapeUtil::GetTupleElementShape(shape, i);

      // Only the outer tuple buffer's target address is obtained from
      // GetEmittedValueFor, to handle the case when Infeed is the root
      // instruction. Target addresses for internal elements can be obtained
      // from EmitTempBufferPointer.
      llvm::Value* tuple_element_address =
          EmitTempBufferPointer(buffer, tuple_element_shape);

      TF_RETURN_IF_ERROR(EmitXfeedTransfer(
          XfeedKind::kInfeed, tuple_element_shape, tuple_element_address));

      tuple_element_addresses.push_back(tuple_element_address);
    }

    llvm_ir::EmitTuple(infeed_array, tuple_element_addresses, &ir_builder_,
                       module_);
  } else {
    TF_RETURN_IF_ERROR(EmitXfeedTransfer(XfeedKind::kInfeed, shape,
                                         GetEmittedValueFor(infeed)));
  }

  return Status::OK();
}

Status IrEmitter::EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                                    llvm::Value* program_buffer_address) {
  int64 length = ByteSizeOf(shape);
  if (length <= 0 || length > std::numeric_limits<int32>::max()) {
    return InvalidArgument(
        "xfeed (infeed or outfeed) buffer length %lld is outside the valid "
        "size range",
        length);
  }
  int32 length_32 = static_cast<int32>(length);

  int32 shape_length;
  TF_ASSIGN_OR_RETURN(llvm::Value * shape_ptr,
                      llvm_ir::EncodeSelfDescribingShapeConstant(
                          shape, &shape_length, &ir_builder_));

  // The signature of the acquire infeed buffer function is:
  //
  //   (void*)(int32 length);
  llvm::Type* int32_type = ir_builder_.getInt32Ty();
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::FunctionType* acquire_type = llvm::FunctionType::get(
      i8_ptr_type, {int32_type, i8_ptr_type, int32_type},
      /*isVarArg=*/false);

  llvm::Function* acquire_func;
  if (kind == XfeedKind::kInfeed) {
    acquire_func = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
        runtime::kAcquireInfeedBufferForDequeueSymbolName, acquire_type));
  } else {
    acquire_func = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
        runtime::kAcquireOutfeedBufferForPopulationSymbolName, acquire_type));
  }
  acquire_func->setCallingConv(llvm::CallingConv::C);

  // The signature of the release infeed buffer function is:
  //
  //   (void)(int32 length, void* buffer);
  llvm::FunctionType* release_type = llvm::FunctionType::get(
      ir_builder_.getVoidTy(),
      {int32_type, i8_ptr_type, i8_ptr_type, int32_type},
      /*isVarArg=*/false);

  llvm::Function* release_func;
  if (kind == XfeedKind::kInfeed) {
    release_func = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
        runtime::kReleaseInfeedBufferAfterDequeueSymbolName, release_type));
  } else {
    release_func = llvm::cast<llvm::Function>(module_->getOrInsertFunction(
        runtime::kReleaseOutfeedBufferAfterPopulationSymbolName, release_type));
  }
  release_func->setCallingConv(llvm::CallingConv::C);

  // Implementation note: this call informs the runtime that it wants a buffer
  // of size exactly 'length_32', and the runtime is responsible for
  // check-failing the process if there is a mismatch, versus passing us back a
  // buffer that we might overrun.
  llvm::Value* acquired_pointer = ir_builder_.CreateCall(
      acquire_func, {ir_builder_.getInt32(length_32), shape_ptr,
                     ir_builder_.getInt32(shape_length)});

  if (kind == XfeedKind::kInfeed) {
    // Copy to the program buffer address from the acquired buffer.
    ir_builder_.CreateMemCpy(program_buffer_address, acquired_pointer,
                             length_32, 1);
  } else {
    // Outfeed -- copy from the in-program address to the acquired buffer.
    ir_builder_.CreateMemCpy(acquired_pointer, program_buffer_address,
                             length_32, 1);
  }

  ir_builder_.CreateCall(release_func,
                         {ir_builder_.getInt32(length_32), acquired_pointer,
                          shape_ptr, ir_builder_.getInt32(shape_length)});

  return Status::OK();
}

Status IrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  HloInstruction* operand = outfeed->operands()[0];
  const Shape& operand_shape = operand->shape();

  llvm::Value* value = GetEmittedValueFor(operand);
  if (!ShapeUtil::IsTuple(operand_shape)) {
    return EmitXfeedTransfer(XfeedKind::kOutfeed, operand_shape, value);
  }

  TF_RET_CHECK(!ShapeUtil::IsNestedTuple(operand_shape));

  for (int64 i = 0; i < operand_shape.tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(operand_shape, i);
    llvm::Value* tuple_element = llvm_ir::EmitGetTupleElement(
        tuple_element_shape, i, MinimumAlignmentForShape(tuple_element_shape),
        value, &ir_builder_, module_);
    TF_RETURN_IF_ERROR(EmitXfeedTransfer(XfeedKind::kOutfeed,
                                         tuple_element_shape, tuple_element));
  }

  return Status::OK();
}

Status IrEmitter::HandleSort(HloInstruction* sort) {
  // TODO(b/26783907): Implement sort on CPU.
  return Unimplemented("Sort is not supported on CPU (b/26783907).");
}

Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(tuple));
  std::vector<llvm::Value*> base_ptrs;
  for (auto operand : tuple->operands()) {
    base_ptrs.push_back(GetEmittedValueFor(operand));
  }
  llvm_ir::EmitTuple(GetIrArrayFor(tuple), base_ptrs, &ir_builder_, module_);
  return Status::OK();
}

Status IrEmitter::HandleMap(HloInstruction* map) {
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(map->operands());
  HloComputation* function = map->to_apply();
  // The called computation should have been emitted previously.
  llvm::Function* mapped_ir_function = FindOrDie(emitted_functions_, function);

  return EmitTargetElementLoop(map, [this, map, operands, mapped_ir_function](
                                        const llvm_ir::IrArray::Index& index) {
    std::vector<llvm::Value*> parameter_addresses;
    for (const HloInstruction* operand : operands) {
      const llvm_ir::IrArray& array = GetIrArrayFor(operand);
      parameter_addresses.push_back(
          array.EmitArrayElementAddress(index, &ir_builder_));
    }
    return EmitElementFunctionCall(mapped_ir_function, map->shape(),
                                   parameter_addresses, "map_function");
  });
}

Status IrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {
  auto operand = reduce_window->operand(0);
  const Window& window = reduce_window->window();
  HloComputation* function = reduce_window->to_apply();
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
            llvm_ir::PrimitiveTypeToIrType(operand_element_type, module_),
            "reduce_window_accumulator_address", &ir_builder_,
            MinimumAlignmentForPrimitiveType(operand_element_type));
        ir_builder_.CreateStore(ir_builder_.CreateLoad(GetEmittedValueFor(
                                    reduce_window->operand(1))),
                                accumulator_address);

        llvm_ir::ForLoopNest loops(IrName(reduce_window, "inner"),
                                   &ir_builder_);
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
        for (size_t i = 0; i < index.size(); ++i) {
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
        llvm_ir::IrArray input_array(GetIrArrayFor(operand));
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
      select_and_scatter, /*desc=*/IrName(select_and_scatter, "init"),
      [this, init_value](const llvm_ir::IrArray::Index& target_index) {
        llvm::Value* init_value_addr = GetEmittedValueFor(init_value);
        return ir_builder_.CreateLoad(init_value_addr);
      }));

  // Create a loop to iterate over the source array to scatter to the output.
  llvm_ir::ForLoopNest source_loops(IrName(select_and_scatter), &ir_builder_);
  const llvm_ir::IrArray::Index source_index =
      source_loops.AddLoopsForShape(source->shape(), "source");
  SetToFirstInsertPoint(source_loops.GetInnerLoopBodyBasicBlock(),
                        &ir_builder_);

  // Allocate space to keep the currently selected value, its index, and
  // the boolean initialized_flag, which is initially set to false.
  llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
      llvm_ir::PrimitiveTypeToIrType(operand_element_type, module_),
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
  llvm_ir::ForLoopNest window_loops(IrName(select_and_scatter, "window"),
                                    &ir_builder_);
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
  llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
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
      result,
      llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0),
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
  llvm_ir::IrArray source_array(GetIrArrayFor(source));
  llvm::Value* source_value_address =
      source_array.EmitArrayElementAddress(source_index, &ir_builder_);
  llvm_ir::IrArray output_array(GetIrArrayFor(select_and_scatter));
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

Status IrEmitter::HandleDot(HloInstruction* dot) {
  auto lhs = dot->operand(0);
  auto rhs = dot->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*dot, /*operands=*/{lhs, rhs},
      /*supported_types=*/{F32, F64, C64}));

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
  return DotOpEmitter::EmitDotOperation(
      *dot, /*transpose_lhs=*/false, /*transpose_rhs=*/false, target_array,
      lhs_array, rhs_array, GetExecutableRunOptionsArgument(), &ir_builder_,
      hlo_module_config_);
}

Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  const auto& window = convolution->window();
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*convolution, /*operands=*/{lhs, rhs},
      /*supported_types=*/{F32, C64}));

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
      int64 input_rows = input_shape.dimensions(dnums.spatial_dimensions(0));
      int64 input_cols =
          one_dim_convolution
              ? 1
              : input_shape.dimensions(dnums.spatial_dimensions(1));
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
          convolution_shape.dimensions(dnums.spatial_dimensions(0));
      int64 output_cols =
          one_dim_convolution
              ? 1
              : convolution_shape.dimensions(dnums.spatial_dimensions(1));

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
      bool multi_threaded_eigen =
          hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
      const char* fn_name =
          (multi_threaded_eigen
               ? runtime::kEigenConvF32SymbolName
               : runtime::kEigenSingleThreadedConvF32SymbolName);
      llvm::Function* conv_func = llvm::cast<llvm::Function>(
          module_->getOrInsertFunction(fn_name, conv_type));
      conv_func->setCallingConv(llvm::CallingConv::C);
      conv_func->setDoesNotThrow();
      conv_func->setOnlyAccessesArgMemory();
      ir_builder_.CreateCall(
          conv_func, {
                         GetExecutableRunOptionsArgument(),
                         ir_builder_.CreateBitCast(
                             GetEmittedValueFor(convolution), float_ptr_type),
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
        llvm::Value* output_feature = index[dnums.output_feature_dimension()];
        llvm::Value* batch = index[dnums.output_batch_dimension()];

        // We will accumulate the products into this sum to calculate
        // the output entry at the given index.
        PrimitiveType lhs_element_type = lhs->shape().element_type();
        llvm::Value* sum_address = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(lhs_element_type, module_),
            "convolution_sum_address", &ir_builder_,
            MinimumAlignmentForPrimitiveType(lhs_element_type));
        ir_builder_.CreateStore(
            llvm::ConstantFP::get(ir_builder_.getFloatTy(), 0.0), sum_address);

        llvm_ir::ForLoopNest loops(IrName(convolution, "inner"), &ir_builder_);
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
                .AddLoop(
                    0, lhs->shape().dimensions(dnums.input_feature_dimension()),
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
        input_index[dnums.input_feature_dimension()] = input_feature;
        input_index[dnums.input_batch_dimension()] = batch;

        llvm_ir::IrArray kernel_array(GetIrArrayFor(rhs));
        llvm_ir::IrArray::Index kernel_index(num_dims);
        for (int i = 0; i < num_spatial_dims; ++i) {
          kernel_index[dnums.kernel_spatial_dimensions(i)] = kernel_spatial[i];
        }
        kernel_index[dnums.kernel_input_feature_dimension()] = input_feature;
        kernel_index[dnums.kernel_output_feature_dimension()] = output_feature;

        llvm_ir::IrArray input_array(GetIrArrayFor(lhs));
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

// Fills up the free variables in 'index_with_free_var' with values from
// 'filler_index'. The size of free variables must be the same as the
// size of 'filler_index'.
//
// This is often used after dimension reduction, where
// 'index_with_free_var' has one or more dimensions reduced, which serves as
// free variables (represented as nullptr). For example, if we have a 4
// dimensional input and index for the dimension being reduced is
// 2 (third dimension), we will have an index like [i, j, NULL, k]
// after reduced dimension.
//
// Here we fill up that free variable by 'filler_index', which contains
// the value in the reduced dimension.
static llvm_ir::IrArray::Index FillReducedDimensionIndex(
    llvm_ir::IrArray::Index index_with_free_var,
    llvm_ir::IrArray::Index filler_index) {
  llvm_ir::IrArray::Index::const_iterator it = filler_index.begin();

  for (size_t i = 0; i < index_with_free_var.size(); ++i) {
    if (index_with_free_var[i] == nullptr) {
      index_with_free_var[i] = *it++;
    }
  }
  CHECK(filler_index.end() == it);
  return index_with_free_var;
}

Status IrEmitter::HandleBatchNormTraining(HloInstruction* batch_norm_training) {
  // The output of BatchNormTraining is a tuple of three element:
  //   - An N-dimensional array containing normalized values.
  //   - A 1 dimensional array containing the mean value for each feature.
  //   - A 1 dimensional array containing the variance value for each feature.
  HloInstruction* operand = batch_norm_training->operands()[0];
  HloInstruction* scale = batch_norm_training->operands()[1];
  HloInstruction* offset = batch_norm_training->operands()[2];
  float epsilon = batch_norm_training->epsilon();
  int64 feature_index = batch_norm_training->feature_index();
  TF_RET_CHECK(ShapeUtil::IsTuple(batch_norm_training->shape()) &&
               ShapeUtil::TupleElementCount(batch_norm_training->shape()) == 3);

  const Shape& output_shape =
      ShapeUtil::GetTupleElementShape(batch_norm_training->shape(), 0);
  const Shape& feature_shape =
      ShapeUtil::GetTupleElementShape(batch_norm_training->shape(), 1);

  // Reduce vector of the non-feature dimensions.
  std::vector<int64> dimensions_to_reduce;

  for (int64 i = 0; i < operand->shape().dimensions_size(); ++i) {
    if (i != feature_index) {
      dimensions_to_reduce.push_back(i);
    }
  }

  // Get the second and third allocations in the output tuple, which should be
  // used to store the result of mean and variance value calculation.
  TF_ASSIGN_OR_RETURN(
      const BufferAllocation::Slice slice_mean,
      assignment_.GetUniqueSlice(batch_norm_training, /*index=*/{1}));
  TF_ASSIGN_OR_RETURN(
      const BufferAllocation::Slice slice_var,
      assignment_.GetUniqueSlice(batch_norm_training, /*index=*/{2}));
  const int feature_count = output_shape.dimensions(feature_index);
  const int size_in_elements = ShapeUtil::ElementsIn(output_shape);
  TF_RET_CHECK(ShapeUtil::ElementsIn(operand->shape()) == size_in_elements);
  const int elements_per_feature = size_in_elements / feature_count;

  llvm::Value* mean = EmitTempBufferPointer(slice_mean, feature_shape);
  llvm_ir::IrArray mean_array(mean, feature_shape);

  llvm::Value* var = EmitTempBufferPointer(slice_var, feature_shape);
  llvm_ir::IrArray var_array(var, feature_shape);

  // This loop calculates mean and variance for each feature.
  //
  // In theory this could be swapped by multi-output fusion. We will evaluate
  // this when it's ready.
  //
  // For variance calculation, we use a simplified formula so we can fuse the
  // computation into the same loop to calculate mean: Var=E(X^2) - E(X)^2.
  TF_RETURN_IF_ERROR(
      llvm_ir::LoopEmitter(
          [&](const llvm_ir::IrArray::Index& index) {
            PrimitiveType element_type = operand->shape().element_type();
            // Used to calculate E(X).
            llvm::Value* sum_address = llvm_ir::EmitAllocaAtFunctionEntry(
                llvm_ir::PrimitiveTypeToIrType(element_type, module_),
                "sum_address", &ir_builder_,
                MinimumAlignmentForPrimitiveType(element_type));

            // Used to calculate E(X^2).
            llvm::Value* sum_square_address =
                llvm_ir::EmitAllocaAtFunctionEntry(
                    llvm_ir::PrimitiveTypeToIrType(element_type, module_),
                    "sum_square_address", &ir_builder_,
                    MinimumAlignmentForPrimitiveType(element_type));

            ir_builder_.CreateStore(
                llvm::ConstantFP::get(ir_builder_.getFloatTy(), 0.0),
                sum_address);

            ir_builder_.CreateStore(
                llvm::ConstantFP::get(ir_builder_.getFloatTy(), 0.0),
                sum_square_address);

            llvm_ir::ForLoopNest loops(IrName(batch_norm_training, "inner"),
                                       &ir_builder_);

            const llvm_ir::IrArray::Index reduced_dims_index =
                loops.AddLoopsForShapeOnDimensions(
                    operand->shape(), dimensions_to_reduce, "reduction_dim");

            SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(),
                                  &ir_builder_);

            llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
            llvm_ir::IrArray::Index input_index =
                FillReducedDimensionIndex(reduced_dims_index, index);
            llvm::Value* new_value =
                operand_array.EmitReadArrayElement(input_index, &ir_builder_);

            llvm::Value* new_value_square =
                ir_builder_.CreateFMul(new_value, new_value);

            llvm::Value* current_sum = ir_builder_.CreateLoad(sum_address);
            llvm::Value* current_sum_square =
                ir_builder_.CreateLoad(sum_square_address);
            // Update sum.
            ir_builder_.CreateStore(
                ir_builder_.CreateFAdd(current_sum, new_value), sum_address);

            // Update sum square.
            ir_builder_.CreateStore(
                ir_builder_.CreateFAdd(current_sum_square, new_value_square),
                sum_square_address);

            SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(),
                                  &ir_builder_);

            llvm::Value* sum = ir_builder_.CreateLoad(sum_address);
            llvm::Value* elements_per_feature_value = llvm::ConstantFP::get(
                ir_builder_.getFloatTy(), elements_per_feature);
            llvm::Value* mean =
                ir_builder_.CreateFDiv(sum, elements_per_feature_value);
            llvm::Value* mean_square = ir_builder_.CreateFMul(mean, mean);
            llvm::Value* sum_square =
                ir_builder_.CreateLoad(sum_square_address);

            // Var=E(X^2) - E(X)^2.
            llvm::Value* var = ir_builder_.CreateFSub(
                ir_builder_.CreateFDiv(sum_square, elements_per_feature_value),
                mean_square);

            var_array.EmitWriteArrayElement(index, var, &ir_builder_);
            return mean;
          },
          mean_array, &ir_builder_)
          .EmitLoop(IrName(batch_norm_training, "mean_var")));

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(batch_norm_training));
  TF_ASSIGN_OR_RETURN(
      const BufferAllocation::Slice slice,
      assignment_.GetUniqueSlice(batch_norm_training, /*index=*/{0}));

  llvm::Value* normalized = EmitTempBufferPointer(slice, output_shape);

  llvm_ir::IrArray target_array(normalized, output_shape);

  AddAliasingInformationToIrArray(*batch_norm_training, &target_array);

  TF_RETURN_IF_ERROR(
      llvm_ir::LoopEmitter(
          [this, mean_array, var_array, epsilon, operand, dimensions_to_reduce,
           feature_index, offset, scale](const llvm_ir::IrArray::Index& index) {
            // The following logic normalizes the input value, scales and shifts
            // it:
            //
            // normalized = (input - mean) / sqrt(variance + epsilon)
            // result = normalized * scale + offset

            // Current index in the feature dimension.
            llvm_ir::IrArray::Index feature_index_value(1,
                                                        index[feature_index]);

            llvm::Value* mean = mean_array.EmitReadArrayElement(
                feature_index_value, &ir_builder_);
            llvm::Value* var = var_array.EmitReadArrayElement(
                feature_index_value, &ir_builder_);

            llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
            llvm::Value* input =
                operand_array.EmitReadArrayElement(index, &ir_builder_);

            llvm::Value* variance_with_epsilon = ir_builder_.CreateFAdd(
                var, llvm::ConstantFP::get(ir_builder_.getFloatTy(), epsilon));
            llvm::Function* func_llvm_sqrt = llvm::Intrinsic::getDeclaration(
                module_, llvm::Intrinsic::sqrt, {ir_builder_.getFloatTy()});
            llvm::Value* variance_sqrt =
                ir_builder_.CreateCall(func_llvm_sqrt, {variance_with_epsilon});
            llvm::Value* normalized = ir_builder_.CreateFDiv(
                ir_builder_.CreateFSub(input, mean), variance_sqrt);
            llvm_ir::IrArray offset_array(GetIrArrayFor(offset));
            llvm::Value* offset = offset_array.EmitReadArrayElement(
                feature_index_value, &ir_builder_);
            llvm_ir::IrArray scale_array(GetIrArrayFor(scale));
            llvm::Value* scale = scale_array.EmitReadArrayElement(
                feature_index_value, &ir_builder_);
            llvm::Value* result = ir_builder_.CreateFAdd(
                ir_builder_.CreateFMul(normalized, scale), offset);

            return result;
          },
          target_array, &ir_builder_)
          .EmitLoop(IrName(batch_norm_training, "normalize")));

  llvm_ir::EmitTuple(GetIrArrayFor(batch_norm_training),
                     {normalized, mean, var}, &ir_builder_, module_);
  return Status::OK();
}

Status IrEmitter::HandleBatchNormGrad(HloInstruction* batch_norm_grad) {
  // TODO(b/62843645) Implement BatchNormGrad on CPU backend.
  return Unimplemented(
      "BatchNormGrad is not implemented on CPU. See b/62843645.");
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
  param_address_untyped->setName(AsStringRef(IrName(parameter, "untyped")));
  if (hlo_module_config_.debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    // We never reassign parameters, so this load is invariant.
    param_address_untyped->setMetadata(
        llvm::LLVMContext::MD_invariant_load,
        llvm::MDNode::get(param_address_untyped->getContext(), /*MDs=*/{}));
  }

  llvm::Value* param_address_typed = ir_builder_.CreateBitCast(
      param_address_untyped, IrShapeType(param_shape)->getPointerTo());
  emitted_value_[parameter] = param_address_typed;

  if (!ShapeUtil::IsOpaque(param_shape)) {
    AttachAlignmentMetadataForLoad(param_address_untyped, param_shape);
    AttachDereferenceableMetadataForLoad(param_address_untyped, param_shape);
  }

  VLOG(2) << "  emitted value: " << llvm_ir::DumpToString(*param_address_typed);
  return Status::OK();
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
    // strided load to get all reals in a vector, all imags in a vector, or use
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
      return [root_is_integral](llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? ir_builder->CreateAdd(lhs, rhs)
                                : ir_builder->CreateFAdd(lhs, rhs);
      };

    case HloOpcode::kMultiply:
      return [root_is_integral](llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? ir_builder->CreateMul(lhs, rhs)
                                : ir_builder->CreateFMul(lhs, rhs);
      };

    case HloOpcode::kAnd:
      return [](llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                llvm::Value* rhs) { return ir_builder->CreateAnd(lhs, rhs); };

    case HloOpcode::kOr:
      return [](llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                llvm::Value* rhs) { return ir_builder->CreateOr(lhs, rhs); };

    case HloOpcode::kMaximum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                 llvm::Value* rhs) {
        if (root_is_floating_point) {
          return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::maxnum,
                                              {lhs, rhs}, {lhs->getType()},
                                              ir_builder);
        }

        return ir_builder->CreateSelect(
            ir_builder->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SGE
                                                  : llvm::ICmpInst::ICMP_UGE,
                                   lhs, rhs),
            lhs, rhs);
      };

    case HloOpcode::kMinimum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* ir_builder, llvm::Value* lhs,
                 llvm::Value* rhs) {
        if (root_is_floating_point) {
          return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::minnum,
                                              {lhs, rhs}, {lhs->getType()},
                                              ir_builder);
        }

        return ir_builder->CreateSelect(
            ir_builder->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SLE
                                                  : llvm::ICmpInst::ICMP_ULE,
                                   lhs, rhs),
            lhs, rhs);
      };
  }
}

IrEmitter::ShardedVectorType IrEmitter::CreateShardedVectorType(
    PrimitiveType element_type, unsigned element_count) {
  // Here we assume that the largest register is a vector register.
  int max_vector_register_size_in_bytes =
      target_machine_features_.largest_register_size_in_bytes(
          compute_function_);

  int vector_register_size_in_elements =
      max_vector_register_size_in_bytes /
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
    HloInstruction* arg, tensorflow::gtl::ArraySlice<int64> dimensions,
    unsigned element_alignment) {
  ShardedVector accumulator;
  accumulator.reserve(accumulator_type.size());
  for (auto accumulator_shard_type : accumulator_type) {
    accumulator.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        accumulator_shard_type, "accumulator", &ir_builder_, 0));
  }

  llvm::Value* init_value_ssa =
      ir_builder_.CreateLoad(GetEmittedValueFor(init_value));

  for (llvm::Value* accumulator_shard : accumulator) {
    llvm::Value* initial_value;
    auto shard_type = accumulator_shard->getType()->getPointerElementType();
    if (auto vector_type = llvm::dyn_cast<llvm::VectorType>(shard_type)) {
      initial_value = ir_builder_.CreateVectorSplat(
          vector_type->getNumElements(), init_value_ssa);
    } else {
      initial_value = init_value_ssa;
    }

    ir_builder_.CreateAlignedStore(initial_value, accumulator_shard,
                                   element_alignment);
  }

  llvm_ir::ForLoopNest reduction_loop_nest(IrName(arg, "vectorized_inner"),
                                           &ir_builder_);
  llvm_ir::IrArray::Index reduced_dims_index =
      reduction_loop_nest.AddLoopsForShapeOnDimensions(arg->shape(), dimensions,
                                                       "reduction_dim");

  SetToFirstInsertPoint(reduction_loop_nest.GetInnerLoopBodyBasicBlock(),
                        &ir_builder_);

  llvm_ir::IrArray arg_array(GetIrArrayFor(arg));
  llvm_ir::IrArray::Index input_index = reduced_dims_index;
  llvm_ir::IrArray::Index::const_iterator it = output_index.begin();

  for (size_t i = 0; i < input_index.size(); ++i) {
    if (input_index[i] == nullptr) {
      input_index[i] = *it++;
    }
  }
  CHECK(output_index.end() == it);

  llvm::Value* input_address = ir_builder_.CreateBitCast(
      arg_array.EmitArrayElementAddress(input_index, &ir_builder_),
      ir_builder_.getInt8PtrTy());

  for (int i = 0; i < accumulator.size(); i++) {
    auto input_address_typed =
        ir_builder_.CreateBitCast(input_address, accumulator[i]->getType());
    auto current_accumulator_value =
        ir_builder_.CreateAlignedLoad(accumulator[i], element_alignment);
    auto addend =
        ir_builder_.CreateAlignedLoad(input_address_typed, element_alignment);
    arg_array.AnnotateLoadStoreInstructionWithMetadata(addend);

    auto reduced_result =
        reduction_generator(&ir_builder_, current_accumulator_value, addend);
    ir_builder_.CreateAlignedStore(reduced_result, accumulator[i],
                                   element_alignment);

    if (i != (accumulator.size() - 1)) {
      input_address = ir_builder_.CreateConstInBoundsGEP1_32(
          reduced_result->getType(), input_address_typed, 1);
    }
  }

  SetToFirstInsertPoint(reduction_loop_nest.GetOuterLoopExitBasicBlock(),
                        &ir_builder_);

  ShardedVector result_ssa;
  result_ssa.reserve(accumulator.size());
  for (auto accumulator_shard : accumulator) {
    result_ssa.push_back(
        ir_builder_.CreateAlignedLoad(accumulator_shard, element_alignment));
  }
  return result_ssa;
}

void IrEmitter::EmitShardedVectorStore(
    llvm::Value* store_address, const std::vector<llvm::Value*>& value_to_store,
    const int alignment, const llvm_ir::IrArray& containing_array) {
  for (int i = 0; i < value_to_store.size(); i++) {
    auto store_address_typed = ir_builder_.CreateBitCast(
        store_address,
        llvm::PointerType::getUnqual(value_to_store[i]->getType()));

    auto store_instruction = ir_builder_.CreateAlignedStore(
        value_to_store[i], store_address_typed, alignment);
    containing_array.AnnotateLoadStoreInstructionWithMetadata(
        store_instruction);

    if (i != (value_to_store.size() - 1)) {
      store_address = ir_builder_.CreateConstInBoundsGEP1_32(
          value_to_store[i]->getType(), store_address_typed, 1);
    }
  }
}

namespace {
// TODO(sanjoy): This is duplicated in tensorflow/core/lib/core/arena.cc.
// Extract out a common implementation to tensorflow/core/lib/math/math_util.h
uint32 GCD(uint32 x, uint32 y) {
  while (y != 0) {
    uint32 r = x % y;
    x = y;
    y = r;
  }
  return x;
}
}  // namespace

StatusOr<bool> IrEmitter::EmitVectorizedReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions, HloComputation* function,
    string* failure_reason) {
  ReductionGenerator reduction_generator =
      MatchReductionGenerator(function, failure_reason);
  if (!reduction_generator) {
    return false;
  }

  int vectorization_factor_in_bytes =
      target_machine_features_.vectorization_factor_in_bytes();

  // We try to process vectorization_factor elements at the same time.
  const int vectorization_factor =
      vectorization_factor_in_bytes /
      ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type());

  bool is_reduction_over_minor_dimension =
      std::find(dimensions.begin(), dimensions.end(),
                arg->shape().layout().minor_to_major(0)) != dimensions.end();

  unsigned element_alignment =
      GCD(ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type()),
          MinimumAlignmentForPrimitiveType(reduce->shape().element_type()));

  if (is_reduction_over_minor_dimension) {
    // TODO(sanjoy): Implement vectorized reduction over the minor dimension.
    *failure_reason = "reduction over minor dimension not implemented";
    return false;
  }

  CHECK(!ShapeUtil::IsTuple(reduce->shape()));
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

  llvm_ir::ForLoopNest loop_nest(IrName(reduce), &ir_builder_);
  llvm_ir::IrArray::Index array_index(reduce->shape().dimensions_size());
  for (int i = reduce->shape().layout().minor_to_major_size() - 1; i > 0; --i) {
    int64 dimension = reduce->shape().layout().minor_to_major(i);
    int64 start_index = 0;
    int64 end_index = reduce->shape().dimensions(dimension);
    std::unique_ptr<llvm_ir::ForLoop> loop =
        loop_nest.AddLoop(start_index, end_index,
                          tensorflow::strings::Printf("dim.%lld", dimension));
    array_index[dimension] = loop->GetIndVarValue();
  }

  int64 innermost_dimension = reduce->shape().layout().minor_to_major(0);
  int64 innermost_dimension_size =
      reduce->shape().dimensions(innermost_dimension);

  if (llvm::BasicBlock* innermost_body_bb =
          loop_nest.GetInnerLoopBodyBasicBlock()) {
    SetToFirstInsertPoint(innermost_body_bb, &ir_builder_);
  }

  auto outermost_loop_exit_block = loop_nest.GetOuterLoopExitBasicBlock();

  if (innermost_dimension_size >= vectorization_factor) {
    int64 start_index = 0;
    int64 end_index = (innermost_dimension_size / vectorization_factor) *
                      vectorization_factor;
    std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
        start_index, end_index, vectorization_factor,
        tensorflow::strings::Printf("dim.%lld", innermost_dimension));
    array_index[innermost_dimension] = loop->GetIndVarValue();

    SetToFirstInsertPoint(loop->GetBodyBasicBlock(), &ir_builder_);

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(), vectorization_factor);
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, &ir_builder_);
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);

    if (auto exit_terminator = loop->GetExitBasicBlock()->getTerminator()) {
      CHECK_GT(reduce->shape().layout().minor_to_major_size(), 1);
      ir_builder_.SetInsertPoint(exit_terminator);
    } else {
      CHECK_EQ(reduce->shape().layout().minor_to_major_size(), 1);
      ir_builder_.SetInsertPoint(loop->GetExitBasicBlock());
    }
  }

  // Since we increment the stride for the inner dimension by more than 1, we
  // may need to peel out an "epilogue" iteration to get the remaining elements
  // in the following case:
  if (innermost_dimension_size % vectorization_factor) {
    // TODO(b/63775531): Consider using a scalar loop here to save on code size.
    array_index[innermost_dimension] =
        ir_builder_.getInt64(innermost_dimension_size -
                             (innermost_dimension_size % vectorization_factor));

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(),
        innermost_dimension_size % vectorization_factor);
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, &ir_builder_);
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);
  }

  if (outermost_loop_exit_block) {
    ir_builder_.SetInsertPoint(outermost_loop_exit_block);
  }

  return true;
}

Status IrEmitter::HandleReduce(HloInstruction* reduce) {
  auto arg = reduce->mutable_operand(0);
  auto init_value = reduce->mutable_operand(1);
  tensorflow::gtl::ArraySlice<int64> dimensions(reduce->dimensions());
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

  // The called computation should have been emitted previously.
  llvm::Function* reducer_function = FindOrDie(emitted_functions_, function);
  return EmitTargetElementLoop(
      reduce, [this, reduce, arg, init_value, dimensions,
               reducer_function](const llvm_ir::IrArray::Index& index) {
        // Initialize an accumulator with init_value.
        PrimitiveType accumulator_type = reduce->shape().element_type();
        llvm::AllocaInst* accumulator_addr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(accumulator_type, module_),
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
        llvm_ir::ForLoopNest loops(IrName(reduce, "inner"), &ir_builder_);
        const llvm_ir::IrArray::Index reduced_dims_index =
            loops.AddLoopsForShapeOnDimensions(arg->shape(), dimensions,
                                               "reduction_dim");

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

        // Build a full index for the input argument, using reduced_dims_index
        // as the base. In reduced_dims_index only the reduction dimensions are
        // filled in. We fill in the rest of the dimensions with induction
        // Value*s taken from 'index' which iterates over the target array.
        // See the high-level description in the XLA documentation for details.
        llvm_ir::IrArray arg_array(GetIrArrayFor(arg));
        llvm_ir::IrArray::Index input_index = reduced_dims_index;
        llvm_ir::IrArray::Index::const_iterator it = index.begin();

        for (size_t i = 0; i < input_index.size(); ++i) {
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

Status IrEmitter::HandleSlice(HloInstruction* slice) {
  VLOG(2) << "HandleSlice: " << slice->ToString();
  auto operand = slice->operand(0);
  // The code below emits a sequential loop nest. For the parallel backend, use
  // EmitParallelTargetElementLoop() which respects dynamic loop bounds.
  if (ShouldEmitParallelLoopFor(*slice)) {
    return DefaultAction(slice);
  }

  // The code below assumes the layouts are equal.
  if (!LayoutUtil::Equal(operand->shape().layout(), slice->shape().layout())) {
    return DefaultAction(slice);
  }

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(slice));

  if (ShapeUtil::HasZeroElements(slice->shape())) {
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

  tensorflow::gtl::FlatSet<int64> inner_dims;
  for (int64 dim : layout.minor_to_major()) {
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
      [&inner_dims](int64 dim) -> bool { return inner_dims.count(dim); },
      operand->shape());

  const int64 primitive_elements_per_logical_element =
      ShapeUtil::ElementsIn(logical_element_shape);

  // memcpy_dim is the innermost (in terms of layout) dimension for which the
  // slice does *not* just copy all the elements along the dimension.
  const int64 memcpy_dim = layout.minor_to_major(inner_dims.size());

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
  llvm_ir::ForLoopNest loops(IrName(slice), &ir_builder_);
  llvm_ir::IrArray::Index target_index =
      loops.AddLoopsForShapeOnDimensions(slice->shape(), outer_dims, "slice");

  // Only the indices for the outer dimensions have been initialized in
  // target_index. The rest of the indices should get initialized to 0, since
  // for the rest of the dimensions the copy writes to the full dimension.
  std::replace(target_index.begin(), target_index.end(),
               static_cast<llvm::Value*>(nullptr),
               static_cast<llvm::Value*>(ir_builder_.getInt64(0)));

  if (num_outer_loops > 0) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);
  }

  llvm_ir::IrArray source_array = GetIrArrayFor(operand);
  const llvm_ir::IrArray::Index source_index = target_index.SourceIndexOfSlice(
      /*shape=*/slice->shape(), /*starts=*/slice->slice_starts(),
      /*strides=*/slice->slice_strides(), /*builder=*/&ir_builder_);

  llvm::Value* memcpy_dest = target_array.EmitArrayElementAddress(
      target_index, &ir_builder_, "slice.dest");
  llvm::Value* memcpy_source = source_array.EmitArrayElementAddress(
      source_index, &ir_builder_, "slice.source");

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
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
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
        IrName(dynamic_update_slice, "in_place"), &ir_builder_);
  }
  return DefaultAction(dynamic_update_slice);
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
      pad, "initialize",
      [this, pad](const llvm_ir::IrArray::Index& target_index) {
        const HloInstruction* padding_value = pad->operand(1);
        llvm::Value* padding_value_addr = GetEmittedValueFor(padding_value);
        return ir_builder_.CreateLoad(padding_value_addr);
      }));

  // Create a loop to iterate over the operand elements and update the output
  // locations where the operand elements should be stored.
  llvm_ir::ForLoopNest loops(IrName(pad, "assign"), &ir_builder_);
  const HloInstruction* operand = pad->operand(0);
  const llvm_ir::IrArray::Index operand_index =
      loops.AddLoopsForShape(operand->shape(), "operand");

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);

  // Load an element from the operand.
  llvm_ir::IrArray operand_array(GetIrArrayFor(operand));
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, &ir_builder_);

  // Compute the output index the operand element should be assigned to.
  // output_index := edge_padding_low + operand_index * (interior_padding + 1)
  const PaddingConfig& padding_config = pad->padding_config();
  llvm_ir::IrArray::Index output_index;
  for (size_t i = 0; i < operand_index.size(); ++i) {
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
  llvm_ir::IrArray output_array(GetIrArrayFor(pad));
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
  auto* root = fusion->fused_expression_root();
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kTransposeDot) {
    DCHECK(root->opcode() == HloOpcode::kDot);
    const HloInstruction* lhs_parameter = StripTranspose(*root->operand(0));
    const HloInstruction* rhs_parameter = StripTranspose(*root->operand(1));
    DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
           rhs_parameter->opcode() == HloOpcode::kParameter);
    const HloInstruction* lhs =
        fusion->operand(lhs_parameter->parameter_number());
    const HloInstruction* rhs =
        fusion->operand(rhs_parameter->parameter_number());

    TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
        /*instruction=*/*root, /*operands=*/{lhs, rhs},
        /*supported_types=*/{F32}));

    llvm_ir::IrArray lhs_array(GetIrArrayFor(lhs));
    llvm_ir::IrArray rhs_array(GetIrArrayFor(rhs));

    Shape target_shape = fusion->shape();
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
    llvm_ir::IrArray target_array = GetIrArrayFor(fusion);
    VLOG(2) << "HandleFusion kTransposeDot: ";
    VLOG(2) << "  lhs operand: "
            << llvm_ir::DumpToString(*lhs_array.GetBasePointer());
    VLOG(2) << "  rhs operand: "
            << llvm_ir::DumpToString(*rhs_array.GetBasePointer());
    VLOG(2) << "  target: "
            << llvm_ir::DumpToString(*target_array.GetBasePointer());

    // Dot operation is complicated so we delegate to a helper class.
    TF_RETURN_IF_ERROR(DotOpEmitter::EmitDotOperation(
        *root, root->operand(0)->IsRank2Transpose(),
        root->operand(1)->IsRank2Transpose(), target_array, lhs_array,
        rhs_array, GetExecutableRunOptionsArgument(), &ir_builder_,
        hlo_module_config_));
    return Status::OK();
  } else if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(fusion,
                                                            assignment_)) {
    VLOG(3) << "HandleFusion FusedDynamicUpdateSliceInPlace";
    CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));

    // Delegate to common implementation of fused in-place dynamic-update-slice.
    auto operands = GetIrArraysForOperandsOf(fusion);
    return llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
        fusion, operands, GetIrArrayFor(fusion), &elemental_emitter,
        &ir_builder_);
  } else if (fusion->fusion_kind() == HloInstruction::FusionKind::kLoop) {
    VLOG(3) << "HandleFusion kLoop";
    CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
    auto operands = GetIrArraysForOperandsOf(fusion);
    FusedIrEmitter fused_emitter(operands, &elemental_emitter);
    TF_RETURN_IF_ERROR(fusion->fused_expression_root()->Accept(&fused_emitter));

    return EmitTargetElementLoop(fusion, fused_emitter.GetRootGenerator());
  } else {
    return Unimplemented("Fusion kind not implemented on CPU");
  }
}

Status IrEmitter::HandleCall(HloInstruction* call) {
  HloComputation* computation = call->to_apply();
  llvm::Function* call_ir_function = FindOrDie(emitted_functions_, computation);

  std::vector<llvm::Value*> parameter_addresses;
  for (const HloInstruction* operand : call->operands()) {
    parameter_addresses.push_back(GetEmittedValueFor(operand));
  }

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(call));

  if (!computation->root_instruction()->outer_dimension_partitions().empty() &&
      !parallel_cpu_backend_) {
    // ParallelTaskAssignment assigned partitions, emit call to
    // ParallelForkJoin.
    TF_RETURN_IF_ERROR(EmitParallelForkJoin(parameter_addresses,
                                            emitted_value_[call], computation,
                                            call_ir_function));
  } else {
    EmitArrayFunctionCallInto(call_ir_function, parameter_addresses,
                              emitted_value_[call], computation->name());
  }

  return Status::OK();
}

Status IrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(
      custom_call->operands());
  tensorflow::StringPiece custom_call_target(custom_call->custom_call_target());
  llvm::Type* i8_ptr_type = ir_builder_.getInt8PtrTy();
  llvm::AllocaInst* operands_alloca =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          i8_ptr_type, ir_builder_.getInt32(operands.size()),
          "cc_operands_alloca", &ir_builder_);
  for (size_t i = 0; i < operands.size(); ++i) {
    const HloInstruction* operand = operands[i];
    llvm::Value* operand_as_i8ptr =
        ir_builder_.CreatePointerCast(GetEmittedValueFor(operand), i8_ptr_type);
    llvm::Value* slot_in_operands_alloca = ir_builder_.CreateInBoundsGEP(
        operands_alloca, {ir_builder_.getInt64(i)});
    ir_builder_.CreateStore(operand_as_i8ptr, slot_in_operands_alloca);
  }
  auto* custom_call_ir_function =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          AsStringRef(custom_call_target),
          llvm::FunctionType::get(
              /*Result=*/ir_builder_.getVoidTy(),
              /*Params=*/{i8_ptr_type, operands_alloca->getType()},
              /*isVarArg=*/false)));

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));
  auto* output_address_arg = ir_builder_.CreatePointerCast(
      GetEmittedValueFor(custom_call), i8_ptr_type);

  ir_builder_.CreateCall(custom_call_ir_function,
                         {output_address_arg, operands_alloca});

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
                a->ToString().c_str(), slice_a.ToString().c_str(),
                b->ToString().c_str(), slice_b.ToString().c_str());
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

  // The called computation should have been emitted previously.
  llvm::Function* condition_ir_function =
      FindOrDie(emitted_functions_, condition);
  llvm::Function* body_ir_function =
      FindOrDie(emitted_functions_, xla_while->while_body());

  // Generating:
  //   while (Condition(while_result)) {
  //     // CopyInsertion pass inserts copies which enable 'while_result' to
  //     // be passed back in as 'Body' parameter.
  //     while_result = Body(while_result);  // Insert
  //   }

  // Terminates the current block with a branch to a while header.
  llvm::BasicBlock* header_bb = llvm::BasicBlock::Create(
      module_->getContext(), AsStringRef(IrName(xla_while, "header")),
      compute_function_);
  ir_builder_.CreateBr(header_bb);
  ir_builder_.SetInsertPoint(header_bb);

  // Calls the condition function to determine whether to proceed with the
  // body.  It must return a bool, so use the scalar call form.
  llvm::Value* while_result = GetEmittedValueFor(xla_while);
  llvm::Value* while_condition = EmitElementFunctionCall(
      condition_ir_function, condition->root_instruction()->shape(),
      {while_result}, IrName(xla_while, "cond"));
  llvm::Value* while_predicate = ir_builder_.CreateICmpNE(
      while_condition,
      llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0));

  // Branches to the body or to the while exit depending on the condition.
  llvm::BasicBlock* body_bb = llvm::BasicBlock::Create(
      module_->getContext(), AsStringRef(IrName(xla_while, "body")),
      compute_function_);
  llvm::BasicBlock* exit_bb = llvm::BasicBlock::Create(
      module_->getContext(), AsStringRef(IrName(xla_while, "exit")));
  ir_builder_.CreateCondBr(while_predicate, body_bb, exit_bb);

  // Calls the body function from the body block.
  ir_builder_.SetInsertPoint(body_bb);

  // Calls the body function.
  EmitArrayFunctionCallInto(body_ir_function, {while_result}, while_result,
                            IrName(xla_while, "body"));
  // Finishes with a branch back to the header.
  ir_builder_.CreateBr(header_bb);

  // Adds the exit block to the function and sets the insert point there.
  compute_function_->getBasicBlockList().push_back(exit_bb);
  ir_builder_.SetInsertPoint(exit_bb);

  return Status::OK();
}

StatusOr<bool> IrEmitter::EmitFastConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
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
    if (LayoutUtil::IsPadded(op->shape())) {
      *failure_reason = "operand has padded layout";
      return false;
    }
  }

  CHECK(!LayoutUtil::IsPadded(concatenate->shape()));

  // We split the dimensions into three categories: the dimension over which we
  // are concatenating (concat_dim), the dimensions that are minor to it
  // (inner_dims) and the dimensions that are major to it (outer_dims).

  int64 concat_dim = concatenate->dimensions(0);
  const Layout& output_layout = output_shape.layout();
  auto concat_dim_layout_itr =
      std::find(output_layout.minor_to_major().begin(),
                output_layout.minor_to_major().end(), concat_dim);

  std::vector<int64> inner_dims(output_layout.minor_to_major().begin(),
                                concat_dim_layout_itr);
  std::vector<int64> outer_dims(std::next(concat_dim_layout_itr),
                                output_layout.minor_to_major().end());

  llvm::Type* i8_ptr_type = ir_builder_.getInt8PtrTy();
  llvm::Type* i8_type = ir_builder_.getInt8Ty();

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(concatenate));
  llvm_ir::IrArray target_array = GetIrArrayFor(concatenate);

  llvm_ir::ForLoopNest loops(IrName(concatenate), &ir_builder_);
  llvm_ir::IrArray::Index outer_dims_index =
      loops.AddLoopsForShapeOnDimensions(output_shape, outer_dims, "concat");
  std::replace(outer_dims_index.begin(), outer_dims_index.end(),
               static_cast<llvm::Value*>(nullptr),
               static_cast<llvm::Value*>(ir_builder_.getInt64(0)));

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &ir_builder_);
  }

  PrimitiveType primitive_type = output_shape.element_type();
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);

  // Contiguous subregions from each operand to the concatenate contribute to a
  // contiguous subregion in the target buffer starting at target_region_begin.
  llvm::Value* target_region_begin = ir_builder_.CreateBitCast(
      target_array.EmitArrayElementAddress(outer_dims_index, &ir_builder_,
                                           "target_region"),
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
    llvm::Value* copy_source_address = ir_builder_.CreateBitCast(
        source_array.EmitArrayElementAddress(outer_dims_index, &ir_builder_,
                                             "src_addr"),
        i8_ptr_type);

    llvm::Value* copy_target_address = ir_builder_.CreateGEP(
        i8_type, target_region_begin,
        ir_builder_.getInt64(byte_offset_into_target_region));

    EmitTransferElements(
        copy_target_address, copy_source_address,
        inner_dims_product * input_shape.dimensions(concat_dim), primitive_type,
        target_array, source_array);

    byte_offset_into_target_region += inner_dims_product *
                                      input_shape.dimensions(concat_dim) *
                                      primitive_type_size;
  }

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &ir_builder_);
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
  unsigned element_alignment = GCD(
      primitive_type_size, MinimumAlignmentForPrimitiveType(primitive_type));
  llvm::Type* primitive_ptr_type = llvm::PointerType::getUnqual(
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module_));

  if (element_count == 1) {
    auto* load_instruction = ir_builder_.CreateAlignedLoad(
        ir_builder_.CreateBitCast(source, primitive_ptr_type),
        element_alignment);
    source_array.AnnotateLoadStoreInstructionWithMetadata(load_instruction);
    auto* store_instruction = ir_builder_.CreateAlignedStore(
        load_instruction, ir_builder_.CreateBitCast(target, primitive_ptr_type),
        element_alignment);
    target_array.AnnotateLoadStoreInstructionWithMetadata(store_instruction);
  } else {
    auto* memcpy_instruction = ir_builder_.CreateMemCpy(
        target, source, element_count * primitive_type_size, element_alignment);

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
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(
      concatenate->operands());
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

  // For the parallel cpu backend, we record the total for each embedded
  // computation callee with its caller kCall HLO.
  HloInstruction* hlo_to_lookup = nullptr;
  if (parallel_cpu_backend_ && is_top_level_computation_) {
    auto* computation = root->parent();
    auto* entry_computation = computation->parent()->entry_computation();
    if (computation != entry_computation) {
      for (HloInstruction* instruction : entry_computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kCall &&
            instruction->to_apply()->root_instruction() == root) {
          hlo_to_lookup = instruction;
          break;
        }
      }
    }
  }
  if (auto* prof_counter = GetProfileCounterFor(hlo_to_lookup)) {
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
    counter_name = IrName("prof_counter", hlo->name());
  } else {
    prof_counter_idx = hlo_to_profile_idx_->size();
    counter_name = "prof_counter.computation";
  }
  return ir_builder_.CreateGEP(GetProfileCountersArgument(),
                               ir_builder_.getInt64(prof_counter_idx),
                               AsStringRef(counter_name));
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
  cycle_start->setName(AsStringRef(IrName(hlo, "cycle_start")));
  cycle_starts_[hlo] = cycle_start;
  if (first_read_cycle_start_ == nullptr) {
    first_read_cycle_start_ = cycle_start;
  }
}

void IrEmitter::ProfilingState::RecordCycleDelta(llvm::IRBuilder<>* ir_builder,
                                                 HloInstruction* hlo,
                                                 llvm::Value* prof_counter) {
  auto* cycle_end = ReadCycleCounter(ir_builder);
  cycle_end->setName(AsStringRef(IrName(hlo, "cycle_end")));
  auto* cycle_start = cycle_starts_[hlo];
  UpdateProfileCounter(ir_builder, prof_counter, cycle_end, cycle_start);
  last_read_cycle_end_ = cycle_end;
}

void IrEmitter::ProfilingState::RecordCompleteComputation(
    llvm::IRBuilder<>* ir_builder, llvm::Value* prof_counter) {
  if (is_top_level_computation_ && last_read_cycle_end_ &&
      first_read_cycle_start_) {
    UpdateProfileCounter(ir_builder, prof_counter, last_read_cycle_end_,
                         first_read_cycle_start_);
  }
}

Status IrEmitter::Preprocess(HloInstruction* hlo) {
  VLOG(3) << "Visiting: " << hlo->ToString();
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

std::vector<llvm::Type*> IrEmitter::GetComputeFunctionParams() {
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(module_->getContext());
  llvm::Type* i8_ptr_ptr_type = i8_ptr_type->getPointerTo();
  llvm::Type* i64_ptr_type = llvm::Type::getInt64PtrTy(module_->getContext());
  std::vector<llvm::Type*> compute_function_params(
      {i8_ptr_type, i8_ptr_type, i8_ptr_ptr_type, i8_ptr_ptr_type});
  if (num_dynamic_loop_bounds_ > 0) {
    compute_function_params.push_back(i64_ptr_type);
  }
  if (hlo_to_profile_idx_) {
    compute_function_params.push_back(i64_ptr_type);
  }
  return compute_function_params;
}

llvm::Argument* IrEmitter::GetResultArgument() {
  return GetArg(compute_function_, 0);
}

llvm::Argument* IrEmitter::GetProfileCountersArgument() {
  const int64 arg_index = num_dynamic_loop_bounds_ > 0 ? 5 : 4;
  return hlo_to_profile_idx_ ? GetArg(compute_function_, arg_index) : nullptr;
}

llvm::Value* IrEmitter::GetTempBuffersArgument() {
  return GetArg(compute_function_, 3);
}

llvm::Value* IrEmitter::GetDynamicLoopBound(const int64 offset) {
  CHECK_GT(num_dynamic_loop_bounds_, 0);
  CHECK_LT(offset, num_dynamic_loop_bounds_ * 2);
  llvm::Argument* loop_bounds_arg = GetArg(compute_function_, 4);
  string name = tensorflow::strings::StrCat("dynamic_loop_bound_", offset);
  return ir_builder_.CreateLoad(ir_builder_.CreateGEP(
      loop_bounds_arg, ir_builder_.getInt64(offset), AsStringRef(name)));
}

llvm::Value* IrEmitter::GetExecutableRunOptionsArgument() {
  return GetArg(compute_function_, 1);
}

llvm::Value* IrEmitter::EmitTempBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {
  llvm::Type* element_type = IrShapeType(target_shape);
  // The alignment and number of bytes within the temporary buffer is determined
  // by the maximal shape as determined by buffer assignment.
  const BufferAllocation& allocation = assignment_.GetAllocation(slice.index());
  if (allocation.is_thread_local()) {
    // Thread-local allocations should only be assigned a single buffer.
    const auto& assigned_buffers = allocation.assigned_buffers();
    CHECK_EQ(1, assigned_buffers.size());
    const Shape& shape = assigned_buffers.begin()->first->shape();

    llvm::AllocaInst*& tempbuf_address = thread_local_buffers_[{
        ir_builder_.GetInsertBlock()->getParent(), slice}];
    if (tempbuf_address == nullptr) {
      tempbuf_address = llvm_ir::EmitAllocaAtFunctionEntry(
          IrShapeType(shape),
          tensorflow::strings::StrCat("thread_local", slice.ToString()),
          &ir_builder_, MinimumAlignmentForShape(target_shape));
    }
    return ir_builder_.CreateBitCast(tempbuf_address,
                                     element_type->getPointerTo());
  }

  llvm::Value* tempbuf_address_ptr = llvm_ir::EmitBufferIndexingGEP(
      GetTempBuffersArgument(), slice.index(), &ir_builder_);
  llvm::LoadInst* tempbuf_address_base =
      ir_builder_.CreateLoad(tempbuf_address_ptr);
  if (hlo_module_config_.debug_options()
          .xla_llvm_enable_invariant_load_metadata()) {
    // Loading the address of a buffer is invariant of the point at which the
    // load is executed in the program because we never reassign buffers.
    tempbuf_address_base->setMetadata(
        llvm::LLVMContext::MD_invariant_load,
        llvm::MDNode::get(tempbuf_address_base->getContext(), /*MDs=*/{}));
  }
  AttachAlignmentMetadataForLoad(tempbuf_address_base, allocation.size());
  AttachDereferenceableMetadataForLoad(tempbuf_address_base, allocation.size());

  llvm::Value* tempbuf_address_untyped = tempbuf_address_base;
  if (slice.offset() > 0) {
    // Adjust the address to account for the slice offset.
    tempbuf_address_untyped = ir_builder_.CreateInBoundsGEP(
        tempbuf_address_base, ir_builder_.getInt64(slice.offset()));
  }
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
      AsStringRef(tensorflow::strings::StrCat(name, "_return_value")));
}

// Emits code to allocate an array of parameter address pointers, and store
// each address from 'parameter_addresses'.
// Returns an array of compute function call arguments (including parameter
// address buffer).
std::vector<llvm::Value*> IrEmitter::GetArrayFunctionCallArguments(
    tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
    llvm::Value* return_value_buffer, tensorflow::StringPiece name) {
  llvm::Value* parameter_addresses_buffer =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          ir_builder_.getInt8PtrTy(),
          ir_builder_.getInt32(parameter_addresses.size()),
          tensorflow::strings::StrCat(name, "_parameter_addresses"),
          &ir_builder_);
  for (size_t i = 0; i < parameter_addresses.size(); ++i) {
    llvm::Value* parameter_as_i8ptr = ir_builder_.CreateBitCast(
        parameter_addresses[i], ir_builder_.getInt8PtrTy(),
        AsStringRef(tensorflow::strings::StrCat(name, "_parameter_", i,
                                                "_address_as_i8ptr")));
    llvm::Value* slot_in_param_adresses = ir_builder_.CreateInBoundsGEP(
        parameter_addresses_buffer, {ir_builder_.getInt64(i)});
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
  return arguments;
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
  ir_builder_.CreateCall(
      function, GetArrayFunctionCallArguments(parameter_addresses,
                                              return_value_buffer, name));
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
          llvm_ir::PrimitiveTypeToIrType(return_type, module_), elements,
          tensorflow::strings::StrCat(name, "_return_value_address"),
          &ir_builder_, MinimumAlignmentForPrimitiveType(return_type));
  EmitArrayFunctionCallInto(function, parameter_addresses, return_value_buffer,
                            name);
  return return_value_buffer;
}

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
Status IrEmitter::EmitParallelForkJoin(
    tensorflow::gtl::ArraySlice<llvm::Value*> parameter_addresses,
    llvm::Value* output_address, HloComputation* computation,
    llvm::Function* parallel_function) {
  HloInstruction* root = computation->root_instruction();

  // Build ParallelForkJoin function type.
  std::vector<llvm::Type*> compute_function_params = GetComputeFunctionParams();
  // Number of parallel compute functions.
  compute_function_params.push_back(ir_builder_.getInt32Ty());
  // Array of partitions. There is an array element for each
  // partition x partition_dim x 2 (for dimension start and limit).
  compute_function_params.push_back(
      llvm::Type::getInt64PtrTy(module_->getContext()));
  // Number of partitioned most-major dimensions in 'root.shape'.
  compute_function_params.push_back(ir_builder_.getInt32Ty());
  // Function pointer for compute function to be dispatched in parallel.
  compute_function_params.push_back(
      llvm::Type::getInt8PtrTy(module_->getContext()));

  llvm::FunctionType* fork_join_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(module_->getContext()),
      /*Params=*/compute_function_params,
      /*isVarArg=*/false);

  llvm::Function* fork_join_func =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          runtime::kParallelForkJoinSymbolName, fork_join_type));
  fork_join_func->setCallingConv(llvm::CallingConv::C);
  fork_join_func->setDoesNotThrow();

  // Add common compute function arguments.
  const string name = computation->name();
  std::vector<llvm::Value*> arguments =
      GetArrayFunctionCallArguments(parameter_addresses, output_address, name);

  // Create ShapePartitionIterator to generate all partitions of 'root.shape'.
  ShapePartitionIterator partition_iterator(root->shape(),
                                            root->outer_dimension_partitions());
  const int64 num_partitions = partition_iterator.GetTotalPartitionCount();
  // Add argument specifying the number of parallel partitions.
  arguments.push_back(ir_builder_.getInt32(num_partitions));

  // The number of partitioned most-major dimensions in 'root.shape'.
  const int32 num_partitioned_dims = root->outer_dimension_partitions().size();
  // A dimension partition consists of two elements: [start_index, limit_index).
  const int32 dim_partition_size = 2;
  // Calculate array partition stride.
  const int32 array_partition_stride =
      num_partitioned_dims * dim_partition_size;
  // Calculate the total number of elements in the partition array.
  const int32 partition_array_size =
      dim_partition_size * num_partitioned_dims * num_partitions;

  // Store dimension partition values as llvm constants in 'partitions'.
  // See comments in runtime_fork_join.cc for array layout description.
  std::vector<llvm::Constant*> partitions(partition_array_size);
  for (int32 i = 0; i < num_partitions; ++i) {
    std::vector<std::pair<int64, int64>> dim_partitions =
        partition_iterator.GetPartition(i);
    CHECK_EQ(num_partitioned_dims, dim_partitions.size());
    const int32 partition_index = i * array_partition_stride;
    for (int32 j = 0; j < num_partitioned_dims; ++j) {
      const std::pair<int64, int64>& dim_partition = dim_partitions[j];
      const int32 index = partition_index + j * dim_partition_size;
      // Store partition [dim_start, dim_limit) intervals for each dimension.
      partitions[index] = ir_builder_.getInt64(dim_partition.first);
      partitions[index + 1] =
          ir_builder_.getInt64(dim_partition.first + dim_partition.second);
    }
  }

  // Create global variable out of dimension partitions in 'partitions'.
  llvm::ArrayType* partitions_array_type =
      llvm::ArrayType::get(ir_builder_.getInt64Ty(), partition_array_size);
  llvm::Constant* partitions_array =
      llvm::ConstantArray::get(partitions_array_type, partitions);
  llvm::GlobalVariable* global_partitions_array = new llvm::GlobalVariable(
      /*Module=*/*module_,
      /*Type=*/partitions_array_type,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/partitions_array,
      /*Name=*/
      AsStringRef(
          tensorflow::strings::StrCat(name, "_parallel_dimension_partitions")));

  // Add argument specifying parallel dimension partitions.
  arguments.push_back(ir_builder_.CreateBitCast(
      global_partitions_array,
      llvm::Type::getInt64PtrTy(module_->getContext())));
  // Add argument specifying the number of partitioned most-major dimensions.
  arguments.push_back(ir_builder_.getInt32(num_partitioned_dims));
  // Add argument for parallel compute function pointer.
  arguments.push_back(
      ir_builder_.CreateBitCast(parallel_function, ir_builder_.getInt8PtrTy()));
  // Emit call to parallel fork/join.
  ir_builder_.CreateCall(fork_join_func, arguments);

  return Status::OK();
}

Status IrEmitter::EmitTargetAddressForOp(const HloInstruction* op) {
  llvm::Value* addr;
  const Shape& target_shape = op->shape();
  if (op == op->parent()->root_instruction()) {
    // For the root node, we write directly to the output buffer of the
    // function.
    llvm::Argument* retval = GetResultArgument();
    if (!ShapeUtil::IsNil(target_shape)) {
      llvm::AttrBuilder attr_builder;
      attr_builder.addAlignmentAttr(MinimumAlignmentForShape(target_shape));
      attr_builder.addDereferenceableAttr(ByteSizeOf(target_shape));
      retval->addAttrs(attr_builder);
    }
    addr = ir_builder_.CreateBitCast(retval,
                                     IrShapeType(target_shape)->getPointerTo());
  } else {
    // For other nodes, we need the temporary buffer allocated for this node to
    // write the result into.
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment_.GetUniqueTopLevelSlice(op));
    addr = EmitTempBufferPointer(slice, target_shape);
  }
  addr->setName(AsStringRef(IrName(op)));
  emitted_value_[op] = addr;
  return Status::OK();
}

Status IrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op,
    const llvm_ir::ElementGenerator& element_generator) {
  return EmitTargetElementLoop(target_op, /*desc=*/"", element_generator);
}

Status IrEmitter::EmitTargetElementLoop(
    HloInstruction* target_op, tensorflow::StringPiece desc,
    const llvm_ir::ElementGenerator& element_generator) {
  VLOG(2) << "EmitTargetElementLoop: " << target_op->ToString();

  const Shape& target_shape = target_op->shape();
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(target_op));
  llvm_ir::IrArray target_array = GetIrArrayFor(target_op);

  if (target_op->IsMultiOutputFusion()) {
    // For multiple outputs fusion, we need to emit each operand and the root.
    TF_RET_CHECK(num_dynamic_loop_bounds_ == 0);
    std::vector<llvm_ir::IrArray> output_arrays;
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(target_shape); ++i) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          assignment_.GetUniqueSlice(target_op, {i}));
      const Shape& element_shape = ShapeUtil::GetSubshape(target_shape, {i});
      llvm::Value* op_target_address =
          EmitTempBufferPointer(slice, element_shape);
      output_arrays.push_back(
          llvm_ir::IrArray(op_target_address, element_shape));
    }
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, output_arrays, &ir_builder_)
            .EmitLoop(IrName(target_op)));

    std::vector<llvm::Value*> tuple_operand_ptrs;
    for (int64 i = 0; i < output_arrays.size(); ++i) {
      tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
    }
    llvm_ir::EmitTuple(target_array, tuple_operand_ptrs, &ir_builder_, module_);

  } else {
    if (ShouldEmitParallelLoopFor(*target_op)) {
      TF_RETURN_IF_ERROR(EmitParallelTargetElementLoop(
          target_shape, element_generator, IrName(target_op), &target_array));
    } else {
      TF_RETURN_IF_ERROR(
          llvm_ir::LoopEmitter(element_generator, target_array, &ir_builder_)
              .EmitLoop(IrName(target_op)));
    }
  }
  return Status::OK();
}

Status IrEmitter::EmitParallelTargetElementLoop(
    const Shape& target_shape,
    const llvm_ir::ElementGenerator& element_generator,
    tensorflow::StringPiece loop_name, llvm_ir::IrArray* target_array) {
  CHECK(!ShapeUtil::IsTuple(target_shape));
  CHECK(!ShapeUtil::IsScalar(target_shape));

  // Emit code to read dynamic loop bounds from function argument 4.
  std::vector<llvm::Value*> dynamic_loop_bounds(2 * num_dynamic_loop_bounds_);
  for (int i = 0; i < 2 * num_dynamic_loop_bounds_; ++i) {
    dynamic_loop_bounds[i] = GetDynamicLoopBound(i);
  }

  llvm_ir::ForLoopNest loop_nest(loop_name, &ir_builder_);
  const int64 num_dims = target_shape.dimensions_size();
  llvm_ir::IrArray::Index array_index(num_dims);

  // Add loops from outer-most to inner-most dimensions.
  for (int i = target_shape.layout().minor_to_major_size() - 1; i >= 0; --i) {
    const int64 dimension = target_shape.layout().minor_to_major(i);
    const int bounds_index = num_dims - 1 - i;
    if (bounds_index < num_dynamic_loop_bounds_) {
      // Emit dynamic loop bounds for this dimension. Dynamic loop bounds
      // are read from ir function dynamic loop bounds argument.
      llvm::Value* start_index = dynamic_loop_bounds[bounds_index * 2 + 0];
      llvm::Value* end_index = dynamic_loop_bounds[bounds_index * 2 + 1];

      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*suffix=*/tensorflow::strings::Printf("dim.%lld", dimension),
          start_index, end_index);
      array_index[dimension] = loop->GetIndVarValue();
    } else {
      // Emit static loop bounds for this dimension.
      std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
          /*start_index=*/0,
          /*end_index=*/target_shape.dimensions(dimension),
          /*suffix=*/tensorflow::strings::Printf("dim.%lld", dimension));
      array_index[dimension] = loop->GetIndVarValue();
    }
  }
  // Point IR builder at inner loop BB.
  SetToFirstInsertPoint(loop_nest.GetInnerLoopBodyBasicBlock(), &ir_builder_);

  // Emit loop body.
  TF_ASSIGN_OR_RETURN(llvm::Value * target_element,
                      element_generator(array_index));
  target_array->EmitWriteArrayElement(array_index, target_element,
                                      &ir_builder_);
  // Point IR builder at outer loop exit BB.
  SetToFirstInsertPoint(loop_nest.GetOuterLoopExitBasicBlock(), &ir_builder_);

  return Status::OK();
}

Status IrEmitter::EmitMemcpy(const HloInstruction& source,
                             const HloInstruction& destination) {
  llvm::Value* source_value = GetEmittedValueFor(&source);
  llvm::Value* destination_value = GetEmittedValueFor(&destination);
  int64 source_size = ByteSizeOf(source.shape());
  // TODO(b/63762267): Be more aggressive about specifying alignment.
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
      return GetIrArrayFor(operand).EmitReadArrayElement(index, &ir_builder_);
    };
  }
  CpuElementalIrEmitter elemental_emitter(hlo_module_config_, this, module_);
  return EmitTargetElementLoop(
      hlo, elemental_emitter.MakeElementGenerator(hlo, operand_to_generator));
}

StatusOr<llvm::Value*> IrEmitter::EmitScalarCall(
    PrimitiveType return_type, HloComputation* computation,
    const std::vector<llvm::Value*>& arguments, tensorflow::StringPiece name) {
  llvm::Function* llvm_function = FindOrDie(emitted_functions_, computation);
  std::vector<llvm::Value*> argument_addrs;
  for (auto argument : arguments) {
    llvm::Value* argument_addr = llvm_ir::EmitAllocaAtFunctionEntry(
        argument->getType(), "arg_addr", &ir_builder_);
    ir_builder_.CreateStore(argument, argument_addr);
    argument_addrs.push_back(argument_addr);
  }
  return EmitElementFunctionCall(llvm_function,
                                 ShapeUtil::MakeShape(return_type, {}),
                                 argument_addrs, name);
}

unsigned TargetMachineFeatures::largest_register_size_in_bytes(
    llvm::Function* function) {
  auto itr = largest_register_size_in_bytes_.find(function);
  if (itr != largest_register_size_in_bytes_.end()) {
    return itr->second;
  }

  int result = largest_register_size_in_bytes_impl(function);

  InsertOrDie(&largest_register_size_in_bytes_, function, result);
  DCHECK_EQ(result, largest_register_size_in_bytes_.begin()->second);
  return result;
}

unsigned TargetMachineFeatures::largest_register_size_in_bytes_impl(
    llvm::Function* function) const {
  auto register_info =
      target_machine_->getSubtargetImpl(*function)->getRegisterInfo();

  unsigned largest_register_size = 0;
  for (const llvm::TargetRegisterClass* register_class :
       register_info->regclasses()) {
    if (register_class->isAllocatable()) {
      largest_register_size =
          std::max(largest_register_size,
                   register_info->getRegSizeInBits(*register_class));
    }
  }

  return largest_register_size / 8;
}
}  // namespace cpu
}  // namespace xla
