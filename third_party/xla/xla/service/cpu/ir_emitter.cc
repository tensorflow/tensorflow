/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emitter.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/map_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/ir_function.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/buffer_assignment_util.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/tuple_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#if defined(INTEL_MKL)
#include "xla/service/cpu/onednn_memory_util.h"
#endif

namespace xla {

namespace {
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

}  // namespace

namespace cpu {

bool IsNativeConvertSupportedOnTargetCPU(std::string feature_string) {
  return (absl::StrContains(feature_string, "+avxneconvert") ||
          absl::StrContains(feature_string, "+amx-bf16"));
}

IrEmitter::IrEmitter(mlir::MLIRContext* mlir_context,
                     const HloModule& hlo_module,
                     const BufferAssignment& assignment,
                     llvm::Module* llvm_module,
                     absl::flat_hash_map<const HloInstruction*, int64_t>
                         instruction_to_profile_idx,
                     absl::flat_hash_map<const HloComputation*, int64_t>
                         computation_to_profile_idx,
                     absl::flat_hash_map<const HloComputation*, bool>
                         computation_transitively_contains_custom_call,
                     const TargetMachineFeatures* target_machine_features,
                     bool emit_code_for_msan,
                     absl::flat_hash_map<BufferAllocation::Slice, int64_t>
                         slice_to_buffer_table_index,
                     bool allow_runtime_calls)
    : assignment_(assignment),
      module_(llvm_module),
      arch_type_(llvm::Triple(llvm_module->getTargetTriple()).getArch()),
      main_builder_(llvm_module->getContext()),
      current_builder_(&main_builder_),
      mlir_context_(mlir_context),
      instruction_to_profile_idx_(std::move(instruction_to_profile_idx)),
      computation_to_profile_idx_(std::move(computation_to_profile_idx)),
      computation_transitively_contains_custom_call_(
          std::move(computation_transitively_contains_custom_call)),
      alias_analysis_(hlo_module, assignment, &llvm_module->getContext()),
      hlo_module_(hlo_module),
      hlo_module_config_(hlo_module.config()),
      is_top_level_computation_(false),
      target_machine_features_(*target_machine_features),
      emit_code_for_msan_(emit_code_for_msan),
      slice_to_buffer_table_index_(std::move(slice_to_buffer_table_index)),
      allow_runtime_calls_(allow_runtime_calls) {
  b()->setFastMathFlags(llvm_ir::GetCpuFastMathFlags(hlo_module_config_));
  absl::Status s = GatherComputationsByAllocationType(
      &hlo_module, &thread_local_computations_, &global_computations_);
  absl::c_sort(thread_local_computations_);
  absl::c_sort(global_computations_);
  TF_CHECK_OK(s) << "Should have failed buffer assignment.";
}

IrEmitter::~IrEmitter() {
  if (!compute_function_.empty()) {
    LOG(WARNING) << "Compute function stack is not empty: "
                 << compute_function_.size();
  }
};

void IrEmitter::EmitThreadLocalFunctionEpilogue(
    const HloComputation* computation) {
  llvm::Argument* out_parameter = compute_function()->result_arg();
  llvm_ir::IrArray root_value = GetIrArrayFor(computation->root_instruction());
  const Shape& return_shape = computation->root_instruction()->shape();

  if (ShapeUtil::IsScalar(return_shape)) {
    llvm::Value* ret_value =
        Load(root_value.GetBasePointeeType(), root_value.GetBasePointer(),
             "load_ret_value");
    Store(ret_value, out_parameter);
  } else {
    CHECK(return_shape.IsTuple());

    llvm::Type* tuple_type =
        llvm_ir::ShapeToIrType(return_shape, module_->getContext());

    for (int i = 0; i < return_shape.tuple_shapes().size(); i++) {
      const Shape& element_shape = return_shape.tuple_shapes(i);
      llvm::Value* destination = llvm_ir::EmitGetTupleElement(
          element_shape,
          /*index=*/i,
          /*alignment=*/MinimumAlignmentForShape(element_shape), out_parameter,
          tuple_type, b());

      llvm::Value* source = llvm_ir::EmitGetTupleElement(
          element_shape,
          /*index=*/i,
          /*alignment=*/MinimumAlignmentForShape(element_shape),
          root_value.GetBasePointer(), root_value.GetBasePointeeType(), b());

      Store(Load(IrShapeType(element_shape), source), destination);
    }
  }
}

absl::StatusOr<llvm::Function*> IrEmitter::EmitComputation(
    const HloComputation* computation, absl::string_view function_name_prefix,
    bool is_top_level_computation,
    absl::Span<HloInstruction* const> instruction_order,
    bool allow_reassociation,
    absl::Span<const llvm::Attribute::AttrKind> function_attributes) {
  std::string function_name = name_uniquer_.GetUniqueName(function_name_prefix);
  VLOG(2) << "Emitting IR for CPU function [" << function_name_prefix << "]";
  is_top_level_computation_ = is_top_level_computation;

  auto cleanup = absl::MakeCleanup(
      [saved_allow_reassociation = allow_reassociation_, this]() {
        allow_reassociation_ = saved_allow_reassociation;
      });
  allow_reassociation_ = allow_reassociation;

  num_dynamic_loop_bounds_ = 0;
  auto backend_config_or =
      computation->root_instruction()->backend_config<BackendConfig>();
  if (backend_config_or.ok() &&
      !backend_config_or->outer_dimension_partitions().empty()) {
    num_dynamic_loop_bounds_ =
        backend_config_or->outer_dimension_partitions().size();
  }

  if (computation->root_instruction()->opcode() != HloOpcode::kOutfeed) {
    TF_ASSIGN_OR_RETURN(
        computation_root_allocation_,
        assignment_.GetUniqueTopLevelSlice(computation->root_instruction()));
  }

  bool has_thread_local_param = false;
  for (const HloInstruction* param : computation->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice param_slice,
                        assignment_.GetUniqueTopLevelSlice(param));
    has_thread_local_param |= param_slice.allocation()->is_thread_local();
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

  llvm::IRBuilderBase::FastMathFlagGuard guard(*b());
  llvm::FastMathFlags flags = b()->getFastMathFlags();
  flags.setAllowReassoc(flags.allowReassoc() || allow_reassociation);
  b()->setFastMathFlags(flags);

  TF_RETURN_IF_ERROR(computation->AcceptOrdered(this, instruction_order));
  llvm::Function* ir_function = compute_function()->function();

  for (llvm::Attribute::AttrKind attr : function_attributes) {
    ir_function->addFnAttr(attr);
  }

  InsertOrDie(&emitted_functions_,
              ComputationToEmit{computation, allow_reassociation}, ir_function);
  // Delete 'compute_function', finalizing 'ir_function' and restoring caller
  // IR insert point.

  // Function epilogue: copying the value over to either the return register,
  // or values pointing from the return register. If we have an allocation for
  // the result, we can cue on whether it is thread_local. If it is a constant,
  // we use the function parameters' allocations to identify a thread_local
  // function.
  const BufferAllocation* root_allocation =
      computation_root_allocation_.allocation();
  if (root_allocation &&
      (root_allocation->is_thread_local() ||
       (root_allocation->is_constant() && has_thread_local_param))) {
    EmitThreadLocalFunctionEpilogue(computation);
  }

  // Destructor for compute_function() terminates the LLVM function definition.
  PopComputeFunction();
  computation_root_allocation_ = BufferAllocation::Slice();
  computation_parameter_allocations_.clear();
  return ir_function;
}

void IrEmitter::InitializeIrFunction(const std::string& function_name) {
  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  llvm::Function::LinkageTypes linkage =
      is_top_level_computation_ ? llvm::GlobalValue::ExternalLinkage
                                : llvm::GlobalValue::InternalLinkage;
  // Create and initialize new IrFunction.
  compute_function_.emplace(function_name, linkage, hlo_module_config_, module_,
                            b(), num_dynamic_loop_bounds_);
}

absl::Status IrEmitter::HandleBitcast(HloInstruction* bitcast) {
  VLOG(2) << "HandleBitcast: " << bitcast->ToString();
  emitted_value_[bitcast] = GetEmittedValueFor(bitcast->operand(0));
  return absl::OkStatus();
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
  return result_global;
}

absl::Status IrEmitter::EmitSmallConstantGlobals() {
  return EmitConstantGlobals(/*max_size_bytes=*/CpuInstructionFusion::
                                 GetLargeConstantThresholdBytes());
}

absl::Status IrEmitter::EmitAllConstantGlobals() {
  return EmitConstantGlobals(/*max_size_bytes=*/std::nullopt);
}

absl::Status IrEmitter::EmitConstantGlobals(
    std::optional<size_t> max_size_bytes) {
  for (const BufferAllocation& allocation : assignment_.Allocations()) {
    // Large constants don't get fused with other instructions, so we don't
    // need to emit them as globals.
    if (!allocation.is_constant() ||
        (max_size_bytes &&
         llvm_ir::LiteralForConstantAllocation(allocation).size_bytes() >
             *max_size_bytes)) {
      continue;
    }

    const Literal& literal = llvm_ir::LiteralForConstantAllocation(allocation);
    llvm::Constant* global_for_const;
    auto it = emitted_literals_.find(LayoutSensitiveLiteralWrapper{literal});
    if (it != emitted_literals_.end()) {
      global_for_const = it->second;
    } else {
      global_for_const = EmitGlobalForLiteral(literal);
      InsertOrDie(&emitted_literals_, LayoutSensitiveLiteralWrapper{literal},
                  global_for_const);
    }

    InsertOrDie(&constant_buffer_to_global_, allocation.index(),
                global_for_const);
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleConstant(HloInstruction* constant) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  // IrEmitter::EmitConstantGlobals has already taken care of emitting the body
  // of the constant.
  return EmitTargetAddressForOp(constant);
}

absl::Status IrEmitter::HandleCopy(HloInstruction* copy) {
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
int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {
  int64_t byte_size = ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  DCHECK_GE(byte_size, 0);
  // Largest scalar is a complex128 so we don't need to worry about the
  // int64_t->int truncation here.
  DCHECK_LE(byte_size, 16);

  // Allocations may be 8-byte aligned if part of a small block.
  return std::min(int64_t{8}, byte_size);
}

// Calculate the alignment of a buffer allocated for a given primitive type.
int IrEmitter::MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {
  return ::xla::cpu::MinimumAlignmentForPrimitiveType(primitive_type);
}

int64_t IrEmitter::ByteSizeOf(const Shape& shape) const {
  return llvm_ir::ByteSizeOf(shape, module_->getDataLayout());
}

// Calculate the alignment of a buffer allocated for a given shape.
int IrEmitter::MinimumAlignmentForShape(const Shape& shape) {
  if (ShapeUtil::IsScalar(shape)) {
    return MinimumAlignmentForPrimitiveType(shape.element_type());
  }

  int64_t buffer_size = ByteSizeOf(shape);
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
                                               int64_t buffer_size) {
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
                                                     int64_t buffer_size) {
  if (buffer_size > 0) {
    llvm_ir::SetDereferenceableMetadataForLoad(load, buffer_size);
  }
}

void IrEmitter::AttachInvariantLoadMetadataForLoad(llvm::LoadInst* load) const {
  AttachInvariantLoadMetadataForLoad(load, hlo_module_config_);
}

/*static*/ void IrEmitter::AttachInvariantLoadMetadataForLoad(
    llvm::LoadInst* load, const HloModuleConfig& config) {
  if (config.debug_options().xla_llvm_enable_invariant_load_metadata()) {
    load->setMetadata(llvm::LLVMContext::MD_invariant_load,
                      llvm::MDNode::get(load->getContext(), /*MDs=*/{}));
  }
}

absl::Status IrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                      assignment_.GetUniqueTopLevelSlice(get_tuple_element));
  llvm::Value* addr = EmitBufferPointer(slice, get_tuple_element->shape());
  emitted_value_[get_tuple_element] = addr;
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleSelect(HloInstruction* select) {
  auto pred = select->operand(0);
  TF_RET_CHECK(pred->shape().element_type() == PRED);
  return DefaultAction(select);
}

absl::Status IrEmitter::HandleInfeed(HloInstruction* instruction) {
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
  llvm::Type* data_type = IrShapeType(data_shape);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice token_slice,
                      assignment_.GetUniqueSlice(infeed, {1}));
  llvm::Value* token_address = EmitBufferPointer(
      token_slice, ShapeUtil::GetTupleElementShape(infeed->shape(), 1));
  llvm_ir::EmitTuple(GetIrArrayFor(infeed), {data_address, token_address}, b());

  if (data_shape.IsTuple()) {
    TF_RET_CHECK(!ShapeUtil::IsNestedTuple(data_shape));

    // For a tuple, we first copy each of the internal elements to
    // their corresponding target locations. We then construct the
    // tuple outer buffer containing pointers to the internal
    // elements.
    std::vector<llvm::Value*> tuple_element_addresses;
    for (int i = 0; i < data_shape.tuple_shapes().size(); ++i) {
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

    llvm_ir::EmitTuple(llvm_ir::IrArray(data_address, data_type, data_shape),
                       tuple_element_addresses, b());
  } else {
    TF_RETURN_IF_ERROR(
        EmitXfeedTransfer(XfeedKind::kInfeed, data_shape, data_address));
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::EmitXfeedTransfer(XfeedKind kind, const Shape& shape,
                                          llvm::Value* program_buffer_address) {
  int64_t length = ByteSizeOf(shape);
  if (length < 0 || length > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument(
        "xfeed (infeed or outfeed) buffer length %d is outside the valid "
        "size range",
        length);
  }
  int32_t length_32 = static_cast<int32_t>(length);

  int32_t shape_length;
  TF_ASSIGN_OR_RETURN(
      llvm::Value * shape_ptr,
      llvm_ir::EncodeSelfDescribingShapeConstant(shape, &shape_length, b()));

  const char* acquire_func_name =
      kind == XfeedKind::kInfeed
          ? runtime::kAcquireInfeedBufferForDequeueSymbolName
          : runtime::kAcquireOutfeedBufferForPopulationSymbolName;

  // Implementation note: this call informs the runtime that it wants a
  // buffer of size exactly 'length_32', and the runtime is responsible for
  // check-failing the process if there is a mismatch, versus passing us
  // back a buffer that we might overrun.
  llvm::Value* acquired_pointer = EmitCallToFunc(
      acquire_func_name,
      {GetExecutableRunOptionsArgument(), b()->getInt32(length_32), shape_ptr,
       b()->getInt32(shape_length)},
      b()->getPtrTy());
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
    if (emit_code_for_msan_) {
      // Mark the outfed data as initialized for msan. The buffer gets read by
      // the host code, which might be msan-instrumented.
      // TODO(b/66051036): Run the msan instrumentation pass instead.
      const llvm::DataLayout& dl = module_->getDataLayout();
      llvm::Type* intptr_type = b()->getIntPtrTy(dl);
      EmitCallToFunc(
          "__msan_unpoison",
          {acquired_pointer, llvm::ConstantInt::get(intptr_type, length)},
          b()->getVoidTy());
    }
  }

  const char* release_func_name =
      kind == XfeedKind::kInfeed
          ? runtime::kReleaseInfeedBufferAfterDequeueSymbolName
          : runtime::kReleaseOutfeedBufferAfterPopulationSymbolName;
  EmitCallToFunc(release_func_name,
                 {GetExecutableRunOptionsArgument(), b()->getInt32(length_32),
                  acquired_pointer, shape_ptr, b()->getInt32(shape_length)},
                 b()->getVoidTy());

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleOutfeed(HloInstruction* outfeed) {
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

  for (int i = 0; i < operand_shape.tuple_shapes().size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(operand_shape, i);
    llvm::Value* tuple_element = llvm_ir::EmitGetTupleElement(
        tuple_element_shape, i, MinimumAlignmentForShape(tuple_element_shape),
        value, IrShapeType(operand_shape), b());
    TF_RETURN_IF_ERROR(EmitXfeedTransfer(XfeedKind::kOutfeed,
                                         tuple_element_shape, tuple_element));
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleSort(HloInstruction* hlo) {
  const HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(sort));
  Shape keys_shape = sort->keys()->shape();
  PrimitiveType keys_type = keys_shape.element_type();
  if (!primitive_util::IsArrayType(keys_type)) {
    return Unimplemented("Element type %s not supported in the Sort op on CPU.",
                         PrimitiveType_Name(keys_type));
  }
  std::vector<llvm::Value*> destination_addresses(sort->operand_count());
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
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
      int64_t primitive_type_size =
          ShapeUtil::ByteSizeOfPrimitiveType(operand->shape().element_type());
      auto source_buffer = GetEmittedValueFor(operand);
      int64_t size = ByteSizeOf(operand->shape());
      MemCpy(destination_addresses[i],
             /*DstAlign=*/llvm::Align(primitive_type_size), source_buffer,
             /*SrcAlign=*/llvm::Align(primitive_type_size), size);
    }
  }

  // Normalize the shape and the dimension to sort.
  Shape normalized_keys_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(keys_shape);
  auto logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(keys_shape.layout());
  TF_RET_CHECK(sort->sort_dimension() < logical_to_physical.size());
  int64_t physical_dimension_to_sort =
      logical_to_physical[sort->sort_dimension()];

  int64_t sort_dimension_elements =
      normalized_keys_shape.dimensions(physical_dimension_to_sort);
  int64_t higher_dimensions = 1;
  for (int64_t i = 0; i < physical_dimension_to_sort; ++i) {
    higher_dimensions *= normalized_keys_shape.dimensions(i);
  }
  int64_t lower_dimensions = 1;
  for (int64_t i = normalized_keys_shape.dimensions().size() - 1;
       i > physical_dimension_to_sort; --i) {
    lower_dimensions *= normalized_keys_shape.dimensions(i);
  }

  CHECK(absl::c_binary_search(thread_local_computations_, sort->to_apply()));
  llvm::Value* values = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      b()->getPtrTy(), b()->getInt32(sort->operand_count()), "cc_values_alloca",
      b());
  llvm::Value* sizes = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      b()->getInt32Ty(), b()->getInt32(sort->operand_count()),
      "cc_sizes_alloca", b());
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    llvm::Value* slot_in_values_alloca =
        ConstInBoundsGEP1_32(b()->getPtrTy(), values, i);
    Store(destination_addresses[i], slot_in_values_alloca);
    llvm::Value* slot_in_sizes_alloca =
        ConstInBoundsGEP1_32(b()->getInt32Ty(), sizes, i);
    llvm::Value* size = b()->getInt32(ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type()));
    Store(size, slot_in_sizes_alloca);
  }

  auto less_than_function =
      FindOrDie(emitted_functions_,
                ComputationToEmit{sort->to_apply(), allow_reassociation_});
  EmitCallToFunc(
      runtime::kKeyValueSortSymbolName,
      {b()->getInt64(higher_dimensions), b()->getInt64(sort_dimension_elements),
       b()->getInt64(lower_dimensions), values,
       b()->getInt32(sort->operand_count()), sizes,
       b()->getInt1(sort->is_stable()), GetExecutableRunOptionsArgument(),
       GetProfileCountersArgument(), less_than_function},
      b()->getVoidTy());

  if (sort->values_count() > 0) {
    llvm_ir::EmitTuple(GetIrArrayFor(sort), destination_addresses, b());
  }
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(tuple));
  llvm::SmallVector<llvm::Value*> base_ptrs;
  for (auto operand : tuple->operands()) {
    base_ptrs.push_back(GetEmittedValueFor(operand));
  }
  llvm_ir::EmitTuple(GetIrArrayFor(tuple), base_ptrs, b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleReduceWindow(HloInstruction* reduce_window) {
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
  bool saved_allow_reassociation = allow_reassociation_;
  allow_reassociation_ = true;
  absl::Status status = DefaultAction(reduce_window);
  allow_reassociation_ = saved_allow_reassociation;
  return status;
}

absl::Status IrEmitter::HandleDot(HloInstruction* dot) {
  auto lhs = dot->operand(0);
  auto rhs = dot->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*dot, /*operands=*/{lhs, rhs},
      /*supported_types=*/
      {PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128}));
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
          << llvm_ir::DumpToString(lhs_array.GetBasePointer());
  VLOG(2) << "  rhs operand: "
          << llvm_ir::DumpToString(rhs_array.GetBasePointer());
  VLOG(2) << "  target: "
          << llvm_ir::DumpToString(target_array.GetBasePointer());

  // Dot operation is complicated so we delegate to a helper class.
  return EmitDotOperation(
      *dot, target_array, lhs_array, rhs_array,
      /*addend_array=*/nullptr, GetExecutableRunOptionsArgument(), b(),
      hlo_module_config_, target_machine_features_, allow_runtime_calls_);
}

absl::Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*convolution, /*operands=*/{lhs, rhs},
      /*supported_types=*/
      {PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128}));

  // TODO(tonywy): Add PotentiallyImplementedAsMKLConvolution to support
  // different data layouts.
  if (allow_runtime_calls_ && PotentiallyImplementedAsEigenConvolution(
                                  *convolution, target_machine_features_)) {
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
      bool one_dim_convolution = lhs_shape.dimensions().size() == 3;
      llvm::Value* lhs_address = GetEmittedValueFor(lhs);
      llvm::Value* rhs_address = GetEmittedValueFor(rhs);
      TF_RETURN_IF_ERROR(EmitTargetAddressForOp(convolution));

      const ConvolutionDimensionNumbers& dnums =
          convolution->convolution_dimension_numbers();

      absl::InlinedVector<int64_t, 2> input_dims;
      absl::InlinedVector<int64_t, 2> kernel_dims;
      absl::InlinedVector<int64_t, 2> output_dims;
      if (one_dim_convolution) {
        input_dims.push_back(1);
        kernel_dims.push_back(1);
        output_dims.push_back(1);
      }

      // Input tensor.
      const Shape& input_shape = convolution->operand(0)->shape();
      int64_t input_batch =
          input_shape.dimensions(dnums.input_batch_dimension());
      for (int d : dnums.input_spatial_dimensions()) {
        input_dims.push_back(input_shape.dimensions(d));
      }
      int64_t input_channels =
          input_shape.dimensions(dnums.input_feature_dimension());

      // Kernel tensor.
      const Shape& kernel_shape = convolution->operand(1)->shape();
      for (int d : dnums.kernel_spatial_dimensions()) {
        kernel_dims.push_back(kernel_shape.dimensions(d));
      }
      int64_t kernel_channels =
          kernel_shape.dimensions(dnums.kernel_input_feature_dimension());
      int64_t kernel_filters =
          kernel_shape.dimensions(dnums.kernel_output_feature_dimension());

      // Output tensor.
      const Shape& convolution_shape = convolution->shape();
      for (int d : dnums.output_spatial_dimensions()) {
        output_dims.push_back(convolution_shape.dimensions(d));
      }

      // Extract the window stride for the convolution.
      const Window& window = convolution->window();
      absl::InlinedVector<int64_t, 2> strides;
      absl::InlinedVector<std::pair<int64_t, int64_t>, 2> padding;
      absl::InlinedVector<int64_t, 2> base_dilation;
      absl::InlinedVector<int64_t, 2> window_dilation;
      if (one_dim_convolution) {
        strides.push_back(1);
        padding.push_back({0, 0});
        base_dilation.push_back(1);
        window_dilation.push_back(1);
      }
      for (const auto& d : window.dimensions()) {
        strides.push_back(d.stride());
        padding.push_back({d.padding_low(), d.padding_high()});
        base_dilation.push_back(d.base_dilation());
        window_dilation.push_back(d.window_dilation());
      }

      PrimitiveType primitive_type = lhs->shape().element_type();
      bool multi_threaded =
          hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
      bool use_acl = hlo_module_config_.debug_options().xla_cpu_use_acl();

      auto valid_num_dims = [](absl::Span<const int64_t> xs) {
        return xs.size() >= 2 && xs.size() <= 3;
      };
      TF_RET_CHECK(valid_num_dims(input_dims)) << input_dims.size();
      TF_RET_CHECK(valid_num_dims(kernel_dims));
      TF_RET_CHECK(valid_num_dims(output_dims));
      TF_RET_CHECK(valid_num_dims(strides));
      TF_RET_CHECK(padding.size() >= 2 && padding.size() <= 3);
      TF_RET_CHECK(valid_num_dims(base_dilation));
      TF_RET_CHECK(valid_num_dims(window_dilation));

      // TODO(b/78639006) Singlethread MKL conv2d is not implemented due to the
      // potential race condition by setting the omp_num_threads.
      const char* fn_name;
      if (input_dims.size() == 2) {
        fn_name =
            primitive_type == F16
                ? (multi_threaded
                       ? runtime::kEigenConv2DF16SymbolName
                       : runtime::kEigenSingleThreadedConv2DF16SymbolName)
                : (multi_threaded
                       ? (use_acl ? runtime::kACLConv2DF32SymbolName
                                  : runtime::kEigenConv2DF32SymbolName)
                       : runtime::kEigenSingleThreadedConv2DF32SymbolName);
      } else if (input_dims.size() == 3) {
        fn_name =
            primitive_type == F16
                ? (multi_threaded
                       ? runtime::kEigenConv3DF16SymbolName
                       : runtime::kEigenSingleThreadedConv3DF16SymbolName)
                : (multi_threaded
                       ? runtime::kEigenConv3DF32SymbolName
                       : runtime::kEigenSingleThreadedConv3DF32SymbolName);
      } else {
        LOG(FATAL) << "Invalid number of dimensions " << input_dims.size();
      }
      std::vector<llvm::Value*> args = {
          GetExecutableRunOptionsArgument(),
          GetEmittedValueFor(convolution),
          lhs_address,
          rhs_address,
          b()->getInt64(input_batch),
      };
      for (int64_t d : input_dims) {
        args.push_back(b()->getInt64(d));
      }
      args.push_back(b()->getInt64(input_channels));
      for (int64_t d : kernel_dims) {
        args.push_back(b()->getInt64(d));
      }
      args.push_back(b()->getInt64(kernel_channels));
      args.push_back(b()->getInt64(kernel_filters));
      for (int64_t d : output_dims) {
        args.push_back(b()->getInt64(d));
      }
      for (int64_t d : strides) {
        args.push_back(b()->getInt64(d));
      }
      for (const auto& p : padding) {
        args.push_back(b()->getInt64(p.first));
        args.push_back(b()->getInt64(p.second));
      }
      for (int64_t d : base_dilation) {
        args.push_back(b()->getInt64(d));
      }
      for (int64_t d : window_dilation) {
        args.push_back(b()->getInt64(d));
      }
      args.push_back(b()->getInt64(convolution->feature_group_count()));

      VLOG(1) << "Ir emitter emitted Convolution to runtime:" << fn_name;
      EmitCallToFunc(fn_name, args, b()->getVoidTy(),
                     /*does_not_throw=*/true,
                     /*only_accesses_arg_memory=*/true);

      return absl::OkStatus();
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

absl::Status IrEmitter::HandleFft(HloInstruction* fft) {
  auto operand = fft->operand(0);
  TF_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*fft, /*operands=*/{operand},
      /*supported_types=*/{F32, F64, C64, C128}));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(operand->shape().layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(fft->shape().layout()));
  VLOG(3) << "operand=" << ShapeUtil::HumanStringWithLayout(operand->shape());
  VLOG(3) << "fft=" << ShapeUtil::HumanStringWithLayout(fft->shape());

  llvm::Value* operand_address = GetEmittedValueFor(operand);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fft));

  const std::vector<int64_t>& fft_length = fft->fft_length();
  const int fft_rank = fft_length.size();

  // Flatten operand batches.
  absl::InlinedVector<int64_t, 4> operand_shape_flat(fft_rank + 1);
  int64_t input_batch = 1;
  int64_t input_batch_length = fft->shape().dimensions().size() - fft_rank;
  for (int i = 0; i < input_batch_length; i++) {
    input_batch *= operand->shape().dimensions(i);
  }
  operand_shape_flat[0] = input_batch;
  for (int i = 0; i < fft_rank; ++i) {
    operand_shape_flat[i + 1] =
        operand->shape().dimensions(i + input_batch_length);
  }

  // Args have been computed, make the call.
  bool multi_threaded_eigen =
      hlo_module_config_.debug_options().xla_cpu_multi_thread_eigen();
  const char* fn_name = multi_threaded_eigen
                            ? runtime::kLegacyDuccFftSymbolName
                            : runtime::kDuccSingleThreadedFftSymbolName;
  auto* fft_lengths =
      EmitGlobalForLiteral(LiteralUtil::CreateR1<int64_t>(fft_length));
  auto* input_shape =
      EmitGlobalForLiteral(LiteralUtil::CreateR1<int64_t>(operand_shape_flat));
  EmitCallToFunc(fn_name,
                 {GetExecutableRunOptionsArgument(), GetEmittedValueFor(fft),
                  operand_address, b()->getInt32(fft->fft_type()),
                  b()->getInt32(operand->shape().element_type() == F64 ||
                                operand->shape().element_type() == C128),
                  b()->getInt32(fft_rank), input_shape, fft_lengths},
                 b()->getVoidTy(), /*does_not_throw=*/true,
                 /*only_accesses_arg_memory=*/false,
                 /*only_accesses_inaccessible_mem_or_arg_mem=*/true);

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAllReduceSingleReplica(HloInstruction* crs) {
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
  for (int64_t i = 0; i < crs->operand_count(); ++i) {
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
  llvm_ir::EmitTuple(GetIrArrayFor(crs), operand_ptrs, b());
  return absl::OkStatus();
}

// Data types supported by ReduceScatter and AllReduce.
static bool DataTypeIsSupportedByReduceScatter(PrimitiveType datatype) {
  // TODO(cheshire): Fix duplication wrt. cpu_runtime
  switch (datatype) {
    case PRED:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

absl::Status IrEmitter::HandleAllReduceMultipleReplica(HloInstruction* crs) {
  CHECK_GE(crs->operand_count(), 1);
  PrimitiveType datatype = crs->operand(0)->shape().element_type();
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(crs));

  if (!DataTypeIsSupportedByReduceScatter(datatype)) {
    return Unimplemented("AllReduce for datatype '%s' is not supported",
                         primitive_util::LowercasePrimitiveTypeName(datatype));
  }

  if (!MatchReductionComputation(crs->to_apply()).has_value()) {
    return Unimplemented("AllReduce for computation '%s' is not supported",
                         crs->to_apply()->ToString());
  }

  std::string replica_groups = ReplicaGroupsToString(crs->replica_groups());
  int32_t replica_groups_size = replica_groups.size();
  llvm::Value* replica_groups_v = b()->CreateGlobalStringPtr(replica_groups);

  bool is_tuple = crs->operand_count() > 1;
  std::vector<llvm::Value*> input_buffer_ptrs;
  std::vector<llvm::Value*> output_buffer_ptrs;
  if (is_tuple) {
    CHECK(crs->shape().IsTuple());

    for (int64_t i = 0; i < crs->operand_count(); i++) {
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
      EncodeArrayFunctionArguments(input_buffer_ptrs, "input_buffers", b());
  llvm::Value* output_buffers =
      EncodeArrayFunctionArguments(output_buffer_ptrs, "output_buffers", b());

  int32_t shape_length;
  TF_ASSIGN_OR_RETURN(llvm::Value * shape_ptr,
                      llvm_ir::EncodeSelfDescribingShapeConstant(
                          crs->shape(), &shape_length, b()));

  bool use_global_device_ids =
      Cast<HloAllReduceInstruction>(crs)->use_global_device_ids();
  EmitCallToFunc(
      runtime::kAllReduceSymbolName,
      {/*run_options=*/GetExecutableRunOptionsArgument(),
       /*replica_groups=*/replica_groups_v,
       /*replica_groups_size=*/b()->getInt32(replica_groups_size),

       /*channel_id_present=*/
       b()->getInt32(static_cast<int32_t>(crs->channel_id().has_value())),
       /*use_global_device_ids=*/
       b()->getInt32(static_cast<int32_t>(use_global_device_ids)),
       /*op_id=*/
       b()->getInt64(crs->channel_id().has_value()
                         ? *crs->channel_id()
                         : crs->GetModule()->unique_id()),
       /*reduction_kind=*/
       b()->getInt32(
           static_cast<int32_t>(*MatchReductionComputation(crs->to_apply()))),
       /*shape_ptr=*/shape_ptr,
       /*shape_length=*/b()->getInt32(shape_length),
       /*num_buffers=*/b()->getInt32(crs->operand_count()),
       /*input_buffers=*/input_buffers,
       /*output_buffers=*/output_buffers},
      b()->getVoidTy());

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  if (hlo_module_config_.replica_count() == 1 &&
      hlo_module_config_.num_partitions() == 1) {
    return HandleAllReduceSingleReplica(crs);
  }
  return HandleAllReduceMultipleReplica(crs);
}

absl::Status IrEmitter::HandleReduceScatter(HloInstruction* rs) {
  CHECK_EQ(rs->operand_count(), 1);
  PrimitiveType datatype = rs->operand(0)->shape().element_type();
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(rs));

  if (!DataTypeIsSupportedByReduceScatter(datatype)) {
    return Unimplemented("ReduceScatter for datatype '%s' is not supported",
                         primitive_util::LowercasePrimitiveTypeName(datatype));
  }

  if (!MatchReductionComputation(rs->to_apply()).has_value()) {
    return Unimplemented("ReduceScatter for computation '%s' is not supported",
                         rs->to_apply()->ToString());
  }

  std::string replica_groups = ReplicaGroupsToString(rs->replica_groups());
  int32_t replica_groups_size = replica_groups.size();
  llvm::Value* replica_groups_v = b()->CreateGlobalStringPtr(replica_groups);

  Shape shape = rs->operand(0)->shape();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                      assignment_.GetUniqueSlice(rs->operand(0), {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(rs, {}));
  llvm::Value* input_buffer = EmitBufferPointer(input_slice, shape);
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, shape);

  bool use_global_device_ids =
      Cast<HloReduceScatterInstruction>(rs)->use_global_device_ids();

  EmitCallToFunc(
      runtime::kReduceScatterSymbolName,
      {/*run_options=*/GetExecutableRunOptionsArgument(),
       /*replica_groups_str=*/replica_groups_v,
       /*replica_groups_str_size=*/b()->getInt32(replica_groups_size),

       /*channel_id_present=*/
       b()->getInt32(static_cast<int32_t>(rs->channel_id().has_value())),
       /*use_global_device_ids=*/
       b()->getInt32(static_cast<int32_t>(use_global_device_ids)),
       /*op_id=*/
       b()->getInt64(rs->channel_id().has_value()
                         ? *rs->channel_id()
                         : rs->GetModule()->unique_id()),
       /*reduction_kind=*/
       b()->getInt32(
           static_cast<int32_t>(*MatchReductionComputation(rs->to_apply()))),
       /*element_type=*/
       b()->getInt32(static_cast<int32_t>(datatype)),
       /*shape=*/b()->getInt64(ShapeUtil::ElementsIn(rs->shape())),
       /*input_buffer=*/input_buffer,
       /*output_buffer=*/output_buffer},
      b()->getVoidTy());

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAllToAll(HloInstruction* instruction) {
  auto* instr = Cast<HloAllToAllInstruction>(instruction);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(instruction));
  CHECK(!instr->split_dimension() && instr->shape().IsTuple())
      << "Only tuple AllToAll is supported";

  std::string replica_groups =
      ReplicaGroupsToString(instruction->replica_groups());
  int32_t replica_groups_size = replica_groups.size();
  llvm::Value* replica_groups_v = b()->CreateGlobalStringPtr(replica_groups);

  int64_t buffer_size = -1;
  std::vector<llvm::Value*> input_buffer_ptrs;
  std::vector<llvm::Value*> output_buffer_ptrs;

  for (int64_t i = 0; i < instruction->operand_count(); i++) {
    const HloInstruction* op = instruction->operand(i);
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_slice,
                        assignment_.GetUniqueSlice(instruction, {i}));
    const Shape& operand_shape = instruction->operand(i)->shape();
    CHECK(operand_shape.IsArray())
        << "Operands to all-to-all must be arrays: " << instruction->ToString();
    output_buffer_ptrs.push_back(EmitBufferPointer(out_slice, operand_shape));
    input_buffer_ptrs.push_back(GetEmittedValueFor(op));
    CHECK(buffer_size == -1 || buffer_size == out_slice.size());
    buffer_size = out_slice.size();
  }

  llvm::Value* input_buffers =
      EncodeArrayFunctionArguments(input_buffer_ptrs, "input_buffers", b());
  llvm::Value* output_buffers =
      EncodeArrayFunctionArguments(output_buffer_ptrs, "output_buffers", b());

  EmitCallToFunc(
      runtime::kAllToAllSymbolName,
      {
          /*run_options=*/GetExecutableRunOptionsArgument(),
          /*channel_id_present=*/
          b()->getInt32(
              static_cast<int32_t>(instruction->channel_id().has_value())),
          /*op_id=*/
          b()->getInt64(instruction->channel_id().has_value()
                            ? *instruction->channel_id()
                            : instruction->GetModule()->unique_id()),
          /*replica_groups=*/replica_groups_v,
          /*replica_groups_size=*/b()->getInt32(replica_groups_size),
          /*num_buffers=*/b()->getInt32(instruction->operand_count()),
          /*buffer_size=*/b()->getInt64(buffer_size),
          /*source_buffers=*/input_buffers,
          /*destination_buffers=*/output_buffers,
      },
      b()->getVoidTy());

  llvm_ir::EmitTuple(GetIrArrayFor(instruction), output_buffer_ptrs, b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAllGather(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(instruction));

  std::string replica_groups =
      ReplicaGroupsToString(instruction->replica_groups());
  int32_t replica_groups_size = replica_groups.size();
  llvm::Value* replica_groups_v = b()->CreateGlobalStringPtr(replica_groups);

  std::vector<llvm::Value*> input_buffer_ptrs;
  std::vector<llvm::Value*> output_buffer_ptrs;

  const HloInstruction* op = instruction->operand(0);
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice in_slice,
                      assignment_.GetUniqueSlice(op, {}));
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_slice,
                      assignment_.GetUniqueSlice(instruction, {}));
  const Shape& operand_shape = op->shape();
  CHECK(op->shape().IsArray())
      << "Operand to all-gather must be arrays: " << instruction->ToString();
  llvm::Value* output_buffer = EmitBufferPointer(out_slice, operand_shape);
  llvm::Value* input_buffer = GetEmittedValueFor(op);
  int64_t buffer_size = in_slice.size();

  bool use_global_device_ids =
      Cast<HloAllGatherInstruction>(instruction)->use_global_device_ids();

  EmitCallToFunc(
      runtime::kAllGatherSymbolName,
      {
          /*run_options=*/GetExecutableRunOptionsArgument(),
          /*channel_id_present=*/
          b()->getInt32(
              static_cast<int32_t>(instruction->channel_id().has_value())),
          /*use_global_device_ids=*/
          b()->getInt32(static_cast<int32_t>(use_global_device_ids)),
          /*op_id=*/
          b()->getInt64(instruction->channel_id().has_value()
                            ? *instruction->channel_id()
                            : instruction->GetModule()->unique_id()),
          /*replica_groups_str=*/replica_groups_v,
          /*replica_groups_str_size=*/b()->getInt32(replica_groups_size),
          /*buffer_size=*/b()->getInt64(buffer_size),
          /*source_buffer=*/input_buffer,
          /*destination_buffer=*/output_buffer,
      },
      b()->getVoidTy());

  llvm_ir::EmitTuple(GetIrArrayFor(instruction), output_buffer_ptrs, b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleCollectivePermute(HloInstruction* crs) {
  auto* instr = Cast<HloCollectivePermuteInstruction>(crs);
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(instr));
  std::string source_target_pairs = absl::StrJoin(
      instr->source_target_pairs(), ",", absl::PairFormatter("="));
  llvm::Value* source_target_pairs_v =
      b()->CreateGlobalStringPtr(source_target_pairs);

  Shape shape = crs->operand(0)->shape();

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                      assignment_.GetUniqueSlice(crs->operand(0), {}));
  llvm::Value* input_buffer = EmitBufferPointer(input_slice, shape);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(crs, {}));
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, shape);

  EmitCallToFunc(
      runtime::kCollectivePermuteSymbolName,
      {/*run_options=*/GetExecutableRunOptionsArgument(),
       /*channel_id_present=*/
       b()->getInt32(static_cast<int32_t>(crs->channel_id().has_value())),
       /*op_id=*/
       b()->getInt64(crs->channel_id().has_value()
                         ? *crs->channel_id()
                         : crs->GetModule()->unique_id()),
       /*byte_size=*/b()->getInt32(ShapeUtil::ByteSizeOf(shape)),
       /*input_buffer=*/input_buffer,
       /*output_buffer=*/output_buffer,
       /*source_target_pairs=*/source_target_pairs_v,
       /*source_target_pairs_size=*/b()->getInt32(source_target_pairs.size())},
      b()->getVoidTy());

  return absl::OkStatus();
}

absl::Status IrEmitter::HandlePartitionId(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(hlo, {}));
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, hlo->shape());
  EmitCallToFunc(runtime::kPartitionIdSymbolName,
                 {/*run_options=*/GetExecutableRunOptionsArgument(),
                  /*output_buffer=*/output_buffer},
                 b()->getVoidTy());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleReplicaId(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      assignment_.GetUniqueSlice(hlo, {}));
  llvm::Value* output_buffer = EmitBufferPointer(output_slice, hlo->shape());
  EmitCallToFunc(runtime::kReplicaIdSymbolName,
                 {/*run_options=*/GetExecutableRunOptionsArgument(),
                  /*output_buffer=*/output_buffer},
                 b()->getVoidTy());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleParameter(HloInstruction* parameter) {
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
  absl::flat_hash_map<int64_t, int64_t> unreduced_dim_map;

  absl::flat_hash_set<int64_t> reduced_dims(reduce.dimensions().begin(),
                                            reduce.dimensions().end());

  const Shape& operand_shape = reduce.operand(0)->shape();
  const Shape& result_shape = reduce.shape();

  int64_t delta = 0;
  for (int64_t i = 0; i < operand_shape.dimensions().size(); i++) {
    if (reduced_dims.contains(i)) {
      delta++;
    } else {
      InsertOrDie(&unreduced_dim_map, i, i - delta);
    }
  }

  // Iterate dimensions minor to major and check that the corresponding
  // dimensions in the source and target shapes are equivalent.
  int64_t result_dim_idx = 0;
  for (int64_t operand_dim_idx = 0;
       operand_dim_idx < operand_shape.dimensions().size(); operand_dim_idx++) {
    int64_t operand_dim =
        operand_shape.layout().minor_to_major(operand_dim_idx);
    if (!reduced_dims.contains(operand_dim)) {
      if (FindOrDie(unreduced_dim_map, operand_dim) !=
          result_shape.layout().minor_to_major(result_dim_idx++)) {
        return false;
      }
    }
  }

  CHECK_EQ(result_dim_idx, result_shape.dimensions().size());

  return true;
}

IrEmitter::ReductionGenerator IrEmitter::MatchReductionGenerator(
    HloComputation* function, std::string* failure_reason) const {
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
      return [root_is_integral](llvm::IRBuilderBase* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateAdd(lhs, rhs)
                                : b->CreateFAdd(lhs, rhs);
      };

    case HloOpcode::kMultiply:
      return [root_is_integral](llvm::IRBuilderBase* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateMul(lhs, rhs)
                                : b->CreateFMul(lhs, rhs);
      };

    case HloOpcode::kAnd:
      return [](llvm::IRBuilderBase* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateAnd(lhs, rhs);
      };

    case HloOpcode::kOr:
      return [](llvm::IRBuilderBase* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateOr(lhs, rhs);
      };

    case HloOpcode::kXor:
      return [](llvm::IRBuilderBase* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateXor(lhs, rhs);
      };

    case HloOpcode::kMaximum:
      return [root_is_floating_point, root_is_signed, this](
                 llvm::IRBuilderBase* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
        if (root_is_floating_point) {
          return llvm_ir::EmitFloatMax(
              lhs, rhs, b,
              hlo_module_config_.debug_options().xla_cpu_enable_fast_min_max());
        }

        return b->CreateSelect(
            b->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SGE
                                         : llvm::ICmpInst::ICMP_UGE,
                          lhs, rhs),
            lhs, rhs);
      };

    case HloOpcode::kMinimum:
      return [root_is_floating_point, root_is_signed, this](
                 llvm::IRBuilderBase* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
        if (root_is_floating_point) {
          return llvm_ir::EmitFloatMin(
              lhs, rhs, b,
              hlo_module_config_.debug_options().xla_cpu_enable_fast_min_max());
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
          *compute_function()->function()) /
      ShapeUtil::ByteSizeOfPrimitiveType(element_type);

  ShardedVectorType sharded_vector_type;
  llvm::Type* element_ir_type =
      llvm_ir::PrimitiveTypeToIrType(element_type, module_->getContext());

  for (int i = 0, e = 1 + Log2Ceiling(element_count); i < e; i++) {
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
          element_ir_type, vector_register_size_in_elements, false);
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
        llvm::VectorType::get(element_ir_type, current_size_fragment, false));
  }
  return sharded_vector_type;
}

absl::StatusOr<IrEmitter::ShardedVector>
IrEmitter::EmitInnerLoopForVectorizedReduction(
    const ReductionGenerator& reduction_generator,
    const llvm_ir::IrArray::Index& output_index,
    const ShardedVectorType& accumulator_type, HloInstruction* init_value,
    HloInstruction* arg, absl::Span<const int64_t> dimensions,
    llvm::Align element_alignment) {
  ShardedVector accumulator;
  accumulator.reserve(accumulator_type.size());
  for (auto accumulator_shard_type : accumulator_type) {
    accumulator.push_back(llvm_ir::EmitAllocaAtFunctionEntry(
        accumulator_shard_type, "accumulator", b(), 0));
  }

  llvm::Value* init_value_ssa =
      Load(IrShapeType(init_value->shape()), GetEmittedValueFor(init_value));

  for (llvm::Value* accumulator_shard : accumulator) {
    llvm::Value* initial_value;
    auto shard_type =
        llvm::cast<llvm::AllocaInst>(accumulator_shard)->getAllocatedType();
    if (auto vector_type = llvm::dyn_cast<llvm::VectorType>(shard_type)) {
      initial_value =
          VectorSplat(vector_type->getElementCount(), init_value_ssa);
    } else {
      initial_value = init_value_ssa;
    }

    AlignedStore(initial_value, accumulator_shard, element_alignment);
  }

  llvm_ir::ForLoopNest reduction_loop_nest(IrName(arg, "vectorized_inner"),
                                           b());
  std::vector<llvm::Value*> input_multi_index =
      reduction_loop_nest.AddLoopsForShapeOnDimensions(arg->shape(), dimensions,
                                                       "reduction_dim");

  SetToFirstInsertPoint(reduction_loop_nest.GetInnerLoopBodyBasicBlock(), b());

  llvm_ir::IrArray arg_array(GetIrArrayFor(arg));
  llvm_ir::IrArray::Index::const_iterator it = output_index.begin();

  for (auto& i : input_multi_index) {
    if (i == nullptr) {
      i = *it++;
    }
  }
  CHECK(output_index.end() == it);
  llvm_ir::IrArray::Index input_index(input_multi_index, arg->shape(),
                                      b()->getInt64Ty());

  llvm::Value* input_address =
      arg_array.EmitArrayElementAddress(input_index, b());

  for (int i = 0; i < accumulator.size(); i++) {
    auto alloca = llvm::cast<llvm::AllocaInst>(accumulator[i]);
    auto current_accumulator_value = AlignedLoad(
        alloca->getAllocatedType(), accumulator[i], element_alignment);
    auto addend = AlignedLoad(alloca->getAllocatedType(), input_address,
                              element_alignment);
    arg_array.AnnotateLoadStoreInstructionWithMetadata(addend);

    auto reduced_result =
        reduction_generator(b(), current_accumulator_value, addend);
    AlignedStore(reduced_result, accumulator[i], element_alignment);

    if (i != (accumulator.size() - 1)) {
      input_address =
          ConstInBoundsGEP1_32(reduced_result->getType(), input_address, 1);
    }
  }

  SetToFirstInsertPoint(reduction_loop_nest.GetOuterLoopExitBasicBlock(), b());

  ShardedVector result_ssa;
  result_ssa.reserve(accumulator.size());
  for (auto accumulator_shard : accumulator) {
    auto alloca = llvm::cast<llvm::AllocaInst>(accumulator_shard);
    result_ssa.push_back(AlignedLoad(alloca->getAllocatedType(),
                                     accumulator_shard, element_alignment));
  }
  return result_ssa;
}

void IrEmitter::EmitShardedVectorStore(
    llvm::Value* store_address, const std::vector<llvm::Value*>& value_to_store,
    llvm::Align alignment, const llvm_ir::IrArray& containing_array) {
  for (int i = 0; i < value_to_store.size(); i++) {
    auto store_instruction =
        AlignedStore(value_to_store[i], store_address, alignment);
    containing_array.AnnotateLoadStoreInstructionWithMetadata(
        store_instruction);

    if (i != (value_to_store.size() - 1)) {
      store_address =
          ConstInBoundsGEP1_32(value_to_store[i]->getType(), store_address, 1);
    }
  }
}

absl::StatusOr<bool> IrEmitter::EmitVectorizedReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    absl::Span<const int64_t> dimensions, HloComputation* function,
    std::string* failure_reason) {
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
          *compute_function()->function()) /
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

  llvm::Align element_alignment(tsl::MathUtil::GCD<unsigned>(
      ShapeUtil::ByteSizeOfPrimitiveType(reduce->shape().element_type()),
      MinimumAlignmentForPrimitiveType(reduce->shape().element_type())));

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

  llvm_ir::ForLoopNest loop_nest(IrName(reduce), b());
  std::vector<llvm::Value*> array_multi_index(
      reduce->shape().dimensions().size());
  for (int i = LayoutUtil::MinorToMajor(reduce->shape()).size() - 1; i > 0;
       --i) {
    int64_t dimension = LayoutUtil::Minor(reduce->shape().layout(), i);
    int64_t start_index = 0;
    int64_t end_index = reduce->shape().dimensions(dimension);
    std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
        start_index, end_index, absl::StrFormat("dim.%d", dimension));
    array_multi_index[dimension] = loop->GetIndVarValue();
  }

  int64_t innermost_dimension = LayoutUtil::Minor(reduce->shape().layout(), 0);
  int64_t innermost_dimension_size =
      reduce->shape().dimensions(innermost_dimension);

  if (llvm::BasicBlock* innermost_body_bb =
          loop_nest.GetInnerLoopBodyBasicBlock()) {
    SetToFirstInsertPoint(innermost_body_bb, b());
  }

  auto outermost_loop_exit_block = loop_nest.GetOuterLoopExitBasicBlock();

  if (innermost_dimension_size >= vectorization_factor) {
    int64_t start_index = 0;
    int64_t end_index = (innermost_dimension_size / vectorization_factor) *
                        vectorization_factor;
    std::unique_ptr<llvm_ir::ForLoop> loop =
        loop_nest.AddLoop(start_index, end_index, vectorization_factor,
                          absl::StrFormat("dim.%d", innermost_dimension));
    array_multi_index[innermost_dimension] = loop->GetIndVarValue();

    SetToFirstInsertPoint(loop->GetBodyBasicBlock(), b());

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(), vectorization_factor);
    llvm_ir::IrArray::Index array_index(array_multi_index, reduce->shape(),
                                        b()->getInt64Ty());
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, b());
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);

    if (auto exit_terminator = loop->GetExitBasicBlock()->getTerminator()) {
      CHECK_GT(LayoutUtil::MinorToMajor(reduce->shape()).size(), 1);
      b()->SetInsertPoint(exit_terminator);
    } else {
      CHECK_EQ(LayoutUtil::MinorToMajor(reduce->shape()).size(), 1);
      b()->SetInsertPoint(loop->GetExitBasicBlock());
    }
  }

  // Since we increment the stride for the inner dimension by more than 1, we
  // may need to peel out an "epilogue" iteration to get the remaining elements
  // in the following case:
  if (innermost_dimension_size % vectorization_factor) {
    // TODO(b/63775531): Consider using a scalar loop here to save on code size.
    array_multi_index[innermost_dimension] =
        b()->getInt64(innermost_dimension_size -
                      (innermost_dimension_size % vectorization_factor));

    ShardedVectorType vector_type = CreateShardedVectorType(
        reduce->shape().element_type(),
        innermost_dimension_size % vectorization_factor);
    llvm_ir::IrArray::Index array_index(array_multi_index, reduce->shape(),
                                        b()->getInt64Ty());
    llvm::IRBuilderBase::FastMathFlagGuard guard(*b());
    llvm::FastMathFlags flags = b()->getFastMathFlags();
    flags.setAllowReassoc(true);
    b()->setFastMathFlags(flags);
    TF_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                        EmitInnerLoopForVectorizedReduction(
                            reduction_generator, array_index, vector_type,
                            init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, b());
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);
  }

  if (outermost_loop_exit_block) {
    b()->SetInsertPoint(outermost_loop_exit_block);
  }

  return true;
}

absl::Status IrEmitter::HandleReduce(HloInstruction* reduce) {
  auto arg = reduce->mutable_operand(0);
  auto init_value = reduce->mutable_operand(1);
  absl::Span<const int64_t> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  bool saved_allow_reassociation = allow_reassociation_;
  allow_reassociation_ = true;
  auto cleanup = absl::MakeCleanup([saved_allow_reassociation, this]() {
    allow_reassociation_ = saved_allow_reassociation;
  });
  if (!options::VectorizedReduceDisabled(hlo_module_config_)) {
    std::string vectorization_failure_reason;
    TF_ASSIGN_OR_RETURN(
        bool vectorization_successful,
        EmitVectorizedReduce(reduce, arg, init_value, dimensions, function,
                             &vectorization_failure_reason));
    if (vectorization_successful) {
      VLOG(1) << "Successfully vectorized reduction " << reduce->ToString()
              << "\n";
      return absl::OkStatus();
    } else {
      VLOG(1) << "Could not vectorize reduction " << reduce->ToString() << ": "
              << vectorization_failure_reason;
    }
  }

  return DefaultAction(reduce);
}

absl::Status IrEmitter::HandleSend(HloInstruction* send) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Send is not implemented on CPU.");
}

absl::Status IrEmitter::HandleSendDone(HloInstruction* send_done) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Send-done is not implemented on CPU.");
}

absl::Status IrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter is not implemented on CPUs.");
}

absl::Status IrEmitter::HandleSlice(HloInstruction* slice) {
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
    return absl::OkStatus();
  }

  const Layout& layout = operand->shape().layout();
  const int64_t num_dims = operand->shape().dimensions().size();

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

  absl::flat_hash_set<int64_t> inner_dims;
  for (int64_t dim : LayoutUtil::MinorToMajor(layout)) {
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
      [&inner_dims](int64_t dim) { return inner_dims.contains(dim); },
      operand->shape());

  const int64_t primitive_elements_per_logical_element =
      ShapeUtil::ElementsIn(logical_element_shape);

  // memcpy_dim is the innermost (in terms of layout) dimension for which the
  // slice does *not* just copy all the elements along the dimension.
  const int64_t memcpy_dim = LayoutUtil::Minor(layout, inner_dims.size());

  const bool memcpy_is_contiguous = slice->slice_strides(memcpy_dim) == 1;
  // The number of logical elements that can be copied in a single call
  // to memcpy. We can only copy 1 element at a time if there is a non-trivial
  // stride.
  const int64_t memcpy_logical_elements =
      memcpy_is_contiguous
          ? slice->slice_limits(memcpy_dim) - slice->slice_starts(memcpy_dim)
          : 1;

  // Determine the dimensions that get lowered as loops.
  llvm::SmallVector<int64_t> outer_dims;
  for (int64_t i = 0; i < num_dims - inner_dims.size() - 1; ++i) {
    outer_dims.push_back(LayoutUtil::Major(layout, i));
  }

  // Is the slice along the memcpy dimension contiguous? If not, then memcpy_dim
  // needs to be wrapped around a loop as well.
  if (!memcpy_is_contiguous) {
    outer_dims.push_back(memcpy_dim);
  }

  llvm_ir::IrArray target_array = GetIrArrayFor(slice);

  const int64_t num_outer_loops = outer_dims.size();
  llvm_ir::ForLoopNest loops(IrName(slice), b());
  std::vector<llvm::Value*> target_multi_index =
      loops.AddLoopsForShapeOnDimensions(slice->shape(), outer_dims, "slice");

  // Only the indices for the outer dimensions have been initialized in
  // target_index. The rest of the indices should get initialized to 0, since
  // for the rest of the dimensions the copy writes to the full dimension.
  std::replace(target_multi_index.begin(), target_multi_index.end(),
               static_cast<llvm::Value*>(nullptr),
               static_cast<llvm::Value*>(b()->getInt64(0)));
  llvm_ir::IrArray::Index target_index(target_multi_index, slice->shape(),
                                       b()->getInt64Ty());

  if (num_outer_loops > 0) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b());
  }

  llvm_ir::IrArray source_array = GetIrArrayFor(operand);
  const llvm_ir::IrArray::Index source_index = target_index.SourceIndexOfSlice(
      /*operand_shape=*/operand->shape(), /*starts=*/slice->slice_starts(),
      /*strides=*/slice->slice_strides(), /*builder=*/b());

  llvm::Value* memcpy_dest =
      target_array.EmitArrayElementAddress(target_index, b(), "slice.dest");
  llvm::Value* memcpy_source =
      source_array.EmitArrayElementAddress(source_index, b(), "slice.source");

  const int64_t memcpy_elements =
      primitive_elements_per_logical_element * memcpy_logical_elements;

  EmitTransferElements(memcpy_dest, memcpy_source, memcpy_elements,
                       slice->shape().element_type(), target_array,
                       source_array);

  if (VLOG_IS_ON(2)) {
    const int64_t memcpy_bytes =
        ShapeUtil::ByteSizeOf(logical_element_shape) * memcpy_elements;
    VLOG(2) << "  emitted copy of " << memcpy_bytes << " bytes inside "
            << num_outer_loops << " loops";
  }

  if (num_outer_loops > 0) {
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b());
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_slice));
    return EmitMemcpy(*dynamic_slice->operand(0), *dynamic_slice);
  }
  return DefaultAction(dynamic_slice);
}

absl::Status IrEmitter::HandleDynamicUpdateSlice(
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
        IrName(dynamic_update_slice, "in_place"), b());
  }
  return DefaultAction(dynamic_update_slice);
}

absl::Status IrEmitter::HandleRecv(HloInstruction* recv) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Recv is not implemented on CPU.");
}

absl::Status IrEmitter::HandleRecvDone(HloInstruction* recv_done) {
  // TODO(b/33942983): Support Send/Recv on CPU.
  return Unimplemented("Recv-done is not implemented on CPU.");
}

absl::Status IrEmitter::HandlePad(HloInstruction* pad) {
  CHECK_EQ(pad->operand_count(), 2);
  const auto operand = pad->operand(0);
  const auto padding_value = pad->operand(1);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(pad));

  return HandlePad(pad, GetIrArrayFor(operand), GetIrArrayFor(padding_value),
                   GetIrArrayFor(pad));
}

absl::Status IrEmitter::HandlePad(HloInstruction* pad,
                                  const llvm_ir::IrArray& operand_array,
                                  const llvm_ir::IrArray& padding_value_array,
                                  const llvm_ir::IrArray& output_array) {
  CHECK_EQ(pad->operand_count(), 2);

  // CPU backend does not properly handle negative padding but this is ok
  // because negative padding should be removed by the algebraic simplifier.
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      return InternalStrCat(
          "Encountered negative padding in IrEmitter on CPU. "
          "This should have been eliminated at the HLO level. ",
          pad->ToString());
    }
  }

  const HloInstruction* padding_value = pad->operand(1);
  const auto index_type = b()->getInt64Ty();
  const auto index = llvm_ir::IrArray::Index(index_type);
  llvm::Value* padding_value_addr = padding_value_array.EmitArrayElementAddress(
      index, b(), "padding_value_addr", true, nullptr);
  const llvm_ir::ElementGenerator element_generator =
      [this, padding_value,
       padding_value_addr](const llvm_ir::IrArray::Index& target_index) {
        return b()->CreateLoad(IrShapeType(padding_value->shape()),
                               padding_value_addr);
      };

  // First, fill in the padding value to all output elements.
  TF_RETURN_IF_ERROR(EmitTargetElementLoop(
      pad, "initialize", element_generator,
      std::optional<const llvm_ir::IrArray>(output_array)));

  // Create a loop to iterate over the operand elements and update the output
  // locations where the operand elements should be stored.
  llvm_ir::ForLoopNest loops(IrName(pad, "assign"), b());
  const HloInstruction* operand = pad->operand(0);
  const llvm_ir::IrArray::Index operand_index =
      loops.AddLoopsForShape(operand->shape(), "operand");

  SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b());

  // Load an element from the operand.
  llvm::Value* operand_data =
      operand_array.EmitReadArrayElement(operand_index, b());

  // Compute the output index the operand element should be assigned to.
  // output_index := edge_padding_low + operand_index * (interior_padding + 1)
  const PaddingConfig& padding_config = pad->padding_config();
  std::vector<llvm::Value*> output_multi_index;
  for (size_t i = 0; i < operand_index.size(); ++i) {
    llvm::Value* offset =
        Mul(operand_index[i],
            b()->getInt64(padding_config.dimensions(i).interior_padding() + 1));
    llvm::Value* index = Add(
        offset, b()->getInt64(padding_config.dimensions(i).edge_padding_low()));
    output_multi_index.push_back(index);
  }

  // Store the operand element to the computed output location.
  llvm_ir::IrArray::Index output_index(
      output_multi_index, output_array.GetShape(), operand_index.GetType());
  output_array.EmitWriteArrayElement(output_index, operand_data, b());

  SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleFusion(HloInstruction* fusion) {
  auto* root = fusion->fused_expression_root();
  if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(fusion, assignment_)) {
    VLOG(3) << "HandleFusion FusedDynamicUpdateSliceInPlace";
    CpuElementalIrEmitter elemental_emitter = ElementalIrEmmiterFactory();
    FusedIrEmitter fused_emitter(elemental_emitter);
    BindFusionArguments(fusion, &fused_emitter);

    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
    // Delegate to common implementation of fused in-place dynamic-update-slice.
    return llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
        fusion, GetIrArrayFor(fusion), &fused_emitter, b());
  } else if (fusion->IsLoopFusion()) {
    VLOG(3) << "HandleFusion kLoop";
    CpuElementalIrEmitter elemental_emitter = ElementalIrEmmiterFactory();
    FusedIrEmitter fused_emitter(elemental_emitter);
    BindFusionArguments(fusion, &fused_emitter);
    TF_ASSIGN_OR_RETURN(auto generator, fused_emitter.GetGenerator(
                                            *fusion->fused_expression_root()));
    return EmitTargetElementLoop(fusion, "kLoop_fusion", generator,
                                 std::nullopt);
  } else if (fusion->IsOutputFusion()) {
    VLOG(3) << "HandleFusion kOutput";
    int64_t dot_op_index =
        root->operand(0)->opcode() == HloOpcode::kDot ? 0 : 1;
    const HloInstruction* dot = root->operand(dot_op_index);
    CHECK_EQ(dot->opcode(), HloOpcode::kDot)
        << dot->ToString() << "  "
        << fusion->fused_instructions_computation()->ToString();

    int64_t dot_lhs_param_number = dot->operand(0)->parameter_number();
    int64_t dot_rhs_param_number = dot->operand(1)->parameter_number();
    int64_t addend_param_number =
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
                         &addend_array, GetExecutableRunOptionsArgument(), b(),
                         hlo_module_config_, target_machine_features_));
    return absl::OkStatus();
  } else {
    return Unimplemented("Fusion kind not implemented on CPU");
  }
}

absl::Status IrEmitter::HandleCall(HloInstruction* call) {
  HloComputation* computation = call->to_apply();
  llvm::Function* call_ir_function = FindOrDie(
      emitted_functions_, ComputationToEmit{computation, allow_reassociation_});

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(call));

  auto backend_config_or =
      computation->root_instruction()->backend_config<BackendConfig>();
  if (backend_config_or.ok() &&
      !backend_config_or->outer_dimension_partitions().empty()) {
    // Having a nonempty set of 'outer_dimension_partitions' means that this
    // computation has been specially selected to be parallelized (one where the
    // root instruction is trivially parallelizable, like elementwise addition
    // of two tensors). The LLVM function generated for this computation accepts
    // an additional set of loop bounds, allowing the caller to control the
    // subset of the output that is generated by each call.

    std::vector<llvm::Value*> call_args = GetArrayFunctionCallArguments(
        {}, b(), computation->name(),
        /*return_value_buffer=*/emitted_value_[call],
        /*exec_run_options_arg=*/GetExecutableRunOptionsArgument(),
        /*buffer_table_arg=*/GetBufferTableArgument(),
        /*status_arg=*/GetStatusArgument(),
        /*profile_counters_arg=*/GetProfileCountersArgument());

    // The parallel fork/join runtime will call the generated function once for
    // each partition in parallel, using an appropriate set of loop bounds for
    // each call such that it only generates one partition of the output.
    HloInstruction* root = computation->root_instruction();
    TF_RETURN_IF_ERROR(EmitCallToParallelForkJoin(
        call_args, root->shape(),
        backend_config_or->outer_dimension_partitions(), b(), call_ir_function,
        computation->name()));

    if (ComputationTransitivelyContainsCustomCall(computation)) {
      EmitEarlyReturnIfErrorStatus();
    }
  } else {
    EmitGlobalCall(*computation, computation->name());
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::EmitSliceToDynamic(
    const HloInstruction* hlo, absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array) {
  std::vector<llvm::Value*> dynamic_dims;
  int32_t raw_data_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::MakeStaticShape(hlo->shape()));
  llvm::Value* dest_buffer = target_array.GetBasePointer();
  for (int64_t i = 1; i < hlo->operand_count(); ++i) {
    const int64_t dim_index = i - 1;
    llvm::Value* source_buffer = source_arrays[i].GetBasePointer();
    llvm::LoadInst* dyn_dim_size = Load(IrShapeType(hlo->operand(i)->shape()),
                                        source_buffer, "dyn_dim_size");

    llvm::Value* metadata = b()->CreateConstInBoundsGEP1_32(
        b()->getInt8Ty(), dest_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    b()->CreateStore(dyn_dim_size, metadata);
    dynamic_dims.push_back(b()->CreateIntCast(dyn_dim_size, b()->getInt64Ty(),
                                              /*isSigned=*/true,
                                              "i64_dyn_dim_size"));
  }

  // Pseudo code for sliceToDynamic:
  //
  //   for (index i in dynamic_dim)
  //     dest_index = delinearize(linearize(i, dynamic_dim), static_dim)
  //     dest[dest_index] = source[i]
  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* source_element =
        source_arrays[0].EmitReadArrayElement(array_index, b());
    llvm::Value* linear_index = array_index.Linearize(dynamic_dims, b());
    // Delinearize the index based on the static shape.
    llvm_ir::IrArray::Index dest_index(linear_index, target_array.GetShape(),
                                       b());
    target_array.EmitWriteArrayElement(dest_index, source_element, b());
    return absl::OkStatus();
  };
  return llvm_ir::LoopEmitter(loop_body_emitter, target_array.GetShape(),
                              dynamic_dims, b())
      .EmitLoop(IrName(hlo));
}

absl::Status IrEmitter::HandleSliceToDynamic(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  llvm_ir::IrArray target_array = GetIrArrayFor(hlo);

  std::vector<llvm_ir::IrArray> source_arrays;
  source_arrays.reserve(hlo->operand_count());
  for (auto operand : hlo->operands()) {
    source_arrays.push_back(GetIrArrayFor(operand));
  }

  return EmitSliceToDynamic(hlo, source_arrays, target_array);
}

absl::Status IrEmitter::HandlePadToStatic(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice data_slice,
                      assignment_.GetUniqueSlice(hlo, {0}));
  std::vector<llvm::Value*> dynamic_dims;
  std::vector<llvm::Value*> tuple_operand_ptrs;
  const Shape& data_shape = ShapeUtil::GetSubshape(hlo->shape(), {0});
  const Shape& input_shape = hlo->operand(0)->shape();
  llvm::Value* data_address = EmitBufferPointer(data_slice, data_shape);
  llvm::Type* data_type = IrShapeType(data_shape);
  llvm_ir::IrArray data_array(data_address, data_type, data_shape);
  llvm::Value* source_buffer = GetEmittedValueFor(hlo->operand(0));
  int64_t raw_data_size =
      ShapeUtil::ByteSizeOf(ShapeUtil::MakeStaticShape(input_shape));

  // Put a placeholder for the data array's pointer
  tuple_operand_ptrs.push_back(data_array.GetBasePointer());
  // PadToStatic has a dynamic tensor as input and variadic size of outputs:
  // (static_tensor, dynamic_dim_0, dynamic_dim_1, ... )
  // Dynamic dimension sizes starts from output index 1.
  for (int i = 1; i < hlo->shape().tuple_shapes().size(); ++i) {
    // Read from the metadata section of the dynamic input (operand 0).
    const Shape& dim_shape = ShapeUtil::GetSubshape(hlo->shape(), {i});
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dim_size_slice,
                        assignment_.GetUniqueSlice(hlo, {i}));
    llvm::Value* dest_dim_size_address =
        EmitBufferPointer(dim_size_slice, data_shape);
    const int64_t dim_index = i - 1;
    llvm::Value* metadata = b()->CreateConstInBoundsGEP1_32(
        b()->getInt8Ty(), source_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        b()->CreateLoad(b()->getInt32Ty(), metadata, "dyn_dim_size");
    b()->CreateStore(dyn_dim_size, dest_dim_size_address);
    dynamic_dims.push_back(b()->CreateIntCast(dyn_dim_size, b()->getInt64Ty(),
                                              /*isSigned=*/true,
                                              "i64_dyn_dim_size"));
    tuple_operand_ptrs.push_back(dest_dim_size_address);
  }

  // Pseudo code for padToStatic:
  //
  //   for (index i in dynamic_dim)
  //     source_index = delinearize(inearize(i, dynamic_dim), static_dim)
  //     dest[i] = source[source_index]
  auto loop_body_emitter =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linear_index = array_index.Linearize(dynamic_dims, b());
    llvm_ir::IrArray::Index source_index(linear_index, input_shape, b());
    llvm::Value* source_element =
        GetIrArrayFor(hlo->operand(0)).EmitReadArrayElement(source_index, b());
    data_array.EmitWriteArrayElement(array_index, source_element, b());
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(
      llvm_ir::LoopEmitter(loop_body_emitter, input_shape, dynamic_dims, b())
          .EmitLoop(IrName(hlo)));

  // Emit static tensor and dynamic sizes as one tuple.
  llvm_ir::EmitTuple(GetIrArrayFor(hlo), tuple_operand_ptrs, b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleTopK(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(hlo));
  const HloInstruction* input = hlo->operand(0);
  const int64_t k = hlo->shape().tuple_shapes(0).dimensions().back();
  const bool has_batch = hlo->shape().tuple_shapes(0).dimensions().size() == 2;
  TF_RET_CHECK(input->shape().element_type() == F32) << hlo->ToString();
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      hlo->shape().tuple_shapes(0).layout()))
      << hlo->ToString();
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      hlo->shape().tuple_shapes(1).layout()))
      << hlo->ToString();
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(hlo->operand(0)->shape().layout()))
      << hlo->ToString();

  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice values_slice,
                      assignment_.GetUniqueSlice(hlo->operand(0), {}));
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_values_slice,
                      assignment_.GetUniqueSlice(hlo, {0}));
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_indices_slice,
                      assignment_.GetUniqueSlice(hlo, {1}));
  llvm::Value* values_ptr =
      EmitBufferPointer(values_slice, hlo->operand(0)->shape());
  llvm::Value* out_values_ptr =
      EmitBufferPointer(out_values_slice, hlo->shape().tuple_shapes(0));
  llvm::Value* out_indices_ptr =
      EmitBufferPointer(out_indices_slice, hlo->shape().tuple_shapes(1));
  EmitCallToFunc(
      runtime::kTopKF32SymbolName,
      {b()->getInt64(has_batch ? input->shape().dimensions(0) : 1),
       b()->getInt64(input->shape().dimensions().back()), b()->getInt64(k),
       values_ptr, out_values_ptr, out_indices_ptr},
      b()->getVoidTy());

  llvm_ir::EmitTuple(GetIrArrayFor(hlo), {out_values_ptr, out_indices_ptr},
                     b());
  return absl::OkStatus();
}

#if defined(INTEL_MKL)

// Emits operands alloca vector for oneDNN custom calls.
std::vector<StackAlloca> IrEmitter::EmitOneDnnOperandsAlloca(
    HloInstruction* custom_call, llvm::Value*& args_val, int& arg_indx) {
  std::vector<StackAlloca> operands_stack_alloca;
  const int num_operands = custom_call->operand_count();
  operands_stack_alloca.reserve(num_operands);
  for (int i = 0; i < num_operands; ++i) {
    llvm_ir::IrArray ir_array(GetIrArrayFor(custom_call->operand(i)));
    StackAlloca stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), ir_array);
    args_val = b()->CreateInsertValue(args_val, stack_alloca.value, arg_indx++);
    operands_stack_alloca.push_back(std::move(stack_alloca));
  }
  return operands_stack_alloca;
}

std::pair<llvm::Value*, StackAlloca> IrEmitter::GetPtrAndAllocaFromBufferSlice(
    const BufferAllocation::Slice& slice, const Shape& shape) {
  llvm::Value* slice_ptr = EmitBufferPointer(slice, shape);
  llvm::Type* type = IrShapeType(shape);
  llvm_ir::IrArray ir_array = llvm_ir::IrArray(slice_ptr, type, shape);
  return {slice_ptr, GetAllocaAndEmitMemrefInfo(*b(), ir_array)};
}

absl::Status IrEmitter::HandleOneDnnMatMulCalls(
    HloInstruction* custom_call, std::string runtime_symbol_name) {
  // We would like to emit LLVM IR for the following function call
  //      custom_call_target(void* result, void** args)
  // args can be thought of an array of pointers allocated on the stack,
  // i.e., alloca [nargs x ptr], as such
  //      args[0]: ptr to nargs
  //      args[1]: ptr to ExecutableRunOptions
  //      args[2]: ptr to OneDnnMatMulConfig
  //      args[3...]: ptrs to operands
  //  This allows us to pass variable number of operands to the
  //  custom_call_target function.
  //
  //  Currently, we assume that neither operands nor results are packed into
  //  tuple(s).

  // First three arguments: nargs, ExecutableRunOptions, and
  // OneDnnMatMulConfig.
  const int nargs_offset = 3;
  const int num_operands = custom_call->operand_count();
  const int nargs = nargs_offset + num_operands;
  int arg_indx = 0;

  llvm::Type* i64_type = b()->getInt64Ty();
  llvm::Type* ptr_type = b()->getPtrTy();
  llvm::ArrayType* ptr_array_type = llvm::ArrayType::get(ptr_type, nargs);
  llvm::Value* args_val = llvm::UndefValue::get(ptr_array_type);

  // Insert nargs.
  llvm::Value* nargs_val = b()->getInt64(nargs);
  llvm::Value* nargs_ptr =
      llvm_ir::EmitAllocaAtFunctionEntry(i64_type, "nargs", b());
  b()->CreateLifetimeStart(nargs_ptr, b()->getInt64(-1));
  b()->CreateStore(nargs_val, nargs_ptr);
  args_val = b()->CreateInsertValue(args_val, nargs_ptr, arg_indx++);

  // Insert ExecutableRunOptions.
  llvm::Value* run_opts_val = GetExecutableRunOptionsArgument();
  args_val = b()->CreateInsertValue(args_val, run_opts_val, arg_indx++);

  // Insert OneDnnMatMulConfig.
  auto typed_custom_call = Cast<HloCustomCallInstruction>(custom_call);
  auto backend_config = typed_custom_call->backend_config<BackendConfig>();
  OneDnnMatMulConfig matmul_config;
  matmul_config.CopyFrom(backend_config->onednn_matmul_config());
  std::string str_config;
  matmul_config.SerializeToString(&str_config);
  llvm::Value* matmul_config_val =
      b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(str_config));
  args_val = b()->CreateInsertValue(args_val, matmul_config_val, arg_indx++);

  // Insert operands.
  auto operands_stack_alloca =
      EmitOneDnnOperandsAlloca(custom_call, args_val, arg_indx);
  TF_RET_CHECK(nargs == arg_indx)
      << "Number of arguments don't equal the last argument index.";

  llvm::Value* args_ptr =
      llvm_ir::EmitAllocaAtFunctionEntry(ptr_array_type, "matmul.args", b());
  b()->CreateLifetimeStart(args_ptr, b()->getInt64(-1));
  b()->CreateStore(args_val, args_ptr);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));

  StackAlloca result_stack_alloca;
  StackAlloca scratch_stack_alloca;
  // Custom-call target for matmul has 3 arguments: void* result, void*
  // scratch and void** args
  std::vector<llvm::Value*> fn_call_args;
  fn_call_args.reserve(3);
  // Add the scratch buffer to the output, so that oneDNN can use it as a
  // user-provided scratchpad
  const bool use_scratchpad = custom_call->shape().IsTuple();
  if (use_scratchpad) {
    llvm::Value* result_slice_ptr;
    llvm::Value* scratch_slice_ptr;
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_.GetUniqueSlice(custom_call, {0}));
    const Shape& result_shape = custom_call->shape().tuple_shapes(0);
    std::tie(result_slice_ptr, result_stack_alloca) =
        GetPtrAndAllocaFromBufferSlice(result_slice, result_shape);
    fn_call_args.push_back(result_stack_alloca.value);

    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice scratch_slice,
                        assignment_.GetUniqueSlice(custom_call, {1}));
    const Shape& scratch_shape = custom_call->shape().tuple_shapes(1);
    std::tie(scratch_slice_ptr, scratch_stack_alloca) =
        GetPtrAndAllocaFromBufferSlice(scratch_slice, scratch_shape);
    fn_call_args.push_back(scratch_stack_alloca.value);
    llvm_ir::EmitTuple(GetIrArrayFor(custom_call),
                       {result_slice_ptr, scratch_slice_ptr}, b());
  } else {
    llvm_ir::IrArray result_array;
    result_array = GetIrArrayFor(custom_call);
    result_stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), result_array);
    fn_call_args.push_back(result_stack_alloca.value);
    fn_call_args.push_back(llvm::ConstantPointerNull::get(b()->getPtrTy()));
  }
  fn_call_args.push_back(args_ptr);
  EmitCallToFunc(std::move(runtime_symbol_name), fn_call_args,
                 b()->getVoidTy());

  // Lifetime ends for all stack allocations.
  b()->CreateLifetimeEnd(nargs_ptr, b()->getInt64(-1));
  b()->CreateLifetimeEnd(args_ptr, b()->getInt64(-1));
  for (auto& alloca : operands_stack_alloca) {
    alloca.EmitLifetimeEnd();
  }
  result_stack_alloca.EmitLifetimeEnd();
  if (use_scratchpad) {
    scratch_stack_alloca.EmitLifetimeEnd();
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleOneDnnConvolution(HloInstruction* custom_call) {
  //      args[0]: ptr to nargs
  //      args[1]: ptr to ExecutableRunOptions
  //      args[2]: ptr to OneDnnConvolutionConfig
  //      args[3...]: ptrs to operands

  // First three arguments: nargs, ExecutableRunOptions, and
  // OneDnnConvolutionConfig.
  const int nargs_offset = 3;
  const int num_operands = custom_call->operand_count();
  const int nargs = nargs_offset + num_operands;
  int arg_indx = 0;

  llvm::Type* i64_type = b()->getInt64Ty();
  llvm::Type* ptr_type = b()->getPtrTy();
  llvm::ArrayType* ptr_array_type = llvm::ArrayType::get(ptr_type, nargs);
  llvm::Value* args_val = llvm::UndefValue::get(ptr_array_type);

  llvm::Value* nargs_val = b()->getInt64(nargs);
  llvm::Value* nargs_ptr =
      llvm_ir::EmitAllocaAtFunctionEntry(i64_type, "nargs", b());
  b()->CreateLifetimeStart(nargs_ptr, b()->getInt64(-1));
  b()->CreateStore(nargs_val, nargs_ptr);
  args_val = b()->CreateInsertValue(args_val, nargs_ptr, arg_indx++);

  llvm::Value* run_opts_val = GetExecutableRunOptionsArgument();
  args_val = b()->CreateInsertValue(args_val, run_opts_val, arg_indx++);

  auto typed_custom_call = Cast<HloCustomCallInstruction>(custom_call);
  auto backend_config = typed_custom_call->backend_config<BackendConfig>();
  OneDnnConvolutionConfig conv_config;
  conv_config.CopyFrom(backend_config->onednn_conv_config());
  std::string str_config;
  conv_config.SerializeToString(&str_config);
  llvm::Value* conv_config_val =
      b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(str_config));
  args_val = b()->CreateInsertValue(args_val, conv_config_val, arg_indx++);

  auto operands_stack_alloca =
      EmitOneDnnOperandsAlloca(custom_call, args_val, arg_indx);
  TF_RET_CHECK(nargs == arg_indx)
      << "Number of arguments don't equal the last argument index.";

  llvm::Value* args_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
      ptr_array_type, "convolution.args", b());
  b()->CreateLifetimeStart(args_ptr, b()->getInt64(-1));
  b()->CreateStore(args_val, args_ptr);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));

  StackAlloca result_stack_alloca;
  StackAlloca scratch_stack_alloca;
  std::vector<llvm::Value*> fn_call_args;
  fn_call_args.reserve(3);
  // Add the scratch buffer to the output, so that oneDNN can use it as a
  // user-provided scratchpad
  const bool use_scratchpad = custom_call->shape().IsTuple();
  if (use_scratchpad) {
    llvm::Value* result_slice_ptr;
    llvm::Value* scratch_slice_ptr;
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_.GetUniqueSlice(custom_call, {0}));
    const Shape& result_shape = custom_call->shape().tuple_shapes(0);
    std::tie(result_slice_ptr, result_stack_alloca) =
        GetPtrAndAllocaFromBufferSlice(result_slice, result_shape);
    fn_call_args.push_back(result_stack_alloca.value);

    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice scratch_slice,
                        assignment_.GetUniqueSlice(custom_call, {1}));
    const Shape& scratch_shape = custom_call->shape().tuple_shapes(1);
    std::tie(scratch_slice_ptr, scratch_stack_alloca) =
        GetPtrAndAllocaFromBufferSlice(scratch_slice, scratch_shape);
    fn_call_args.push_back(scratch_stack_alloca.value);
    llvm_ir::EmitTuple(GetIrArrayFor(custom_call),
                       {result_slice_ptr, scratch_slice_ptr}, b());
  } else {
    llvm_ir::IrArray result_array;
    result_array = GetIrArrayFor(custom_call);
    result_stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), result_array);
    fn_call_args.push_back(result_stack_alloca.value);
    fn_call_args.push_back(llvm::ConstantPointerNull::get(b()->getPtrTy()));
  }
  fn_call_args.push_back(args_ptr);
  EmitCallToFunc(runtime::kOneDnnConvolutionSymbolName, fn_call_args,
                 b()->getVoidTy());

  // Lifetime ends for all stack allocations.
  b()->CreateLifetimeEnd(nargs_ptr, b()->getInt64(-1));
  b()->CreateLifetimeEnd(args_ptr, b()->getInt64(-1));
  for (StackAlloca& alloca : operands_stack_alloca) {
    alloca.EmitLifetimeEnd();
  }
  result_stack_alloca.EmitLifetimeEnd();
  if (use_scratchpad) {
    scratch_stack_alloca.EmitLifetimeEnd();
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleOneDnnLayerNorm(HloInstruction* custom_call) {
  //      args[0]: ptr to nargs
  //      args[1]: ptr to ExecutableRunOptions
  //      args[2]: ptr to OneDnnNormConfig
  //      args[3...]: ptrs to operands

  // First three arguments: nargs, ExecutableRunOptions, and
  // OneDnnNormConfig.
  const int nargs_offset = 3;
  const int num_operands = custom_call->operand_count();
  const int nargs = nargs_offset + num_operands;
  int arg_indx = 0;

  llvm::Type* i64_type = b()->getInt64Ty();
  llvm::Type* ptr_type = b()->getPtrTy();
  llvm::ArrayType* ptr_array_type = llvm::ArrayType::get(ptr_type, nargs);
  llvm::Value* args_val = llvm::UndefValue::get(ptr_array_type);

  // Insert nargs.
  llvm::Value* nargs_val = b()->getInt64(nargs);
  llvm::Value* nargs_ptr =
      llvm_ir::EmitAllocaAtFunctionEntry(i64_type, "nargs", b());
  b()->CreateLifetimeStart(nargs_ptr, b()->getInt64(-1));
  b()->CreateStore(nargs_val, nargs_ptr);
  args_val = b()->CreateInsertValue(args_val, nargs_ptr, arg_indx++);

  // Insert ExecutableRunOptions.
  llvm::Value* run_opts_val = GetExecutableRunOptionsArgument();
  args_val = b()->CreateInsertValue(args_val, run_opts_val, arg_indx++);

  // Insert OneDnnNormConfig.
  auto typed_custom_call = Cast<HloCustomCallInstruction>(custom_call);
  auto backend_config = typed_custom_call->backend_config<BackendConfig>();
  OneDnnNormConfig ln_config;
  ln_config.CopyFrom(backend_config->onednn_layer_norm_config());
  std::string str_config;
  ln_config.SerializeToString(&str_config);
  llvm::Value* ln_config_val =
      b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(str_config));
  args_val = b()->CreateInsertValue(args_val, ln_config_val, arg_indx++);

  // Insert operands.
  auto operands_stack_alloca =
      EmitOneDnnOperandsAlloca(custom_call, args_val, arg_indx);
  TF_RET_CHECK(nargs == arg_indx)
      << "Number of arguments don't equal the last argument index.";

  llvm::Value* args_ptr =
      llvm_ir::EmitAllocaAtFunctionEntry(ptr_array_type, "layernorm.args", b());
  b()->CreateLifetimeStart(args_ptr, b()->getInt64(-1));
  b()->CreateStore(args_val, args_ptr);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));
  llvm_ir::IrArray result_array = GetIrArrayFor(custom_call);
  auto result_stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), result_array);

  EmitCallToFunc(runtime::kOneDnnLayerNormSymbolName,
                 {result_stack_alloca.value, args_ptr}, b()->getVoidTy());

  // Lifetime ends for all stack allocations.
  b()->CreateLifetimeEnd(nargs_ptr, b()->getInt64(-1));
  for (int i = 0; i < num_operands; ++i) {
    operands_stack_alloca[i].EmitLifetimeEnd();
  }
  b()->CreateLifetimeEnd(args_ptr, b()->getInt64(-1));
  result_stack_alloca.EmitLifetimeEnd();

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleOneDnnSoftmax(HloInstruction* custom_call) {
  // Serialize and emit OneDnnSoftmaxConfig.
  auto typed_custom_call = Cast<HloCustomCallInstruction>(custom_call);
  auto backend_config = typed_custom_call->backend_config<BackendConfig>();
  OneDnnSoftmaxConfig softmax_config;
  softmax_config.CopyFrom(backend_config->onednn_softmax_config());
  std::string str_config;
  softmax_config.SerializeToString(&str_config);
  llvm::Value* softmax_config_val =
      b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(str_config));

  auto input = custom_call->operand(0);
  llvm_ir::IrArray input_array(GetIrArrayFor(input));
  auto input_stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), input_array);

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));
  llvm_ir::IrArray result_array = GetIrArrayFor(custom_call);
  auto result_stack_alloca = GetAllocaAndEmitMemrefInfo(*b(), result_array);

  EmitCallToFunc(runtime::kOneDnnSoftmaxSymbolName,
                 {GetExecutableRunOptionsArgument(), input_stack_alloca.value,
                  result_stack_alloca.value, softmax_config_val},
                 b()->getVoidTy());

  input_stack_alloca.EmitLifetimeEnd();
  result_stack_alloca.EmitLifetimeEnd();

  return absl::OkStatus();
}
#endif  // INTEL_MKL

absl::Status IrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  if (custom_call->custom_call_target() == "PadToStatic") {
    return HandlePadToStatic(custom_call);
  }
  if (custom_call->custom_call_target() == "SliceToDynamic") {
    return HandleSliceToDynamic(custom_call);
  }
  if (custom_call->custom_call_target() == "TopK") {
    return HandleTopK(custom_call);
  }
#if defined(INTEL_MKL)
  if (custom_call->custom_call_target() == "__onednn$matmul") {
    return HandleOneDnnMatMulCalls(custom_call,
                                   runtime::kOneDnnMatMulSymbolName);
  }
  if (custom_call->custom_call_target() == "__onednn$softmax") {
    return HandleOneDnnSoftmax(custom_call);
  }
  if (custom_call->custom_call_target() == "__onednn$layernorm") {
    return HandleOneDnnLayerNorm(custom_call);
  }
  if (custom_call->custom_call_target() == "__onednn$convolution") {
    return HandleOneDnnConvolution(custom_call);
  }
  if (custom_call->custom_call_target() == "__onednn$matmul_reorder") {
    return HandleOneDnnMatMulCalls(custom_call,
                                   runtime::kOneDnnMatMulReorderSymbolName);
  }
#endif  // INTEL_MKL
  absl::Span<HloInstruction* const> operands(custom_call->operands());
  auto typed_custom_call = Cast<HloCustomCallInstruction>(custom_call);
  auto is_typed_ffi = typed_custom_call->api_version() ==
                      CustomCallApiVersion::API_VERSION_TYPED_FFI;
  std::vector<llvm::Value*> operand_values;
  operand_values.reserve(operands.size());

  for (int64_t i = 0; i < operands.size(); ++i) {
    HloInstruction* operand = operands[i];
    if (is_typed_ffi) {
      // Emit nested tuples as flat buffer pointers
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          operand->shape(), [&](const Shape& shape, const ShapeIndex& index) {
            if (!shape.IsArray() && !shape.IsToken()) {
              return absl::OkStatus();
            }
            TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                                assignment_.GetUniqueSlice(operand, index));
            operand_values.push_back(EmitBufferPointer(slice, shape));
            return absl::OkStatus();
          }));
    } else {
      operand_values.push_back(GetEmittedValueFor(operand));
    }
  }
  llvm::AllocaInst* operands_alloca =
      llvm_ir::EmitAllocaAtFunctionEntryWithCount(
          b()->getPtrTy(), b()->getInt32(operand_values.size()),
          "cc_operands_alloca", b());
  if (emit_code_for_msan_) {
    // Mark the alloca as initialized for msan. The buffer gets read by the
    // custom callee, which might be msan-instrumented.
    // TODO(b/66051036): Run the msan instrumentation pass instead.
    const llvm::DataLayout& dl = module_->getDataLayout();
    llvm::Type* intptr_type = b()->getIntPtrTy(dl);
    EmitCallToFunc("__msan_unpoison",
                   {operands_alloca,
                    llvm::ConstantInt::get(
                        intptr_type, *operands_alloca->getAllocationSize(dl))},
                   b()->getVoidTy());
  }
  for (int64_t i = 0; i < operand_values.size(); ++i) {
    llvm::Value* slot_in_operands_alloca =
        InBoundsGEP(operands_alloca->getAllocatedType(), operands_alloca,
                    {b()->getInt64(i)});
    Store(operand_values[i], slot_in_operands_alloca);
  }

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(custom_call));
  // Write the tuple table if the output is a tuple.
  std::vector<llvm::Value*> tuple_ptrs;
  if (custom_call->shape().IsTuple()) {
    for (int i = 0; i < ShapeUtil::TupleElementCount(custom_call->shape());
         ++i) {
      const Shape& elem_shape =
          ShapeUtil::GetTupleElementShape(custom_call->shape(), i);
      if (!is_typed_ffi) {
        TF_RET_CHECK(!elem_shape.IsTuple()) << "Nested tuples not implemented";
      }
      TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                          assignment_.GetUniqueSlice(custom_call, {i}));
      tuple_ptrs.push_back(EmitBufferPointer(slice, elem_shape));
    }
    llvm_ir::EmitTuple(GetIrArrayFor(custom_call), tuple_ptrs, b());
  }
  auto* output_address = GetEmittedValueFor(custom_call);

  switch (typed_custom_call->api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
#ifdef PLATFORM_GOOGLE
      LOG(FATAL)
#else
      LOG(ERROR)
#endif
          << "Custom call API version `API_VERSION_ORIGINAL` is not supported "
             "by XLA:CPU. Prefer https://docs.jax.dev/en/latest/ffi.html. It "
             "will be fully removed in November 2025.";

      EmitCallToFunc(custom_call->custom_call_target(),
                     {output_address, operands_alloca}, b()->getVoidTy());
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
      EmitCallToFunc(custom_call->custom_call_target(),
                     {output_address, operands_alloca, GetStatusArgument()},
                     b()->getVoidTy());
      EmitEarlyReturnIfErrorStatus();
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED: {
      absl::string_view opaque = typed_custom_call->opaque();
      EmitCallToFunc(custom_call->custom_call_target(),
                     {output_address, operands_alloca,
                      b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(opaque)),
                      b()->getInt64(opaque.size()), GetStatusArgument()},
                     b()->getVoidTy());
      EmitEarlyReturnIfErrorStatus();
      break;
    }
    case CustomCallApiVersion::API_VERSION_TYPED_FFI: {
      // Flatten into raw buffers to avoid (nested) tuples
      std::vector<llvm::Value*> buffer_ptrs;
      if (custom_call->shape().IsTuple()) {
        buffer_ptrs.reserve(ShapeUtil::TupleElementCount(custom_call->shape()));
      }
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          custom_call->shape(),
          [&](const Shape& shape, const ShapeIndex& index) {
            if (!shape.IsArray() && !shape.IsToken()) {
              return absl::OkStatus();
            }
            TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                                assignment_.GetUniqueSlice(custom_call, index));
            buffer_ptrs.push_back(EmitBufferPointer(slice, shape));
            return absl::OkStatus();
          }));
      llvm::AllocaInst* results_alloca =
          llvm_ir::EmitAllocaAtFunctionEntryWithCount(
              b()->getPtrTy(), b()->getInt32(buffer_ptrs.size()),
              "ffi_results_alloca", b());
      if (emit_code_for_msan_) {
        // Mark the alloca as initialized for msan
        // TODO(b/66051036): Run the msan instrumentation pass instead.
        const llvm::DataLayout& dl = module_->getDataLayout();
        llvm::Type* intptr_type = b()->getIntPtrTy(dl);
        EmitCallToFunc(
            "__msan_unpoison",
            {results_alloca,
             llvm::ConstantInt::get(intptr_type,
                                    *results_alloca->getAllocationSize(dl))},
            b()->getVoidTy());
      }
      for (int i = 0; i < buffer_ptrs.size(); ++i) {
        llvm::Value* tuple_slot_in_results_alloca =
            InBoundsGEP(results_alloca->getAllocatedType(), results_alloca,
                        {b()->getInt64(i)});
        Store(buffer_ptrs[i], tuple_slot_in_results_alloca);
      }
      EmitCallToFfi(typed_custom_call, results_alloca, operands_alloca);
      EmitEarlyReturnIfErrorStatus();
      break;
    }
    default:
      return Internal(
          "Unknown custom-call API version enum value: %d (%s)",
          typed_custom_call->api_version(),
          CustomCallApiVersion_Name(typed_custom_call->api_version()));
  }

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleWhile(HloInstruction* xla_while) {
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
                         const ShapeIndex& index) -> absl::Status {
        auto check = [this](const HloInstruction* a, const HloInstruction* b,
                            const ShapeIndex& index) -> absl::Status {
          const BufferAllocation::Slice slice_a =
              assignment_.GetUniqueSlice(a, index).value();
          const BufferAllocation::Slice slice_b =
              assignment_.GetUniqueSlice(b, index).value();
          if (slice_a != slice_b) {
            return Internal(
                "instruction %s %s does not share slice with "
                "instruction %s %s",
                a->ToString(), slice_a.ToString(), b->ToString(),
                slice_b.ToString());
          }
          return absl::OkStatus();
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
        return absl::OkStatus();
      }));

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(xla_while));

  // Generating:
  //   while (Condition(while_result)) {
  //     // CopyInsertion pass inserts copies which enable 'while_result' to
  //     // be passed back in as 'Body' parameter.
  //     while_result = Body(while_result);  // Insert
  //   }

  // Terminates the current block with a branch to a while header.
  llvm::BasicBlock* header_bb = llvm::BasicBlock::Create(
      module_->getContext(), IrName(xla_while, "header"),
      compute_function()->function());
  Br(header_bb);
  b()->SetInsertPoint(header_bb);

  // TODO(willfroom): Use trip count if known.

  // Calls the condition function to determine whether to proceed with the
  // body.  It must return a bool, so use the scalar call form.
  EmitGlobalCall(*xla_while->while_condition(), IrName(xla_while, "cond"));
  llvm::Value* while_predicate = ICmpNE(
      Load(IrShapeType(
               xla_while->while_condition()->root_instruction()->shape()),
           GetBufferForGlobalCallReturnValue(*xla_while->while_condition())),
      llvm::ConstantInt::get(
          llvm_ir::PrimitiveTypeToIrType(PRED, module_->getContext()), 0));

  // Branches to the body or to the while exit depending on the condition.
  llvm::BasicBlock* body_bb =
      llvm::BasicBlock::Create(module_->getContext(), IrName(xla_while, "body"),
                               compute_function()->function());
  llvm::BasicBlock* exit_bb = llvm::BasicBlock::Create(
      module_->getContext(), IrName(xla_while, "exit"));
  CondBr(while_predicate, body_bb, exit_bb);

  // Calls the body function from the body block.
  b()->SetInsertPoint(body_bb);

  // Calls the body function.
  EmitGlobalCall(*xla_while->while_body(), IrName(xla_while, "body"));

  // Finishes with a branch back to the header.
  Br(header_bb);

  // Adds the exit block to the function and sets the insert point there.
  llvm::Function* llvm_fn = compute_function()->function();
  llvm_fn->insert(llvm_fn->end(), exit_bb);
  b()->SetInsertPoint(exit_bb);

  return absl::OkStatus();
}

absl::Status IrEmitter::EmitFastConcatenate(
    const HloInstruction* instr,
    absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array) {
  return ::xla::cpu::EmitFastConcatenate(instr, source_arrays, target_array,
                                         module_, *b());
}

absl::Status EmitFastConcatenate(
    const HloInstruction* instr,
    absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array, llvm::Module* module,
    llvm::IRBuilderBase& b) {
  // We split the dimensions into three categories: the dimension over which we
  // are concatenating (concat_dim), the dimensions that are minor to it
  // (inner_dims) and the dimensions that are major to it (outer_dims).

  auto* concatenate = Cast<HloConcatenateInstruction>(instr);
  const Shape& output_shape = concatenate->shape();
  int64_t concat_dim = concatenate->concatenate_dimension();
  const Layout& output_layout = output_shape.layout();
  auto output_min2maj = LayoutUtil::MinorToMajor(output_layout);
  auto concat_dim_layout_itr = absl::c_find(output_min2maj, concat_dim);

  std::vector<int64_t> inner_dims(output_min2maj.begin(),
                                  concat_dim_layout_itr);
  std::vector<int64_t> outer_dims(std::next(concat_dim_layout_itr),
                                  output_min2maj.end());

  llvm_ir::ForLoopNest loops(IrName(concatenate), &b);
  std::vector<llvm::Value*> target_multi_index =
      loops.AddLoopsForShapeOnDimensions(output_shape, outer_dims, "concat");
  absl::c_replace(target_multi_index, static_cast<llvm::Value*>(nullptr),
                  static_cast<llvm::Value*>(b.getInt64(0)));
  llvm_ir::IrArray::Index target_index(target_multi_index, output_shape,
                                       b.getInt64Ty());

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b);
  }

  PrimitiveType primitive_type = output_shape.element_type();
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);

  // Contiguous subregions from each operand to the concatenate contribute to a
  // contiguous subregion in the target buffer starting at target_region_begin.
  llvm::Value* target_region_begin =
      target_array.EmitArrayElementAddress(target_index, &b, "target_region");
  int64_t byte_offset_into_target_region = 0;

  int64_t inner_dims_product = absl::c_accumulate(
      inner_dims, int64_t{1}, [&](int64_t product, int64_t inner_dim) {
        return product * output_shape.dimensions(inner_dim);
      });

  // For each operand, emit a memcpy from the operand to the target of size
  // equal to the product of inner dimensions.
  for (int64_t i = 0; i < source_arrays.size(); ++i) {
    const Shape& input_shape = concatenate->operand(i)->shape();
    const llvm_ir::IrArray& source_array = source_arrays[i];
    llvm_ir::IrArray::Index source_index(target_multi_index, input_shape,
                                         b.getInt64Ty());
    llvm::Value* copy_source_address =
        source_array.EmitArrayElementAddress(source_index, &b, "src_addr");

    llvm::Value* copy_target_address =
        b.CreateGEP(b.getInt8Ty(), target_region_begin,
                    b.getInt64(byte_offset_into_target_region));

    ::xla::cpu::EmitTransferElements(
        copy_target_address, copy_source_address,
        inner_dims_product * input_shape.dimensions(concat_dim), primitive_type,
        target_array, source_array, module, b);

    byte_offset_into_target_region += inner_dims_product *
                                      input_shape.dimensions(concat_dim) *
                                      primitive_type_size;
  }

  if (!outer_dims.empty()) {
    SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b);
  }
  return absl::OkStatus();
}

llvm::Value* IrEmitter::EmitPrintf(absl::string_view fmt,
                                   absl::Span<llvm::Value* const> arguments) {
  std::vector<llvm::Value*> call_args;
  call_args.push_back(b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)));
  absl::c_copy(arguments, std::back_inserter(call_args));
  return b()->CreateCall(
      b()->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "printf",
          llvm::FunctionType::get(b()->getInt32Ty(), {b()->getPtrTy()},
                                  /*isVarArg=*/true)),
      call_args);
}

llvm::Value* IrEmitter::EmitPrintfToStderr(
    absl::string_view fmt, absl::Span<llvm::Value* const> arguments) {
  std::vector<llvm::Value*> call_args;
  call_args.push_back(b()->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)));
  absl::c_copy(arguments, std::back_inserter(call_args));
  return b()->CreateCall(
      b()->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          runtime::kPrintfToStderrSymbolName,
          llvm::FunctionType::get(b()->getInt32Ty(), {b()->getPtrTy()},
                                  /*isVarArg=*/true)),
      call_args);
}

llvm::Value* IrEmitter::EmitCallToFunc(
    std::string func_name, const std::vector<llvm::Value*>& arguments,
    llvm::Type* return_type, bool does_not_throw, bool only_accesses_arg_memory,
    bool only_accesses_inaccessible_mem_or_arg_mem) {
  std::vector<llvm::Type*> types;
  types.reserve(arguments.size());
  absl::c_transform(arguments, std::back_inserter(types),
                    [&](llvm::Value* val) { return val->getType(); });
  llvm::FunctionType* func_type =
      llvm::FunctionType::get(return_type, types, /*isVarArg=*/false);
  auto func = llvm::dyn_cast<llvm::Function>(
      module_->getOrInsertFunction(func_name, func_type).getCallee());
  func->setCallingConv(llvm::CallingConv::C);
  if (does_not_throw) {
    func->setDoesNotThrow();
  }
  if (only_accesses_arg_memory) {
    func->setOnlyAccessesArgMemory();
  }
  if (only_accesses_inaccessible_mem_or_arg_mem) {
    func->setOnlyAccessesInaccessibleMemOrArgMem();
  }
  return b()->CreateCall(func, arguments);
}

template <typename T>
static const Shape& GetShape(T&& arg) {
  if constexpr (std::is_convertible_v<absl::remove_cvref_t<decltype(arg)>,
                                      Shape>) {
    return arg;  // convertible to shape, so just return
  } else {
    return arg->shape();
  }
};

struct EncodedInfo {
  llvm::AllocaInst* alloca;
  int64_t size;
};

template <typename Args>
static EncodedInfo StoreEncodedTypes(absl::string_view alloca_name,
                                     const Args& args,
                                     llvm::IRBuilderBase& ir) {
  // Store the types of `args` into the allocated memory. These types are stored
  // as int32_t values contiguously. All tuples are flattened to bare elements.
  int64_t total_elements = 0;
  for (int64_t i = 0; i < args.size(); ++i) {
    total_elements += ShapeUtil::GetLeafCount(GetShape(args[i]));
  }
  llvm::AllocaInst* types_alloca = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      ir.getInt32Ty(), ir.getInt64(total_elements), alloca_name, &ir);
  int64_t element_id = 0;
  auto store_type = [&](const Shape& shape, const ShapeIndex& index) {
    if (shape.IsTuple()) {
      return;
    }
    llvm::Value* slot_in_types_alloca = ir.CreateConstInBoundsGEP1_32(
        ir.getInt32Ty(), types_alloca, element_id++);
    ir.CreateStore(ir.getInt32(shape.element_type()), slot_in_types_alloca);
  };

  for (int64_t i = 0; i < args.size(); ++i) {
    ShapeUtil::ForEachSubshape(GetShape(args[i]), store_type);
  }
  CHECK_EQ(element_id, total_elements);
  return {types_alloca, total_elements};
};

template <typename Args>
static EncodedInfo StoreEncodedShapes(absl::string_view alloca_name,
                                      const Args& args,
                                      llvm::IRBuilderBase& ir) {
  // Prepare metadata for all buffers. A tuple shape is flattened to only encode
  // information about its elements (buffers). Shapes metadata is encoded using
  // contiguous flattened dimension values:
  //    {
  // 1:   DIMCOUNT_1, DIM_1[1], DIM_1[2], ..., DIM_1[DIMCOUNT_1],
  //                  \______________DIMCOUNT_1 _______________/
  // 2:   DIMCOUNT_2, DIM_2[1], DIM_2[2], ..., DIM_2[DIMCOUNT_2],
  //                  \______________DIMCOUNT_2 _______________/
  // .:   ...
  // N:   DIMCOUNT_N, DIM_N[1], DIM_N[2], ..., DIM_N[DIMCOUNT_N],
  //                  \______________DIMCOUNT_N _______________/
  //    }
  //  where N is `operand_count`, and `DIMCOUNT_i` is the # of dimensions
  int64_t total_dims = 0;
  int64_t total_dim_counts = 0;
  for (int64_t i = 0; i < args.size(); ++i) {
    ShapeUtil::ForEachSubshape(
        GetShape(args[i]), [&](const Shape& shape, const ShapeIndex& index) {
          if (!shape.IsArray()) {
            return;
          }
          total_dims += shape.dimensions().size();
          ++total_dim_counts;
        });
  }
  int64_t shapes_encoding_size = total_dim_counts  // the # of dimension counts
                                 + total_dims;     // the # of dimension values

  llvm::AllocaInst* shapes_alloca = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
      ir.getInt64Ty(), ir.getInt64(shapes_encoding_size), alloca_name, &ir);

  int64_t slot_id = 0;
  auto store_shape = [&](const Shape& shape, const ShapeIndex& index) {
    if (!shape.IsArray()) {
      return;
    }
    llvm::Value* alloca_slot = ir.CreateConstInBoundsGEP1_64(
        ir.getInt64Ty(), shapes_alloca, slot_id++);
    // Store the operand count
    ir.CreateStore(ir.getInt64(shape.dimensions().size()), alloca_slot);
    // Store the operand dimensions
    for (int64_t dim : shape.dimensions()) {
      alloca_slot = ir.CreateConstInBoundsGEP1_64(ir.getInt64Ty(),
                                                  shapes_alloca, slot_id++);
      ir.CreateStore(ir.getInt64(dim), alloca_slot);
    }
  };

  for (int64_t i = 0; i < args.size(); ++i) {
    ShapeUtil::ForEachSubshape(GetShape(args[i]), store_shape);
  }
  CHECK_EQ(slot_id, shapes_encoding_size);  // All slots are filled
  return {shapes_alloca, shapes_encoding_size};
};

llvm::Value* IrEmitter::EmitCallToFfi(HloCustomCallInstruction* custom_call,
                                      llvm::AllocaInst* results_alloca,
                                      llvm::AllocaInst* operands_alloca) {
  const auto& operands = absl::MakeSpan(custom_call->operands());
  const auto& shape = custom_call->shape();
  const auto& result_shapes =
      shape.IsTuple() ? shape.tuple_shapes() : std::vector<Shape>({shape});

  EncodedInfo operand_types_encoded =
      StoreEncodedTypes("operands_types", operands, *b());
  EncodedInfo operand_shapes_encoded =
      StoreEncodedShapes("operands_shapes", operands, *b());

  EncodedInfo result_types_encoded =
      StoreEncodedTypes("results_types", result_shapes, *b());
  EncodedInfo result_shapes_encoded =
      StoreEncodedShapes("results_shapes", result_shapes, *b());

  const absl::string_view target = custom_call->custom_call_target();  // name
  const absl::string_view opaque = custom_call->opaque();

  const auto target_ref = llvm_ir::AsStringRef(target);
  const auto opaque_ref = llvm_ir::AsStringRef(opaque);

  std::vector<llvm::Value*> arguments = {
      /*run_options_ptr=*/GetExecutableRunOptionsArgument(),
      /*target_name_ptr=*/b()->CreateGlobalStringPtr(target_ref),
      /*target_name_len=*/b()->getInt64(target.size()),
      /*outputs=*/results_alloca,
      /*inputs=*/operands_alloca,
      /*opaque_str_ptr=*/b()->CreateGlobalStringPtr(opaque_ref),
      /*opaque_str_len=*/b()->getInt64(opaque.size()),
      /*status_opaque=*/GetStatusArgument(),
      /*operand_types=*/operand_types_encoded.alloca,
      /*operand_count=*/b()->getInt64(operand_types_encoded.size),
      /*operand_dims=*/operand_shapes_encoded.alloca,
      /*result_types=*/result_types_encoded.alloca,
      /*result_count=*/b()->getInt64(result_types_encoded.size),
      /*result_dims=*/result_shapes_encoded.alloca,
  };

  return EmitCallToFunc(runtime::kHandleFfiCallSymbolName, arguments,
                        b()->getVoidTy(),
                        /*does_not_throw=*/false,
                        /*only_accesses_arg_memory=*/true);
}

void IrEmitter::EmitTransferElements(llvm::Value* target, llvm::Value* source,
                                     int64_t element_count,
                                     PrimitiveType primitive_type,
                                     const llvm_ir::IrArray& target_array,
                                     const llvm_ir::IrArray& source_array) {
  ::xla::cpu::EmitTransferElements(target, source, element_count,
                                   primitive_type, target_array, source_array,
                                   module_, *b());
}

void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                          int64_t element_count, PrimitiveType primitive_type,
                          const llvm_ir::IrArray& target_array,
                          const llvm_ir::IrArray& source_array,
                          llvm::Module* module, llvm::IRBuilderBase& b) {
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  llvm::Align element_alignment(tsl::MathUtil::GCD<unsigned>(
      primitive_type_size,
      ::xla::cpu::MinimumAlignmentForPrimitiveType(primitive_type)));
  llvm::Type* primitive_llvm_type =
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module->getContext());

  if (element_count == 1) {
    auto* load_instruction =
        b.CreateAlignedLoad(primitive_llvm_type, source, element_alignment);
    source_array.AnnotateLoadStoreInstructionWithMetadata(load_instruction);
    auto* store_instruction =
        b.CreateAlignedStore(load_instruction, target, element_alignment);
    target_array.AnnotateLoadStoreInstructionWithMetadata(store_instruction);
  } else {
    auto* memcpy_instruction = b.CreateMemCpy(
        target, /*DstAlign=*/llvm::Align(element_alignment), source,
        /*SrcAlign=*/llvm::Align(element_alignment),
        element_count * primitive_type_size);

    // The memcpy does the load and the store internally.  The aliasing related
    // metadata has to reflect that.
    std::map<int, llvm::MDNode*> merged_metadata =
        llvm_ir::MergeMetadata(&module->getContext(), source_array.metadata(),
                               target_array.metadata());
    for (const auto& kind_md_pair : merged_metadata) {
      memcpy_instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
    }
  }
}

absl::Status IrEmitter::CanDoFastConcatenate(
    const HloInstruction* instr) const {
  if (ShouldEmitParallelLoopFor(*instr)) {
    return absl::Status(
        absl::StatusCode::kFailedPrecondition,
        "Cannot generate memcpy-based concat for the parallel CPU backend");
  }
  const auto* concatenate = Cast<HloConcatenateInstruction>(instr);

  const Shape& output_shape = concatenate->shape();
  for (auto* op : concatenate->operands()) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Operand has mismatching layouts");
    }
  }
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleConcatenate(HloInstruction* concatenate) {
  absl::Status fast_impl_reason = CanDoFastConcatenate(concatenate);
  if (fast_impl_reason.ok()) {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(concatenate));
    llvm_ir::IrArray target_array = GetIrArrayFor(concatenate);
    std::vector<llvm_ir::IrArray> source_arrays;
    source_arrays.reserve(concatenate->operands().size());
    for (HloInstruction* operand : concatenate->operands()) {
      source_arrays.emplace_back(GetIrArrayFor(operand));
    }
    TF_RETURN_IF_ERROR(::xla::cpu::EmitFastConcatenate(
        concatenate, source_arrays, target_array, module_, *b()));
    VLOG(1) << "Emitted fast concatenate for " << concatenate->ToString();
    return absl::OkStatus();
  }
  VLOG(1) << "Could not emit fast concatenate for " << concatenate->ToString()
          << ": " << fast_impl_reason.message();

  return DefaultAction(concatenate);
}

absl::Status IrEmitter::HandleConditional(HloInstruction* conditional) {
  auto branch_index = conditional->operand(0);
  int num_branches = conditional->branch_count();
  TF_RET_CHECK(ShapeUtil::IsScalar(branch_index->shape()) &&
               (branch_index->shape().element_type() == PRED ||
                branch_index->shape().element_type() == S32))
      << "Branch index on a conditional must be scalar bool or int32_t; got: "
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
        GetIrArrayFor(branch_index).GetBasePointeeType(),
        GetIrArrayFor(branch_index).GetBasePointer(), "load_predicate_value");
    llvm::Value* pred_cond = ICmpNE(
        pred_value,
        llvm::ConstantInt::get(
            llvm_ir::PrimitiveTypeToIrType(PRED, module_->getContext()), 0),
        "boolean_predicate");
    llvm_ir::LlvmIfData if_data =
        llvm_ir::EmitIfThenElse(pred_cond, "conditional", b());

    SetToFirstInsertPoint(if_data.true_block, b());
    EmitGlobalCall(*conditional->branch_computation(0),
                   IrName(conditional, "_true"));

    SetToFirstInsertPoint(if_data.false_block, b());
    EmitGlobalCall(*conditional->branch_computation(1),
                   IrName(conditional, "_false"));

    SetToFirstInsertPoint(if_data.after_block, b());
    return absl::OkStatus();
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
      GetIrArrayFor(branch_index).GetBasePointeeType(),
      GetIrArrayFor(branch_index).GetBasePointer(), "load_branch_index_value");

  auto case_block = b()->GetInsertBlock();
  llvm::BasicBlock* after_block;
  // Add a terminator to the case block, if necessary.
  if (case_block->getTerminator() == nullptr) {
    after_block = llvm_ir::CreateBasicBlock(nullptr, "case-after", b());
    b()->SetInsertPoint(case_block);
    b()->CreateBr(after_block);
  } else {
    after_block =
        case_block->splitBasicBlock(b()->GetInsertPoint(), "case-after");
  }
  // Our basic block should now end with an unconditional branch.  Remove it;
  // we're going to replace it with a switch based branch.
  case_block->getTerminator()->eraseFromParent();

  // Lower the default branch computation.
  auto default_block = llvm_ir::CreateBasicBlock(nullptr, "case-default", b());
  b()->SetInsertPoint(default_block);
  EmitGlobalCall(*conditional->branch_computation(num_branches - 1),
                 IrName(conditional, "_default"));
  b()->CreateBr(after_block);

  // Prepare the switch (branch_index) { ... } instruction.
  b()->SetInsertPoint(case_block);
  llvm::SwitchInst* case_inst =
      b()->CreateSwitch(branch_index_value, default_block, num_branches - 1);
  // Lower each branch's computation.
  for (int br = 0; br < num_branches - 1; ++br) {  // last branch is default
    // Lower the case br: { ... ; break; } computation.
    auto branch_block = llvm_ir::CreateBasicBlock(
        nullptr, absl::StrCat("case-branch", br), b());
    b()->SetInsertPoint(branch_block);
    EmitGlobalCall(*conditional->branch_computation(br),
                   IrName(conditional, absl::StrCat("_branch", br)));
    b()->CreateBr(after_block);
    case_inst->addCase(b()->getInt32(br), branch_block);
  }

  SetToFirstInsertPoint(after_block, b());
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAfterAll(HloInstruction* after_all) {
  TF_RET_CHECK(ByteSizeOf(after_all->shape()) == 0);
  // No code to generate, but we need to emit an address for book-keeping.
  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(after_all));
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleBatchNormGrad(HloInstruction* batch_norm_grad) {
  return Unimplemented("BatchNormGrad should be rewritten for CPU.");
}

absl::Status IrEmitter::HandleBatchNormTraining(
    HloInstruction* batch_norm_training) {
  return Unimplemented("BatchNormTraining should be rewritten for CPU.");
}

absl::Status IrEmitter::HandleGetDimensionSize(HloInstruction* get_size) {
  return Unimplemented("GetDimensionSize should be rewritten for CPU.");
}

absl::Status IrEmitter::HandleSetDimensionSize(HloInstruction* set_size) {
  return Unimplemented("SetDimensionSize should be rewritten for CPU.");
}

absl::Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  // AddDedendency just forwards its zero-th operand.
  emitted_value_[add_dependency] =
      GetEmittedValueFor(add_dependency->operand(0));
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleRng(HloInstruction* rng) {
  return Unimplemented("Rng should be expanded for CPU.");
}

absl::Status IrEmitter::HandleRngBitGenerator(HloInstruction* rng) {
  return Unimplemented("RngBitGenerator should be expanded for CPU.");
}

absl::Status IrEmitter::HandleRngGetAndUpdateState(HloInstruction* rng_state) {
  VLOG(2) << "RngGetAndUpdateState: " << rng_state->ToString();
  llvm::Value* old_state = llvm_ir::RngGetAndUpdateState(
      Cast<HloRngGetAndUpdateStateInstruction>(rng_state)->delta(), module_,
      b());

  TF_RETURN_IF_ERROR(EmitTargetAddressForOp(rng_state));
  llvm::Value* address = GetEmittedValueFor(rng_state);

  // The buffer has an array type while the value has a i128. Cast the
  // buffer to i128 type to store the value.
  llvm::StoreInst* store = Store(old_state, address);
  store->setAlignment(llvm::Align(IrEmitter::MinimumAlignmentForPrimitiveType(
      rng_state->shape().element_type())));

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleStochasticConvert(HloInstruction* instruction) {
  return Unimplemented("StochasticConvert should be decomposed for CPU.");
}

absl::Status IrEmitter::FinishVisit(HloInstruction* root) {
  // When this method is called, we should have already emitted an IR value for
  // the root (return) op. The IR value holds the address of the buffer holding
  // the value. If the root is a constant or parameter, we perform a memcpy from
  // this buffer to the retval buffer of the computation. Otherwise, there's
  // nothing to do since the result was already written directly into the output
  // buffer.
  VLOG(2) << "FinishVisit root: " << root->ToString();
  if (root->opcode() == HloOpcode::kOutfeed) {
    VLOG(2) << "  outfeed with value: "
            << llvm_ir::DumpToString(GetEmittedValueFor(root->operand(0)));
  } else {
    VLOG(2) << "  value: " << llvm_ir::DumpToString(GetEmittedValueFor(root));
  }

  auto record_complete_computation = [&](llvm::Value* prof_counter) {
    if (prof_counter) {
      profiling_state_.RecordCompleteComputation(b(), prof_counter);
    }
  };

  // For the entry computation this increment is cumulative of embedded
  // computations since it includes cycles spent in computations invoked by
  // While, Call etc.
  record_complete_computation(GetProfileCounterFor(*root->parent()));
  return absl::OkStatus();
}

template <typename T>
llvm::Value* IrEmitter::GetProfileCounterCommon(
    const T& hlo,
    const absl::flat_hash_map<const T*, int64_t>& profile_index_map) {
  auto it = profile_index_map.find(&hlo);
  if (it == profile_index_map.end()) {
    return nullptr;
  }

  int64_t prof_counter_idx = it->second;
  std::string counter_name = IrName("prof_counter", hlo.name());
  return GEP(b()->getInt64Ty(), GetProfileCountersArgument(),
             b()->getInt64(prof_counter_idx), counter_name);
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

void IrEmitter::ProfilingState::UpdateProfileCounter(llvm::IRBuilderBase* b,
                                                     llvm::Value* prof_counter,
                                                     llvm::Value* cycle_end,
                                                     llvm::Value* cycle_start) {
  auto* cycle_diff = b->CreateSub(cycle_end, cycle_start);
  llvm::LoadInst* old_cycle_count = b->CreateLoad(
      llvm::cast<llvm::GetElementPtrInst>(prof_counter)->getSourceElementType(),
      prof_counter, "old_cycle_count");
  auto* new_cycle_count =
      b->CreateAdd(cycle_diff, old_cycle_count, "new_cycle_count");
  b->CreateStore(new_cycle_count, prof_counter);
}

llvm::Value* IrEmitter::ProfilingState::ReadCycleCounter(
    llvm::IRBuilderBase* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  if (!use_rdtscp_) {
    llvm::Function* func_llvm_readcyclecounter =
        llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::readcyclecounter);
    return b->CreateCall(func_llvm_readcyclecounter);
  }
  llvm::Function* func_llvm_x86_rdtscp =
      llvm::Intrinsic::getOrInsertDeclaration(module,
                                              llvm::Intrinsic::x86_rdtscp);
  llvm::Value* rdtscp_call = b->CreateCall(func_llvm_x86_rdtscp);
  return b->CreateExtractValue(rdtscp_call, {0});
}

void IrEmitter::ProfilingState::RecordCycleStart(llvm::IRBuilderBase* b,
                                                 HloInstruction* hlo) {
  auto* cycle_start = ReadCycleCounter(b);
  cycle_start->setName(IrName(hlo, "cycle_start"));
  cycle_starts_[hlo] = cycle_start;
  if (first_read_cycle_start_ == nullptr) {
    first_read_cycle_start_ = cycle_start;
  }
}

void IrEmitter::ProfilingState::RecordCycleDelta(llvm::IRBuilderBase* b,
                                                 HloInstruction* hlo,
                                                 llvm::Value* prof_counter) {
  auto* cycle_end = ReadCycleCounter(b);
  cycle_end->setName(IrName(hlo, "cycle_end"));
  auto* cycle_start = cycle_starts_[hlo];
  UpdateProfileCounter(b, prof_counter, cycle_end, cycle_start);
  last_read_cycle_end_ = cycle_end;
}

void IrEmitter::ProfilingState::RecordCompleteComputation(
    llvm::IRBuilderBase* b, llvm::Value* prof_counter) {
  if (last_read_cycle_end_ && first_read_cycle_start_) {
    UpdateProfileCounter(b, prof_counter, last_read_cycle_end_,
                         first_read_cycle_start_);
  }
}

void IrEmitter::TracingState::EmitTracingStart(llvm::IRBuilderBase* b,
                                               HloInstruction* hlo,
                                               llvm::Value* run_options) {
  if (!enabled_) {
    return;
  }

  llvm::Type* void_ptr_type = b->getPtrTy();
  llvm::FunctionType* fn_type = llvm::FunctionType::get(
      b->getInt64Ty(),
      {void_ptr_type, void_ptr_type, void_ptr_type, b->getInt64Ty()},
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
  auto* hlo_module = b->CreateGlobalStringPtr(hlo->GetModule()->name());
  auto* program_id = b->getInt64(hlo->GetModule()->unique_id());
  auto* activity_id = b->CreateCall(
      trace_func, {run_options, hlo_name, hlo_module, program_id});
  activity_id->setName(IrName(hlo, "activity_id"));
  activity_ids_[hlo] = activity_id;
}

void IrEmitter::TracingState::EmitTracingEnd(llvm::IRBuilderBase* b,
                                             HloInstruction* hlo,
                                             llvm::Value* run_options) {
  if (!enabled_) {
    return;
  }

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(b->getVoidTy(), {b->getPtrTy(), b->getInt64Ty()},
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
  b->CreateCall(trace_func, {run_options, activity_id});
}

namespace {
bool IsHloVeryCheap(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kBitcast ||
         hlo->opcode() == HloOpcode::kTuple ||
         hlo->opcode() == HloOpcode::kGetTupleElement ||
         hlo->opcode() == HloOpcode::kParameter ||
         hlo->opcode() == HloOpcode::kConstant ||
         hlo->opcode() == HloOpcode::kReplicaId;
}
}  // namespace

absl::Status IrEmitter::Preprocess(HloInstruction* hlo) {
  VLOG(3) << "Visiting: " << hlo->ToString();
  // When profiling is enabled, trace the same HLOs that the profiler does.
  if (instruction_to_profile_idx_.count(hlo) ||
      (hlo_module_config_.cpu_traceme_enabled() && !IsHloVeryCheap(hlo) &&
       hlo->parent()->IsEntryComputation())) {
    tracing_state_.EmitTracingStart(b(), hlo,
                                    GetExecutableRunOptionsArgument());
    profiling_state_.RecordCycleStart(b(), hlo);
  }
  return absl::OkStatus();
}

absl::Status IrEmitter::Postprocess(HloInstruction* hlo) {
  if (auto* prof_counter = GetProfileCounterFor(*hlo)) {
    profiling_state_.RecordCycleDelta(b(), hlo, prof_counter);
  }
  // When profiling is enabled, trace the same HLOs that the profiler does.
  if (instruction_to_profile_idx_.count(hlo) ||
      (hlo_module_config_.cpu_traceme_enabled() && !IsHloVeryCheap(hlo) &&
       hlo->parent()->IsEntryComputation())) {
    tracing_state_.EmitTracingEnd(b(), hlo, GetExecutableRunOptionsArgument());
  }
  return absl::OkStatus();
}

llvm_ir::IrArray IrEmitter::GetIrArrayFor(const HloInstruction* hlo) {
  llvm::Value* value_for_op = GetEmittedValueFor(hlo);

  llvm::Type* ir_type = IrShapeType(hlo->shape());
  llvm_ir::IrArray array(value_for_op, ir_type, hlo->shape());
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
  return llvm_ir::ShapeToIrType(shape, module_->getContext());
}

llvm::Value* IrEmitter::GetProfileCountersArgument() {
  return compute_function()->profile_counters_arg();
}

llvm::Value* IrEmitter::GetStatusArgument() {
  return compute_function()->status_arg();
}

llvm::Value* IrEmitter::GetBufferTableArgument() {
  return compute_function()->buffer_table_arg();
}

llvm::Value* IrEmitter::GetExecutableRunOptionsArgument() {
  return compute_function()->exec_run_options_arg();
}

llvm::BasicBlock* IrEmitter::GetReturnBlock() {
  return compute_function()->return_block();
}

void IrEmitter::EmitEarlyReturnIfErrorStatus() {
  // Use the runtime helper to get the success/failure state as a boolean.
  llvm::Value* succeeded =
      EmitCallToFunc(runtime::kStatusIsSuccessSymbolName, {GetStatusArgument()},
                     b()->getInt1Ty(), /*does_not_throw=*/true,
                     /*only_accesses_arg_memory=*/true);
  llvm_ir::EmitEarlyReturn(succeeded, b(), GetReturnBlock());
}

llvm::Value* IrEmitter::EmitThreadLocalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {
  const BufferAllocation& allocation = *slice.allocation();
  llvm::Value* tempbuf_address = [&]() -> llvm::Value* {
    auto param_it =
        computation_parameter_allocations_.find(slice.allocation()->index());
    if (param_it != computation_parameter_allocations_.end()) {
      int64_t param_number = param_it->second;
      // We have to access the parameter at offset param_number in the params
      // array. The code generated here is equivalent to this C code:
      //
      //   i8* param_address_untyped = params[param_number];
      //   Param* param_address_typed = (Param*)param_address_untyped;
      //
      // Where Param is the actual element type of the underlying buffer (for
      // example, float for an XLA F32 element type).
      llvm::Value* params = compute_function()->parameters_arg();
      llvm::Value* param_address_offset = llvm_ir::EmitBufferIndexingGEP(
          params, b()->getPtrTy(), param_number, b());
      llvm::LoadInst* param_address_untyped =
          Load(b()->getPtrTy(), param_address_offset);

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
        compute_function()->function(), slice};
    auto buf_it = thread_local_buffers_.find(key);
    if (buf_it == thread_local_buffers_.end()) {
      llvm::Value* buffer = llvm_ir::EmitAllocaAtFunctionEntry(
          IrShapeType(shape), absl::StrCat("thread_local", slice.ToString()),
          b(), MinimumAlignmentForShape(target_shape));
      auto it_inserted_pair = thread_local_buffers_.insert({key, buffer});
      CHECK(it_inserted_pair.second);
      buf_it = it_inserted_pair.first;
    }
    return buf_it->second;
  }();
  return tempbuf_address;
}

llvm::Value* IrEmitter::EmitGlobalBufferPointer(
    const BufferAllocation::Slice& slice, const Shape& target_shape) {
  const BufferAllocation& allocation = *slice.allocation();

  const auto explicit_index_it = slice_to_buffer_table_index_.find(slice);
  bool has_explicit_index =
      explicit_index_it != slice_to_buffer_table_index_.end();

  CHECK(slice_to_buffer_table_index_.empty() || has_explicit_index)
      << "All or none of the slices should have an explicit index.";

  int64_t index =
      has_explicit_index ? explicit_index_it->second : slice.index();
  llvm::Value* tempbuf_address_ptr = llvm_ir::EmitBufferIndexingGEP(
      GetBufferTableArgument(), b()->getPtrTy(), index, b());
  llvm::LoadInst* tempbuf_address_base =
      Load(b()->getPtrTy(), tempbuf_address_ptr);

  AttachInvariantLoadMetadataForLoad(tempbuf_address_base);
  AttachAlignmentMetadataForLoad(tempbuf_address_base, allocation.size());
  AttachDereferenceableMetadataForLoad(tempbuf_address_base, allocation.size());

  llvm::Value* tempbuf_address_untyped = tempbuf_address_base;
  // Any explicit buffer pointer should point to the start of the slice.
  if (!has_explicit_index && slice.offset() > 0) {
    // Adjust the address to account for the slice offset.
    tempbuf_address_untyped = InBoundsGEP(
        b()->getInt8Ty(), tempbuf_address_base, b()->getInt64(slice.offset()));
  }
  return tempbuf_address_untyped;
}

llvm::Value* IrEmitter::EmitBufferPointer(const BufferAllocation::Slice& slice,
                                          const Shape& target_shape) {
  if (slice.allocation()->is_thread_local()) {
    return EmitThreadLocalBufferPointer(slice, target_shape);
  } else if (const auto itr =
                 constant_buffer_to_global_.find(slice.allocation()->index());
             itr != constant_buffer_to_global_.end()) {
    return itr->second;
  } else {
    return EmitGlobalBufferPointer(slice, target_shape);
  }
}

absl::Status IrEmitter::EmitTargetAddressForOp(const HloInstruction* op) {
  const Shape& target_shape = op->shape();
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                      assignment_.GetUniqueTopLevelSlice(op));
  llvm::Value* addr = EmitBufferPointer(slice, target_shape);
  addr->setName(IrName(op));
  emitted_value_[op] = addr;
  return absl::OkStatus();
}

absl::Status IrEmitter::EmitTargetElementLoop(
    const HloInstruction* target_op, absl::string_view desc,
    const llvm_ir::ElementGenerator& element_generator,
    std::optional<llvm_ir::IrArray> result_array_opt) {
  VLOG(2) << "EmitTargetElementLoop: " << target_op->ToString();

  llvm_ir::IrArray target_array;
  if (result_array_opt.has_value()) {
    target_array = result_array_opt.value();
  } else {
    TF_RETURN_IF_ERROR(EmitTargetAddressForOp(target_op));
    target_array = GetIrArrayFor(target_op);
  }

  const Shape& target_shape = target_op->shape();

  if (target_shape.IsTuple() &&
      (target_op->opcode() == HloOpcode::kFusion ||
       target_op->opcode() == HloOpcode::kReduce ||
       target_op->opcode() == HloOpcode::kReduceWindow)) {
    // For multiple outputs fusion, we need to emit each operand and the root.
    TF_RET_CHECK(num_dynamic_loop_bounds_ == 0);
    std::vector<llvm_ir::IrArray> output_arrays;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(target_shape); ++i) {
      TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                          assignment_.GetUniqueSlice(target_op, {i}));
      const Shape& element_shape = ShapeUtil::GetSubshape(target_shape, {i});
      llvm::Value* op_target_address = EmitBufferPointer(slice, element_shape);
      llvm::Type* op_target_type = IrShapeType(element_shape);
      output_arrays.push_back(
          llvm_ir::IrArray(op_target_address, op_target_type, element_shape));
    }
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, output_arrays, b())
            .EmitLoop(IrName(target_op, desc)));

    std::vector<llvm::Value*> tuple_operand_ptrs;
    tuple_operand_ptrs.reserve(output_arrays.size());
    for (int64_t i = 0; i < output_arrays.size(); ++i) {
      tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
    }
    llvm_ir::EmitTuple(target_array, tuple_operand_ptrs, b());

  } else {
    if (ShouldEmitParallelLoopFor(*target_op)) {
      // Emit code to read dynamic loop bounds from compute function argument.
      std::vector<std::pair<llvm::Value*, llvm::Value*>> dynamic_loop_bounds =
          compute_function()->GetDynamicLoopBounds();
      // Emit parallel loop with dynamic loop bounds for most-major dimensions.
      TF_RETURN_IF_ERROR(ParallelLoopEmitter(element_generator, target_array,
                                             &dynamic_loop_bounds, b())
                             .EmitLoop(IrName(target_op, desc)));
    } else {
      TF_RETURN_IF_ERROR(
          llvm_ir::LoopEmitter(element_generator, target_array, b())
              .EmitLoop(IrName(target_op, desc)));
    }
  }
  return absl::OkStatus();
}

absl::Status IrEmitter::EmitMemcpy(const HloInstruction& source,
                                   const HloInstruction& destination) {
  llvm::Value* source_value = GetEmittedValueFor(&source);
  llvm::Value* destination_value = GetEmittedValueFor(&destination);
  int64_t source_size = ByteSizeOf(source.shape());
  // TODO(b/63762267): Be more aggressive about specifying alignment.
  MemCpy(destination_value, /*DstAlign=*/llvm::Align(1), source_value,
         /*SrcAlign=*/llvm::Align(1), source_size);
  return absl::OkStatus();
}

absl::Status IrEmitter::ElementTypesSameAndSupported(
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
  return absl::OkStatus();
}

absl::Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  CpuElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArrayFor(operand).EmitReadArrayElement(index, b());
    };
  }
  CpuElementalIrEmitter elemental_emitter = ElementalIrEmmiterFactory();
  return EmitTargetElementLoop(
      hlo, "elemental_loop",
      elemental_emitter.MakeElementGenerator(hlo, operand_to_generator),
      std::nullopt);
}

llvm::Value* IrEmitter::EmitScalarReturningThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name) {
  std::vector<llvm::Value*> return_value =
      EmitThreadLocalCall(callee, parameters, name, /*is_reducer=*/false);
  CHECK_EQ(return_value.size(), 1);
  return return_value[0];
}

std::vector<llvm::Value*> IrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view name, bool is_reducer, bool in_compute_function) {
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
        parameter->getType(), "arg_addr", b());
    Store(parameter, parameter_addr);
    parameter_addrs.push_back(parameter_addr);
  }

  llvm::Type* return_value_buffer_type =
      llvm_ir::ShapeToIrType(return_shape, module_->getContext());
  std::string retval_alloca_name = absl::StrCat(name, "_return_value_addr");
  int retval_alignment =
      is_scalar_return
          ? MinimumAlignmentForPrimitiveType(return_shape.element_type())
          : 0;
  llvm::AllocaInst* return_value_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
      return_value_buffer_type, retval_alloca_name, b(), retval_alignment);

  std::vector<llvm::Value*> allocas_for_returned_scalars;
  if (is_scalar_return) {
    allocas_for_returned_scalars.push_back(return_value_buffer);
  } else {
    constexpr int max_tuple_size = 1000;
    CHECK_LT(return_shape.tuple_shapes().size(), max_tuple_size)
        << "Multivalue function can not return more than 1000 elements to avoid"
        << " stack smashing";
    allocas_for_returned_scalars =
        llvm_ir::EmitTupleAllocasAtFunctionEntry(return_shape, b());
    llvm_ir::IrArray tuple_array(return_value_buffer, return_value_buffer_type,
                                 return_shape);

    EmitTuple(tuple_array, allocas_for_returned_scalars, b());
  }

  llvm::Value* null_ptr = llvm::Constant::getNullValue(b()->getPtrTy());

  Call(
      FindOrDie(emitted_functions_,
                ComputationToEmit{&callee, allow_reassociation_ || is_reducer}),
      GetArrayFunctionCallArguments(
          parameter_addrs, b(), name,
          /*return_value_buffer=*/return_value_buffer,
          /*exec_run_options_arg=*/
          in_compute_function ? GetExecutableRunOptionsArgument() : null_ptr,
          /*buffer_table_arg=*/null_ptr,
          /*status_arg=*/in_compute_function ? GetStatusArgument() : null_ptr,
          /*profile_counters_arg=*/
          in_compute_function ? GetProfileCountersArgument() : null_ptr));

  if (ComputationTransitivelyContainsCustomCall(&callee)) {
    DCHECK(!in_compute_function) << "Custom call inside nested computations "
                                    "are not supported by Thunks runtime";
    EmitEarlyReturnIfErrorStatus();
  }

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    returned_scalars.push_back(
        Load(llvm::cast<llvm::AllocaInst>(addr)->getAllocatedType(), addr));
  }
  return returned_scalars;
}

void IrEmitter::EmitGlobalCall(const HloComputation& callee,
                               absl::string_view name) {
  CHECK(absl::c_binary_search(global_computations_, &callee));

  Call(FindOrDie(emitted_functions_,
                 ComputationToEmit{&callee, allow_reassociation_}),
       GetArrayFunctionCallArguments(
           /*parameter_addresses=*/{}, b(), name,
           /*return_value_buffer=*/
           llvm::Constant::getNullValue(b()->getPtrTy()),
           /*exec_run_options_arg=*/GetExecutableRunOptionsArgument(),
           /*buffer_table_arg=*/GetBufferTableArgument(),
           /*status_arg=*/GetStatusArgument(),
           /*profile_counters_arg=*/GetProfileCountersArgument()));

  if (ComputationTransitivelyContainsCustomCall(&callee)) {
    EmitEarlyReturnIfErrorStatus();
  }
}

llvm::Value* IrEmitter::GetBufferForGlobalCallReturnValue(
    const HloComputation& callee) {
  const HloInstruction* root_inst = callee.root_instruction();
  if (root_inst->opcode() == HloOpcode::kOutfeed) {
    return llvm::Constant::getNullValue(b()->getPtrTy());
  }

  const BufferAllocation::Slice root_buffer =
      assignment_.GetUniqueTopLevelSlice(root_inst).value();
  return EmitBufferPointer(root_buffer, root_inst->shape());
}

void IrEmitter::BindFusionArguments(const HloInstruction* fusion,
                                    FusedIrEmitter* fused_emitter) {
  for (int i = 0; i < fusion->operand_count(); i++) {
    const HloInstruction* operand = fusion->operand(i);
    fused_emitter->BindGenerator(
        *fusion->fused_parameter(i),
        [this, operand](llvm_ir::IrArray::Index index) {
          return GetIrArrayFor(operand).EmitReadArrayElement(index, b());
        });
  }
}

CpuElementalIrEmitter IrEmitter::ElementalIrEmmiterFactory() {
  auto thread_local_call_fn = [this](const HloComputation& callee,
                                     absl::Span<llvm::Value* const> parameters,
                                     absl::string_view name, bool is_reducer) {
    return EmitThreadLocalCall(callee, parameters, name, is_reducer);
  };

  bool use_truncate_f32_to_bf16_conversion =
      !IsNativeConvertSupportedOnTargetCPU(
          target_machine_features_.get_target_feature_string());

  return CpuElementalIrEmitter(
      module_, b(), std::move(thread_local_call_fn),
      use_truncate_f32_to_bf16_conversion,
      hlo_module_config_.debug_options().xla_cpu_enable_fast_min_max());
}

absl::StatusOr<llvm::Function*> IrEmitter::EmitNestedComputation(
    const HloComputation& callee, absl::string_view name, bool is_reducer) {
  // Module must be scheduled to emit thread local computation.
  if (!hlo_module_.has_schedule()) {
    return absl::InternalError(
        "HLO module must be scheduled to emit thread local computation.");
  }

  if (const auto itr = emitted_functions_.find({&callee, is_reducer});
      itr != emitted_functions_.end()) {
    return itr->second;
  }

  for (HloInstruction* instr : callee.instructions()) {
    bool nested_is_reducer = instr->opcode() == HloOpcode::kReduce ||
                             instr->opcode() == HloOpcode::kReduceWindow;
    for (HloComputation* called_computation : instr->called_computations()) {
      // reassociation is transitive so we "or" the caller and the callee.
      TF_RETURN_IF_ERROR(EmitNestedComputation(*called_computation,
                                               llvm_ir::IrName(instr),
                                               is_reducer || nested_is_reducer)
                             .status());
    }
  }

  if (callee.IsFusionComputation()) {
    return nullptr;
  }

  VLOG(2) << "Emit nested computation: " << callee.name();
  return EmitComputation(
      const_cast<HloComputation*>(&callee), name, false,
      hlo_module_.schedule().sequence(&callee).instructions(),
      /*allow_reassociation=*/is_reducer,
      /*function_attributes=*/{llvm::Attribute::AlwaysInline});
}

// Implementation detail for ComputationsTransitivelyContainCustomCall, which
// recursively checks whether a computation contains a custom call.
bool RecursivelyCheckForCustomCall(
    const HloComputation& computation,
    absl::flat_hash_map<const HloComputation*, bool>& custom_call_map) {
  if (const auto itr = custom_call_map.find(&computation);
      itr != custom_call_map.end()) {
    return itr->second;
  }

  bool contains_custom_call =
      !computation.caller_instructions(HloOpcode::kCustomCall).empty();

  for (const HloInstruction* instruction : computation.instructions()) {
    contains_custom_call |= instruction->opcode() == HloOpcode::kCustomCall;
    for (const HloComputation* nested_computation :
         instruction->called_computations()) {
      contains_custom_call |=
          RecursivelyCheckForCustomCall(*nested_computation, custom_call_map);
    }
  }

  custom_call_map[&computation] = contains_custom_call;
  return contains_custom_call;
}

absl::flat_hash_map<const HloComputation*, bool>
ComputationsTransitivelyContainCustomCall(const HloInstruction* instr) {
  absl::flat_hash_map<const HloComputation*, bool> custom_call_map;

  for (const HloComputation* computation : instr->called_computations()) {
    RecursivelyCheckForCustomCall(*computation, custom_call_map);
  }

  return custom_call_map;
}

}  // namespace cpu
}  // namespace xla
