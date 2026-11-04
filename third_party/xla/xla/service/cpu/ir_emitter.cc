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
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/map_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_instruction_fusion.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/dot_op_emitter.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/ir_emission_utils.h"
#include "xla/service/cpu/ir_function.h"
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
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

IrEmitter::IrEmitter(const HloModule& hlo_module,
                     const BufferAssignment& assignment,
                     llvm::Module* llvm_module,
                     const TargetMachineFeatures* target_machine_features,
                     absl::flat_hash_map<BufferAllocation::Slice, int64_t>
                         slice_to_buffer_table_index)
    : assignment_(assignment),
      module_(llvm_module),
      main_builder_(llvm_module->getContext()),
      current_builder_(&main_builder_),
      alias_analysis_(hlo_module, assignment, &llvm_module->getContext()),
      hlo_module_(hlo_module),
      hlo_module_config_(hlo_module.config()),
      is_top_level_computation_(false),
      target_machine_features_(*target_machine_features),
      slice_to_buffer_table_index_(std::move(slice_to_buffer_table_index)) {
  b()->setFastMathFlags(llvm_ir::GetCpuFastMathFlags(hlo_module_config_));
  absl::Status s = GatherComputationsByAllocationType(
      &hlo_module, &thread_local_computations_, &global_computations_);
  absl::c_sort(thread_local_computations_);
  absl::c_sort(global_computations_);
  CHECK_OK(s) << "Should have failed buffer assignment.";
  SetModuleMemoryRegionName(*module_, "ir_emitter");
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

  if (computation->root_instruction()->opcode() != HloOpcode::kOutfeed) {
    ABSL_ASSIGN_OR_RETURN(
        computation_root_allocation_,
        assignment_.GetUniqueTopLevelSlice(computation->root_instruction()));
  }

  bool has_thread_local_param = false;
  for (const HloInstruction* param : computation->parameter_instructions()) {
    ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice param_slice,
                     assignment_.GetUniqueTopLevelSlice(param));
    has_thread_local_param |= param_slice.allocation()->is_thread_local();
    computation_parameter_allocations_[param_slice.allocation()->index()] =
        param->parameter_number();
  }

  InitializeIrFunction(function_name);

  llvm::IRBuilderBase::FastMathFlagGuard guard(*b());
  llvm::FastMathFlags flags = b()->getFastMathFlags();
  flags.setAllowReassoc(flags.allowReassoc() || allow_reassociation);
  b()->setFastMathFlags(flags);

  ABSL_RETURN_IF_ERROR(computation->AcceptOrdered(this, instruction_order));
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
                            b());
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

absl::Status IrEmitter::EmitConstantGlobals(
    std::optional<size_t> max_size_bytes) {
  for (const BufferAllocation& allocation : assignment_.Allocations()) {
    // Large constants don't get fused with other instructions, so we don't
    // need to emit them as globals.
    if (!allocation.is_constant() ||
        (max_size_bytes &&
         llvm_ir::LiteralForConstantAllocation(assignment_, allocation)
                 .size_bytes() > *max_size_bytes)) {
      continue;
    }

    const Literal& literal =
        llvm_ir::LiteralForConstantAllocation(assignment_, allocation);
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
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(copy));
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
  ABSL_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
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
  return Unimplemented("Infeed is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleOutfeed(HloInstruction* outfeed) {
  return Unimplemented("Outfeed is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleSort(HloInstruction* hlo) {
  return Unimplemented("Sort is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(tuple));
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
  ABSL_RETURN_IF_ERROR(ElementTypesSameAndSupported(
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

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(dot));
  llvm_ir::IrArray target_array = GetIrArrayFor(dot);

  VLOG(2) << "HandleDot: ";
  VLOG(2) << "  lhs operand: "
          << llvm_ir::DumpToString(lhs_array.GetBasePointer());
  VLOG(2) << "  rhs operand: "
          << llvm_ir::DumpToString(rhs_array.GetBasePointer());
  VLOG(2) << "  target: "
          << llvm_ir::DumpToString(target_array.GetBasePointer());

  // Dot operation is complicated so we delegate to a helper class.
  ABSL_ASSIGN_OR_RETURN(
      DotOpWorkGroupDim num_workgroups,
      EmitDotOperation(*dot, target_array, lhs_array, rhs_array,
                       /*addend_array=*/nullptr,
                       /*work_group_id=*/{b()->getInt64(0), b()->getInt64(0)},
                       b(), hlo_module_config_, target_machine_features_,
                       /*allow_parallelism=*/false));
  DCHECK_EQ(num_workgroups.x, 1);
  DCHECK_EQ(num_workgroups.y, 1);

  return absl::OkStatus();
}

absl::Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  auto lhs = convolution->operand(0);
  auto rhs = convolution->operand(1);
  ABSL_RETURN_IF_ERROR(ElementTypesSameAndSupported(
      /*instruction=*/*convolution, /*operands=*/{lhs, rhs},
      /*supported_types=*/
      {PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32, F64, C64, C128}));

  // This is a completely un-optimized version of convolution just to
  // have an early version that works. E.g. the input index and
  // padding calculation is not hoisted out of the inner loop.
  //
  // See the description of convolution in the XLA documentation for the pseudo
  // code for convolution.
  return DefaultAction(convolution);
}

absl::Status IrEmitter::HandleFft(HloInstruction* fft) {
  return Unimplemented("Fft is not implemented in the legacy emitter");
}

absl::Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented("AllReduce is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleReduceScatter(HloInstruction* rs) {
  return Unimplemented(
      "ReduceScatter is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleAllToAll(HloInstruction* instruction) {
  return Unimplemented("AllToAll is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleAllGather(HloInstruction* instruction) {
  return Unimplemented("AllGather is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleCollectivePermute(HloInstruction* crs) {
  return Unimplemented(
      "CollectivePermute is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandlePartitionId(HloInstruction* hlo) {
  return Unimplemented("PartitionId is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleReplicaId(HloInstruction* hlo) {
  return Unimplemented("ReplicaId is not implemented in the legacy emitter.");
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
  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(reduce));

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
    ABSL_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
                     EmitInnerLoopForVectorizedReduction(
                         reduction_generator, array_index, vector_type,
                         init_value, arg, dimensions, element_alignment));

    llvm_ir::IrArray target_array = GetIrArrayFor(reduce);
    llvm::Value* output_address =
        target_array.EmitArrayElementAddress(array_index, b());
    EmitShardedVectorStore(output_address, accumulator, element_alignment,
                           target_array);

    if (auto exit_terminator =
            loop->GetExitBasicBlock()->getTerminatorOrNull()) {
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
    ABSL_ASSIGN_OR_RETURN(std::vector<llvm::Value*> accumulator,
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
    ABSL_ASSIGN_OR_RETURN(
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

  // The code below assumes the layouts are equal.
  if (!LayoutUtil::Equal(operand->shape().layout(), slice->shape().layout())) {
    return DefaultAction(slice);
  }

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(slice));

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
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_slice));
    return EmitMemcpy(*dynamic_slice->operand(0), *dynamic_slice);
  }
  return DefaultAction(dynamic_slice);
}

absl::Status IrEmitter::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  auto update = dynamic_update_slice->operand(1);
  if (ShapeUtil::IsScalar(dynamic_update_slice->shape())) {
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_update_slice));
    return EmitMemcpy(*update, *dynamic_update_slice);
  } else if (llvm_ir::CanUpdateDynamicSliceInPlace(dynamic_update_slice,
                                                   assignment_)) {
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(dynamic_update_slice));
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

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(pad));

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
  ABSL_RETURN_IF_ERROR(EmitTargetElementLoop(
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
  if (fusion->IsLoopFusion()) {
    VLOG(3) << "HandleFusion kLoop";
    CpuElementalIrEmitter elemental_emitter = ElementalIrEmmiterFactory();
    FusedIrEmitter fused_emitter(elemental_emitter);
    BindFusionArguments(fusion, &fused_emitter);
    if (llvm_ir::CanEmitFusedDynamicUpdateSliceInPlace(fusion, assignment_)) {
      ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
      return llvm_ir::EmitFusedDynamicUpdateSliceInPlace(
          fusion, GetIrArrayFor(fusion), &fused_emitter, b());
    }
    ABSL_ASSIGN_OR_RETURN(auto generator, fused_emitter.GetGenerator(
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
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(fusion));
    llvm_ir::IrArray target_array = GetIrArrayFor(fusion);

    llvm_ir::IrArray lhs_array(
        GetIrArrayFor(fusion->operand(dot_lhs_param_number)));
    llvm_ir::IrArray rhs_array(
        GetIrArrayFor(fusion->operand(dot_rhs_param_number)));
    llvm_ir::IrArray addend_array(
        GetIrArrayFor(fusion->operand(addend_param_number)));

    ABSL_ASSIGN_OR_RETURN(
        DotOpWorkGroupDim num_workgroups,
        EmitDotOperation(*dot, target_array, lhs_array, rhs_array,
                         &addend_array,
                         /*work_group_id=*/{b()->getInt64(0), b()->getInt64(0)},
                         b(), hlo_module_config_, target_machine_features_,
                         /*allow_parallelism=*/false));
    DCHECK_EQ(num_workgroups.x, 1);
    DCHECK_EQ(num_workgroups.y, 1);

    return absl::OkStatus();
  } else {
    return Unimplemented("Fusion kind not implemented on CPU");
  }
}

absl::Status IrEmitter::HandleCall(HloInstruction* call) {
  ABSL_RETURN_IF_ERROR(CheckGlobalCallee(*call->to_apply()));
  HloComputation* computation = call->to_apply();

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(call));

  EmitGlobalCall(*computation, computation->name());

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

absl::Status IrEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return Unimplemented("CustomCall is not implemented in the legacy emitter.");
}

absl::Status IrEmitter::HandleWhile(HloInstruction* xla_while) {
  ABSL_RETURN_IF_ERROR(CheckGlobalCallee(*xla_while->while_body()));
  // Precondition: Condition computation must return a scalar bool.
  HloComputation* condition = xla_while->while_condition();
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool; got: "
      << ShapeUtil::HumanString(condition->root_instruction()->shape());
  // Check that all while-related buffers share an allocation slice.
  ABSL_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
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
        ABSL_RETURN_IF_ERROR(check(xla_while, xla_while->operand(0), index));
        ABSL_RETURN_IF_ERROR(check(
            xla_while, xla_while->while_condition()->parameter_instruction(0),
            index));
        ABSL_RETURN_IF_ERROR(check(xla_while,
                              xla_while->while_body()->parameter_instruction(0),
                              index));
        ABSL_RETURN_IF_ERROR(check(
            xla_while, xla_while->while_body()->root_instruction(), index));
        return absl::OkStatus();
      }));

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(xla_while));

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
                                         module_, *b())
      .status();
}

absl::StatusOr<bool> EmitFastConcatenate(
    const HloInstruction* instr,
    absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array, llvm::Module* module,
    llvm::IRBuilderBase& b, llvm::Value* workgroup_id, int64_t num_workgroups) {
  // We split the dimensions into three categories: the dimension over which we
  // are concatenating (concat_dim), the dimensions that are minor to it
  // (inner_dims) and the dimensions that are major to it (outer_dims).

  if (workgroup_id != nullptr && num_workgroups <= 0) {
    return absl::UnimplementedError("Missing number of workgroups");
  }

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

  bool has_workgroup_id = workgroup_id != nullptr;
  bool has_multiple_workers = num_workgroups > 1;
  bool has_outer_dims = !outer_dims.empty();
  bool is_parallel = has_workgroup_id && has_multiple_workers && has_outer_dims;

  llvm::Value* workgroup_ind_var = nullptr;
  if (is_parallel) {
    int64_t outer_dim_size = output_shape.dimensions(outer_dims.back());
    int64_t workgroup_size = CeilOfRatio(outer_dim_size, num_workgroups);
    llvm::Value* workgroup_size_value =
        llvm::ConstantInt::get(b.getInt64Ty(), workgroup_size);
    llvm::Value* constant_1 = llvm::ConstantInt::get(b.getInt64Ty(), 1);
    llvm::Value* constant_dim_size =
        llvm::ConstantInt::get(b.getInt64Ty(), outer_dim_size);
    llvm::Value* workgroup_start_idx =
        b.CreateMul(workgroup_id, workgroup_size_value);
    llvm::Value* workgroup_end_idx = b.CreateBinaryIntrinsic(
        llvm::Intrinsic::smin,
        b.CreateMul(b.CreateAdd(workgroup_id, constant_1),
                    workgroup_size_value),
        constant_dim_size);

    auto workgroup_loop =
        loops.AddLoop("workgroup", workgroup_start_idx, workgroup_end_idx);
    workgroup_ind_var = workgroup_loop->GetIndVarValue();
  }

  std::vector<llvm::Value*> target_multi_index =
      loops.AddLoopsForShapeOnDimensions(
          output_shape,
          workgroup_ind_var
              ? absl::MakeSpan(outer_dims).first(outer_dims.size() - 1)
              : absl::MakeSpan(outer_dims),
          "concat");

  if (workgroup_ind_var) {
    target_multi_index[outer_dims.back()] = workgroup_ind_var;
  }

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
  return is_parallel;
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
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(concatenate));
    llvm_ir::IrArray target_array = GetIrArrayFor(concatenate);
    std::vector<llvm_ir::IrArray> source_arrays;
    source_arrays.reserve(concatenate->operands().size());
    for (HloInstruction* operand : concatenate->operands()) {
      source_arrays.emplace_back(GetIrArrayFor(operand));
    }
    ABSL_RETURN_IF_ERROR(::xla::cpu::EmitFastConcatenate(concatenate, source_arrays,
                                                    target_array, module_, *b())
                        .status());
    VLOG(1) << "Emitted fast concatenate for " << concatenate->ToString();
    return absl::OkStatus();
  }
  VLOG(1) << "Could not emit fast concatenate for " << concatenate->ToString()
          << ": " << fast_impl_reason.message();

  return DefaultAction(concatenate);
}

absl::Status IrEmitter::HandleConditional(HloInstruction* conditional) {
  ABSL_RETURN_IF_ERROR(CheckGlobalCallee(*conditional->branch_computation(0)));
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

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(conditional));

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
  if (!case_block->hasTerminator()) {
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
  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(after_all));
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

  ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(rng_state));
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
  VLOG(2) << "  value: " << llvm_ir::DumpToString(GetEmittedValueFor(root));
  return absl::OkStatus();
}

absl::Status IrEmitter::Preprocess(HloInstruction* hlo) {
  VLOG(3) << "Visiting: " << hlo->ToString();
  return absl::OkStatus();
}

absl::Status IrEmitter::Postprocess(HloInstruction* hlo) {
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
  absl::c_transform(
      hlo->operands(), std::back_inserter(arrays),
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

llvm::Value* IrEmitter::GetBufferTableArgument() {
  return compute_function()->buffer_table_arg();
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

    std::pair<llvm::Function*, BufferAllocation::Slice> key = {
        compute_function()->function(), slice};
    auto buf_it = thread_local_buffers_.find(key);
    if (buf_it == thread_local_buffers_.end()) {
      // A thread-local allocation may hold several aliased values (in-place
      // ops such as dynamic-update-slice share one buffer between operand and
      // result), so size the alloca by the allocation, not by one value.
      llvm::Value* buffer = llvm_ir::EmitAllocaAtFunctionEntry(
          llvm::ArrayType::get(b()->getInt8Ty(), allocation.size()),
          absl::StrCat("thread_local", slice.ToString()), b(),
          MinimumAlignmentForShape(target_shape));
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
  ABSL_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
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
    ABSL_RETURN_IF_ERROR(EmitTargetAddressForOp(target_op));
    target_array = GetIrArrayFor(target_op);
  }

  const Shape& target_shape = target_op->shape();

  if (target_shape.IsTuple() &&
      (target_op->opcode() == HloOpcode::kFusion ||
       target_op->opcode() == HloOpcode::kReduce ||
       target_op->opcode() == HloOpcode::kReduceWindow)) {
    // For multiple outputs fusion, we need to emit each operand and the root.
    std::vector<llvm_ir::IrArray> output_arrays;
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(target_shape); ++i) {
      ABSL_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                       assignment_.GetUniqueSlice(target_op, {i}));
      const Shape& element_shape = ShapeUtil::GetSubshape(target_shape, {i});
      llvm::Value* op_target_address = EmitBufferPointer(slice, element_shape);
      llvm::Type* op_target_type = IrShapeType(element_shape);
      output_arrays.push_back(
          llvm_ir::IrArray(op_target_address, op_target_type, element_shape));
    }
    ABSL_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, output_arrays, b())
                        .EmitLoop(IrName(target_op, desc)));

    std::vector<llvm::Value*> tuple_operand_ptrs;
    tuple_operand_ptrs.reserve(output_arrays.size());
    for (int64_t i = 0; i < output_arrays.size(); ++i) {
      tuple_operand_ptrs.push_back(output_arrays[i].GetBasePointer());
    }
    llvm_ir::EmitTuple(target_array, tuple_operand_ptrs, b());

  } else {
    ABSL_RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, target_array, b())
                        .EmitLoop(IrName(target_op, desc)));
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
    absl::string_view name, bool is_reducer) {
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
      GetArrayFunctionCallArguments(parameter_addrs, b(), name,
                                    /*return_value_buffer=*/return_value_buffer,
                                    /*exec_run_options_arg=*/null_ptr,
                                    /*buffer_table_arg=*/null_ptr,
                                    /*status_arg=*/null_ptr,
                                    /*profile_counters_arg=*/null_ptr));

  std::vector<llvm::Value*> returned_scalars;
  returned_scalars.reserve(allocas_for_returned_scalars.size());
  for (llvm::Value* addr : allocas_for_returned_scalars) {
    returned_scalars.push_back(
        Load(llvm::cast<llvm::AllocaInst>(addr)->getAllocatedType(), addr));
  }
  return returned_scalars;
}

absl::Status IrEmitter::CheckGlobalCallee(const HloComputation& callee) {
  if (absl::c_binary_search(global_computations_, &callee)) {
    return absl::OkStatus();
  }
  // Embedded computations (e.g. sort comparators) are emitted as thread-local
  // functions, which cannot call the global function of `callee`.
  return Unimplemented(
      "Calling computation %s from an embedded computation is not supported "
      "on CPU.",
      callee.name());
}

void IrEmitter::EmitGlobalCall(const HloComputation& callee,
                               absl::string_view name) {
  CHECK(absl::c_binary_search(global_computations_, &callee));

  llvm::Value* null_ptr = llvm::Constant::getNullValue(b()->getPtrTy());
  Call(FindOrDie(emitted_functions_,
                 ComputationToEmit{&callee, allow_reassociation_}),
       GetArrayFunctionCallArguments(
           /*parameter_addresses=*/{}, b(), name,
           /*return_value_buffer=*/null_ptr,
           /*exec_run_options_arg=*/null_ptr,
           /*buffer_table_arg=*/GetBufferTableArgument(),
           /*status_arg=*/null_ptr,
           /*profile_counters_arg=*/null_ptr));
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
      ABSL_RETURN_IF_ERROR(EmitNestedComputation(*called_computation,
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

}  // namespace cpu
}  // namespace xla
