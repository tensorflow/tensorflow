/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/elemental/concatenate_kernel_emitter.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Alignment.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

int MinimumAlignmentForPrimitiveType(PrimitiveType primitive_type) {
  int64_t byte_size = ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  return std::min(int64_t{8}, byte_size);
}

void EmitTransferElements(llvm::Value* target, llvm::Value* source,
                          int64_t element_count, PrimitiveType primitive_type,
                          const llvm_ir::IrArray& target_array,
                          const llvm_ir::IrArray& source_array,
                          llvm::Module* module, llvm::IRBuilderBase& b) {
  unsigned primitive_type_size =
      ShapeUtil::ByteSizeOfPrimitiveType(primitive_type);
  llvm::Align element_alignment(tsl::MathUtil::GCD<unsigned>(
      primitive_type_size, MinimumAlignmentForPrimitiveType(primitive_type)));
  llvm::Type* primitive_llvm_type =
      llvm_ir::PrimitiveTypeToIrType(primitive_type, module->getContext());

  if (element_count == 1) {
    llvm::LoadInst* load_instruction =
        b.CreateAlignedLoad(primitive_llvm_type, source, element_alignment);
    source_array.AnnotateLoadStoreInstructionWithMetadata(load_instruction);
    llvm::StoreInst* store_instruction =
        b.CreateAlignedStore(load_instruction, target, element_alignment);
    target_array.AnnotateLoadStoreInstructionWithMetadata(store_instruction);
  } else {
    auto* memcpy_instruction = b.CreateMemCpy(
        target, /*DstAlign=*/llvm::Align(element_alignment), source,
        /*SrcAlign=*/llvm::Align(element_alignment),
        element_count * primitive_type_size);

    // The memcpy does the load and the store internally. The aliasing related
    // metadata has to reflect that.
    std::map<int, llvm::MDNode*> merged_metadata =
        llvm_ir::MergeMetadata(&module->getContext(), source_array.metadata(),
                               target_array.metadata());
    for (const auto& kind_md_pair : merged_metadata) {
      memcpy_instruction->setMetadata(kind_md_pair.first, kind_md_pair.second);
    }
  }
}

absl::StatusOr<bool> EmitFastConcatenate(
    const HloInstruction* instr,
    absl::Span<const llvm_ir::IrArray> source_arrays,
    const llvm_ir::IrArray& target_array, llvm::Module* module,
    llvm::IRBuilderBase& b, llvm::Value* workgroup_id, int64_t num_workgroups) {
  // We split the dimensions into three categories: the dimension over which we
  // are concatenating (concat_dim), the dimensions that are minor to it
  // (inner_dims), and the dimensions that are major to it (outer_dims).
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

  llvm_ir::ForLoopNest loops(llvm_ir::IrName(concatenate), &b);

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
    llvm_ir::SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), &b);
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

    EmitTransferElements(
        copy_target_address, copy_source_address,
        inner_dims_product * input_shape.dimensions(concat_dim), primitive_type,
        target_array, source_array, module, b);

    byte_offset_into_target_region += inner_dims_product *
                                      input_shape.dimensions(concat_dim) *
                                      primitive_type_size;
  }

  if (!outer_dims.empty()) {
    llvm_ir::SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), &b);
  }
  return is_parallel;
}

}  // namespace

absl::Status CanDoFastConcatenate(const HloInstruction& instruction) {
  const Shape& output_shape = instruction.shape();
  for (auto* op : instruction.operands()) {
    if (!LayoutUtil::Equal(op->shape().layout(), output_shape.layout())) {
      return absl::Status(absl::StatusCode::kFailedPrecondition,
                          "Operand has mismatching layouts");
    }
  }
  return absl::OkStatus();
}

ConcatenateKernelEmitter::ConcatenateKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<ConcatenateKernelEmitter::KernelDefinition>
ConcatenateKernelEmitter::EmitKernelDefinition() {
  if (absl::Status status = CanDoFastConcatenate(*instr_); !status.ok()) {
    return Internal(
        "Concatenate is not supported by ConcatenateKernelEmitter: %s",
        status.message());
  }

  auto ctx = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  const auto backend_config =
      instr_->backend_config<BackendConfig>().value_or(BackendConfig());
  const auto& partitions = backend_config.outer_dimension_partitions();
  auto total_workgroups =
      absl::c_accumulate(partitions, 1, std::multiplies<int64_t>());

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_elemental_kernel_module"), *ctx,
      target_machine_);

  ABSL_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(
          *llvm_module, instr_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> ir_builder(*ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  llvm_ir::IrArray output_array = kernel_prototype.results[0];
  ABSL_ASSIGN_OR_RETURN(
      bool is_parallel,
      EmitFastConcatenate(instr_, kernel_prototype.arguments, output_array,
                          llvm_module.get(), ir_builder,
                          kernel_prototype.workgroup_id.x, total_workgroups));

  LlvmKernelSource source(std::move(ctx), std::move(llvm_module));
  NumWorkGroups num_workgroups;
  if (is_parallel) {
    num_workgroups.x = total_workgroups;
  }

  KernelSpec::Buffers arguments, results;
  for (const auto& buffer : kernel_prototype.argument_buffers) {
    arguments.push_back({buffer.slice, buffer.shape});
  }
  for (const auto& buffer : kernel_prototype.result_buffers) {
    results.push_back({buffer.slice, buffer.shape});
  }

  KernelSpec spec(kernel_prototype.function->getName(), num_workgroups,
                  std::move(arguments), std::move(results),
                  std::move(kernel_prototype.invariant_arguments));

  return KernelDefinition(std::move(spec), std::move(source));
}

}  // namespace xla::cpu
