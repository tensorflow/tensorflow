/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/elemental_ir_emitter.h"
#include "xla/service/cpu/parallel_loop_emitter.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "xla/shape_partition.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

namespace {

struct ParallelConfig {
  std::vector<int64_t> outer_dimension_partitions;
};

// Parallel partition bounds for parallelized outer dimensions:
//   vector<[i64 lower_bound, i64 upper_bound]>
using ParallelPartitionBounds =
    std::vector<std::pair<llvm::Value*, llvm::Value*>>;

std::optional<ParallelConfig> GetParallelConfig(const HloInstruction* instr) {
  // Check if the instruction is marked for parallel execution.
  auto backend_config = instr->backend_config<BackendConfig>();
  if (!backend_config.ok() ||
      backend_config->outer_dimension_partitions().empty()) {
    return std::nullopt;
  }

  ParallelConfig config;
  config.outer_dimension_partitions.assign(
      backend_config->outer_dimension_partitions().begin(),
      backend_config->outer_dimension_partitions().end());

  return config;
}

ParallelPartitionBounds EmitParallelPartitionBounds(
    llvm::IRBuilderBase& b,
    const KernelApiIrBuilder::KernelPrototype& kernel_prototype,
    const ParallelConfig& parallel_config, const Shape& shape,
    absl::string_view name) {
  ShapePartitionIterator it(shape, parallel_config.outer_dimension_partitions);

  size_t num_parallel_dimensions =
      parallel_config.outer_dimension_partitions.size();

  // Create a constant array of all partition bounds. We will be indexing into
  // this array using workgroup id passed in a call frame.
  //
  // Type: [#partitions x [#outer_dimensions x [lower_bound, upper_bound]]]
  //
  llvm::ArrayType* dim_bounds_ty = llvm::ArrayType::get(b.getInt64Ty(), 2);
  llvm::ArrayType* partition_bounds_ty =
      llvm::ArrayType::get(dim_bounds_ty, num_parallel_dimensions);
  llvm::ArrayType* parallel_bounds_ty =
      llvm::ArrayType::get(partition_bounds_ty, it.GetTotalPartitionCount());

  // Build a nested array of partition bounds from shape partition iterator.
  std::vector<llvm::Constant*> partition_bounds;
  for (int64_t i = 0; i < it.GetTotalPartitionCount(); ++i) {
    std::vector<llvm::Constant*> dim_counts;
    for (auto [lower, size] : it.GetPartition(i)) {
      dim_counts.push_back(llvm::ConstantArray::get(
          dim_bounds_ty, {b.getInt64(lower), b.getInt64(lower + size)}));
    }
    partition_bounds.push_back(
        llvm::ConstantArray::get(partition_bounds_ty, dim_counts));
  }

  llvm::Constant* parallel_bounds =
      llvm::ConstantArray::get(parallel_bounds_ty, partition_bounds);

  llvm::Module* module = b.GetInsertBlock()->getParent()->getParent();
  llvm::GlobalVariable* parallel_bounds_global = new llvm::GlobalVariable(
      /*M=*/*module,
      /*Ty=*/parallel_bounds_ty,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/parallel_bounds,
      /*Name=*/absl::StrCat(name, "_parallel_bounds"));

  // Construct IR to load bounds for all parallel dimensions.
  ParallelPartitionBounds bounds;
  for (size_t i = 0; i < num_parallel_dimensions; ++i) {
    llvm::Value* partition = kernel_prototype.workgroup_id.x;
    llvm::Value* parallel_dim = b.getInt32(i);

    llvm::Value* lower_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(0)},
        absl::StrCat("lo_dim_", i, "_gep"));

    llvm::Value* upper_gep = b.CreateInBoundsGEP(
        parallel_bounds_ty, parallel_bounds_global,
        {b.getInt32(0), partition, parallel_dim, b.getInt32(1)},
        absl::StrCat("up_dim_", i, "_gep"));

    bounds.emplace_back(
        b.CreateLoad(b.getInt64Ty(), lower_gep, absl::StrCat("lo_dim_", i)),
        b.CreateLoad(b.getInt64Ty(), upper_gep, absl::StrCat("up_dim_", i)));
  }

  return bounds;
}

}  // namespace

ElementalKernelEmitter::ElementalKernelEmitter(
    const HloInstruction* instr, const BufferAssignment* buffer_assignment,
    const TargetMachineFeatures* target_machine)
    : instr_(instr),
      buffer_assignment_(buffer_assignment),
      target_machine_(target_machine) {}

absl::StatusOr<ElementalKernelEmitter::KernelDefinition>
ElementalKernelEmitter::EmitKernelDefinition() {
  VLOG(2) << "Emit elemental host kernel: " << instr_->name();

  auto ctx = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  KernelApiIrBuilder kernel_api_ir_builder(
      *ctx,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> llvm_module = KernelApiIrBuilder::CreateModule(
      absl::StrCat(instr_->name(), "_elemental_kernel_module"), *ctx);

  ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(
          *llvm_module, instr_, buffer_assignment_, name(), "_kernel"));

  llvm::IRBuilder<> ir_builder(*ctx);
  ir_builder.SetInsertPoint(
      kernel_prototype.function->getEntryBlock().getTerminator());

  // Set fast-math flags on the builder.
  ir_builder.setFastMathFlags(
      llvm_ir::GetCpuFastMathFlags(hlo_module->config()));

  ASSIGN_OR_RETURN(
      CpuElementalIrEmitter::ThreadLocalCallCallback thread_local_call_fn,
      ThreadLocalCallbackFactory(ir_builder, *llvm_module));

  CpuElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    operand_to_generator[operand] = [&, i](const llvm_ir::IrArray::Index& idx) {
      return kernel_prototype.arguments[i].EmitReadArrayElement(idx,
                                                                &ir_builder);
    };
  }

  bool enable_fast_min_max =
      hlo_module
          ? hlo_module->config().debug_options().xla_cpu_enable_fast_min_max()
          : true;
  CpuElementalIrEmitter elemental_ir_emitter(llvm_module.get(), &ir_builder,
                                             std::move(thread_local_call_fn),
                                             true, enable_fast_min_max);

  llvm_ir::ElementGenerator element_generator =
      elemental_ir_emitter.MakeElementGenerator(instr_, operand_to_generator);

  ASSIGN_OR_RETURN(NumWorkGroups num_workgroups,
                   EmitElementalLoops(ir_builder, instr_, kernel_prototype,
                                      element_generator));

  LlvmKernelSource source(std::move(ctx), std::move(llvm_module));

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

absl::StatusOr<NumWorkGroups> ElementalKernelEmitter::EmitElementalLoops(
    llvm::IRBuilderBase& b, const HloInstruction* instr,
    const KernelApiIrBuilder::KernelPrototype& kernel_prototype,
    const llvm_ir::ElementGenerator& element_generator) {
  // We can emit loops for instruction with multiple results only if it is a
  // fusion, reduce or reduce window.
  bool multiple_results = kernel_prototype.results.size() > 1;
  bool support_multiple_results = instr->opcode() == HloOpcode::kFusion ||
                                  instr->opcode() == HloOpcode::kReduce ||
                                  instr->opcode() == HloOpcode::kReduceWindow;

  auto parallel_config = GetParallelConfig(instr);
  bool has_parallel_config = parallel_config.has_value();

  if (multiple_results && !support_multiple_results) {
    return Internal(
        "Multi-output host kernels are not supported for %s instruction",
        HloOpcodeString(instr->opcode()));
  }

  // TODO(ezhulenev): Support multiple results for parallel loops.
  if (multiple_results) {
    RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, kernel_prototype.results, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return NumWorkGroups();
  }

  const llvm_ir::IrArray& result = kernel_prototype.results.front();

  // Emit a loop for a single parallel partition with dynamic bounds computed
  // from workgroup id.
  if (has_parallel_config) {
    ParallelPartitionBounds parallel_bounds = EmitParallelPartitionBounds(
        b, kernel_prototype, *parallel_config, instr->shape(), instr->name());
    RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, result, &parallel_bounds, &b)
            .EmitLoop(llvm_ir::IrName(instr)));
    return NumWorkGroups{
        static_cast<uint64_t>(ShapePartitionAssigner::GetTotalPartitionCount(
            parallel_config->outer_dimension_partitions))};
  }

  // Emit a whole loop for the instruction.
  RETURN_IF_ERROR(llvm_ir::LoopEmitter(element_generator, result, &b)
                      .EmitLoop(llvm_ir::IrName(instr)));
  return NumWorkGroups();
}

absl::StatusOr<CpuElementalIrEmitter::ThreadLocalCallCallback>
ElementalKernelEmitter::ThreadLocalCallbackFactory(llvm::IRBuilderBase& builder,
                                                   llvm::Module& module) const {
  return [&builder, &module](
             const HloComputation& callee,
             absl::Span<llvm::Value* const> parameters, absl::string_view name,
             bool is_reducer) -> absl::StatusOr<std::vector<llvm::Value*>> {
    CpuElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
    CpuElementalIrEmitter sub_emitter(&module, &builder, nullptr,
                                      /*allow_fast_math=*/false,
                                      /*hlo_fast_math=*/false);
    for (const HloInstruction* inst : callee.MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kParameter) {
        int64_t param_no = inst->parameter_number();
        TF_RET_CHECK(param_no < parameters.size())
            << "Parameter index " << param_no << " out of bounds ("
            << parameters.size() << ")";
        llvm::Value* param_val = parameters[param_no];
        llvm::Type* elem_type = llvm_ir::PrimitiveTypeToIrType(
            inst->shape().element_type(), module.getContext());
        if (param_val->getType()->isPointerTy()) {
          param_val = builder.CreateLoad(elem_type, param_val);
        }
        if (param_val->getType() != elem_type) {
          if (param_val->getType()->isIntegerTy() && elem_type->isIntegerTy()) {
            bool is_signed = primitive_util::IsSignedIntegralType(
                inst->shape().element_type());
            param_val = builder.CreateIntCast(param_val, elem_type, is_signed);
          } else if (param_val->getType()->isFloatingPointTy() &&
                     elem_type->isFloatingPointTy()) {
            param_val = builder.CreateFPCast(param_val, elem_type);
          } else if (param_val->getType()->isIntegerTy() &&
                     elem_type->isFloatingPointTy()) {
            param_val = builder.CreateSIToFP(param_val, elem_type);
          } else if (param_val->getType()->isFloatingPointTy() &&
                     elem_type->isIntegerTy()) {
            param_val = builder.CreateFPToSI(param_val, elem_type);
          }
        }
        operand_to_generator[inst] =
            [param_val](const llvm_ir::IrArray::Index& idx) {
              return param_val;
            };
      } else if (inst->opcode() == HloOpcode::kConstant) {
        llvm::Type* elem_type = llvm_ir::PrimitiveTypeToIrType(
            inst->shape().element_type(), module.getContext());
        llvm::Constant* const_val = nullptr;
        if (auto val_double = inst->literal().GetAsDouble({})) {
          if (elem_type->isFloatingPointTy()) {
            const_val = llvm::ConstantFP::get(elem_type, *val_double);
          } else if (elem_type->isIntegerTy()) {
            const_val = llvm::ConstantInt::get(
                elem_type, static_cast<int64_t>(*val_double));
          }
        } else if (auto val_s64 = inst->literal().GetIntegralAsS64({})) {
          if (elem_type->isIntegerTy()) {
            const_val = llvm::ConstantInt::get(elem_type, *val_s64);
          } else if (elem_type->isFloatingPointTy()) {
            const_val =
                llvm::ConstantFP::get(elem_type, static_cast<double>(*val_s64));
          }
        }

        if (const_val != nullptr) {
          operand_to_generator[inst] =
              [const_val](const llvm_ir::IrArray::Index& idx) {
                return const_val;
              };
        } else {
          llvm::Constant* initializer =
              llvm_ir::ConvertLiteralToIrConstant(inst->literal(), &module);
          llvm::GlobalVariable* global = new llvm::GlobalVariable(
              module, initializer->getType(), /*isConstant=*/true,
              llvm::GlobalValue::PrivateLinkage, initializer, "");
          global->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::Global);
          llvm_ir::IrArray array(global, elem_type, inst->shape());
          operand_to_generator[inst] = [&builder, array](
                                           const llvm_ir::IrArray::Index& idx) {
            llvm_ir::IrArray::Index array_idx = idx;
            if (array_idx.size() != array.GetShape().dimensions_size()) {
              array_idx =
                  array_idx.SourceIndexOfBitcast(array.GetShape(), &builder);
            }
            return array.EmitReadArrayElement(array_idx, &builder);
          };
        }
      } else {
        operand_to_generator[inst] =
            sub_emitter.MakeElementGenerator(inst, operand_to_generator);
      }
    }
    llvm_ir::IrArray::Index dummy_index(builder.getInt32Ty());
    ASSIGN_OR_RETURN(
        llvm::Value * result_val,
        operand_to_generator.at(callee.root_instruction())(dummy_index));
    llvm::Type* root_type = llvm_ir::PrimitiveTypeToIrType(
        callee.root_instruction()->shape().element_type(), module.getContext());
    if (result_val->getType() != root_type) {
      if (result_val->getType()->isFloatingPointTy() &&
          root_type->isFloatingPointTy()) {
        result_val = builder.CreateFPCast(result_val, root_type);
      } else if (result_val->getType()->isIntegerTy() &&
                 root_type->isIntegerTy()) {
        bool is_signed = primitive_util::IsSignedIntegralType(
            callee.root_instruction()->shape().element_type());
        result_val = builder.CreateIntCast(result_val, root_type, is_signed);
      } else if (result_val->getType()->isIntegerTy() &&
                 root_type->isFloatingPointTy()) {
        result_val = builder.CreateSIToFP(result_val, root_type);
      } else if (result_val->getType()->isFloatingPointTy() &&
                 root_type->isIntegerTy()) {
        result_val = builder.CreateFPToSI(result_val, root_type);
      }
    }
    return std::vector<llvm::Value*>{result_val};
  };
}

}  // namespace xla::cpu
