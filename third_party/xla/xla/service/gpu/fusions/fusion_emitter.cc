/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/fusion_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

void AnnotateWithInt32Value(std::string name, int64_t value,
                            const std::string& kernel_name,
                            llvm::Module* llvm_module) {
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Function* ir_kernel = llvm_module->getFunction(kernel_name.c_str());
  llvm::LLVMContext& llvm_context = llvm_module->getContext();

  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, name),
       llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
           llvm::IntegerType::get(llvm_context, /*NumBits=*/32), value))}));
}

}  // namespace

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, const std::string& kernel_name,
    llvm::Module* llvm_module) {
  TF_RET_CHECK(device_info.block_dim_limit().x == 0 ||
               launch_dims.block_counts().x < device_info.block_dim_limit().x)
      << "Kernel '" << kernel_name << "' launch needs more blocks ("
      << launch_dims.block_counts().x << ") than allowed by hardware ("
      << device_info.block_dim_limit().x << ").";
  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.

  // Our launch bounds are exact, so we can specify them as
  // reqntid[xyz] rather than maxntid[xyz].
  AnnotateWithInt32Value("reqntidx", launch_dims.thread_counts_per_block().x,
                         kernel_name, llvm_module);
  if (launch_dims.thread_counts_per_block().y > 1) {
    AnnotateWithInt32Value("reqntidy", launch_dims.thread_counts_per_block().y,
                           kernel_name, llvm_module);
  }
  if (launch_dims.thread_counts_per_block().z > 1) {
    AnnotateWithInt32Value("reqntidz", launch_dims.thread_counts_per_block().z,
                           kernel_name, llvm_module);
  }
  // Maybe we want to set "reqnctapercluster" here, but not sure if needed or if
  // LLVM supports that yet. Let's do that later when needed.
  return absl::OkStatus();
}

IndexingMap KernelFusionInterface::GetDefaultThreadIdIndexingMap(
    const LaunchDimensions& launch_dims, int unroll_factor, const Shape& shape,
    mlir::MLIRContext* ctx) {
  std::vector<mlir::AffineExpr> output_dims(shape.rank());

  std::array<uint64_t, 3> thread_counts{
      launch_dims.thread_counts_per_block().x,
      launch_dims.thread_counts_per_block().y,
      launch_dims.thread_counts_per_block().z,
  };

  std::array<uint64_t, 3> total_sizes{
      launch_dims.thread_counts_per_block().x * launch_dims.block_counts().x,
      launch_dims.thread_counts_per_block().y * launch_dims.block_counts().y,
      launch_dims.thread_counts_per_block().z * launch_dims.block_counts().z,
  };

  // ParallelLoopEmitter makes some assumptions about launch dimensions and
  // computes the linear index using only the x and y components.
  //
  // We implement the general formula instead and rely on the simplifier to
  // fix it.
  //
  // This means that this code supports some launch grids that the parallel
  // loop emitter doesn't support. This is safe, since the latter CHECK fails
  // if its assumptions are not fulfilled.
  mlir::AffineExpr c0 = mlir::getAffineConstantExpr(0, ctx);
  mlir::AffineExpr linear_index = c0;
  uint64_t stride = 1;
  for (int i = 0; i < 3; ++i) {
    auto coord = mlir::getAffineDimExpr(kIndexingMapThreadIdxDims[i], ctx) +
                 mlir::getAffineDimExpr(kIndexingMapBlockIdxDims[i], ctx) *
                     thread_counts[i];
    auto linear_component = coord * stride;
    linear_index = linear_index + linear_component;
    stride *= total_sizes[i];
  }
  mlir::AffineExpr chunk_id = mlir::getAffineSymbolExpr(0, ctx);
  mlir::AffineExpr unroll_elem_id = mlir::getAffineSymbolExpr(1, ctx);

  linear_index = linear_index * unroll_factor +
                 chunk_id * unroll_factor * launch_dims.launch_bound() +
                 unroll_elem_id;

  // See IndexUtil::LinearIndexToMultidimensionalIndex.
  uint64_t divisor = 1;
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    output_dims[dimension] = (linear_index.floorDiv(divisor)) %
                             static_cast<uint64_t>(shape.dimensions(dimension));
    divisor *= shape.dimensions(dimension);
  }

  std::vector<DimVar> dim_vars = {
      {{0, static_cast<int64_t>(launch_dims.thread_counts_per_block().x) - 1}},
      {{0, static_cast<int64_t>(launch_dims.thread_counts_per_block().y) - 1}},
      {{0, static_cast<int64_t>(launch_dims.thread_counts_per_block().z) - 1}},
      {{0, static_cast<int64_t>(launch_dims.block_counts().x) - 1}},
      {{0, static_cast<int64_t>(launch_dims.block_counts().y) - 1}},
      {{0, static_cast<int64_t>(launch_dims.block_counts().z) - 1}},
  };
  std::vector<RangeVar> range_vars;
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  range_vars.push_back(
      {{0, CeilOfRatio(num_elements,
                       static_cast<int64_t>(launch_dims.launch_bound()) *
                           unroll_factor) -
               1}});
  range_vars.push_back({0, unroll_factor - 1});
  IndexingMap indexing_map(
      mlir::AffineMap::get(/*dimCount=*/6,
                           /*symbolCount=*/2, output_dims, ctx),
      dim_vars, range_vars, /*rt_vars=*/{});
  // Remove the unroll_elem_id symbol if unrolling divides num_elements.
  if (num_elements % unroll_factor == 0) {
    indexing_map.AddConstraint(linear_index.replace({{unroll_elem_id, c0}}),
                               Interval{0, num_elements - unroll_factor});
  } else {
    indexing_map.AddConstraint(linear_index, Interval{0, num_elements - 1});
  }
  indexing_map.Simplify(GetIndexingMapForInstruction);
  return indexing_map;
}

absl::StatusOr<std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray>,
                          std::vector<llvm_ir::IrArray>>>
BuildKernelPrototype(IrEmitterContext& ir_emitter_context,
                     const std::string& suggested_name,
                     absl::Span<const KernelArgument> arguments,
                     size_t num_inputs,
                     const LaunchDimensions& launch_dimensions,
                     llvm::IRBuilder<>* builder) {
  // If some arguments have the same buffer, we will pass them only once.
  llvm::SmallVector<int> to_llvm_arg_no(arguments.size());
  llvm::SmallVector<int> to_arg_no;
  to_arg_no.reserve(arguments.size());
  for (const auto& [arg_no, argument] : llvm::enumerate(arguments)) {
    if (argument.first_with_same_slice().has_value()) {
      to_llvm_arg_no[arg_no] =
          to_llvm_arg_no[argument.first_with_same_slice().value()];
      continue;
    }

    to_llvm_arg_no[arg_no] = to_arg_no.size();
    to_arg_no.push_back(arg_no);
  }
  const int kNumLlvmArgs = to_arg_no.size();

  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  std::string kernel_name = ir_emitter_context.name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(suggested_name));

  // Create the kernel and add it to the module.
  auto* llvm_module = ir_emitter_context.llvm_module();
  llvm::LLVMContext& context = llvm_module->getContext();
  // Explicitly set global addrspace for SPIR backend.
  int addrspace = llvm::Triple(llvm_module->getTargetTriple()).isSPIR() ? 1 : 0;
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(kNumLlvmArgs, builder->getPtrTy(addrspace)),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name, llvm_module);

  AnnotateFunctionAsGpuKernel(llvm_module, kernel, builder);
  TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
      ir_emitter_context.gpu_device_info(), launch_dimensions, kernel_name,
      llvm_module));

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  builder->SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

  for (size_t llvm_arg_no = 0; llvm_arg_no < kernel->arg_size();
       ++llvm_arg_no) {
    const KernelArgument& kernel_argument = arguments[to_arg_no[llvm_arg_no]];
    llvm::Argument& llvm_arg = *kernel->getArg(llvm_arg_no);

    llvm_arg.setName(absl::StrCat("arg", llvm_arg_no));

    kernel->addDereferenceableParamAttr(llvm_arg_no,
                                        kernel_argument.slice().size());

    kernel->addParamAttr(
        llvm_arg_no,
        llvm::Attribute::get(llvm_arg.getContext(), llvm::Attribute::Alignment,
                             kernel_argument.alignment()));

    if (!kernel_argument.aliased()) {
      kernel->addParamAttr(llvm_arg_no,
                           llvm::Attribute::get(llvm_arg.getContext(),
                                                llvm::Attribute::NoAlias));
    }
  }

  std::vector<llvm_ir::IrArray> inputs, outputs;
  for (size_t arg_no = 0; arg_no < arguments.size(); ++arg_no) {
    const KernelArgument& kernel_argument = arguments[arg_no];
    llvm::Argument& llvm_arg = *kernel->getArg(to_llvm_arg_no[arg_no]);

    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape(), llvm_module);
    llvm_ir::IrArray ir_array(&llvm_arg, ir_type, kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }

    (arg_no < num_inputs ? inputs : outputs).push_back(ir_array);
  }

  return {{kernel, std::move(inputs), std::move(outputs)}};
}

absl::StatusOr<FusionEmissionResult> KernelFusionEmitterBase::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  llvm::IRBuilder<> builder(ir_emitter_context.llvm_module()->getContext());
  std::string suggested_kernel_name = std::string(fusion.name());

  TF_ASSIGN_OR_RETURN(
      KernelArguments kernel_arguments,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));

  auto* fused_computation = fusion.fused_instructions_computation();

  TF_ASSIGN_OR_RETURN(auto result,
                      EmitInitializers(ir_emitter_context, fusion));
  auto launch_dims = launch_dimensions();
  std::vector<llvm_ir::IrArray> inputs, outputs;
  auto [status_or_entry, cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          fused_computation, kernel_arguments.args(), /*discriminator=*/"",
          [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
            llvm::Function* kernel;
            TF_ASSIGN_OR_RETURN(
                std::tie(kernel, inputs, outputs),
                BuildKernelPrototype(ir_emitter_context, suggested_kernel_name,
                                     kernel_arguments.args(),
                                     fusion.operand_count(), launch_dims,
                                     &builder));
            if (ir_emitter_context.emit_kernels()) {
              TF_RETURN_IF_ERROR(EmitKernel(ir_emitter_context, fusion,
                                            launch_dims, std::move(inputs),
                                            std::move(outputs), &builder));
            } else {
              VLOG(3) << "Skipped kernel compilation: "
                      << suggested_kernel_name;
            }
            // TODO(jreiffers): Return shmem_bytes from EmitKernel when
            // converting the Triton emitters to this infrastructure.
            return KernelReuseCache::Entry{kernel->getName().str(), launch_dims,
                                           /*cluster_dim=*/std::nullopt,
                                           /*shmem_bytes=*/0};
          });
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  if (cached) {
    VLOG(3) << "Reuse: " << suggested_kernel_name << " -> "
            << entry->kernel_name;
  }

  result.thunks.emplace_back(std::make_unique<KernelThunk>(
      &fusion, entry->kernel_name, kernel_arguments.args(), launch_dims,
      entry->cluster_dim, entry->shmem_bytes));

  return result;
}

}  // namespace gpu
}  // namespace xla
