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
#include "xla/backends/gpu/codegen/fusion_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, llvm::Function* kernel,
    llvm::Module* llvm_module) {
  TF_RET_CHECK(
      (device_info.block_dim_limit().x == 0 ||
       launch_dims.block_counts().x < device_info.block_dim_limit().x) &&
      (device_info.block_dim_limit().y == 0 ||
       launch_dims.block_counts().y < device_info.block_dim_limit().y))
      << "Kernel '" << kernel->getName().str() << "' launch needs more blocks ("
      << launch_dims.block_counts().x << ", " << launch_dims.block_counts().y
      << ") than allowed by hardware (" << device_info.block_dim_limit().x
      << ", " << device_info.block_dim_limit().y << ").";
  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.

  // Our launch bounds are exact, so we can specify them as
  // reqntid[xyz] rather than maxntid[xyz].
  const std::string attr =
      absl::StrCat(launch_dims.thread_counts_per_block().x, ",",
                   launch_dims.thread_counts_per_block().y, ",",
                   launch_dims.thread_counts_per_block().z);
  kernel->addFnAttr("nvvm.reqntid", attr);
  // Maybe we want to set "reqnctapercluster" here, but not sure if needed or if
  // LLVM supports that yet. Let's do that later when needed.
  return absl::OkStatus();
}

IndexingMap KernelFusionInterface::GetDefaultThreadIdIndexingMap(
    const LaunchDimensions& launch_dims, int unroll_factor, const Shape& shape,
    mlir::MLIRContext* ctx) {
  std::vector<mlir::AffineExpr> output_dims(shape.dimensions().size());

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

  std::vector<IndexingMap::Variable> dim_vars = DimVarsFromGPUGrid(
      {static_cast<int64_t>(launch_dims.thread_counts_per_block().x),
       static_cast<int64_t>(launch_dims.thread_counts_per_block().y),
       static_cast<int64_t>(launch_dims.thread_counts_per_block().z),
       static_cast<int64_t>(launch_dims.block_counts().x),
       static_cast<int64_t>(launch_dims.block_counts().y),
       static_cast<int64_t>(launch_dims.block_counts().z)});
  std::vector<IndexingMap::Variable> range_vars;
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  range_vars.push_back(IndexingMap::Variable{
      {0, CeilOfRatio(num_elements,
                      static_cast<int64_t>(launch_dims.launch_bound()) *
                          unroll_factor) -
              1}});
  range_vars.push_back({0, unroll_factor - 1});
  IndexingMap indexing_map(
      mlir::AffineMap::get(/*dimCount=*/6,
                           /*symbolCount=*/2, output_dims, ctx),
      dim_vars, range_vars, /*rt_vars=*/{});
  indexing_map.AddConstraint(linear_index, Interval{0, num_elements - 1});
  indexing_map.Simplify();
  return indexing_map;
}

std::string GetSanitizedUniqueName(IrEmitterContext& ir_emitter_context,
                                   const std::string& suggested_name) {
  return ir_emitter_context.name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(suggested_name));
}

absl::StatusOr<std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray>,
                          std::vector<llvm_ir::IrArray>>>
BuildKernelPrototype(IrEmitterContext& ir_emitter_context,
                     const std::string& impl_fn_name,
                     const std::string& suggested_name,
                     absl::Span<const emitters::KernelArgument> arguments,
                     size_t num_inputs,
                     const LaunchDimensions& launch_dimensions,
                     llvm::IRBuilderBase* builder) {
  return BuildKernelPrototypeFromUniqueName(
      ir_emitter_context, impl_fn_name,
      GetSanitizedUniqueName(ir_emitter_context, suggested_name), arguments,
      num_inputs, launch_dimensions, builder);
}

absl::StatusOr<std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray>,
                          std::vector<llvm_ir::IrArray>>>
BuildKernelPrototypeFromUniqueName(
    IrEmitterContext& ir_emitter_context, const std::string& impl_fn_name,
    const std::string& unique_kernel_name,
    absl::Span<const emitters::KernelArgument> arguments, size_t num_inputs,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilderBase* builder) {
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
                             unique_kernel_name, llvm_module);

  AnnotateFunctionAsGpuKernel(llvm_module, kernel, builder);
  TF_RETURN_IF_ERROR(
      AnnotateKernelLaunchDimensions(ir_emitter_context.gpu_device_info(),
                                     launch_dimensions, kernel, llvm_module));

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  builder->SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));
  // Get the original function to extract attributes.
  auto impl_func = llvm_module->getFunction(impl_fn_name);

  for (size_t llvm_arg_no = 0; llvm_arg_no < kernel->arg_size();
       ++llvm_arg_no) {
    const emitters::KernelArgument& kernel_argument =
        arguments[to_arg_no[llvm_arg_no]];
    // Get the original argument to extract attributes from if they exist.
    llvm::Argument* impl_arg =
        impl_func ? impl_func->getArg(llvm_arg_no) : nullptr;
    llvm::Argument& new_arg = *kernel->getArg(llvm_arg_no);
    new_arg.setName(absl::StrCat("arg", llvm_arg_no));

    if (impl_arg && impl_arg->hasByValAttr()) {
      kernel->addParamAttr(llvm_arg_no,
                           impl_arg->getAttribute(llvm::Attribute::ByVal));
    } else {
      kernel->addDereferenceableParamAttr(llvm_arg_no,
                                          kernel_argument.slice().size());
    }
    // If the alignment has been specified in the original function, use it.
    // Otherwise, use the alignment from the kernel argument.
    if (impl_arg && impl_arg->hasAttribute(llvm::Attribute::Alignment)) {
      kernel->addParamAttr(llvm_arg_no,
                           impl_arg->getAttribute(llvm::Attribute::Alignment));
    } else {
      kernel->addParamAttr(
          llvm_arg_no,
          llvm::Attribute::get(new_arg.getContext(), llvm::Attribute::Alignment,
                               kernel_argument.alignment()));
    }
    if (!kernel_argument.aliased()) {
      kernel->addParamAttr(
          llvm_arg_no,
          llvm::Attribute::get(new_arg.getContext(), llvm::Attribute::NoAlias));
    }
  }

  std::vector<llvm_ir::IrArray> inputs, outputs;
  for (size_t arg_no = 0; arg_no < arguments.size(); ++arg_no) {
    const emitters::KernelArgument& kernel_argument = arguments[arg_no];
    llvm::Argument& llvm_arg = *kernel->getArg(to_llvm_arg_no[arg_no]);

    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape(), context);
    llvm_ir::IrArray ir_array(&llvm_arg, ir_type, kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }

    (arg_no < num_inputs ? inputs : outputs).push_back(ir_array);
  }

  return {{kernel, std::move(inputs), std::move(outputs)}};
}

}  // namespace gpu
}  // namespace xla
