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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

void CopySelectAttrs(const llvm::Function& src, llvm::Function& dst) {
  for (uint32_t arg_idx = 0; arg_idx < src.arg_size(); arg_idx++) {
    // Get the original argument to extract attributes from if they exist.
    llvm::Argument* src_arg = src.getArg(arg_idx);
    llvm::Argument& dst_arg = *dst.getArg(arg_idx);
    dst_arg.setName(absl::StrCat("arg", arg_idx));

    if (src_arg->hasByValAttr()) {
      dst.addParamAttr(arg_idx, src_arg->getAttribute(llvm::Attribute::ByVal));
    }

    // If the alignment has been specified in the original function, use it.
    // Otherwise, use the alignment from the kernel argument.
    if (src_arg->hasAttribute(llvm::Attribute::Alignment)) {
      dst.addParamAttr(arg_idx,
                       src_arg->getAttribute(llvm::Attribute::Alignment));
    }
    if (src_arg->hasAttribute("nvvm.grid_constant")) {
      dst.addParamAttr(arg_idx, llvm::Attribute::get(dst_arg.getContext(),
                                                     "nvvm.grid_constant"));
    }
  }
}

void AnnotateAttrsIfUnset(const emitters::KernelArguments& arguments,
                          llvm::Function& dst) {
  for (auto&& [arg_idx, kernel_argument] : llvm::enumerate(arguments.args())) {
    llvm::Argument& dst_arg = *dst.getArg(arg_idx);
    dst_arg.setName(absl::StrCat("arg", arg_idx));

    if (!dst_arg.hasByValAttr()) {
      dst.addDereferenceableParamAttr(arg_idx, kernel_argument.slice().size());
    }
    // If the alignment has been specified in the original function, use it.
    // Otherwise, use the alignment from the kernel argument.
    if (!dst_arg.hasAttribute(llvm::Attribute::Alignment) &&
        kernel_argument.alignment()) {
      dst.addParamAttr(
          arg_idx,
          llvm::Attribute::get(dst_arg.getContext(), llvm::Attribute::Alignment,
                               kernel_argument.alignment()));
    }
    if (!kernel_argument.aliased()) {
      dst.addParamAttr(arg_idx, llvm::Attribute::get(dst_arg.getContext(),
                                                     llvm::Attribute::NoAlias));
    }
  }
}

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

  llvm::Triple target_triple = llvm::Triple(llvm_module->getTargetTriple());

  if (target_triple.isNVPTX()) {
    // Our launch bounds are exact, so we can specify them as
    // reqntid[xyz] rather than maxntid[xyz].
    const std::string attr =
        absl::StrCat(launch_dims.thread_counts_per_block().x, ",",
                     launch_dims.thread_counts_per_block().y, ",",
                     launch_dims.thread_counts_per_block().z);
    kernel->addFnAttr("nvvm.reqntid", attr);
    // Maybe we want to set "reqnctapercluster" here, but not sure if needed or
    // if LLVM supports that yet. Let's do that later when needed.
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    kernel->addFnAttr("amdgpu-flat-work-group-size",
                      absl::StrJoin({launch_dims.num_threads_per_block(),
                                     launch_dims.num_threads_per_block()},
                                    ","));
    kernel->addFnAttr("amdgpu-max-num-workgroups",
                      absl::StrJoin({launch_dims.block_counts().x,
                                     launch_dims.block_counts().y,
                                     launch_dims.block_counts().z},
                                    ","));
  }
  return absl::OkStatus();
}

IndexingMap KernelFusionInterface::GetDefaultThreadIdIndexingMap(
    const LaunchDimensions& launch_dims, int unroll_factor, const Shape& shape,
    mlir::MLIRContext* mlir_context) {
  WorkDimensions work_dimensions = launch_dims.AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(unroll_factor);
  return emitters::GetDefaultWorkItemIndexingMap(work_dimensions, shape,
                                                 mlir_context);
}

absl::StatusOr<llvm::Function*> BuildKernelPrototypeFromUniqueName(
    llvm::Module* llvm_module, const se::DeviceDescription& gpu_device_info,
    const std::string& impl_fn_name, const std::string& unique_kernel_name,
    const emitters::KernelArguments& arguments,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilderBase* builder) {
  // Create the kernel and add it to the module.
  llvm::LLVMContext& context = llvm_module->getContext();
  // Explicitly set global addrspace for SPIR backend.
  int addrspace = llvm::Triple(llvm_module->getTargetTriple()).isSPIR() ? 1 : 0;
  std::vector<llvm::Type*> kernel_args;
  kernel_args.reserve(arguments.args().size());
  for (const auto& arg : arguments.args()) {
    // Handle pointer arguments.
    // Either managed OR unmanaged with non-scalar shape.
    if (arg.kind() == emitters::KernelArgument::Kind::kManaged ||
        !arg.shape().dimensions().empty()) {
      kernel_args.push_back(builder->getPtrTy(addrspace));
      continue;
    }
    // Handle scalars.
    llvm::Type* ir_type =
        llvm_ir::PrimitiveTypeToIrType(arg.shape().element_type(), context);
    if (!ir_type) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported scalar type: ",
                       PrimitiveType_Name(arg.shape().element_type())));
    }
    kernel_args.push_back(ir_type);
  }
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context), std::move(kernel_args),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             unique_kernel_name, llvm_module);

  AnnotateFunctionAsGpuKernel(llvm_module, kernel, builder);
  TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
      gpu_device_info, launch_dimensions, kernel, llvm_module));

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  builder->SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));
  // Get the original function to extract attributes.
  llvm::Function* impl_func = llvm_module->getFunction(impl_fn_name);

  if (impl_func) {
    CopySelectAttrs(*impl_func, *kernel);
  }
  AnnotateAttrsIfUnset(arguments, *kernel);
  return kernel;
}

absl::StatusOr<llvm::Function*> BuildKernelPrototype(
    llvm::Module* llvm_module, const se::DeviceDescription& gpu_device_info,
    const std::string& impl_fn_name, const std::string& unique_kernel_name,
    const emitters::KernelArguments& arguments,
    const LaunchDimensions& launch_dimensions, llvm::IRBuilderBase* builder) {
  return BuildKernelPrototypeFromUniqueName(
      llvm_module, gpu_device_info, impl_fn_name, unique_kernel_name, arguments,
      launch_dimensions, builder);
}

// Triton's kernel ABI expects additional scratchpad global memory for
// TMA and profiling information.
// For now it is only used for on-device creation of TMA descriptors, which
// we do not use yet, so we are just replacing this argument with a null
// pointer.
// TODO: b/381242007 - Allocate a proper buffer if we want to use
// device-side TMA APIs.
absl::StatusOr<llvm::Function*> RemoveUnusedTritonAbiArguments(
    llvm::Module* llvm_module, IrEmitterContext& ir_emitter_context,
    const std::string& sanitized_kernel_name) {
  llvm::Function* impl_fn = llvm_module->getFunction(sanitized_kernel_name);
  TF_RET_CHECK(impl_fn);
  impl_fn->setName(ir_emitter_context.GetSanitizedUniqueName(
      sanitized_kernel_name + "_impl"));

  constexpr int arg_to_remove = 2;

  auto fn_attrs = impl_fn->getAttributes();
  llvm::SmallVector<llvm::Type*, 8> arg_types;

  for (uint32_t i = 0; i < impl_fn->arg_size(); i++) {
    if (i < impl_fn->arg_size() - arg_to_remove) {
      arg_types.push_back(impl_fn->getArg(i)->getType());
    } else {
      fn_attrs = fn_attrs.removeParamAttributes(llvm_module->getContext(), i);

      auto arg = impl_fn->getArg(i);
      arg->replaceAllUsesWith(llvm::ConstantPointerNull::get(
          llvm::cast<llvm::PointerType>(arg->getType())));
    }
  }

  llvm::FunctionType* new_type =
      llvm::FunctionType::get(impl_fn->getReturnType(), arg_types, false);

  auto inserted =
      llvm_module
          ->getOrInsertFunction(sanitized_kernel_name, new_type, fn_attrs)
          .getCallee();
  llvm::Function* new_function = static_cast<llvm::Function*>(inserted);

  new_function->copyMetadata(impl_fn, 0);
  new_function->setAttributes(impl_fn->getAttributes());

  // Set the correct calling convention for the target GPU.
  // Triton generates PTX_Kernel CC even for AMD, so we need to use
  // AnnotateFunctionAsGpuKernel to set the correct CC based on target triple.
  llvm::IRBuilder<> builder(llvm_module->getContext());
  AnnotateFunctionAsGpuKernel(llvm_module, new_function, &builder);

  new_function->splice(new_function->begin(), impl_fn);

  for (const auto& [impl_fn_arg, kernel_arg] :
       llvm::zip(impl_fn->args(), new_function->args())) {
    kernel_arg.setName(impl_fn_arg.getName());
    impl_fn_arg.replaceAllUsesWith(&kernel_arg);
  }

  impl_fn->eraseFromParent();

  return new_function;
}

}  // namespace gpu
}  // namespace xla
