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

#include "absl/container/flat_hash_set.h"
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
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla {
namespace gpu {
namespace {

absl::flat_hash_set<int64_t> NonInvariantSliceIndices(
    const emitters::KernelArguments& arguments) {
  absl::flat_hash_set<int64_t> non_invariant;
  for (const emitters::KernelArgument& arg : arguments.args()) {
    if (arg.kind() == emitters::KernelArgument::Kind::kManaged &&
        !arg.invariant()) {
      non_invariant.insert(arg.slice_index());
    }
  }
  return non_invariant;
}

}  // namespace

void AnnotateAttrsIfUnset(const emitters::KernelArguments& arguments,
                          llvm::Function& dst) {
  const absl::flat_hash_set<int64_t> non_invariant_slices =
      NonInvariantSliceIndices(arguments);
  for (auto&& [arg_idx, kernel_argument] : llvm::enumerate(arguments.args())) {
    llvm::Argument& dst_arg = *dst.getArg(arg_idx);
    dst_arg.setName(absl::StrCat("arg", arg_idx));

    if (!dst_arg.getType()->isPointerTy()) {
      continue;
    }

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
    // NVPTX derives ld.global.nc from readonly plus noalias; omitting both on
    // slices marked not invariant prevents inferred readonly from restoring it.
    bool slice_invariant =
        kernel_argument.kind() != emitters::KernelArgument::Kind::kManaged ||
        !non_invariant_slices.contains(kernel_argument.slice_index());
    if (!kernel_argument.aliased() && slice_invariant) {
      dst.addParamAttr(arg_idx, llvm::Attribute::get(dst_arg.getContext(),
                                                     llvm::Attribute::NoAlias));
    }
    if (!kernel_argument.written() && slice_invariant &&
        !dst_arg.hasByValAttr() &&
        !dst_arg.hasAttribute(llvm::Attribute::ReadNone) &&
        !dst_arg.hasAttribute(llvm::Attribute::WriteOnly) &&
        !dst_arg.hasAttribute(llvm::Attribute::Writable)) {
      dst.addParamAttr(arg_idx,
                       llvm::Attribute::get(dst_arg.getContext(),
                                            llvm::Attribute::ReadOnly));
    }
  }
}

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
absl::Status AnnotateKernelLaunchDimensions(
    const se::DeviceDescription& device_info,
    const LaunchDimensions& launch_dims, llvm::Function* kernel,
    llvm::Module* llvm_module) {
  const se::BlockDim &limit = device_info.block_dim_limit(),
                     &block_count = launch_dims.block_counts();
  TF_RET_CHECK((limit.x == 0 || block_count.x <= limit.x) &&
               (limit.y == 0 || block_count.y <= limit.y))
      << "Kernel '" << kernel->getName().str() << "' launch needs more blocks ("
      << block_count.x << ", " << block_count.y
      << ") than allowed by hardware (" << limit.x << ", " << limit.y << ").";
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
  } else if (target_triple.getArch() == llvm::Triple::amdgpu) {
    kernel->addFnAttr("amdgpu-flat-work-group-size",
                      absl::StrJoin({launch_dims.num_threads_per_block(),
                                     launch_dims.num_threads_per_block()},
                                    ","));
    kernel->addFnAttr(
        "amdgpu-max-num-workgroups",
        absl::StrJoin({block_count.x, block_count.y, block_count.z}, ","));
  }
  return absl::OkStatus();
}

absl::StatusOr<llvm::Function*> RemoveUnusedTritonAbiArguments(
    llvm::Module* llvm_module, const std::string& sanitized_kernel_name,
    const std::string& sanitized_kernel_impl_name, bool keep_scratch) {
  llvm::Function* impl_fn = llvm_module->getFunction(sanitized_kernel_name);
  TF_RET_CHECK(impl_fn);
  impl_fn->setName(sanitized_kernel_impl_name);

  constexpr int arg_to_remove = 2;

  auto fn_attrs = impl_fn->getAttributes();
  llvm::SmallVector<llvm::Type*, 8> arg_types;

  for (uint32_t i = 0; i < impl_fn->arg_size(); i++) {
    bool is_scratch = (i == impl_fn->arg_size() - 2);
    bool should_keep = (i < impl_fn->arg_size() - arg_to_remove) ||
                       (is_scratch && keep_scratch);

    if (should_keep) {
      arg_types.push_back(impl_fn->getArg(i)->getType());
      if (is_scratch) {
        fn_attrs = fn_attrs.addParamAttribute(
            llvm_module->getContext(), i,
            llvm::Attribute::getWithAlignment(llvm_module->getContext(),
                                              llvm::Align(128)));
      }
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
