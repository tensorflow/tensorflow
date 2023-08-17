/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/gpu/fusions/fusion_emitter.h"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_arguments.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_reuse_cache.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

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

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
void AnnotateKernelLaunchDimensions(const LaunchDimensions& launch_dims,
                                    const std::string& kernel_name,
                                    llvm::Module* llvm_module) {
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
}

}  // namespace

std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray>,
           std::vector<llvm_ir::IrArray>>
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
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(kNumLlvmArgs, builder->getInt8PtrTy()),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name, llvm_module);

  AnnotateFunctionAsGpuKernel(llvm_module, kernel, builder);
  AnnotateKernelLaunchDimensions(launch_dimensions, kernel_name, llvm_module);

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
    llvm_ir::IrArray ir_array(
        CastToTypedValue(kernel_argument.shape(), &llvm_arg, builder), ir_type,
        kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }

    (arg_no < num_inputs ? inputs : outputs).push_back(ir_array);
  }

  return {kernel, std::move(inputs), std::move(outputs)};
}

StatusOr<FusionEmissionResult> KernelFusionEmitterBase::Emit(
    IrEmitterContext& ir_emitter_context, ElementalIrEmitter& elemental_emitter,
    mlir::lmhlo::FusionOp fusion_op, const HloFusionInstruction& fusion,
    KernelReuseCache& kernel_cache, llvm::IRBuilder<>* builder) const {
  std::string suggested_kernel_name = GetIrNameFromLoc(fusion_op->getLoc());

  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.allocations(), fusion_op));
  auto* fused_computation = fusion.fused_instructions_computation();

  FusionEmissionResult result;
  for (int i = 0, n = num_kernels(); i < n; ++i) {
    TF_ASSIGN_OR_RETURN(auto launch_dims,
                        launch_dimensions(ir_emitter_context, i));
    std::vector<llvm_ir::IrArray> inputs, outputs;
    auto [entry, cached] = kernel_cache.Get(
        fused_computation, kernel_arguments.args(), absl::StrCat(i),
        [&]() -> KernelReuseCache::Entry {
          llvm::Function* kernel;
          std::tie(kernel, inputs, outputs) = BuildKernelPrototype(
              ir_emitter_context, suggested_kernel_name,
              kernel_arguments.args(), fusion_op.getInputBuffers().size(),
              launch_dims, builder);
          return {kernel->getName().str(), launch_dims};
        });

    if (cached) {
      VLOG(3) << "Reuse: " << suggested_kernel_name << " -> "
              << entry.kernel_name;
    }

    result.thunks.emplace_back(std::make_unique<KernelThunk>(
        fusion_op, entry.kernel_name, kernel_arguments.args(), launch_dims));
    if (!cached) {
      TF_RETURN_IF_ERROR(EmitKernel(
          ir_emitter_context, elemental_emitter, fusion_op, fusion, launch_dims,
          std::move(inputs), std::move(outputs), builder, i));
    }
  }

  return result;
}

}  // namespace gpu
}  // namespace xla
