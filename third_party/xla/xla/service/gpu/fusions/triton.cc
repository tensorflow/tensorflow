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
#include "xla/service/gpu/fusions/triton.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// Derives the number of blocks and threads to use for processing a Triton
// Softmax fusion.
LaunchDimensions CalculateSoftMaxLaunchDimensions(
    const HloFusionAdaptor& fusion) {
  auto reduce = HloFindIf(fusion.GetRoots(), fusion, [](auto node) {
    return node.opcode() == HloOpcode::kReduce;
  });

  CHECK(reduce.has_value());
  const Shape& reduce_input_shape = reduce->GetOperand(0).instruction().shape();

  CHECK_EQ(reduce->instruction().dimensions().size(), 1);
  CHECK_EQ(reduce->instruction().dimensions()[0],
           reduce_input_shape.rank() - 1);

  int reduction_dim = reduce_input_shape.dimensions_minor(0);

  unsigned num_rows = 1;
  for (unsigned minor_axis = 1; minor_axis < reduce_input_shape.rank();
       ++minor_axis) {
    num_rows *= reduce_input_shape.dimensions_minor(minor_axis);
  }

  unsigned num_warps = 32;

  if (reduction_dim <= 512) {
    num_warps = 1;
  } else if (reduction_dim <= 1024) {
    num_warps = 2;
  } else if (reduction_dim <= 16384) {
    num_warps = 4;
  } else if (reduction_dim <= 32768) {
    num_warps = 8;
  } else if (reduction_dim <= 65536) {
    num_warps = 16;
  }

  return {num_rows, static_cast<uint64_t>(num_warps * WarpSize())};
}

}  // namespace

absl::StatusOr<FusionEmissionResult> TritonFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  llvm::IRBuilder builder(ir_emitter_context.llvm_module()->getContext());
  VLOG(3) << fusion.ToString();
  std::string suggested_kernel_name = std::string(fusion.name());
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));

  const HloComputation* hlo_computation =
      fusion.fused_instructions_computation();

  auto generate = [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
    VLOG(3) << "Generating: " << suggested_kernel_name;

    const std::string impl_fn_name =
        ir_emitter_context.name_uniquer()->GetUniqueName(
            llvm_ir::SanitizeFunctionName(
                absl::StrCat(suggested_kernel_name, "_impl")));

    auto backend_config = analysis_.fusion_backend_config();
    absl::string_view fusion_kind = backend_config.kind();

    TritonWrapperResult triton_wrapper_result;
    LaunchDimensions launch_dimensions;
    if (fusion_kind == kTritonFusionKind ||
        fusion_kind == kTritonSoftmaxFusionKind) {
      launch_dimensions = *this->launch_dimensions();

      // This is a hack, we use TritonGemmConfig also for the Softmax and
      // Generic emitters, but we ignore most parameters.
      TritonGemmConfig config;
      config.num_stages = 1;
      // Thread count per block is always a multiple of WarpSize.
      config.num_warps = launch_dimensions.num_threads_per_block() / WarpSize();
      config.num_ctas = 1;

      TF_ASSIGN_OR_RETURN(
          triton_wrapper_result,
          TritonWrapper(
              /*analysis=*/{}, impl_fn_name, hlo_computation,
              ir_emitter_context.gpu_compute_capability(),
              ir_emitter_context.gpu_device_info(), config,
              /*output_tile_sizes=*/{},  // TODO(b/332649307): Pass useful data.
              ir_emitter_context.llvm_module(),
              (fusion_kind == kTritonFusionKind ? &EmitGeneric : &EmitSoftMax),
              *ir_emitter_context.mlir_context()));
    } else {  // Must be a MatMul
      CHECK_EQ(fusion_kind, kTritonGemmFusionKind);
      if (!backend_config.has_triton_gemm_config()) {
        LOG(WARNING) << "Using fallback triton GEMM config for op "
                     << fusion.name();
        auto& triton_config = *backend_config.mutable_triton_gemm_config();
        triton_config.set_block_m(64);
        triton_config.set_block_k(64);
        triton_config.set_block_n(64);
        triton_config.set_split_k(1);
        triton_config.set_num_stages(1);
        triton_config.set_num_warps(2);
        triton_config.set_num_ctas(1);
      }
      TF_ASSIGN_OR_RETURN(
          TritonGemmConfig config,
          TritonGemmConfig::FromProto(backend_config.triton_gemm_config()));

      TF_ASSIGN_OR_RETURN(auto analysis, TritonFusionAnalysis::Execute(
                                             *hlo_computation, config.split_k));
      TF_ASSIGN_OR_RETURN(
          triton_wrapper_result,
          TritonWrapper(analysis, impl_fn_name, hlo_computation,
                        ir_emitter_context.gpu_compute_capability(),
                        ir_emitter_context.gpu_device_info(), config,
                        /*output_tile_sizes=*/{},
                        ir_emitter_context.llvm_module(), &EmitMatMul,
                        *ir_emitter_context.mlir_context()));
      TF_ASSIGN_OR_RETURN(
          launch_dimensions,
          GetMatMulLaunchDimensions(analysis, analysis_.fusion(), config));
    }

    llvm::Function* impl_fn =
        ir_emitter_context.llvm_module()->getFunction(impl_fn_name);
    TF_RET_CHECK(impl_fn);

    llvm::Function* kernel;
    std::vector<llvm_ir::IrArray> inputs;
    std::vector<llvm_ir::IrArray> outputs;
    TF_ASSIGN_OR_RETURN(
        std::tie(kernel, inputs, outputs),
        BuildKernelPrototype(ir_emitter_context, suggested_kernel_name,
                             kernel_arguments.args(), impl_fn->arg_size(),
                             launch_dimensions, &builder));

    // Move function body into kernel prototype.
    llvm::Function* prototype_func = builder.GetInsertBlock()->getParent();
    prototype_func->splice(prototype_func->begin(), impl_fn);
    for (const auto& [arg, ir_array] : llvm::zip(impl_fn->args(), inputs)) {
      arg.replaceAllUsesWith(ir_array.GetBasePointer());
    }
    impl_fn->eraseFromParent();

    return {{kernel->getName().str(), launch_dimensions,
             triton_wrapper_result.cluster_dim,
             triton_wrapper_result.shmem_bytes}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          hlo_computation, kernel_arguments.args(),
          /*discriminator=*/"", generate);
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<KernelThunk>(
      &fusion, entry->kernel_name, kernel_arguments.args(),
      entry->launch_dimensions, entry->cluster_dim, entry->shmem_bytes));

  return result;
}

std::optional<LaunchDimensions> TritonFusion::launch_dimensions() const {
  if (analysis_.fusion_backend_config().kind() == kTritonFusionKind) {
    // TODO(b/332649307): Change the line below to something more generic that
    // can handle different instructions (not just Reduce) and different
    // dimensions.
    //
    // One rough idea is to have a grid where:
    // - 1 grid dimension corresponds to all batch dimensions in the HLO.
    // - 1-2 grid dimension corresponds to block-able dimensions from the HLO.
    return CalculateSoftMaxLaunchDimensions(analysis_.fusion());
  } else if (analysis_.fusion_backend_config().kind() ==
             kTritonSoftmaxFusionKind) {
    return CalculateSoftMaxLaunchDimensions(analysis_.fusion());
  }

  // MatMul is not yet supported.
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
