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
#include "xla/layout_util.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
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
std::optional<TritonFusion::LaunchConfig> CalculateSoftMaxLaunchConfig(
    const HloFusionAdaptor& fusion) {
  // Assumptions we make about the matcher:
  //   * matches Softmax "diamonds" on the last axis, along with any number of
  //     elementwise operations/bitcasts on any edge
  //   * within a given fusion, every argument to a Softmax diamond has the same
  //     shape
  //   * every reduction is on the last axis
  //   * the last axis of every reduction parameter has the same length
  //   * reductions only reduce a single operand
  //   * all the shapes have canonical layout (logical layout = physical layout)
  //   * the computation has a single output
  //   * we tile along a single dimension

  std::optional<HloInstructionAdaptor> reduce_adaptor =
      HloFindIf(fusion.GetRoots(), fusion,
                [](auto node) { return node.opcode() == HloOpcode::kReduce; });

  if (!reduce_adaptor.has_value()) {
    LOG(ERROR) << "No reduce instruction found.";
    return std::nullopt;
  }

  const HloInstruction& reduce = reduce_adaptor->instruction();

  const Shape& reduce_input_shape = reduce.operand(0)->shape();

  if (reduce.dimensions().size() != 1 ||
      reduce.dimensions(0) != reduce_input_shape.rank() - 1) {
    LOG(ERROR) << "Reduce instruction must reduce inner-most dimension. "
               << reduce.ToString();
    return std::nullopt;
  }

  auto roots = fusion.GetRoots();
  if (roots.size() != 1) {
    LOG(ERROR) << "Multi-output fusions are not supported. "
               << fusion.ToString();
    return std::nullopt;
  }

  const HloInstruction& root = roots[0].instruction();
  const Shape& root_shape = root.shape();
  if (!root_shape.IsArray() ||
      LayoutUtil::IsMonotonicWithDim0Minor(root_shape.layout())) {
    LOG(ERROR) << "Root shape is not supported. " << root_shape.ToString();
    return std::nullopt;
  }

  TritonFusion::LaunchConfig launch_config;

  int row_len = reduce_input_shape.dimensions_minor(0);
  launch_config.output_tile_sizes.resize(root_shape.rank(), 1);
  launch_config.output_tile_sizes.back() = row_len;

  unsigned num_rows = 1;
  for (unsigned minor_axis = 1; minor_axis < reduce_input_shape.rank();
       ++minor_axis) {
    num_rows *= reduce_input_shape.dimensions_minor(minor_axis);
  }

  unsigned num_warps = 32;

  if (row_len <= 512) {
    num_warps = 1;
  } else if (row_len <= 1024) {
    num_warps = 2;
  } else if (row_len <= 16384) {
    num_warps = 4;
  } else if (row_len <= 32768) {
    num_warps = 8;
  } else if (row_len <= 65536) {
    num_warps = 16;
  }

  launch_config.launch_dimensions =
      LaunchDimensions(num_rows, static_cast<uint64_t>(num_warps * WarpSize()));

  return launch_config;
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

    auto backend_config =
        fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
    absl::string_view fusion_kind = backend_config.kind();

    TritonWrapperResult triton_wrapper_result;
    LaunchDimensions launch_dimensions;
    if (fusion_kind == kTritonFusionKind) {
      auto launch_config = *this->launch_config();
      launch_dimensions = launch_config.launch_dimensions;

      // TODO(bchetioui): parse block-level parameters from backend config
      // where available.
      BlockLevelParameters block_level_parameters;
      block_level_parameters.output_tile_sizes = std::vector<int64_t>(
          hlo_computation->root_instruction()->shape().rank() - 1, 1);
      block_level_parameters.output_tile_sizes.push_back(
          hlo_computation->root_instruction()->shape().dimensions().back());
      block_level_parameters.num_warps =
          launch_dimensions.num_threads_per_block() / WarpSize();
      block_level_parameters.num_ctas = 1;
      block_level_parameters.num_stages = 1;

      TF_ASSIGN_OR_RETURN(
          triton_wrapper_result,
          TritonWrapper(impl_fn_name, &fusion,
                        ir_emitter_context.gpu_compute_capability(),
                        ir_emitter_context.gpu_device_info(),
                        block_level_parameters,
                        ir_emitter_context.llvm_module(),
                        *ir_emitter_context.mlir_context()));
    } else {  // Must be a MatMul
      CHECK_EQ(fusion_kind, kTritonGemmFusionKind);
      // TODO(bchetioui): port matmul emitter to fully use the new
      // infrastructure.
      BlockLevelParameters block_level_parameters;
      if (!backend_config.has_triton_gemm_config()) {
        LOG(WARNING) << "Using fallback triton GEMM config for op "
                     << fusion.name();
        // TODO(bchetioui): deduplicate default matmul config information.
        auto& triton_config = *backend_config.mutable_triton_gemm_config();
        triton_config.set_block_m(64);
        triton_config.set_block_k(64);
        triton_config.set_block_n(64);
        triton_config.set_split_k(1);

        block_level_parameters.num_ctas = 1;
        block_level_parameters.num_stages = 1;
        block_level_parameters.num_warps = 2;
      } else {
        const auto& triton_config = backend_config.triton_gemm_config();
        block_level_parameters.num_ctas = triton_config.num_ctas();
        block_level_parameters.num_stages = triton_config.num_stages();
        block_level_parameters.num_warps = triton_config.num_warps();
      }

      TF_ASSIGN_OR_RETURN(
          triton_wrapper_result,
          TritonWrapper(impl_fn_name, &fusion,
                        ir_emitter_context.gpu_compute_capability(),
                        ir_emitter_context.gpu_device_info(),
                        block_level_parameters,
                        ir_emitter_context.llvm_module(),
                        *ir_emitter_context.mlir_context()));

      // TODO(bchetioui): move calculation of launch dimensions to
      // 'launch_config()'.
      TF_ASSIGN_OR_RETURN(
          TritonGemmConfig config,
          TritonGemmConfig::FromProto(backend_config.triton_gemm_config()));

      TF_ASSIGN_OR_RETURN(auto analysis, TritonFusionAnalysis::Execute(
                                             *hlo_computation, config.split_k));

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

std::optional<TritonFusion::LaunchConfig> TritonFusion::launch_config() const {
  if (analysis_.fusion_backend_config().kind() == kTritonFusionKind) {
    // TODO(b/332649307): Change the line below to something more generic that
    // can handle different instructions (not just Reduce) and different
    // dimensions.
    //
    // One rough idea is to have a grid where:
    // - 1 grid dimension corresponds to all batch dimensions in the HLO.
    // - 1-2 grid dimension corresponds to block-able dimensions from the HLO.
    return CalculateSoftMaxLaunchConfig(analysis_.fusion());
  }

  // MatMul is not yet supported.
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
