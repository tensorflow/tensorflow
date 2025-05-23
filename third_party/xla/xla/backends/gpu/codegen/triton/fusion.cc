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
#include "xla/backends/gpu/codegen/triton/fusion.h"

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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

// Since we are creating the kernel and splicing the impl_fn into it, we
// need to manually annotate the kernel with the nvvm.annotations.
static void PopulateNvvmAnnotations(
    llvm::Module* llvm_module, llvm::Function* kernel,
    TritonWrapperResult& triton_wrapper_result) {
  llvm::NamedMDNode* dest_nvvm_annotations =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  for (auto md : triton_wrapper_result.nvvm_annotations) {
    if (auto node = llvm::dyn_cast<llvm::MDNode>(md)) {
      if (node->getNumOperands() >= 1) {
        std::vector<llvm::Metadata*> new_operands;
        new_operands.reserve(node->getNumOperands());
        new_operands.push_back(llvm::ValueAsMetadata::get(kernel));
        for (unsigned i = 1; i < node->getNumOperands(); ++i) {
          new_operands.push_back(node->getOperand(i));
        }
        llvm::MDNode* new_node =
            llvm::MDNode::get(llvm_module->getContext(), new_operands);
        dest_nvvm_annotations->addOperand(new_node);
      }
    }
  }
}

absl::StatusOr<TritonWrapperResult>
TritonFusion::GenerateTritonKernelAndWrapper(
    const HloFusionInstruction& fusion, absl::string_view impl_fn_name,
    const se::DeviceDescription& device_info, llvm::Module* llvm_module,
    mlir::MLIRContext* mlir_context) const {
  const se::GpuComputeCapability& cc = device_info.gpu_compute_capability();
  auto backend_config =
      fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
  absl::string_view fusion_kind = backend_config.kind();
  TritonWrapperResult triton_wrapper_result;

  if (fusion_kind == kTritonFusionKind ||
      fusion_kind == kTritonNestedGemmFusionKind) {
    std::optional<LaunchConfig> launch_config = this->launch_config();
    if (!launch_config.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Block level fusion config is required for Triton fusions: ",
          fusion.ToString()));
    }
    TF_ASSIGN_OR_RETURN(triton_wrapper_result,
                        TritonWrapper(impl_fn_name, &fusion, cc, device_info,
                                      launch_config->block_level_parameters,
                                      llvm_module, *mlir_context));
  } else {  // Must be a MatMul
    CHECK_EQ(fusion_kind, kTritonGemmFusionKind);
    // TODO(bchetioui): port matmul emitter to fully use the new
    // infrastructure.
    BlockLevelParameters block_level_parameters;
    if (!backend_config.has_triton_gemm_config()) {
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
        TritonWrapper(impl_fn_name, &fusion, cc, device_info,
                      block_level_parameters, llvm_module, *mlir_context));
  }

  return triton_wrapper_result;
};

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
  VLOG(3) << "hlo_computation: " << hlo_computation->ToString();

  auto generate = [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
    VLOG(3) << "Generating: " << suggested_kernel_name;

    const std::string impl_fn_name =
        ir_emitter_context.name_uniquer()->GetUniqueName(
            llvm_ir::SanitizeFunctionName(
                absl::StrCat(suggested_kernel_name, "_impl")));

    TF_ASSIGN_OR_RETURN(
        TritonWrapperResult triton_wrapper_result,
        GenerateTritonKernelAndWrapper(fusion, impl_fn_name,
                                       ir_emitter_context.gpu_device_info(),
                                       ir_emitter_context.llvm_module(),
                                       ir_emitter_context.mlir_context()));

    auto backend_config =
        fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
    absl::string_view fusion_kind = backend_config.kind();

    LaunchDimensions launch_dimensions;
    if (fusion_kind == kTritonFusionKind ||
        fusion_kind == kTritonNestedGemmFusionKind) {
      std::optional<LaunchConfig> launch_config = this->launch_config();
      // This check should be enforced by `GenerateTritonKernelWrapper`.
      CHECK(launch_config.has_value());
      launch_dimensions = std::move(launch_config->launch_dimensions);
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
        triton_config.set_num_stages(1);
        triton_config.set_num_warps(2);
        triton_config.set_num_ctas(1);
      }

      // TODO(bchetioui): move calculation of launch dimensions to
      // 'launch_config()'.
      TF_ASSIGN_OR_RETURN(
          TritonGemmConfig config,
          TritonGemmConfig::FromProto(backend_config.triton_gemm_config()));

      TF_ASSIGN_OR_RETURN(auto analysis, TritonFusionAnalysis::Execute(
                                             *hlo_computation, config.split_k));

      TF_ASSIGN_OR_RETURN(
          launch_dimensions,
          GetMatMulLaunchDimensions(analysis, analysis_.fusion(), config,
                                    analysis_.device_info()));
    }

    llvm::Function* impl_fn =
        ir_emitter_context.llvm_module()->getFunction(impl_fn_name);
    TF_RET_CHECK(impl_fn);

    llvm::Function* kernel;
    std::vector<llvm_ir::IrArray> inputs;
    std::vector<llvm_ir::IrArray> outputs;
    TF_ASSIGN_OR_RETURN(
        std::tie(kernel, inputs, outputs),
        BuildKernelPrototype(ir_emitter_context, impl_fn_name,
                             suggested_kernel_name, kernel_arguments.args(),
                             impl_fn->arg_size(), launch_dimensions, &builder));

    PopulateNvvmAnnotations(ir_emitter_context.llvm_module(), kernel,
                            triton_wrapper_result);

    // Move function body into kernel prototype.
    llvm::Function* prototype_func = builder.GetInsertBlock()->getParent();
    prototype_func->splice(prototype_func->begin(), impl_fn);
    for (const auto& [arg, ir_array] : llvm::zip(impl_fn->args(), inputs)) {
      arg.replaceAllUsesWith(ir_array.GetBasePointer());
    }
    // Triton's kernel ABI expects an additional scratchpad global memory.
    // For now it is only used for on-device creation of TMA descriptors, which
    // we do not use yet, so we are just replacing this argument with a null
    // pointer.
    // TODO: b/381242007 - Allocate a proper buffer if we want to use
    // device-side TMA APIs.
    auto scratchpad_arg = impl_fn->getArg(impl_fn->arg_size() - 1);
    scratchpad_arg->replaceAllUsesWith(llvm::ConstantPointerNull::get(
        llvm::cast<llvm::PointerType>(scratchpad_arg->getType())));

    return {{kernel->getName().str(), launch_dimensions,
             triton_wrapper_result.cluster_dim,
             triton_wrapper_result.shmem_bytes, /*binary=*/"",
             triton_wrapper_result.tma_metadata}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          hlo_computation, kernel_arguments.args(),
          /*discriminator=*/"", generate);
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(&fusion), entry->kernel_name,
      kernel_arguments.args(), entry->launch_dimensions, entry->cluster_dim,
      entry->shmem_bytes, entry->tma_metadata));

  return result;
}

namespace {
int64_t GetNumberOfBlocks(absl::Span<const int64_t> dimensions,
                          absl::Span<const int64_t> tile_sizes) {
  int64_t num_blocks = 1;
  for (auto [dim_size, dim_tile_size] : llvm::zip(dimensions, tile_sizes)) {
    num_blocks *= (dim_size + dim_tile_size - 1) / dim_tile_size;
  }
  return num_blocks;
}
}  // namespace

std::optional<TritonFusion::LaunchConfig> TritonFusion::launch_config() const {
  if (analysis_.fusion_backend_config().has_block_level_fusion_config()) {
    BlockLevelParameters block_level_parameters =
        BlockLevelParameters::FromBlockLevelFusionConfig(
            analysis_.fusion_backend_config().block_level_fusion_config());

    // We expect all roots to have the same number of blocks. Otherwise we
    // cannot codegen it.
    int64_t num_blocks =
        GetNumberOfBlocks(analysis_.fusion_root(0).shape().dimensions(),
                          block_level_parameters.output_tile_sizes[0]);
    for (int64_t i = 1; i < analysis_.fusion_root_count(); ++i) {
      CHECK_EQ(GetNumberOfBlocks(analysis_.fusion_root(i).shape().dimensions(),
                                 block_level_parameters.output_tile_sizes[i]),
               num_blocks);
    }

    LaunchConfig launch_config;
    launch_config.launch_dimensions = LaunchDimensions{
        static_cast<uint64_t>(num_blocks),
        static_cast<uint64_t>(block_level_parameters.num_warps *
                              WarpSize(analysis_.device_info()))};
    launch_config.block_level_parameters = std::move(block_level_parameters);
    return launch_config;
  }

  // MatMul is not yet supported.
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
