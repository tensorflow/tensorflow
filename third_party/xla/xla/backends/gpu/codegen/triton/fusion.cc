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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

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
    const se::DeviceDescription& device_info, const llvm::Triple& target_triple,
    const std::string& data_layout, llvm::LLVMContext* llvm_context,
    mlir::MLIRContext* mlir_context) const {
  const se::GpuComputeCapability& cc = device_info.gpu_compute_capability();

  if (!analysis_.fusion_backend_config().has_block_level_fusion_config()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Block level fusion config is required for Triton fusions: ",
        fusion.ToString()));
  }
  return TritonWrapper(
      impl_fn_name, &fusion, cc, device_info,
      BlockLevelParameters::FromBlockLevelFusionConfig(
          analysis_.fusion_backend_config().block_level_fusion_config()),
      target_triple, data_layout, *llvm_context, *mlir_context);
};

absl::StatusOr<FusionEmissionResult> TritonFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  TF_ASSIGN_OR_RETURN(EmitResult kernel_and_module,
                      Emit(ir_emitter_context, fusion, nullptr, {}));
  FusionEmissionResult result;
  result.thunks.push_back(std::move(kernel_and_module.kernel_thunk));
  result.module = std::move(kernel_and_module.llvm_module);
  return result;
}

absl::StatusOr<TritonFusion::EmitResult> TritonFusion::Emit(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const HloInstruction* instr_override,
    absl::Span<const Shape> unmanaged_arguments) const {
  std::string suggested_kernel_name = std::string(fusion.name());
  llvm::IRBuilder builder(*ir_emitter_context.llvm_context());
  VLOG(3) << fusion.ToString();
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      emitters::KernelArguments::Create(
          ir_emitter_context.buffer_assignment(), GetDefaultBufferAlignment(),
          instr_override != nullptr ? instr_override : &fusion,
          unmanaged_arguments));

  const HloComputation* hlo_computation =
      fusion.fused_instructions_computation();
  VLOG(3) << "hlo_computation: " << hlo_computation->ToString();

  std::unique_ptr<llvm::Module> local_module;
  auto generate = [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
    VLOG(3) << "Generating: " << suggested_kernel_name;

    const std::string sanitized_kernel_name =
        ir_emitter_context.GetSanitizedUniqueName(suggested_kernel_name);

    TF_ASSIGN_OR_RETURN(
        TritonWrapperResult triton_wrapper_result,
        GenerateTritonKernelAndWrapper(
            fusion, sanitized_kernel_name, ir_emitter_context.gpu_device_info(),
            ir_emitter_context.target_triple(),
            ir_emitter_context.data_layout(), ir_emitter_context.llvm_context(),
            ir_emitter_context.mlir_context()));
    local_module = std::move(triton_wrapper_result.llvm_module);

    auto backend_config =
        fusion.backend_config<GpuBackendConfig>()->fusion_backend_config();
    absl::string_view fusion_kind = backend_config.kind();

    LaunchDimensions launch_dimensions;

    // TODO(bchetioui,pifon): this list should be consolidated; why do we need
    // so many different fusion kinds?
    const std::vector<absl::string_view> kSupportedFusionKinds = {
        kTritonFusionKind,
        kTritonNestedGemmFusionKind,
        kTritonCollectiveFusionKind,
    };

    if (!absl::c_linear_search(kSupportedFusionKinds, fusion_kind)) {
      return Internal("Unsupported fusion kind: %s", fusion_kind);
    }

    std::optional<LaunchConfig> launch_config;
    // Currently GetLaunchConfig will compute the same value as the extracted
    // one. They are different only when warp specialization is enabled.
    // Ideally we should always pass the thread_dims value extracted from
    // the Triton compilation. However, we are keeping the old code path
    // to maintain the current behavior and be safe.
    if (fusion.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_enable_triton_warp_specialization()) {
      launch_config = this->GetLaunchConfig(triton_wrapper_result.thread_dims);
    } else {
      launch_config = this->GetLaunchConfig();
    }
    // This check should be enforced by `GenerateTritonKernelWrapper`.
    CHECK(launch_config.has_value());
    launch_dimensions = std::move(launch_config->launch_dimensions);

    TF_ASSIGN_OR_RETURN(
        llvm::Function * kernel,
        RemoveUnusedTritonAbiArguments(local_module.get(), ir_emitter_context,
                                       sanitized_kernel_name,
                                       /*keep_scratch=*/false));

    AnnotateAttrsIfUnset(kernel_arguments, *kernel);
    PopulateNvvmAnnotations(local_module.get(), kernel, triton_wrapper_result);

    TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
        ir_emitter_context.gpu_device_info(), launch_dimensions, kernel,
        local_module.get()));

    return {{kernel->getName().str(), launch_dimensions,
             /*cluster_dim=*/std::nullopt, triton_wrapper_result.shmem_bytes,
             /*binary=*/"", triton_wrapper_result.tma_metadata}};
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          hlo_computation, kernel_arguments.args(),
          /*discriminator=*/"", generate);
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);
  return EmitResult{
      std::make_unique<KernelThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              &fusion, ir_emitter_context.GetNextThunkId()),
          entry->kernel_name, kernel_arguments, entry->launch_dimensions,
          /*cluster_dim=*/std::nullopt, entry->shmem_bytes,
          entry->tma_metadata),
      was_cached ? nullptr : std::move(local_module)};
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

std::optional<TritonFusion::LaunchConfig> TritonFusion::GetLaunchConfig(
    std::optional<se::ThreadDim> thread_dims_override) const {
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
    // TODO(b/451901200): We eventually also want to be able to predict this
    // value without compiling so the cost model can rely on it. Currently, we
    // need the override for auto warp specialization.
    if (thread_dims_override) {
      launch_config.launch_dimensions = LaunchDimensions{
          se::BlockDim(num_blocks), thread_dims_override.value()};
    } else {
      int64_t estimated_threads_per_block =
          block_level_parameters.num_warps * WarpSize(analysis_.device_info());
      launch_config.launch_dimensions =
          LaunchDimensions{static_cast<uint64_t>(num_blocks),
                           static_cast<uint64_t>(estimated_threads_per_block)};
    }

    launch_config.block_level_parameters = std::move(block_level_parameters);
    return launch_config;
  }

  // MatMul is not yet supported.
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
