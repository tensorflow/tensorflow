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
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/frontend_attributes.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

struct EmitArgs {
  TritonFusion::EmitThunk emit_thunk;
  std::vector<Shape> unmanaged_arguments;
};

absl::StatusOr<EmitArgs> EmitCollectiveFusion(
    Thunk::ThunkInfo info, const HloFusionAnalysis& analysis,
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion) {
  VLOG(5) << "EmitCollectiveFusion: " << info.profile_annotation;
  TF_RET_CHECK(analysis.fusion_heroes().size() == 1)
      << "EmitCollectiveFusion: Expected exactly one hero in a collective "
         "fusion. Found: "
      << analysis.fusion_heroes().size() << " for: " << fusion.ToString();

  const HloInstructionAdaptor collective_hero = analysis.fusion_hero(0);

  bool use_global_device_ids = collective_hero.use_global_device_ids();

  CollectiveConfig config = GetCollectiveConfig(&collective_hero.instruction(),
                                                use_global_device_ids);
  const HloFusionInstruction* fusion_instr = &fusion;
  std::vector<CollectiveThunk::Buffer> buffers;
  ABSL_ASSIGN_OR_RETURN(buffers, GetCollectiveBuffers(
                                ir_emitter_context.buffer_assignment(),
                                fusion_instr, Thunk::Kind::kCollectiveKernel,
                                /*has_dynamic_root=*/false));

  TritonFusion::EmitThunk make_thunk =
      [info = std::move(info), buffers = std::move(buffers), config,
       fusion_instr](TritonFusion::EmitResult result) mutable
      -> absl::StatusOr<ThunkSequence> {
    ABSL_ASSIGN_OR_RETURN(CollectiveKernelSpec kernel_spec,
                     CreateCollectiveKernelSpec(
                         fusion_instr, result.entry.launch_dimensions));
    auto cubin = result.entry.binary.empty()
                     ? std::nullopt
                     : std::make_optional(std::move(result.entry.binary));
    return ThunkSequence::Of<CollectiveKernelThunk>(
        std::move(info), config, std::move(kernel_spec), std::move(buffers),
        /*is_collective_kernel_enabled=*/true, result.entry.kernel_name,
        result.entry.launch_dimensions, result.entry.shmem_bytes,
        std::move(cubin), result.entry.use_pdl);
  };
  ABSL_ASSIGN_OR_RETURN(std::vector<Shape> unmanaged_arguments,
                   GetCollectiveUnmanagedKernelArguments(fusion_instr));
  return EmitArgs{std::move(make_thunk), std::move(unmanaged_arguments)};
}
}  // namespace

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

xla::Future<TritonWrapperResult> TritonFusion::GenerateTritonKernelAndWrapper(
    const HloFusionInstruction& fusion, absl::string_view impl_fn_name,
    const se::DeviceDescription& device_info, const llvm::Triple& target_triple,
    const std::string& data_layout, BorrowedMlirContext borrowed_context,
    KernelCompiler* compiler) const {
  if (!analysis_.fusion_backend_config().has_block_level_fusion_config()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Block level fusion config is required for Triton fusions: ",
        fusion.ToString()));
  }

  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          analysis_.fusion_backend_config().block_level_fusion_config());
  ABSL_ASSIGN_OR_RETURN(
      TritonKernelSource kernel_source,
      CreateTritonModule(impl_fn_name, fusion, device_info,
                         block_level_parameters, **borrowed_context));

  // Forward PDL launch annotation from HLO to MLIR.
  if (DoesPdlLaunch(fusion)) {
    kernel_source.module()->setAttr(
        kXlaPdlLaunch, mlir::UnitAttr::get(borrowed_context->get()));
  }
  return compiler->CompileTritonToLlvm(
      impl_fn_name, *fusion.GetModule(), device_info, block_level_parameters,
      target_triple, data_layout, std::move(kernel_source),
      std::move(borrowed_context),
      /*is_xla_fusion=*/true);
};

AsyncThunkSequence TritonFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  VLOG(5) << "TritonFusion::Emit: " << thunk_info.profile_annotation;
  EmitArgs emit_args;
  if (analysis_.fusion_backend_config().kind() == kTritonCollectiveFusionKind) {
    ABSL_ASSIGN_OR_RETURN(emit_args,
                     EmitCollectiveFusion(std::move(thunk_info), analysis_,
                                          ir_emitter_context, fusion));
  } else {
    EmitThunk make_thunk =
        [thunk_info = std::move(thunk_info)](
            EmitResult result) -> absl::StatusOr<ThunkSequence> {
      ABSL_ASSIGN_OR_RETURN(
          CustomKernel custom_kernel,
          kernel::CreateOwnedCubinCustomKernel(
              result.entry.kernel_name, result.entry.binary,
              result.kernel_arguments.args().size(),
              result.entry.launch_dimensions.block_counts(),
              result.entry.launch_dimensions.thread_counts_per_block(),
              result.entry.shmem_bytes));
      return ThunkSequence::Of<CustomKernelThunk>(
          thunk_info, std::move(custom_kernel), result.kernel_arguments,
          result.entry.use_pdl, std::vector<int64_t>{},
          result.entry.tma_metadata);
    };
    emit_args = {
        std::move(make_thunk),
        /*unmanaged_arguments=*/{},
    };
  }
  return Emit(ir_emitter_context, fusion, nullptr,
              std::move(emit_args.unmanaged_arguments))
      .Map(std::move(emit_args.emit_thunk));
}

xla::Future<TritonFusion::EmitResult> TritonFusion::Emit(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const HloInstruction* instr_override,
    absl::Span<const Shape> unmanaged_arguments) const {
  std::string suggested_kernel_name = std::string(fusion.name());
  VLOG(3) << fusion.ToString();
  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments kernel_arguments,
      emitters::KernelArguments::Create(
          ir_emitter_context.buffer_assignment(), GetDefaultBufferAlignment(),
          instr_override != nullptr ? instr_override : &fusion,
          unmanaged_arguments));

  const HloComputation* hlo_computation =
      fusion.fused_instructions_computation();
  VLOG(3) << "hlo_computation: " << hlo_computation->ToString();

  auto generate = [&]() -> xla::Future<KernelReuseCache::Entry> {
    VLOG(3) << "Generating: " << suggested_kernel_name;

    const std::string sanitized_kernel_name =
        ir_emitter_context.GetSanitizedUniqueName(suggested_kernel_name);

    return GenerateTritonKernelAndWrapper(
               fusion, sanitized_kernel_name,
               ir_emitter_context.gpu_device_info(),
               ir_emitter_context.target_triple(),
               ir_emitter_context.data_layout(),
               ir_emitter_context.BorrowMlirContext(),
               ir_emitter_context.kernel_compiler())
        .Map([analysis = &analysis_,
              kernel_compiler = ir_emitter_context.kernel_compiler(),
              &gpu_device_info = ir_emitter_context.gpu_device_info(),
              fusion = &fusion, sanitized_kernel_name,
              sanitized_kernel_impl_name =
                  ir_emitter_context.GetSanitizedUniqueName(
                      sanitized_kernel_name + "_impl"),
              kernel_arguments](
                 TritonWrapperResult triton_wrapper_result) mutable
                 -> xla::Future<KernelReuseCache::Entry> {
          auto local_module = std::move(triton_wrapper_result.kernel_source)
                                  .thread_safe_module();

          ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                           fusion->backend_config<GpuBackendConfig>());
          absl::string_view fusion_kind =
              gpu_backend_config.fusion_backend_config().kind();

          LaunchDimensions launch_dimensions;

          // TODO(bchetioui,pifon): this list should be consolidated; why do we
          // need so many different fusion kinds?
          const std::vector<absl::string_view> kSupportedFusionKinds = {
              kTritonFusionKind,
              kTritonNestedGemmFusionKind,
              kTritonCollectiveFusionKind,
          };

          if (!absl::c_linear_search(kSupportedFusionKinds, fusion_kind)) {
            return Internal("Unsupported fusion kind: %s", fusion_kind);
          }

          std::optional<LaunchConfig> launch_config;
          // Currently GetLaunchConfig will compute the same value as the
          // extracted one. They are different only when warp specialization is
          // enabled. Ideally we should always pass the thread_dims value
          // extracted from the Triton compilation. However, we are keeping the
          // old code path to maintain the current behavior and be safe.
          if (fusion->GetModule()
                  ->config()
                  .debug_options()
                  .xla_gpu_experimental_enable_triton_warp_specialization()) {
            launch_config =
                GetLaunchConfig(analysis, triton_wrapper_result.thread_dims);
          } else {
            launch_config = GetLaunchConfig(analysis);
          }
          // This check should be enforced by `GenerateTritonKernelWrapper`.
          CHECK(launch_config.has_value());
          launch_dimensions = std::move(launch_config->launch_dimensions);

          ABSL_ASSIGN_OR_RETURN(
              llvm::Function * kernel,
              RemoveUnusedTritonAbiArguments(local_module.getModuleUnlocked(),
                                             sanitized_kernel_name,
                                             sanitized_kernel_impl_name,
                                             /*keep_scratch=*/false));

          AnnotateAttrsIfUnset(kernel_arguments, *kernel);
          PopulateNvvmAnnotations(local_module.getModuleUnlocked(), kernel,
                                  triton_wrapper_result);

          ABSL_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
              gpu_device_info, launch_dimensions, kernel,
              local_module.getModuleUnlocked()));

          return kernel_compiler
              ->CompileToTargetBinary(LlvmKernelSource{std::move(local_module)})
              .Map([kernel_name = std::move(sanitized_kernel_name),
                    launch_dims = std::move(launch_dimensions),
                    tma_metadata = triton_wrapper_result.tma_metadata,
                    shmem_bytes = triton_wrapper_result.shmem_bytes,
                    use_pdl = triton_wrapper_result.use_pdl](
                       const std::vector<uint8_t>& cubin) mutable {
                return KernelReuseCache::Entry{std::move(kernel_name),
                                               launch_dims,
                                               /*cluster_dim=*/std::nullopt,
                                               shmem_bytes,
                                               cubin,
                                               tma_metadata,
                                               use_pdl};
              });
        });
  };

  auto [status_or_entry, was_cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          hlo_computation, kernel_arguments.args(),
          /*discriminator=*/"TritonFusion", generate);
  return status_or_entry.Map([kernel_arguments = std::move(kernel_arguments)](
                                 const KernelReuseCache::Entry* entry) mutable
                                 -> absl::StatusOr<EmitResult> {
    return EmitResult{*entry, std::move(kernel_arguments)};
  });
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
    const HloFusionAnalysis* analysis,
    std::optional<se::ThreadDim> thread_dims_override) {
  if (!analysis->fusion_backend_config().has_block_level_fusion_config()) {
    // MatMul is not yet supported.
    return std::nullopt;
  }
  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          analysis->fusion_backend_config().block_level_fusion_config());

  // We expect all roots to have the same number of blocks. Otherwise we
  // cannot codegen it.
  int64_t num_blocks =
      GetNumberOfBlocks(analysis->fusion_root(0).shape().dimensions(),
                        block_level_parameters.output_tile_sizes[0]);
  for (int64_t i = 1; i < analysis->fusion_root_count(); ++i) {
    CHECK_EQ(GetNumberOfBlocks(analysis->fusion_root(i).shape().dimensions(),
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
        block_level_parameters.num_warps * WarpSize(analysis->device_info());
    launch_config.launch_dimensions =
        LaunchDimensions{static_cast<uint64_t>(num_blocks),
                         static_cast<uint64_t>(estimated_threads_per_block)};
  }

  launch_config.block_level_parameters = std::move(block_level_parameters);
  return launch_config;
}

}  // namespace gpu
}  // namespace xla
