/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/tensor_ir/fusion.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Options/OptionsEnums.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/tensor_ir/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/tensor_ir/custom_kernel.h"
#include "xla/backends/gpu/codegen/tensor_ir/hlo_to_tensor_ir.h"
#include "xla/backends/gpu/codegen/tensor_ir/layout_utils.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::gpu {

namespace {

// Imports, lowers and compiles the fusion to a cubin.
absl::StatusOr<KernelReuseCache::Entry> CompileFusion(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const TensorIrFusionConfig& tensor_ir_config,
    const emitters::KernelArguments& kernel_arguments) {
  const HloComputation* computation = fusion.fused_instructions_computation();

  // Create the MLIR context and module.
  BorrowedMlirContext borrowed_context = ir_emitter_context.BorrowMlirContext();
  mlir::MLIRContext& context = **borrowed_context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::cuda_tile::CudaTileDialect,
                      mlir::arith::ArithDialect>();

  // `OwningOpRef` so that the module is destroyed when we leave this function;
  // the MLIR context is borrowed from a pool and outlives it.
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(&context));
  mlir::ModuleOp module = *module_ref;
  ABSL_ASSIGN_OR_RETURN(
      mlir::nv_tensor_ir::GraphOp graph_op,
      tensor_ir::ImportAndLegalizeComputation(*computation, module));
  {
    mlir::BaseScopedDiagnosticHandler diagnostic_handler(&context);
    if (llvm::failed(graph_op.verify())) {
      return diagnostic_handler.Combine(absl::InternalError(
          absl::StrCat("TensorIrFusion: invalid TensorIR graph for fusion: ",
                       fusion.ToString(), ": ")));
    }
  }

  const std::vector<emitters::KernelArgument>& kernel_args =
      kernel_arguments.args();
  // The graph has one argument per fused parameter, which the HLO verifier
  // ties to the fusion's operands. A fusion is permitted to have operands that
  // no fused parameter reads, though, in which case the counts disagree; fail
  // loudly here rather than letting the alignments be attached off-by-one.
  int64_t num_inputs = computation->num_parameters();
  if (kernel_args.size() != num_inputs + 1) {
    return absl::InternalError(absl::StrCat(
        "TensorIrFusion: expected ", num_inputs + 1,
        " kernel arguments (one per fused parameter plus a single output), "
        "got ",
        kernel_args.size(), " for fusion: ", fusion.ToString()));
  }
  // Layout strides were already attached by `ImportAndLegalizeComputation`;
  // alignments need buffer assignment and so can only be added here.
  ABSL_RETURN_IF_ERROR(tensor_ir::AttachBufferAlignments(graph_op, kernel_args));

  // Run the conversion pipeline.
  llvm::SmallVector<int32_t> tile_size(tensor_ir_config.tile_size().begin(),
                                       tensor_ir_config.tile_size().end());
  // An unset `reduction_tile_size` is 0, which the pipeline rejects ("must be
  // a positive power of two") even for fusions without a reduction. Fall back
  // to the pipeline's own default instead.
  int64_t reduction_tile_size = tensor_ir_config.reduction_tile_size();
  if (reduction_tile_size == 0) {
    reduction_tile_size = mlir::nv_tensor_ir::kDefaultReductionTileSize;
  }
  mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions pipeline_options =
      tensor_ir::GetPipelineOptions(
          ir_emitter_context.gpu_device_info().gpu_compute_capability(),
          tile_size, reduction_tile_size);

  // Note: the graph is *not* lowered here. `ICompiler::compile()` runs the
  // TensorIR-to-CudaTile pipeline itself and requires the module to still
  // contain exactly one (unlowered) `nv_tensor_ir.graph`; pre-lowering it here
  // makes the compiler reject the module.

  // The CudaTile bytecode writer only accepts `DILocAttr`, `FileLineColLoc`,
  // `UnknownLoc` and `CallSiteLoc` locations, while the HLO importer attaches
  // `NameLoc`s derived from HLO instruction and parameter names. Drop them,
  // otherwise writing the kernel's TileIR bytecode fails. Block arguments need
  // the same treatment: the lowering propagates their locations onto the ops
  // it creates.
  mlir::Location unknown_loc = mlir::UnknownLoc::get(&context);
  module.walk([&](mlir::Operation* op) {
    op->setLoc(unknown_loc);
    for (mlir::Region& region : op->getRegions()) {
      for (mlir::Block& block : region) {
        for (mlir::BlockArgument arg : block.getArguments()) {
          arg.setLoc(unknown_loc);
        }
      }
    }
  });

  // Compile the kernel.
  using ::mlir::nv_tensor_ir::ArchPortability;
  using ::mlir::nv_tensor_ir::backend::cuda_tile::CudaTileArtifactKind;
  using ::mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions;
  mlir::FailureOr<mlir::nv_tensor_ir::SmTarget> sm_target =
      mlir::nv_tensor_ir::SmTarget::fromCc(pipeline_options.computeCapability,
                                           ArchPortability::arch_conditional);
  if (llvm::failed(sm_target)) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrFusion: unsupported compute capability: ",
                     pipeline_options.computeCapability));
  }
  CudaTileCompileOptions compile_options(*sm_target, pipeline_options.numCTAs,
                                         pipeline_options.numWarps, tile_size);
  // Ask the compiler for device code rather than TileIR bytecode. Cubin
  // assembly requires an arch-conditional target, which `SmTarget::fromCc`
  // only grants from sm_90 on; below that it silently clamps the portability
  // down and `CudaTileCompileOptions::validate()` then rejects the request.
  // `IsSupportedComputeCapability` above has already ruled those out.
  if (sm_target->getPortability() != ArchPortability::arch_conditional) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrFusion: unsupported compute capability: ",
                     pipeline_options.computeCapability));
  }
  compile_options.artifactKind = CudaTileArtifactKind::Cubin;
  // `ICompiler::compile()` rebuilds the pipeline options from the compile
  // options (`makePipelineOptions`), so the reduction tile size has to travel
  // through here as well; `pipeline_options.reductionTileSize` alone would be
  // dropped. The field comes from third_party/tensor_ir/patches/hotfixes.patch.
  compile_options.reductionTileSize = reduction_tile_size;

  tensor_ir::SetUpTileIrAssembler();
  std::unique_ptr<mlir::nv_tensor_ir::ICompiler> compiler =
      mlir::nv_tensor_ir::ICompiler::create(
          mlir::nv_tensor_ir::CompilerBackend::CudaTile);
  auto kernel_or = compiler->compile(module, compile_options);
  if (!kernel_or.ok()) {
    return absl::InternalError(
        absl::StrCat("TensorIrFusion: failed to compile fusion ", fusion.name(),
                     ": ", kernel_or.status().message()));
  }

  return tensor_ir::MakeKernelCacheEntry(**kernel_or);
}

}  // namespace

AsyncThunkSequence TensorIrFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  // Verify the fusion is supported.
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_backend_config.fusion_backend_config();
  if (backend_config.kind() != kTensorIrFusionKind) {
    return absl::InternalError(absl::StrCat(
        "TensorIrFusion: unsupported fusion kind: ", backend_config.kind()));
  }
  if (!backend_config.has_tensor_ir_fusion_config()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TensorIrFusion: missing tensor_ir_fusion_config for fusion: ",
        fusion.ToString()));
  }
  const TensorIrFusionConfig& tensor_ir_config =
      backend_config.tensor_ir_fusion_config();

  if (auto decision = tensor_ir::IsSupportedComputeCapability(
          ir_emitter_context.gpu_device_info().gpu_compute_capability());
      !decision.IsAllowed()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrFusion: ", decision.Explain()));
  }

  const HloComputation* computation = fusion.fused_instructions_computation();
  if (auto decision = tensor_ir::IsSupportedFusionComputation(*computation);
      !decision.IsAllowed()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrFusion: ", decision.Explain()));
  }

  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));

  // Kernels are shared between identical fusions; the tiling is part of the
  // key because the same computation can be compiled with different tilings.
  std::string discriminator = absl::StrCat(
      "TensorIrFusion:", absl::StrJoin(tensor_ir_config.tile_size(), ","), ":",
      tensor_ir_config.reduction_tile_size());
  auto [future_entry, cached] = ir_emitter_context.kernel_cache().GetWithStatus(
      computation, kernel_arguments.args(), discriminator,
      [&]() -> xla::Future<KernelReuseCache::Entry> {
        return CompileFusion(ir_emitter_context, fusion, tensor_ir_config,
                             kernel_arguments);
      });
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  return future_entry.Map(
      [fusion_name = std::string(fusion.name()),
       thunk_info = std::move(thunk_info),
       kernel_arguments = std::move(kernel_arguments), cached = cached,
       devices_in_process =
           ir_emitter_context.gpu_topology().num_devices_per_process()](
          const KernelReuseCache::Entry& entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        if (cached) {
          VLOG(3) << "Reuse: " << fusion_name << " -> " << entry.kernel_name;
        }
        ABSL_ASSIGN_OR_RETURN(
            CustomKernel custom_kernel,
            kernel::CreateSharedCubinCustomKernel(
                entry.kernel_name, entry.binary, kernel_arguments.args().size(),
                entry.launch_dimensions.block_counts(),
                entry.launch_dimensions.thread_counts_per_block(),
                entry.shmem_bytes));
        return ThunkSequence::Of<CustomKernelThunk>(
            thunk_info, std::move(custom_kernel), kernel_arguments,
            devices_in_process);
      });
}

}  // namespace xla::gpu
