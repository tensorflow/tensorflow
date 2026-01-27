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

#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/lowering_util.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/ir_printing.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/codegen/xtile/codegen/fusion_emitter.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/llvm_gpu_backend/nvptx_libdevice_path.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace ttir = ::mlir::triton;
namespace stablehlo = ::mlir::stablehlo;
namespace xgt = ::xla::gpu::triton;

using ::llvm::SmallVector;
using ::mlir::MLIRContext;

using ::xla::gpu::ir_emitter_triton_internal::GetModuleIrString;

void LoadMlirDialectsForTriton(mlir::MLIRContext& mlir_context) {
  mlir_context.loadDialect<
      ttir::TritonDialect, ttir::gpu::TritonGPUDialect,
      mlir::arith::ArithDialect, mlir::affine::AffineDialect,
      mlir::LLVM::LLVMDialect, xla::XlaDialect, xla::gpu::XlaGpuDialect,
      ttir::xla::XlaTritonDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, xla::xtile::XTileDialect,
      mlir::NVVM::NVVMDialect, stablehlo::StablehloDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir_context.appendDialectRegistry(registry);
}

// Simplified copy of translateLLVMToLLVMIR which in addition takes
// path to libdevice directly as an argument.
absl::StatusOr<std::unique_ptr<llvm::Module>> TranslateLLVMToLLVMIR(
    llvm::LLVMContext* llvmContext, mlir::ModuleOp module) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    return Internal("Failed to emit LLVM IR.");
  }
  // TODO: b/363203060 - Upstream Triton sets specific flags for the LLVM
  // optimizer to get best performance. Figure out if we can gain any of it by
  // propagating these flags to
  // xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc.
  return llvmModule;
}

absl::Status CreateInternalError(absl::string_view message,
                                 const HloFusionInstruction* fusion,
                                 mlir::ModuleOp triton_module) {
  std::string err;
  llvm::raw_string_ostream os(err);
  os << message << "\n";
  os << "fusion instruction: " << fusion->ToString() << "\n";
  os << "HLO module to reproduce:\n"
     << ExtractInstructionIntoNewModule(*fusion)->ToString();
  os << "triton_module>>>\n";
  triton_module->print(os, mlir::OpPrintingFlags().enableDebugInfo(true, true));
  os << "<<<triton_module\n";
  return absl::InternalError(err);
}

absl::Status IsTritonSupportedFusion(const HloFusionInstruction& fusion,
                                     const se::DeviceDescription& device_info) {
  const HloComputation* computation = fusion.fused_instructions_computation();
  for (const HloInstruction* hlo : computation->instructions()) {
    // Skip generating nested fusions, they are emitted by their consumer.
    if (hlo->parent()->IsFusionComputation() &&
        hlo->opcode() == HloOpcode::kFusion) {
      if (hlo->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_experimental_scaled_dot_with_triton()) {
        continue;
      }
      CodegenDecision decision = IsTritonSupportedInstruction(
          *hlo, device_info.gpu_compute_capability());
      if (!decision.CanFuse()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Fusion ", hlo->ToString(),
                         " is not supported: ", decision.Explain()));
      }
      VLOG(1) << "Skipping nested fusion: " << hlo->ToString();
      continue;
    }

    if (hlo->opcode() == HloOpcode::kPad) {
      if (!IsTritonSupportedInstruction(*hlo,
                                        device_info.gpu_compute_capability())) {
        return absl::FailedPreconditionError(
            absl::StrCat("Pad is not supported: ", hlo->ToString()));
      }
    }

    if (hlo->opcode() == HloOpcode::kReduce && hlo->dimensions().size() != 1) {
      return absl::FailedPreconditionError(
          absl::StrCat("Reduction with only a single dimension is supported: ",
                       hlo->ToString()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    MLIRContext& mlir_context) {
  TF_RETURN_IF_ERROR(IsTritonSupportedFusion(*fusion, device_info));

  LoadMlirDialectsForTriton(mlir_context);
  RegisterSymbolicExprStorage(&mlir_context);

  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  std::string fusion_kind(kTritonFusionKind);
  if (fusion->has_backend_config()) {
    auto backend_config = fusion->backend_config<GpuBackendConfig>();
    if (backend_config.ok()) {
      fusion_kind = backend_config->fusion_backend_config().kind();
    }
  }

  if (fusion_kind == kTritonGemmFusionKind) {
    return Internal(
        "Attempted to emit a GEMM fusion through the legacy Triton "
        "emitter, but it has been deleted. This is a bug.");
  }

  // TODO(bchetioui,pifon): this list should be consolidated; why do we need so
  // many different fusion kinds?
  const std::vector<absl::string_view> kSupportedFusionKinds = {
      kTritonFusionKind,
      kTritonNestedGemmFusionKind,
      kTritonCollectiveFusionKind,
  };

  if (!absl::c_linear_search(kSupportedFusionKinds, fusion_kind)) {
    return Internal("Unsupported fusion kind: %s", fusion_kind);
  }

  llvm::SmallVector<mlir::Type> opaque_args_types;
  // Add metadata arguments for collectives.
  // This is done after the input and output arguments but before the tile
  // index.
  int32_t num_metadata_arguments = 0;
  if (fusion_kind == kTritonCollectiveFusionKind) {
    auto loc = mlir::NameLoc::get(
        mlir::StringAttr::get(&mlir_context, hlo_computation->name()));
    mlir::ImplicitLocOpBuilder b(loc, &mlir_context);

    TF_ASSIGN_OR_RETURN(
        num_metadata_arguments,
        AddCollectiveMetadataArguments(opaque_args_types, b, hlo_computation));
  }

  const HloComputation* computation = fusion->fused_instructions_computation();
  SymbolicTileAnalysisOrError symbolic_tile_analysis_or =
      SymbolicTileAnalysis::AnalyzeComputation(
          *computation, &mlir_context,
          TritonEmitterConstraints::GetBuilder(device_info));

  if (std::holds_alternative<FusionDecision>(symbolic_tile_analysis_or)) {
    return Internal(
        "Unsupported fusion in EmitGeneric: %s",
        std::get<FusionDecision>(symbolic_tile_analysis_or).Explain());
  }

  const auto& symbolic_tile_analysis =
      std::get<SymbolicTileAnalysis>(symbolic_tile_analysis_or);

  TF_ASSIGN_OR_RETURN(
      Tiling tiling,
      ir_emitter_triton_internal::TilingFromAnnotatedFusion(
          fusion, symbolic_tile_analysis, block_level_parameters));

  TF_ASSIGN_OR_RETURN(
      auto triton_module,
      EmitXTileModule(fn_name, fusion, symbolic_tile_analysis, tiling,
                      mlir_context, absl::MakeSpan(opaque_args_types)));

  const auto debug_options = fusion->GetModule()->config().debug_options();

  if (DumpingEnabledForHloModule(*hlo_computation->parent()) &&
      DumpingEnabledForEmitter("triton-fusion", debug_options)) {
    auto suffix = absl::StrCat(fusion->name(), ".before_validation.ttir.txt");
    DumpToFileInDirOrStdout(*hlo_computation->parent(), "", suffix,
                            GetModuleIrString(triton_module.get()));
    std::string fusion_suffix = absl::StrCat(fusion->name(), ".hlo");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", fusion_suffix,
        ExtractInstructionIntoNewModule(*fusion)->ToString());
  }

  TF_RETURN_IF_ERROR(ir_emitter_triton_internal::LowerXTileToTriton(
      triton_module.get(), mlir_context, *fusion, device_info,
      block_level_parameters));

  VLOG(6) << GetModuleIrString(triton_module.get());
  if (DumpingEnabledForHloModule(*hlo_computation->parent()) &&
      DumpingEnabledForEmitter("triton-fusion", debug_options)) {
    std::string suffix = absl::StrCat(fusion->name(), ".ttir.txt");
    DumpToFileInDirOrStdout(*hlo_computation->parent(), "", suffix,
                            GetModuleIrString(triton_module.get()));
  }

  return std::move(triton_module);
}

absl::Status CheckAtLeastAmpere(const se::GpuComputeCapability& gpu_cc) {
  if (auto* cuda_cc = gpu_cc.cuda_compute_capability();
      cuda_cc != nullptr && !cuda_cc->IsAtLeastAmpere()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Triton support is only enabled for Ampere GPUs (compute ",
                     "capability 8.0) and up, but got compute capability ",
                     cuda_cc->ToString(), "."));
  }
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const TritonWrapperResult& result) {
  os << "\nTritonWrapperResult: " << "\n";
  os << "  shmem_bytes: " << result.shmem_bytes << "\n";
  auto tma_metadata = result.tma_metadata.ToProto();
  os << "  tma_metadata: {\n";
  for (const auto& tma_entry : tma_metadata.arg_index_to_tma_info()) {
    os << "    " << tma_entry.first << " : " << tma_entry.second.DebugString()
       << "\n";
  }
  os << "  }\n";
  os << "  thread_dims: " << result.thread_dims.ToString() << "\n";
  os << "  nvvm_annotations: " << result.nvvm_annotations.size() << "\n";
  os << "  llvm_module: " << result.llvm_module->getName().str() << "\n";
  return os;
}

absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& gpu_cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    llvm::LLVMContext& llvm_context, MLIRContext& mlir_context) {
  TF_RETURN_IF_ERROR(CheckAtLeastAmpere(gpu_cc));

  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> triton_module,
                      CreateTritonModule(fn_name, fusion, device_info,
                                         block_level_parameters, mlir_context));

  VLOG(3) << fusion->ToString(HloPrintOptions::ShortParsable());
  VLOG(3) << fusion->fused_instructions_computation()->ToString(
      HloPrintOptions::ShortParsable());

  // Compile Triton kernel to LLVM.
  const HloModule* hlo_module = fusion->GetModule();
  return CompileTritonToLLVM(fn_name, *hlo_module, device_info,
                             block_level_parameters, triton_module.get(),
                             target_triple, data_layout, llvm_context,
                             mlir_context,
                             /*is_xla_fusion=*/true);
}

absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, const llvm::Triple& target_triple,
    const std::string& data_layout, llvm::LLVMContext& llvm_context,
    mlir::MLIRContext& mlir_context, bool is_xla_fusion, bool emit_kernel) {
  const auto& gpu_cc = device_info.gpu_compute_capability();
  TF_RETURN_IF_ERROR(CheckAtLeastAmpere(gpu_cc));
  std::string arch_name = gpu_cc.ToString();

  const HloModuleConfig& hlo_config = hlo_module.config();

  bool should_verify =
      (hlo_config.debug_options().xla_gpu_llvm_verification_level() >= 1);
#ifndef NDEBUG
  should_verify = true;
#endif

  mlir::PassManager pm(&mlir_context);
  EnableIRPrintingIfRequested(pm, &mlir_context, hlo_module, kernel_name,
                              "triton-to-llvm");
  pm.enableVerifier(should_verify);
  CreateTritonXlaPipeline(
      &pm, gpu_cc, /*rewrite_int4=*/is_xla_fusion,
      block_level_parameters.is_tma_allowed, block_level_parameters.num_stages,
      block_level_parameters.is_warp_specialization_allowed);

  int num_warps = block_level_parameters.num_warps;
  int num_ctas = block_level_parameters.num_ctas;
  int num_stages = block_level_parameters.num_stages;
  if (num_warps <= 0 || num_ctas <= 0 || num_stages <= 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "(num_warps, num_ctas, num_stages) must be positive, but got: (",
        num_warps, ", ", num_ctas, ", ", num_stages, ")"));
  }
  CreateTritonPipeline(&pm, gpu_cc, num_warps, num_ctas, num_stages);

  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(mlir::triton::xla::CreateGeneralizeKernelSignaturePass());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  if (failed(pm.run(triton_module))) {
    return Internal("Failed to compile Triton kernel.");
  }

  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.shared").getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin()) {
    return absl::ResourceExhaustedError(absl::StrFormat(
        "Shared memory size limit exceeded: requested %d, available: %d",
        shared_mem_bytes, device_info.shared_memory_per_block_optin()));
  }

  if (auto* cuda_cc = gpu_cc.cuda_compute_capability();
      cuda_cc != nullptr && cuda_cc->IsBlackwell()) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory
    constexpr int kTensorMemoryColumns = 512;
    const int tensor_mem_columns =
        triton_module
            ->getAttrOfType<mlir::IntegerAttr>("ttg.tensor_memory_size")
            .getInt();
    if (tensor_mem_columns > 0) {
      VLOG(2) << "Tensor memory usage: " << tensor_mem_columns << " columns";
    }
    if (tensor_mem_columns > kTensorMemoryColumns) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Tensor memory size limit exceeded: requested %d, available: %d",
          tensor_mem_columns, kTensorMemoryColumns));
    }
  }

  std::vector<llvm::Metadata*> captured_nvvm_annotations;
  std::unique_ptr<llvm::Module> ll_triton_module;
  if (emit_kernel) {
    TF_ASSIGN_OR_RETURN(ll_triton_module,
                        TranslateLLVMToLLVMIR(&llvm_context, triton_module));

    XLA_VLOG_LINES(5, llvm_ir::DumpToString(ll_triton_module.get()));
    if (should_verify) {
      VerifyModule(*ll_triton_module);
    }

    // Integrate LLVM matmul kernel into XLA's LLVM module.
    captured_nvvm_annotations =
        xgt::ExtractNvvmAnnotations(ll_triton_module.get());
    ll_triton_module->setDataLayout(data_layout);
    ll_triton_module->setTargetTriple(target_triple);
    // Use override flag because libdevice functions can be present in both.
    XLA_VLOG_LINES(5, llvm_ir::DumpToString(ll_triton_module.get()));
    if (should_verify) {
      VerifyModule(*ll_triton_module);
    }
  }

  SmallVector<mlir::LLVM::LLVMFuncOp> func_ops;
  for (auto func : triton_module.getOps<mlir::LLVM::LLVMFuncOp>()) {
    // Custom calls will also match to LLVMFuncOp, so we are only interested in
    // the entry function.
    if (func.getName().str() == kernel_name) {
      func_ops.push_back(func);
    }
  }
  CHECK_EQ(func_ops.size(), 1)
      << "Expected a single LLVMFuncOp in the module for the entry function.";
  mlir::LLVM::LLVMFuncOp func_op = func_ops[0];

  TF_ASSIGN_OR_RETURN(se::ThreadDim thread_dims,
                      xgt::ExtractThreadDims(triton_module, func_op));
  TF_ASSIGN_OR_RETURN(stream_executor::gpu::TmaMetadata tma_metadata,
                      xgt::ExtractTmaMetadata(func_op));

  // Propagate the following extracted information from the Triton module:
  // - TMA metadata.
  // - Total threads per block. Computed from module attributes.
  // - Captured NVVM annotations.
  TritonWrapperResult result = {shared_mem_bytes, tma_metadata, thread_dims,
                                captured_nvvm_annotations,
                                std::move(ll_triton_module)};
  return result;
}

std::string GetLibdevicePath(const HloModuleConfig& hlo_config,
                             const se::DeviceDescription& device_info) {
  if (device_info.gpu_compute_capability().IsCuda()) {
    return nvptx::LibDevicePath(
        hlo_config.debug_options().xla_gpu_cuda_data_dir());
  }
  return "";
}

namespace ir_emitter_triton_internal {

namespace {
absl::StatusOr<absl::InlinedVector<int64_t, 4>> DotTilingParameters(
    const HloInstruction* hlo,
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters) {
  const HloInstruction* lhs = hlo->operand(0);
  // When encountering a `dot`, we always expect its operands to be nests.
  auto backend_config = lhs->backend_config<GpuBackendConfig>();
  if (!backend_config.ok() || !backend_config->fusion_backend_config()
                                   .has_block_level_fusion_config()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No block_level_fusion_config in ", lhs->ToString()));
  }
  std::vector<int64_t> lhs_output_tile_sizes =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          backend_config->fusion_backend_config().block_level_fusion_config())
          .output_tile_sizes.front();

  absl::InlinedVector<int64_t, 4> dot_tiling_parameters;
  dot_tiling_parameters.reserve(
      hlo->dot_dimension_numbers().lhs_contracting_dimensions().size());
  for (int64_t contracting_dim_id :
       hlo->dot_dimension_numbers().lhs_contracting_dimensions()) {
    if (contracting_dim_id >= lhs_output_tile_sizes.size()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Output tile sizes index ", contracting_dim_id,
                       " is out of bounds for ", lhs->ToString()));
    }
    dot_tiling_parameters.push_back(lhs_output_tile_sizes[contracting_dim_id]);
  }
  return dot_tiling_parameters;
}
}  // namespace

absl::StatusOr<Tiling> TilingFromAnnotatedFusion(
    const HloFusionInstruction* fusion,
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters) {
  Tiling::TileMapping tile_mapping;
  int64_t real_root_index = symbolic_tile_analysis.real_root_index();
  const HloInstruction* real_root =
      symbolic_tile_analysis.GetRoots()[real_root_index];

  for (const auto& [hlo, num_tiling_parameters] :
       symbolic_tile_analysis.GetTilingSpecification().parameter_mapping()) {
    // TODO(b/419026602): handle reductions.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kScaledDot) {
      ASSIGN_OR_RETURN(tile_mapping[hlo],
                       DotTilingParameters(hlo, symbolic_tile_analysis,
                                           block_level_parameters));
    }

    // TODO(b/390559452): this should change for generalized multi-output
    // fusions.
    if (hlo == real_root) {
      if (real_root_index >= block_level_parameters.output_tile_sizes.size()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Output tile sizes index ", real_root_index,
            " is out of bounds for block level fusion config: ",
            block_level_parameters.ToBlockLevelFusionConfig().DebugString()));
      }
      absl::Span<const int64_t> output_tile_sizes =
          block_level_parameters.output_tile_sizes[real_root_index];
      tile_mapping[hlo].insert(tile_mapping[hlo].end(),
                               output_tile_sizes.begin(),
                               output_tile_sizes.end());
    }
  }

  return Tiling(std::move(tile_mapping));
}

absl::Status LowerXTileToTriton(
    mlir::ModuleOp xtile_dialect_module, mlir::MLIRContext& mlir_context,
    const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters) {
  {
    const HloModule& hlo_module = *fusion.GetModule();
    // Convert xTile ops to Triton ops.
    mlir::PassManager pm(&mlir_context);
    EnableIRPrintingIfRequested(pm, &mlir_context, hlo_module, fusion.name(),
                                "xtile-to-triton");
    // Disable verifier because the Triton code may be invalid due to the
    // unsupported types.
    pm.enableVerifier(/*enabled=*/false);
    pm.addPass(mlir::triton::xla::CreateTensorLowerToTritonPass());
    pm.addPass(mlir::triton::xla::CreateStableHLOLowerToTritonPass(
        block_level_parameters.is_warp_specialization_allowed));
    pm.addPass(xtile::createStablehloLowerToArithPass());
    pm.addPass(xtile::createStablehloLowerToXtilePass());
    pm.addPass(xtile::createConvertElementwise0DTensorToScalarPass());
    pm.addPass(mlir::triton::xla::CreateArithFP8ConversionToTritonPass());
    pm.addPass(mlir::triton::xla::CreateXTileLowerToTritonPass());

    std::string libdevice_path =
        GetLibdevicePath(fusion.GetModule()->config(), device_info);
    absl::string_view triple = device_info.gpu_compute_capability().IsRocm()
                                   ? "amdgcn-unknown-unknown"
                                   : "nvptx64-unknown-unknown";
    pm.addPass(mlir::triton::xla::CreateTritonXLAMathToLibdevicePass(
        libdevice_path, triple));

    tsl::StatusScopedDiagnosticHandler diagnostic_handler(&mlir_context);
    if (absl::Status status =
            diagnostic_handler.consumeStatus(pm.run(xtile_dialect_module));
        !status.ok()) {
      return CreateInternalError(
          "Failed to lower from shared dialect to Triton.", &fusion,
          xtile_dialect_module);
    }
  }

  {
    if (fusion.GetModule()
            ->config()
            .debug_options()
            .xla_gpu_experimental_scaled_dot_with_triton()) {
      // Convert unsupported types before verification.
      mlir::PassManager pm(&mlir_context);

      EnableIRPrintingIfRequested(pm, &mlir_context, *fusion.GetModule(),
                                  fusion.name(),
                                  "convert-scaled-dot-unsupported-types");
      pm.addPass(
          mlir::triton::xla::CreateTritonXLAConvertUnsupportedTypesPass());
      if (mlir::failed(pm.run(xtile_dialect_module))) {
        return CreateInternalError(
            "Failed to fix unsupported types in Triton module for fusion:",
            &fusion, xtile_dialect_module);
      }
    }

    if (mlir::failed(mlir::verify(xtile_dialect_module))) {
      return CreateInternalError("Failed to verify Triton module for fusion:",
                                 &fusion, xtile_dialect_module);
    }
    mlir::PassManager pm(&mlir_context);
    EnableIRPrintingIfRequested(pm, &mlir_context, *fusion.GetModule(),
                                fusion.name(), "canonicalize-cse");
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (mlir::failed(pm.run(xtile_dialect_module))) {
      return CreateInternalError("Failed to create Triton module for fusion:",
                                 &fusion, xtile_dialect_module);
    }
  }
  return absl::OkStatus();
}

}  // namespace ir_emitter_triton_internal

}  // namespace gpu
}  // namespace xla
