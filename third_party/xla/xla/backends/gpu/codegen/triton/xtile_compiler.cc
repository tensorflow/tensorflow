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

#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
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
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/lowering_util.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
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
#include "xla/status_macros.h"
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
#include "tsl/platform/path.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla {
namespace gpu {

namespace ttir = ::mlir::triton;
namespace stablehlo = ::mlir::stablehlo;
namespace xgt = ::xla::gpu::triton;

using ::llvm::SmallVector;
using ::mlir::MLIRContext;

using ::xla::gpu::ir_emitter_triton_internal::DumpTritonIR;

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

  // TODO: b/451959933 - Use reference or check pointer.

  TF_ASSIGN_OR_RETURN(
      auto triton_module,
      EmitXTileModule(fn_name,
                      TritonEmitterConstraints::GetBuilder(device_info), fusion,
                      block_level_parameters, mlir_context));

  const HloComputation* hlo_computation =
      fusion->fused_instructions_computation();

  const auto debug_options = fusion->GetModule()->config().debug_options();

  if (DumpingEnabledForHloModule(*hlo_computation->parent()) &&
      DumpingEnabledForEmitter("triton-fusion", debug_options)) {
    auto suffix = absl::StrCat(fusion->name(), ".before_validation.ttir.txt");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", suffix,
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
    std::string fusion_suffix = absl::StrCat(fusion->name(), ".hlo");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", fusion_suffix,
        ExtractInstructionIntoNewModule(*fusion)->ToString());
  }

  TF_RETURN_IF_ERROR(ir_emitter_triton_internal::LowerXTileToTriton(
      triton_module.get(), mlir_context, *fusion, device_info));

  VLOG(6) << DumpTritonIR(triton_module.get(),
                          fusion->GetModule()
                              ->config()
                              .debug_options()
                              .xla_gpu_unsupported_annotate_with_emitter_loc());
  if (DumpingEnabledForHloModule(*hlo_computation->parent()) &&
      DumpingEnabledForEmitter("triton-fusion", debug_options)) {
    std::string suffix = absl::StrCat(fusion->name(), ".ttir.txt");
    DumpToFileInDirOrStdout(
        *hlo_computation->parent(), "", suffix,
        DumpTritonIR(triton_module.get(),
                     fusion->GetModule()
                         ->config()
                         .debug_options()
                         .xla_gpu_unsupported_annotate_with_emitter_loc()));
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

absl::StatusOr<TritonWrapperResult> TritonWrapper(
    absl::string_view fn_name, const HloFusionInstruction* fusion,
    const se::GpuComputeCapability& gpu_cc,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    llvm::Module* llvm_module, MLIRContext& mlir_context) {
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
                             llvm_module, mlir_context,
                             /*is_xla_fusion=*/true);
}

absl::StatusOr<TritonWrapperResult> CompileTritonToLLVM(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    mlir::ModuleOp triton_module, llvm::Module* llvm_module,
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

  bool should_dump_mlir_passes =
      hlo_config.debug_options().xla_enable_dumping() &&
      DumpingEnabledForHloModule(hlo_module) &&
      DumpingEnabledForEmitter("triton-fusion", hlo_config.debug_options());

  mlir::PassManager pm(&mlir_context);
  pm.enableVerifier(should_verify);

  std::optional<llvm::raw_fd_ostream> log_stream;
  if (should_dump_mlir_passes) {
    std::string outputs_dir = hlo_config.debug_options().xla_dump_to();
    if (outputs_dir == "sponge") {
      if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
        LOG(ERROR) << "Failed to get test undeclared outputs dir. Lets skip "
                      "dumping triton passes.";
        outputs_dir = "";
      }
    }
    if (!outputs_dir.empty()) {
      const std::string basename =
          absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module.name())),
                       ".", kernel_name, ".triton-passes.log");
      std::string path = tsl::io::JoinPath(outputs_dir, basename);
      std::error_code err;
      log_stream.emplace(path, err, llvm::sys::fs::OF_None);
      if (err) {
        log_stream.reset();
        LOG(ERROR) << "Failed to dump triton passes to " << path << ": "
                   << err.message();
      } else {
        pm.getContext()->disableMultithreading();
        auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
        pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                            /*shouldPrintAfterPass=*/print_always,
                            /*printModuleScope=*/true,
                            /*printAfterOnlyOnChange=*/false,
                            /*printAfterOnlyOnFailure=*/true, *log_stream);
      }
    } else {
      LOG(ERROR)
          << "--xla_dump_emitter_re=triton-fusion is set, but neither "
          << "the environment variable TEST_UNDECLARED_OUTPUTS_DIR nor the "
          << "flag --xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  CreateTritonXlaPipeline(&pm, gpu_cc, /*rewrite_int4=*/is_xla_fusion,
                          block_level_parameters.is_tma_allowed,
                          block_level_parameters.num_stages);

  int num_warps = block_level_parameters.num_warps;
  int num_ctas = block_level_parameters.num_ctas;
  int num_stages = block_level_parameters.num_stages;
  if (num_warps <= 0 || num_ctas <= 0 || num_stages <= 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "(num_warps, num_ctas, num_stages) must be positive, but got: (",
        num_warps, ", ", num_ctas, ", ", num_stages, ")"));
  }
  mlir::triton::nvidia_gpu::ClusterInfo cluster_info;
  CreateTritonPipeline(&pm, gpu_cc, num_warps, num_ctas, num_stages,
                       cluster_info);

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
  if (emit_kernel) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<llvm::Module> ll_triton_module,
        TranslateLLVMToLLVMIR(&llvm_module->getContext(), triton_module));

    XLA_VLOG_LINES(5, llvm_ir::DumpToString(ll_triton_module.get()));
    if (should_verify) {
      VerifyModule(*ll_triton_module);
    }

    // Integrate LLVM matmul kernel into XLA's LLVM module.
    captured_nvvm_annotations =
        xgt::ExtractNvvmAnnotations(ll_triton_module.get());
    ll_triton_module->setDataLayout(llvm_module->getDataLayout());
    ll_triton_module->setTargetTriple(llvm_module->getTargetTriple());
    // Use override flag because libdevice functions can be present in both.
    TF_RET_CHECK(
        !llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));

    XLA_VLOG_LINES(5, llvm_ir::DumpToString(llvm_module));
    if (should_verify) {
      VerifyModule(*llvm_module);
    }
  }

  // `cluster_info` must be read after pm.run().
  std::optional<se::ClusterDim> cluster_dim;
  if (block_level_parameters.num_ctas > 1) {
    VLOG(3) << "num_ctas: " << block_level_parameters.num_ctas
            << ", cluster_info: " << cluster_info.clusterDimX << ","
            << cluster_info.clusterDimY << "," << cluster_info.clusterDimZ;
    if (cluster_info.clusterDimX > 1 || cluster_info.clusterDimY > 1 ||
        cluster_info.clusterDimZ > 1) {
      cluster_dim =
          se::ClusterDim(cluster_info.clusterDimX, cluster_info.clusterDimY,
                         cluster_info.clusterDimZ);
    }
  } else {
    TF_RET_CHECK(cluster_info.clusterDimX == 1 &&
                 cluster_info.clusterDimY == 1 &&
                 cluster_info.clusterDimZ == 1);
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
  TritonWrapperResult result = {
      shared_mem_bytes,          cluster_dim, tma_metadata, thread_dims,
      captured_nvvm_annotations,
  };
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

absl::Status LowerXTileToTriton(mlir::ModuleOp xtile_dialect_module,
                                mlir::MLIRContext& mlir_context,
                                const HloFusionInstruction& fusion,
                                const se::DeviceDescription& device_info) {
  {
    // Convert xTile ops to Triton ops.
    mlir::PassManager pm(&mlir_context);
    // Disable verifier because the Triton code may be invalid due to the
    // unsupported types.
    pm.enableVerifier(/*enabled=*/false);
    pm.addPass(xtile::createConvertElementwise0DTensorToScalarPass());
    pm.addPass(mlir::triton::xla::CreateArithFP8ConversionToTritonPass());
    pm.addPass(mlir::triton::xla::CreateTensorLowerToTritonPass());
    pm.addPass(mlir::triton::xla::CreateStableHLOLowerToTritonPass());
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

  if (fusion.GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_scaled_dot_with_triton()) {
    // Convert unsupported types before verification.
    mlir::PassManager pm(&mlir_context);
    pm.addPass(mlir::triton::xla::CreateTritonXLAConvertUnsupportedTypesPass());
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

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (mlir::failed(pm.run(xtile_dialect_module))) {
    return CreateInternalError("Failed to create Triton module for fusion:",
                               &fusion, xtile_dialect_module);
  }

  return absl::OkStatus();
}

}  // namespace ir_emitter_triton_internal

}  // namespace gpu
}  // namespace xla
