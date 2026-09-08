/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/stablehlo_tfl_pipeline.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/Timing.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/debug/debug.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/python/conversion_failure_reporter.h"
#include "tensorflow/compiler/mlir/lite/python/pass_debug_instrumentation.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/cast_bf16_ops_to_f32_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/large_constant_fold_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/tsl/platform/env.h"

namespace mlir::TFL {

void AddPipelinePasses(mlir::OpPassManager& pass_manager,
                       const mlir::TFL::PassConfig& pass_config) {
  // =========================================================================
  // 1. Skip-to-TFLite & Pre-Lowering Passes
  // =========================================================================
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateLegalizeChloToTflPass());
  // Inline private functions before lowering quant annotations to eliminate
  // func.call boundaries that cause func.call result type mismatches when
  // quantized types are introduced.
  pass_manager.addPass(mlir::createInlinerPass());
  pass_manager.addPass(mlir::TFL::CreateLowerQuantAnnotationsPass());
  pass_manager.addPass(mlir::createSymbolDCEPass());

  // =========================================================================
  // 2. HLO Optimization & Canonicalization Passes
  // =========================================================================
  // Drop shape assertion custom calls before VHLO legalization
  pass_manager.addPass(mlir::odml::CreateDropShapeAssertionsPass());

  // Legalize VHLO quant custom calls to StableHLO custom calls before VHLO
  // legalization
  pass_manager.addPass(mlir::odml::CreateLegalizeVhloQuantCustomCallsPass());

  // VHLO -> StableHLO
  pass_manager.addPass(mlir::stablehlo::createVhloLegalizeToStablehloPass());

  // Decompose newer StableHLO ops into equivalent older ops for baseline
  // compatibility with the TFLite legalization backend.
  pass_manager.addPass(
      mlir::stablehlo::createStablehloCompatibilityExpanderPass(
          {tflite_supported_stablehlo_version}));

  // CHLO -> StableHLO
  mlir::stablehlo_ext::createChloLegalizeToStablehloPipeline(pass_manager);

  pass_manager.addPass(mlir::odml::CreateTransposeCommuteOpsPass());

  // Uniform Quantization support
  pass_manager.addPass(mlir::odml::CreateComposeUniformQuantizedTypePass());

  // Jax Random legalization
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLegalizeJaxRandomPass());

  // Optimization and Canonicalization
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pass_manager.addPass(mlir::createSymbolDCEPass());
  pass_manager.addPass(mlir::createInlinerPass());

  // Import cleanup (Tuple flattening)
  // Defaulting to "main" for the entry function name for now.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass(
          {"main"}));

  // High-level StableHLO optimizations
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createStablehloUnfuseBatchNormPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createStablehloFuseConvolutionPass());

  // StableHLO -> MHLO bridge
  pass_manager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());

  // Composite lowering when IR is in MHLO
  pass_manager.addPass(mlir::odml::CreateCompositeLoweringPass());
  pass_manager.addPass(mlir::createSymbolDCEPass());

  // MHLO algebraic optimizations
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeEinsumToDotGeneralPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createOptimizePass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // =========================================================================
  // 3. HLO to TFLite Legalization Passes
  // =========================================================================
  // HLO -> TFLite legalization
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateUniformQuantizedStableHloToTflPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreatePrepareHloPass());
  // This pass must be added right before the legalization because pattern
  // rewriter driver applies folding by default.
  pass_manager.addPass(mlir::odml::CreateUnfoldSplatConstantPass());
  pass_manager.addPass(mlir::odml::CreateLegalizeHloToTfLitePass());

  // Legalize remaining MHLO ops to StableHLO
  pass_manager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createLegalizeCompositeToCustomOpPass());

  // =========================================================================
  // 4. TFLite Optimization & Quantization Passes
  // =========================================================================
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateCastBf16OpsToF32Pass());

  // Final TFLite optimizations
  pass_manager.addPass(mlir::TFL::CreatePushTransposeThroughEwisePass());
  {
    mlir::TFL::OptimizeBroadcastLikePassOptions options;
    options.unsafe_fuse_dynamic_shaped_broadcast =
        pass_config.unsafe_fuse_dynamic_shaped_broadcast;
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::Create<mlir::TFL::OptimizeBroadcastLikePass>(options));
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizePass());

  if (!pass_config.unfold_batch_matmul) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizePass());
  }

  // Quantization
  pass_manager.addPass(mlir::TFL::CreatePropagateQParamsPass());
  pass_manager.addPass(mlir::TFL::CreateBiasQuantizerPass());
  pass_manager.addPass(mlir::TFL::CreateFuseQDQPass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(/*emit_quant_adaptor_ops=*/true));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());

  // Some optimizations need to happen on the quantized graph.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizePass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Fold operations on Large DenseResourceElementsAttr constants (Cast, Add,
  // Transpose, Reshape).
  pass_manager.addPass(mlir::TFL::CreateLargeConstantFoldPass(
      pass_config.fold_fp16_resource_casts));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pass_manager.addPass(mlir::TFL::CreateCleanupOptimizationBarrierPass());
  pass_manager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

static absl::Status VerifyInputModule(mlir::ModuleOp module,
                                      absl::string_view debug_dir,
                                      bool enable_debug,
                                      int64_t elide_elements_larger_than = 8) {
  mlir::MLIRContext* context = module->getContext();
  mlir::StatusScopedDiagnosticHandler initial_status_handler(
      context,
      /*propagate=*/false);
  bool verification_failed = mlir::failed(module.verify());
  absl::Status initial_status = initial_status_handler.ConsumeStatus();
  if (verification_failed) {
    std::string err_msg =
        initial_status.ok()
            ? "Input MLIR module verification failed."
            : absl::StrCat("Input MLIR module verification failed: ",
                           initial_status.message());
    std::string status_code =
        initial_status.ok() ? "INVALID_ARGUMENT"
                            : absl::StatusCodeToString(initial_status.code());
    ConversionFailureReporter::WriteFailureJson(
        debug_dir, module, err_msg, "JAX_Export_Module_Verification",
        status_code, /*failing_pass=*/"", /*failing_pass_arg=*/"",
        /*write_module_artifacts=*/enable_debug, /*failing_function=*/"",
        elide_elements_larger_than);
    return absl::InvalidArgumentError(err_msg);
  }
  return absl::OkStatus();
}

struct PassTimingSession {
  std::unique_ptr<std::ofstream> file_stream;
  std::unique_ptr<llvm::raw_os_ostream> timing_stream;
};

static PassTimingSession CreatePassTimingSession(absl::string_view debug_dir) {
  PassTimingSession session;
  if (!tsl::Env::Default()->RecursivelyCreateDir(std::string(debug_dir)).ok()) {
    return session;
  }
  std::string main_path = absl::StrCat(debug_dir, "/mlir_pass_timing.log");
  auto file = std::make_unique<std::ofstream>(main_path);
  if (file->is_open()) {
    session.timing_stream = std::make_unique<llvm::raw_os_ostream>(*file);
    session.file_stream = std::move(file);
  }
  return session;
}

static void AttachPassTiming(mlir::PassManager& pm,
                             PassTimingSession& session) {
  auto timing_manager = std::make_unique<mlir::DefaultTimingManager>();
  timing_manager->setEnabled(true);
  timing_manager->setDisplayMode(mlir::DefaultTimingManager::DisplayMode::List);

  if (session.timing_stream) {
    timing_manager->setOutput(mlir::createOutputStrategy(
        mlir::DefaultTimingManager::OutputFormat::Text,
        *session.timing_stream));
  }

  pm.enableTiming(std::move(timing_manager));
}

absl::Status ConvertStableHloToTFLite(
    mlir::ModuleOp module, const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    llvm::raw_pwrite_stream& export_stream) {
  mlir::MLIRContext* context = module->getContext();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  registry.insert<mlir::TFL::TensorFlowLiteDialect, mlir::mhlo::MhloDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect,
                  mlir::quant::QuantDialect,
                  mlir::quantfork::QuantizationForkDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  bool enable_debug = converter_flags.enable_debug();
  std::string debug_dir = ConversionFailureReporter::GetOrCreateDebugDir(
      converter_flags.debug_dir());

  int64_t elide_elements_larger_than =
      converter_flags.debug_options().has_elide_elementsattrs_if_larger()
          ? converter_flags.debug_options().elide_elementsattrs_if_larger()
          : 8;

  if (auto status = VerifyInputModule(module, debug_dir, enable_debug,
                                      elide_elements_larger_than);
      !status.ok()) {
    return status;
  }

  mlir::PassManager pm(context);
  PassTimingSession timing_session;
  if (enable_debug) {
    timing_session = CreatePassTimingSession(debug_dir);
    AttachPassTiming(pm, timing_session);
  }

  tensorflow::converter::DebugOptions debug_options =
      converter_flags.debug_options();
  if (enable_debug) {
    pm.getContext()->disableMultithreading();
    debug_options.clear_print_ir_before();
    debug_options.clear_print_ir_after();
  }
  tensorflow::InitPassManager(pm, debug_options, llvm::nulls());

  AddPipelinePasses(pm, pass_config);

  PipelineFailureCoordinator failure_coordinator(debug_dir, enable_debug,
                                                 elide_elements_larger_than);
  pm.addInstrumentation(failure_coordinator.CreateInstrumentation(
      converter_flags.debug_options().print_ir_before(),
      converter_flags.debug_options().print_ir_after()));

  mlir::StatusScopedDiagnosticHandler status_handler(context,
                                                     /*propagate=*/false);
  bool pass_failed = mlir::failed(pm.run(module));
  absl::Status pass_status = status_handler.ConsumeStatus();

  if (timing_session.timing_stream) {
    timing_session.timing_stream->flush();
  }
  if (timing_session.file_stream) {
    timing_session.file_stream->flush();
  }

  if (pass_failed) {
    return failure_coordinator.ReportFailure(module, pass_status);
  }

  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_allow_custom_ops(true);
  options.converter_flags.set_use_buffer_offset(true);

  std::string diag_errors;
  SerializationDiagHandler diag_handler(module.getContext(), &diag_errors);

  auto status =
      tflite::MlirToFlatBufferTranslateFunction(module, options, export_stream);

  if (!status.ok()) {
    return failure_coordinator.ReportSerializationFailure(module, status,
                                                          diag_errors);
  }

  return absl::OkStatus();
}

}  // namespace mlir::TFL
