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
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/Timing.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/debug/debug.h"
#include "tensorflow/compiler/mlir/lite/export/export.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/metrics/error_collector_inst.h"
#include "tensorflow/compiler/mlir/lite/python/conversion_failure_reporter.h"
#include "tensorflow/compiler/mlir/lite/python/pass_debug_instrumentation.h"
#include "tensorflow/compiler/mlir/lite/python/preserve_regions.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/cast_bf16_ops_to_f32_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/downcast_x64_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/large_constant_fold_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"

namespace mlir::TFL {
namespace {

// Adds the keys of the resource blobs `module` references to `keys`, including
// resources nested in array or dictionary attributes (e.g. composite
// attributes).
void CollectResourceKeys(mlir::ModuleOp module, llvm::StringSet<>& keys) {
  module.walk([&](mlir::Operation* op) {
    op->getAttrDictionary().walk([&](mlir::DenseResourceElementsAttr attr) {
      keys.insert(attr.getRawHandle().getKey());
    });
  });
}

class PruneDeadResourcesPass
    : public mlir::PassWrapper<PruneDeadResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruneDeadResourcesPass)

  // `extra_live_keys` are kept in addition to the keys the module references,
  // for blobs that other modules in the same context still use.
  explicit PruneDeadResourcesPass(
      std::shared_ptr<const llvm::StringSet<>> extra_live_keys = nullptr)
      : extra_live_keys_(std::move(extra_live_keys)) {}

  llvm::StringRef getArgument() const final {
    return "tfl-prune-dead-resources";
  }
  llvm::StringRef getDescription() const final {
    return "Prunes unreferenced resource blobs from the "
           "DialectResourceBlobManager to release memory.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = module.getContext();

    auto* builtin_dialect = context->getLoadedDialect<mlir::BuiltinDialect>();
    if (!builtin_dialect) return;

    auto* interface = builtin_dialect->getRegisteredInterface<
        mlir::ResourceBlobManagerDialectInterface>();
    if (!interface) return;

    // Collect all referenced resource keys in the module.
    llvm::StringSet<> live_resource_keys;
    CollectResourceKeys(module, live_resource_keys);
    auto is_live = [&](llvm::StringRef key) {
      return live_resource_keys.contains(key) ||
             (extra_live_keys_ && extra_live_keys_->contains(key));
    };

    // Prune unreferenced entries from the blob manager.
    mlir::DialectResourceBlobManager& blob_mgr = interface->getBlobManager();
    blob_mgr.getBlobMap(
        [&](const llvm::StringMap<mlir::DialectResourceBlobManager::BlobEntry>&
                map) {
          for (auto& entry_pair : map) {
            if (!is_live(entry_pair.first())) {
              const_cast<mlir::DialectResourceBlobManager::BlobEntry&>(
                  entry_pair.second)
                  .setBlob(mlir::AsmResourceBlob());
            }
          }
        });
  }

 private:
  std::shared_ptr<const llvm::StringSet<>> extra_live_keys_;
};

std::unique_ptr<mlir::Pass> CreatePruneDeadResourcesPass(
    std::shared_ptr<const llvm::StringSet<>> extra_live_keys = nullptr) {
  return std::make_unique<PruneDeadResourcesPass>(std::move(extra_live_keys));
}

// Adds the passes that lower the input module to TFL (stages 1-3).
void AddLoweringPasses(mlir::OpPassManager& pass_manager) {
  // =========================================================================
  // 1. Skip-to-TFLite & Pre-Lowering Passes
  // =========================================================================
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateLegalizeChloToTflPass());
  // Inline private functions before lowering quant annotations to eliminate
  // func.call boundaries that cause func.call result type mismatches when
  // quantized types are introduced.
  pass_manager.addPass(mlir::createInlinerPass());
  pass_manager.addPass(mlir::TFL::CreateFoldStablehloConstantTransformsPass());
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

  // VHLO -> StableHLO, upgrading VHLO to the current version first.
  mlir::stablehlo::createStablehloDeserializePipeline(pass_manager);

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
  // Unfusing training batch norms emits TF ops (mean and variance), which the
  // flatbuffer exporter doesn't support.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLegalizeTFPass(/*run_tfl_runtime_verification=*/false));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createStablehloFuseConvolutionPass());

  // Build StableHLO composite from PyTorch mark_tensor ops before MHLO bridge
  pass_manager.addPass(mlir::odml::createBuildStableHLOCompositePass());
  pass_manager.addPass(mlir::createInlinerPass());

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
  // TODO(markoristic): Temporary. PrepareHlo and the HLO->TFL legalization
  // consume StableHLO while the preceding passes still produce MHLO. Remove
  // once the whole pipeline runs on StableHLO.
  pass_manager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  // Fold and simplify StableHLO into the canonical form the legalization
  // expects (MHLO did this through its folders and canonicalizations).
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateSimplifyHloPass());
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
}

// Adds the TFL optimization and quantization passes (stage 4), up to but not
// including `AddFinalizePasses`.
//
// Pruning only sees the module it runs on; `extra_live_resource_keys` names
// the blobs that other modules in the same context (the preserve-region
// bodies) still reference, so that they are not released.
void AddOptimizationPasses(mlir::OpPassManager& pass_manager,
                           const mlir::TFL::PassConfig& pass_config,
                           const tflite::ConverterFlags& converter_flags,
                           std::shared_ptr<const llvm::StringSet<>>
                               extra_live_resource_keys = nullptr) {
  // =========================================================================
  // 4. TFLite Optimization & Quantization Passes
  // =========================================================================
  pass_manager.addPass(mlir::TFL::CreateLargeConstantFoldPass(
      /*fold_fp16_resource_casts=*/false, /*fold_elementwise_ops=*/true));
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
    pass_manager.addPass(mlir::TFL::CreateLargeConstantFoldPass(
        /*fold_fp16_resource_casts=*/false, /*fold_elementwise_ops=*/true));
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
    pass_manager.addPass(mlir::TFL::CreateLargeConstantFoldPass(
        /*fold_fp16_resource_casts=*/false, /*fold_elementwise_ops=*/true));
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
  pass_manager.addPass(CreatePruneDeadResourcesPass(extra_live_resource_keys));

  // Some optimizations need to happen on the quantized graph.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizePass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Fold operations on Large DenseResourceElementsAttr constants.
  pass_manager.addPass(mlir::TFL::CreateLargeConstantFoldPass(
      /*fold_fp16_resource_casts=*/false, /*fold_elementwise_ops=*/true));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pass_manager.addPass(CreatePruneDeadResourcesPass(extra_live_resource_keys));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateFuseA4W2DRQFullyConnectedPass());
  // Blockwise dequantizes of constants that no fusion above claimed (e.g.
  // weight-only blockwise quantization) move their parameters onto the tensor
  // type so that they can be exported.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLowerBlockwiseDequantizePass());
  if (pass_config.fold_fp16_casts_into_fully_connected) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateFoldFp16CastsIntoFullyConnectedPass());
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addPass(mlir::createSymbolDCEPass());
  pass_manager.addPass(mlir::TFL::CreateCleanupOptimizationBarrierPass());
  if (!converter_flags.enable_x64()) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateDowncastX64Pass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
  }
}

// Adds the fixed pipeline run on preserve-region bodies in place of
// `AddOptimizationPasses`.
//
// It legalizes what the flatbuffer cannot represent otherwise and fuses
// authored Q/DQ ops into quantized kernels, without propagating quantization
// parameters to ops the author left in float. It has no
// canonicalizer, CSE, constant folding pass or TFL optimization patterns,
// which would rewrite authored ops (e.g. merge adjacent reshapes or fuse
// activations). The quantization passes do run op folders through their greedy
// drivers, so identity ops and ops whose operands are all constants can still
// fold. Fusions that the main pipeline runs after quantization (e.g.
// FuseA4W2DRQFullyConnected, FoldFp16CastsIntoFullyConnected) do not apply in
// regions.
//
// The region boundary must end up with the same types on both sides when the
// bodies are stitched back. That holds because `CastBf16OpsToF32` leaves
// `stablehlo.composite` ops and function signatures alone in both pipelines,
// and `DowncastX64` narrows both function signatures and all op results,
// composites included, in both pipelines.
void AddPreserveRegionPasses(mlir::OpPassManager& pass_manager,
                             const tflite::ConverterFlags& converter_flags) {
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateCastBf16OpsToF32Pass());

  // Quantization, so that authored Q/DQ ops fuse into quantized kernels.
  // `PropagateQParams` is left out on purpose: it would wrap float ops next
  // to a quantized value in new Q/DQ pairs and rewrite authored Q/DQ chains
  // into requantize ops, so it would quantize more than the author did. Only
  // ops whose operands and results the author quantized are fused.
  // `BiasQuantizer` only fires on ops whose non-bias operands the author
  // already quantized, and gives them the int32 bias their kernel needs.
  pass_manager.addPass(mlir::TFL::CreateBiasQuantizerPass());
  pass_manager.addPass(mlir::TFL::CreateFuseQDQPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(/*emit_quant_adaptor_ops=*/true));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLowerBlockwiseDequantizePass());

  pass_manager.addPass(mlir::TFL::CreateCleanupOptimizationBarrierPass());
  if (!converter_flags.enable_x64()) {
    // Unlike the main pipeline this is not followed by a canonicalizer, so a
    // 64-bit float constant in a region keeps the `tfl.cast` this inserts.
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateDowncastX64Pass());
  }
}

// Adds the passes that run on the whole module right before export, after the
// preserve-region bodies have been stitched back.
void AddFinalizePasses(mlir::OpPassManager& pass_manager) {
  pass_manager.addPass(mlir::odml::createLegalizeStablehloToVhloPass());
  pass_manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  pass_manager.addPass(CreatePruneDeadResourcesPass());
}

}  // namespace

static absl::Status VerifyInputModule(mlir::ModuleOp module,
                                      absl::string_view debug_dir,
                                      bool enable_debug,
                                      int64_t elide_elements_larger_than = 8) {
  mlir::MLIRContext* context = module->getContext();
  mlir::BaseScopedDiagnosticHandler initial_status_handler(context,
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
  std::unique_ptr<std::string> timing_buffer;
  std::unique_ptr<llvm::raw_string_ostream> timing_stream;
  std::string debug_dir;
};

static PassTimingSession CreatePassTimingSession(absl::string_view debug_dir) {
  PassTimingSession session;
  session.debug_dir = std::string(debug_dir);
  session.timing_buffer = std::make_unique<std::string>();
  session.timing_stream =
      std::make_unique<llvm::raw_string_ostream>(*session.timing_buffer);
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

  PassTimingSession timing_session;
  if (enable_debug) {
    timing_session = CreatePassTimingSession(debug_dir);
  }

  PipelineFailureCoordinator failure_coordinator(debug_dir, enable_debug,
                                                 elide_elements_larger_than);

  // Runs the passes `add_passes` adds on `target`, with the converter's
  // instrumentation. The pipeline runs in several pieces so that preserve
  // regions can be split off after lowering; each piece gets the same setup.
  // Debug artifacts are therefore per piece: IR dump step numbers restart and
  // `ir_dump_dir` gets one timestamped directory per pass manager.
  auto run_passes =
      [&](mlir::ModuleOp target,
          absl::FunctionRef<void(mlir::OpPassManager&)> add_passes)
      -> absl::Status {
    mlir::PassManager pm(context);

    if (enable_debug) {
      pm.getContext()->disableMultithreading();
      pm.addInstrumentation(
          std::make_unique<mlir::TFL::ErrorCollectorInstrumentation>(
              pm.getContext()));
      AttachPassTiming(pm, timing_session);
      pm.addInstrumentation(failure_coordinator.CreateInstrumentation(
          converter_flags.debug_options().print_ir_before(),
          converter_flags.debug_options().print_ir_after()));
    } else {
      tensorflow::InitPassManager(pm, converter_flags.debug_options());
    }

    add_passes(pm);

    mlir::BaseScopedDiagnosticHandler status_handler(context,
                                                     /*propagate=*/false);
    if (mlir::succeeded(pm.run(target))) return absl::OkStatus();
    return failure_coordinator.ReportFailure(target,
                                             status_handler.ConsumeStatus());
  };

  // Runs a step outside of a pass manager, turning its diagnostics into the
  // returned status.
  auto run_step =
      [&](absl::FunctionRef<mlir::LogicalResult()> step) -> absl::Status {
    mlir::BaseScopedDiagnosticHandler status_handler(context,
                                                     /*propagate=*/false);
    if (mlir::succeeded(step())) return absl::OkStatus();
    return failure_coordinator.ReportFailure(module,
                                             status_handler.ConsumeStatus());
  };

  auto run_pipeline = [&]() -> absl::Status {
    TF_RETURN_IF_ERROR(run_passes(module, AddLoweringPasses));

    if (!HasPreserveRegions(module)) {
      return run_passes(module, [&](mlir::OpPassManager& pm) {
        AddOptimizationPasses(pm, pass_config, converter_flags);
        AddFinalizePasses(pm);
      });
    }

    // Preserve regions: optimize the main module and the region bodies with
    // different pipelines, in separate modules, then stitch them back.
    mlir::OwningOpRef<mlir::ModuleOp> regions;
    TF_RETURN_IF_ERROR(run_step([&]() -> mlir::LogicalResult {
      auto split = SplitPreserveRegions(module);
      if (mlir::failed(split)) return mlir::failure();
      regions = std::move(*split);
      return mlir::success();
    }));
    TF_RETURN_IF_ERROR(run_passes(*regions, [&](mlir::OpPassManager& pm) {
      AddPreserveRegionPasses(pm, converter_flags);
    }));
    // The region bodies are final now. The blobs they reference must survive
    // the main module's pruning until the bodies are stitched back.
    auto region_resource_keys = std::make_shared<llvm::StringSet<>>();
    CollectResourceKeys(*regions, *region_resource_keys);
    TF_RETURN_IF_ERROR(run_passes(module, [&](mlir::OpPassManager& pm) {
      AddOptimizationPasses(pm, pass_config, converter_flags,
                            region_resource_keys);
    }));
    TF_RETURN_IF_ERROR(
        run_step([&] { return StitchPreserveRegions(module, *regions); }));
    return run_passes(module, AddFinalizePasses);
  };
  absl::Status pipeline_status = run_pipeline();

  if (timing_session.timing_stream) {
    timing_session.timing_stream->flush();
    if (!timing_session.debug_dir.empty() && timing_session.timing_buffer &&
        !timing_session.timing_buffer->empty()) {
      (void)tsl::Env::Default()->RecursivelyCreateDir(timing_session.debug_dir);
      std::string timing_file =
          absl::StrCat(timing_session.debug_dir, "/mlir_pass_timing.log");
      (void)tsl::WriteStringToFile(tsl::Env::Default(), timing_file,
                                   *timing_session.timing_buffer);
    }
  }

  if (!pipeline_status.ok()) {
    return pipeline_status;
  }

  std::string diag_errors;
  SerializationDiagHandler diag_handler(module.getContext(), &diag_errors);

  auto status = mlir::TFL::ExportFlatbuffer(module, export_stream);

  if (!status.ok()) {
    return failure_coordinator.ReportSerializationFailure(module, status,
                                                          diag_errors);
  }

  return absl::OkStatus();
}

}  // namespace mlir::TFL
