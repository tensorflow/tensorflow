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

#include <unistd.h>

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/cast_bf16_ops_to_f32_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/large_constant_fold_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"

namespace mlir::TFL {

void AddSkipToTflitePasses(mlir::OpPassManager& pass_manager) {
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateLegalizeChloToTflPass());
  pass_manager.addPass(mlir::odml::CreateCompositeLoweringPass());
  pass_manager.addPass(mlir::TFL::CreateLowerQuantAnnotationsPass());
  pass_manager.addPass(mlir::createSymbolDCEPass());
}

void AddHloOptimizationPasses(mlir::OpPassManager& pass_manager) {
  // Drop shape assertion custom calls before VHLO legalization
  pass_manager.addPass(mlir::odml::CreateDropShapeAssertionsPass());

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

  // MHLO algebraic optimizations
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeEinsumToDotGeneralPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createOptimizePass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  // Undo the MHLO::BroadcastInDimOp folding pattern on splat constants.
  pass_manager.addPass(mlir::odml::CreateUnfoldSplatConstantPass());
}

void AddHloToTfLiteLegalizationPasses(mlir::OpPassManager& pass_manager) {
  // HLO -> TFLite legalization
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateUniformQuantizedStableHloToTflPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreatePrepareHloPass());
  pass_manager.addPass(mlir::odml::CreateLegalizeHloToTfLitePass());
}

void AddTfLiteOptimizationPasses(mlir::OpPassManager& pass_manager,
                                 const mlir::TFL::PassConfig& pass_config) {
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

  // Quantization
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareQuantizePass(pass_config.quant_specs));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(pass_config.quant_specs));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(/*emit_quant_adaptor_ops=*/true));

  if (!pass_config.unfold_batch_matmul) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
  }

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
}

absl::Status ConvertStableHloToTFLite(
    mlir::ModuleOp module, const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    llvm::raw_pwrite_stream& export_stream) {
  mlir::MLIRContext* context = module->getContext();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  registry.insert<mlir::TFL::TFLDialect, mlir::mhlo::MhloDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  mlir::PassManager pm(context);
  tensorflow::InitPassManager(pm, converter_flags.debug_options(),
                              llvm::errs());

  AddSkipToTflitePasses(pm);
  AddHloOptimizationPasses(pm);
  AddHloToTfLiteLegalizationPasses(pm);
  AddTfLiteOptimizationPasses(pm, pass_config);

  if (mlir::failed(pm.run(module))) {
    return absl::InvalidArgumentError("StableHLO to TFLite pipeline failed.");
  }

  tflite::FlatbufferExportOptions options;
  options.converter_flags.set_allow_custom_ops(true);
  options.converter_flags.set_use_buffer_offset(true);

  auto status =
      tflite::MlirToFlatBufferTranslateFunction(module, options, export_stream);

  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to serialize to FlatBuffer: ", status.message()));
  }

  return absl::OkStatus();
}

}  // namespace mlir::TFL
