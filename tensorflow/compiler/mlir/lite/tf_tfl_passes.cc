/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/tf_tfl_passes.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/core/macros.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_passes.h"
#include "tensorflow/compiler/mlir/lite/quantization/tensorflow/passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"
#include "tensorflow/compiler/mlir/lite/transforms/converter_pass_options_setter.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/variable_freezing_pipeline.h"
#include "tensorflow/compiler/mlir/lite/utils/fake_quant_utils.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {
namespace {
// Data layout supported by TFLite.
constexpr mlir::StringRef kTFLiteDataLayout = "NHWC";
}  // namespace

void AddOptimizationPasses(const tflite::ConverterFlags& converter_flags,
                           const mlir::TFL::PassConfig& pass_config,
                           mlir::OpPassManager* pass_manager) {
  mlir::TFL::ConverterPassOptionsSetter converter_pass_options_setter(
      converter_flags, pass_config);

  pass_manager->addPass(mlir::TFL::CreatePushTransposeThroughEwisePass());

  // Add BroadcastLike optimization pass.
  {
    mlir::TFL::OptimizeBroadcastLikePassOptions options;
    options.unsafe_fuse_dynamic_shaped_broadcast =
        pass_config.unsafe_fuse_dynamic_shaped_broadcast;
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::Create<mlir::TFL::OptimizeBroadcastLikePass>(options));
  }

  // Add TFLite optimize pass.
  mlir::TFL::OptimizePassOptions optimize_pass_options;
  optimize_pass_options.enable_strict_qdq_mode =
      (pass_config.quant_specs.qdq_conversion_mode ==
       mlir::quant::QDQConversionMode::kQDQStrict);
  std::unique_ptr<mlir::Pass> optimize_pass =
      mlir::TFL::Create<mlir::TFL::OptimizePass>(optimize_pass_options);
  auto pass_ptr =
      dynamic_cast<mlir::TFL::MutableOptionsPass*>(optimize_pass.get());
  if (pass_ptr) pass_ptr->ApplyOptionsVisitor(converter_pass_options_setter);
  pass_manager->addNestedPass<mlir::func::FuncOp>(std::move(optimize_pass));

  if (!pass_config.unfold_batch_matmul) {
    // Enable an optimization pass that transforms BatchMatmul to FC only
    // when `unfold_batch_matmul=false`.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
  }
}

void AddStrictQDQQuantizationPasses(
    const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager& pass_manager) {
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareQuantizePass(pass_config.quant_specs));

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(pass_config.quant_specs));

  // clean up DRQ FQ decomposition functions
  pass_manager.addPass(mlir::createSymbolDCEPass());

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(true));

  // So that quantized clipping activations get fused into preceding ops.
  AddOptimizationPasses(converter_flags, pass_config, &pass_manager);

  // TODO(b/399468842): This is a temporary fix to enable constant folding on
  // quantized TFL_TransposeOp and TFL_ReshapeOp. Ideally TFL constant folders
  // should handle quantized types (or anything that is supported by TFLite
  // runtime).
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(true));
}

void AddQuantizationPasses(const mlir::TFL::PassConfig& pass_config,
                           mlir::OpPassManager& pass_manager) {
  const mlir::quant::QuantizationSpecs& quant_specs = pass_config.quant_specs;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareQuantizePass(quant_specs));
  if (quant_specs.default_ranges.first.has_value() ||
      quant_specs.default_ranges.second.has_value()) {
    mlir::TFL::DefaultQuantParamsPassOptions default_quant_params_pass_options;
    default_quant_params_pass_options.default_min_ =
        quant_specs.default_ranges.first.value_or(0.0);
    default_quant_params_pass_options.default_max_ =
        quant_specs.default_ranges.second.value_or(0.0);
    default_quant_params_pass_options.is_signed_ =
        quant_specs.IsSignedInferenceType();
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateDefaultQuantParamsPass(
            default_quant_params_pass_options));
  }

  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  // TODO(b/265081639): When added PrepareQuantizeVariablesPass before adding
  // PrepareQuantizePass, an error occurs in certain model. It could fix it by
  // roll-back to run PrepareQuantizeVariablesPass, QuantizePass,
  // PostQuantizePass as suggested in cl/479634700. Need to figure out the
  // fundamental reason of the error, and (if needed) fix it without this
  // rollback.
  if (quant_specs.enable_mlir_variable_quantization) {
    pass_manager.addPass(mlir::TFL::CreatePrepareQuantizeVariablesPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateQuantizePass(quant_specs));
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizeOpOrderPass());
  // Add optimization pass after quantization for additional fusing
  // opportunities.
  // Add TFLite optimize pass.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizePass());

  if (!pass_config.unfold_batch_matmul) {
    // Enable an optimization pass that transforms BatchMatmul to FC only when
    // `unfold_batch_matmul=false`.
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
  }
}

void AddVariableFreezingFromGlobalTensorsPasses(
    const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
  // This pass does resource analysis of saved model global tensors and marks
  // those deemed read-only as immutable.
  auto pass = mlir::TFL::Create<mlir::TFL::VariableFreezingPipeline,
                                mlir::TFL::VariableFreezingPipelineOptions>();
  auto pass_ptr = dynamic_cast<mlir::TFL::MutableOptionsPass*>(pass.get());
  if (pass_ptr)
    pass_ptr->ApplyOptionsVisitor(
        mlir::TFL::ConverterPassOptionsSetter(converter_flags, pass_config));
  pass_manager->addPass(std::move(pass));
}

void AddDynamicRangeQuantizationPasses(const mlir::TFL::PassConfig& pass_config,
                                       mlir::OpPassManager& pass_manager) {
  const mlir::quant::QuantizationSpecs& quant_specs = pass_config.quant_specs;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePrepareDynamicRangeQuantizePass(quant_specs));
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateQuantizePass(quant_specs));
  bool emit_quant_adaptor_ops =
      quant_specs.inference_type != quant_specs.inference_input_type;
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops,
                                        quant_specs.custom_map));
  // TODO(b/265081639): When added PrepareQuantizeVariablesPass before adding
  // PrepareQuantizePass, an error occurs in certain model. It could fix it by
  // roll-back to run PrepareQuantizeVariablesPass, QuantizePass,
  // PostQuantizePass as suggested in cl/479634700. Need to figure out the
  // fundamental reason of the error, and (if needed) fix it without this
  // rollback.
  if (quant_specs.enable_mlir_variable_quantization) {
    pass_manager.addPass(mlir::TFL::CreatePrepareQuantizeVariablesPass());
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateQuantizePass(quant_specs));
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreatePostQuantizePass(emit_quant_adaptor_ops));
  }
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizeOpOrderPass());
  // Add optimization pass after quantization for additional fusing
  // opportunities.
  // Add TFLite optimize pass.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateOptimizePass());

  if (!pass_config.unfold_batch_matmul) {
    // Enable an optimization pass that transforms BatchMatmul to FC only when
    // `unfold_batch_matmul=false`.
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateOptimizeBatchMatmulPass());
  }
}

void AddPytorchPasses(mlir::OpPassManager& pass_manager) {
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pass_manager.addPass(mlir::odml::createBuildStableHLOCompositePass());
  pass_manager.addPass(mlir::createInlinerPass());
  pass_manager.addPass(mlir::odml::createLiftCallSiteLocCallerPass());
  pass_manager.addPass(mlir::odml::createLegalizeQDQCustomCallPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
}

void AddPreQuantizationStableHloToTfPasses(
    const mlir::StringRef entry_function_name,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager& pass_manager) {
  pass_manager.addPass(
      mlir::odml::CreateLegalizeTFXlaCallModuleToStablehloPass());

  if (pass_config.model_origin_framework == tflite::ConverterFlags::PYTORCH) {
    AddPytorchPasses(pass_manager);
  }

  // Legalize MHLO to StableHLO should be moved closer to where it is needed
  // There are some entry points that start with HLO->MHLO like
  // jax_to_tfl_flatbuffer.cc which can likely be updated to emit StableHLO
  // to be consistent with other entrypoints.
  pass_manager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

  // Expand backward compatibility with the given StableHLO version by
  // decomposing newer StableHLO operations into equivalent operations supported
  // by that older version.
  pass_manager.addPass(
      mlir::stablehlo::createStablehloCompatibilityExpanderPass(
          {tflite_supported_stablehlo_version}));

  // Decompose CHLO into StableHLO ops
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateLegalizeChloToTflPass());
  // TODO(b/331843141): There are some CHLO's like TopK which we could instead
  // lower to TFL ops.
  mlir::stablehlo_ext::createChloLegalizeToStablehloPipeline(pass_manager);

  pass_manager.addPass(mlir::odml::CreateTransposeCommuteOpsPass());
  // The following two passes find specific uniform quantization patterns in
  // StableHLO and converts them to TFLite ops that accept or produce uniform
  // quantized types. They only target a specific set of models that contain
  // "decomposed" quantized ops produced from the framework level. This is why
  // they are placed right after the `LegalizeTFXlaCallModuleToStablehloPass`
  // because the quantization patterns should be identified before any
  // optimizations kick in.
  //
  // There are future plans to make the framework to directly produce StableHLO
  // uniform quantized ops and deprecate `ComposeUniformQuantizedTypePass`. If
  // no quantization patterns are found, it is a no-op.
  pass_manager.addPass(mlir::odml::CreateComposeUniformQuantizedTypePass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::CreateUniformQuantizedStableHloToTflPass());

  // Legalize jax random to tflite custom op.
  // The CreateLegalizeJaxRandom Pass has to stay at because we need to replace
  // the random function body before being inlined.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateLegalizeJaxRandomPass());

  // Canonicalize, CSE etc.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // DCE for private symbols.
  pass_manager.addPass(mlir::createSymbolDCEPass());
  pass_manager.addPass(mlir::TF::CreateStripNoinlineAttributePass());
  // Add inline pass.
  pass_manager.addPass(mlir::createInlinerPass());
  // Flatten tuples in entry computations and custom calls.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass(
          {entry_function_name.str()}));
  mlir::odml::AddMhloOptimizationPasses(
      pass_manager,
      /*add_fold_broadcast_pass=*/pass_config.enable_stablehlo_quantizer);

  // Undo the MHLO::BroadcastInDimOp folding pattern on splat constants. This
  // pass must be added right before the legalization because pattern rewriter
  // driver applies folding by default.
  // TODO: b/295966255 - Remove this pass after moving MHLO folders to a
  // separate pass.
  pass_manager.addPass(mlir::odml::CreateUnfoldSplatConstantPass());

  if (pass_config.enable_stablehlo_quantizer) {
    // When using StableHLO Quantizer, MHLO ops should be transformed back into
    // StableHLO because the quantizer takes StableHLO dialect as its input.
    pass_manager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  }
}

void AddPostQuantizationStableHloToTfPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager& pass_manager) {
  if (pass_config.enable_stablehlo_quantizer) {
    // Convert StableHLO -> TFLite for fused quantization patterns early so that
    // quantized types do not go through the TF dialect which doesn't support
    // quantized types.
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::odml::CreateUniformQuantizedStableHloToTflPass());

    // StableHLO -> MHLO
    pass_manager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  }

  if (pass_config.enable_composite_direct_lowering) {
    pass_manager.addPass(mlir::odml::CreateCompositeLoweringPass());
    pass_manager.addPass(mlir::createSymbolDCEPass());
  }

  // TFLite dialect passes.
  if (!pass_config.disable_hlo_to_tfl_conversion) {
    pass_manager.addNestedPass<mlir::func::FuncOp>(
        mlir::odml::CreatePrepareHloPass());
    // This pass must be added right before the legalization because pattern
    // rewriter driver applies folding by default.
    // TODO: b/354280588 - Rewrite this pass into a pattern in PrepareHloPass.
    pass_manager.addPass(mlir::odml::CreateUnfoldSplatConstantPass());
    pass_manager.addPass(mlir::odml::CreateLegalizeHloToTfLitePass());
    // Folds tfl.BroadcastTo ops with subsequent ops if they have built in
    // broadcasting support. This needs to be run immediately after HLO->TFL
    // legalization, otherwise the newly generated TFL broadcast ops can fold
    // and materialize the weights.
    {
      mlir::TFL::OptimizeBroadcastLikePassOptions options;
      options.unsafe_fuse_dynamic_shaped_broadcast =
          pass_config.unsafe_fuse_dynamic_shaped_broadcast;
      pass_manager.addNestedPass<mlir::func::FuncOp>(
          mlir::TFL::Create<mlir::TFL::OptimizeBroadcastLikePass>(options));
    }
  }
  // folds tf.BroadcastTo ops with subsequent ops if they have built in
  // broadcasting support. This needs to be run immediately after HLO->TF
  // legalization; otherwise other passes like `ConvertTFBroadcastTo` will
  // constant fold the newly generated TF broadcast ops and materialize the
  // weights.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateBroadcastFoldPass());

  // Canonicalization after TF legalization.
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());

  // Legalize all remaining mhlo ops to stableHLO
  pass_manager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

  // Translate "stablehlo.custom_call @stablehlo.composite" to
  // "stablehlo.composite"
  // TODO: b/330741524 - clean this up when "stablehlo.composite" is emitted
  // directly. Additionally remove the composite to custom once ODML long term
  // solution lands.
  pass_manager.addPass(
      mlir::odml::createLegalizeStablehloCustomCallToCompositePass());
  pass_manager.addNestedPass<mlir::func::FuncOp>(
      mlir::odml::createLegalizeCompositeToCustomOpPass());
}

// This is the early part of the conversion in isolation. This enables a caller
// to inject more information in the middle of the conversion before resuming
// it.
void AddPreVariableFreezingTFToTFLConversionPasses(
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
  // This pass wraps all the tf.FakeQuant ops in a custom op so they are not
  // folded before being converted to tfl.quantize and tfl.dequantize ops.
  std::vector<std::string> target_ops = mlir::TFL::AllTfFakeQuantOps();
  mlir::TFL::RaiseCustomOpsPassOptions raise_custom_ops_pass_options;
  raise_custom_ops_pass_options.target_ops_ = llvm::to_vector(target_ops);
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateRaiseCustomOpsPass(raise_custom_ops_pass_options));

  mlir::TF::StandardPipelineOptions standard_pipeline_options;
  standard_pipeline_options.enable_inliner = false;
  standard_pipeline_options.form_clusters = pass_config.form_clusters;
  mlir::TF::CreateTFStandardPipeline(*pass_manager, standard_pipeline_options);

  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateDeviceIndexSelectorPass());

  // Add canonicalize pass to remove no-op session initializer pass.
  pass_manager->addPass(mlir::createCanonicalizerPass());

  if (pass_config.guarantee_all_funcs_one_use) {
    pass_manager->addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  }
  if (pass_config.shape_inference) {
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Keep this pass after the shape inference pass, which couldn't do shape
  // inference for non-tf ops.
  if (!pass_config.quant_specs.serialized_quant_stats.empty()) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::quant::CreateImportQuantStatsPassForTFControlDialect(
            pass_config.quant_specs.serialized_quant_stats));
  }

  pass_manager->addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  // The conversion pipeline has to follow the following orders:
  // 1) Saved model related optimization like decompose resource ops
  // 2) Convert composite functions like lstm/rnns, along with proper function
  // inlining & dce.
  // 3) Lower static tensor list pass.

  // This decomposes resource ops like ResourceGather into read-variable op
  // followed by gather. This is used when the saved model import path is used
  // during which resources don't get frozen in the python layer.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  pass_manager->addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
}

// This is the later part of the conversion in isolation. This enables a caller
// to resume the conversion after injecting more information in the middle of
// it.
void AddPostVariableFreezingTFToTFLConversionPasses(
    llvm::StringRef saved_model_dir,
    const tflite::ConverterFlags& converter_flags,
    const mlir::TFL::PassConfig& pass_config,
    mlir::OpPassManager* pass_manager) {
  // Note:
  // We need to fuse composite ops before LowerStaticTensorList pass.
  // The tensorflow list is not supported right now by that pass.
  // Enable fusing composite ops that can be lowered to built-in TFLite ops.
  if (pass_config.emit_builtin_tflite_ops &&
      converter_flags.tf_quantization_mode().empty()) {
    pass_manager->addPass(mlir::TFL::CreatePrepareCompositeFunctionsPass());
  }

  pass_manager->addPass(mlir::createInlinerPass());
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (pass_config.legalize_custom_tensor_list_ops) {
    pass_manager->addPass(mlir::TFL::CreateLegalizeTensorListPass());
  }

  if (pass_config.lower_tensor_list_ops &&
      converter_flags.tf_quantization_mode().empty()) {
    // TODO(haoliang): Add this pass by default.
    mlir::TFL::LowerStaticTensorListPassOptions
        lower_static_tensor_list_pass_options;
    lower_static_tensor_list_pass_options.allow_tensorlist_pass_through_ =
        converter_flags.force_select_tf_ops() ||
        converter_flags.enable_select_tf_ops();
    lower_static_tensor_list_pass_options.default_to_single_batch_ =
        converter_flags.default_to_single_batch_in_tensor_list_ops();
    lower_static_tensor_list_pass_options.enable_dynamic_update_slice_ =
        converter_flags.enable_dynamic_update_slice();
    pass_manager->addPass(mlir::TFL::CreateLowerStaticTensorListPass(
        lower_static_tensor_list_pass_options));
  }

  if (pass_config.shape_inference) {
    // Add a shape inference pass to optimize away the unnecessary casts.
    pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
  }

  // Legalize while early to allow further constant folding.
  // TODO(jpienaar): This may not actually matter as we do canonicalization
  // after the legalize below, for now it needs to be below the above passes
  // that work on TF dialect and before inliner so that the function calls in
  // body and cond are inlined for optimization.
  pass_manager->addPass(mlir::TFL::CreateLegalizeTFWhilePass());

  // Add function inlining pass. Both TF and TFLite dialects are opted into
  // function inliner interface.
  pass_manager->addPass(mlir::createInlinerPass());
  // Reduce operands of TFL::While without changing the outcome.
  // It needs to stay here because:
  // 1. WhileOps are in TFL dialect.
  // 2. The body and cond are inlined.
  // 3. We need to do this before while canonicalization, otherwise it would be
  //   difficult to find dependencies.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::TFL::CreateReduceWhileOperandsPass());
  // Canonicalization includes const folding, which is utilized here to optimize
  // away ops that can't get constant folded after PrepareTF pass. For example,
  // tf.Conv2D is split into tf.Transpose and tfl.Conv2D.
  pass_manager->addNestedPass<mlir::func::FuncOp>(
      mlir::createCanonicalizerPass());
  pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  // This pass does dead code elimination based on symbol visibility.
  pass_manager->addPass(mlir::createSymbolDCEPass());

  if (!saved_model_dir.empty()) {
    // This pass 'freezes' tf saved model asset ops and inlines as string values
    // in a format of the tf constant op.
    pass_manager->addPass(
        mlir::tf_saved_model::CreateFreezeAssetsPass(saved_model_dir.str()));
  }
  // For TF Quantization, convert unsupported ops to Flex ops before other
  // conversion passes.
  if (!converter_flags.tf_quantization_mode().empty()) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateFallbackToFlexOpsPass(
            converter_flags.tf_quantization_mode()));
  }
  // The below passes only make sense if Builtin TFLite ops are enabled
  // for emission.
  if (pass_config.emit_builtin_tflite_ops) {
    // Run shape inference after variables are converted to constants.
    if (pass_config.shape_inference) {
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }

    // Force layout supported by TFLite, this will transpose the data
    // to match 'kTFLiteDataLayout'
    mlir::TF::LayoutOptimizationPipelineOptions layout_optimization_options;
    layout_optimization_options.force_data_format = kTFLiteDataLayout.str();
    layout_optimization_options.skip_fold_transpose_in_ops = true;
    mlir::TF::CreateLayoutOptimizationPipeline(
        pass_manager->nest<mlir::func::FuncOp>(), layout_optimization_options);

    // Prepare for TFLite dialect, rerun canonicalization, and then legalize to
    // the TFLite dialect.
    mlir::TFL::PrepareTFPassOptions prepare_tf_pass_options;
    prepare_tf_pass_options.unfold_batch_matmul_ =
        pass_config.unfold_batch_matmul;
    prepare_tf_pass_options.allow_bf16_and_f16_type_legalization_ =
        !pass_config.runtime_verification;
    prepare_tf_pass_options.use_fake_quant_num_bits_ =
        converter_flags.use_fake_quant_num_bits();
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreatePrepareTFPass(prepare_tf_pass_options));
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    if (pass_config.shape_inference) {
      // Add a shape inference pass to optimize away the unnecessary casts.
      // This also fixes the unranked shapes due to TF ops constant folding.
      // TODO(fengliuai): remove this pass if TableGen patterns have a better
      // to control the shapes for the intermediate results.
      pass_manager->addPass(mlir::TF::CreateTFShapeInferencePass());
    }

    // Inline function calls that left in the graph after folding functional
    // control flow ops (IfOp, CaseOp).
    pass_manager->addPass(mlir::createInlinerPass());

    // This pass removes the asset file dependencies in hash table use cases.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TF::CreateInitTextFileToImportPass(saved_model_dir.str()));

    // Add legalize TF pass to TFL dialect.
    mlir::TFL::LegalizeTFPassOptions legalize_tf_pass_options;
    legalize_tf_pass_options.run_tfl_runtime_verification_ =
        pass_config.runtime_verification;
    legalize_tf_pass_options.preserve_assert_op_ =
        pass_config.preserve_assert_op;
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateLegalizeTFPass(legalize_tf_pass_options));

    pass_manager->addPass(mlir::TFL::CreateAnalyzeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeVariablesPass());
    pass_manager->addPass(mlir::TFL::CreateLegalizeHashTablesPass());

    if (pass_config.quant_specs.qdq_conversion_mode ==
        mlir::quant::QDQConversionMode::kQDQStrict) {
      pass_manager->addPass(mlir::TFL::CreateLowerQuantAnnotationsPass());

      // To remove the quant annotation decompositions.
      pass_manager->addPass(mlir::createSymbolDCEPass());
    }

    // Run TFL optimization passes set multiple times as op fusion and
    // reordering in later passes may enable further optimizations with earlier
    // passes.
    AddOptimizationPasses(converter_flags, pass_config, pass_manager);
    AddOptimizationPasses(converter_flags, pass_config, pass_manager);

    // This pass operates on TensorFlow ops but is triggered after legalization
    // so that it can target constants introduced once TensorFlow Identity ops
    // are removed during legalization.
    pass_manager->addPass(mlir::TFL::CreateOptimizeFunctionalOpsPass());
    std::vector<std::string> empty_wrapped_ops({});
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateRaiseCustomOpsPass(empty_wrapped_ops));
    pass_manager->addPass(mlir::createSymbolDCEPass());
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::createCanonicalizerPass());
    pass_manager->addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

    if (pass_config.quant_specs.qdq_conversion_mode ==
        mlir::quant::QDQConversionMode::kQDQStrict) {
      AddStrictQDQQuantizationPasses(converter_flags, pass_config,
                                     *pass_manager);
    } else {
      // Run quantization after all the floating point model conversion is
      // completed. Add either full integer quantization or dynamic range
      // quantization passes based on quant_specs.
      if (pass_config.quant_specs
              .RunPropagationAndRewriteQuantizationPasses() ||
          pass_config.quant_specs.qdq_conversion_mode !=
              mlir::quant::QDQConversionMode::kQDQNone) {
        AddQuantizationPasses(pass_config, *pass_manager);
        // Remove unnecessary QDQs while handling QAT models.
        pass_manager->addNestedPass<mlir::func::FuncOp>(
            mlir::TFL::CreatePostQuantizeRemoveQDQPass());
      } else if (pass_config.quant_specs
                     .RunAndRewriteDynamicRangeQuantizationPasses()) {
        AddDynamicRangeQuantizationPasses(pass_config, *pass_manager);
      }
    }
    pass_manager->addPass(mlir::createCanonicalizerPass());

    if (pass_config.reduce_type_precision ||
        converter_flags.reduce_type_precision()) {
      pass_manager->addPass(mlir::TFL::CreateReduceTypePrecisionPass());
    }

    // This pass should alway run before the end of the model conversion but
    // not after the CreateSplitMergedOperandsPass below.
    if (pass_config.canonicalizing_inf_as_min_max_float)
      pass_manager->addPass(mlir::TFL::CreateCanonicalizeBoundaryValuePass());

    // This pass should be always at the end of the model
    // conversion (even after quantization). Some TFL ops like unidirectional
    // sequence lstm will have stateful operands and some optimization passes
    // will merge those operands if they have identical values & types. However,
    // it's not desired by TFL. This pass serves as a "fix" pass to split the
    // merged inputs until we have 1st class variable support or reuse
    // tf.variable to model this.
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateSplitMergedOperandsPass());

    // Add CallOnceOp when there is a session initializer function in tf saved
    // model dialect.
    pass_manager->addPass(
        mlir::TFL::CreateInsertCallOnceOpFromSessionInitializerPass());
  } else {
    // This pass should alway run before the end of the model conversion.
    if (pass_config.canonicalizing_inf_as_min_max_float)
      pass_manager->addPass(mlir::TFL::CreateCanonicalizeBoundaryValuePass());
  }

  if (pass_config.unfold_large_splat_constant) {
    pass_manager->addPass(mlir::TFL::CreateUnfoldLargeSplatConstantPass());
  }
  if (pass_config.outline_tf_while) {
    pass_manager->addPass(mlir::TFL::CreateWhileOutlinePass());
  }
  pass_manager->addPass(mlir::TFL::CreateIfOutlinePass());
  if (pass_config.runtime_verification) {
    pass_manager->addNestedPass<mlir::func::FuncOp>(
        mlir::TFL::CreateRuntimeVerifyPass());
  }
}

void AddTFToTFLConversionPasses(llvm::StringRef saved_model_dir,
                                const tflite::ConverterFlags& converter_flags,
                                const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
  AddPreVariableFreezingTFToTFLConversionPasses(pass_config, pass_manager);
  AddPostVariableFreezingTFToTFLConversionPasses(
      saved_model_dir, converter_flags, pass_config, pass_manager);
}
void AddTFToTFLConversionPasses(const mlir::TFL::PassConfig& pass_config,
                                mlir::OpPassManager* pass_manager) {
  const tflite::ConverterFlags converter_flags;
  AddTFToTFLConversionPasses(/*saved_model_dir=*/"", converter_flags,
                             pass_config, pass_manager);
}

}  // namespace tensorflow
