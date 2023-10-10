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

#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"

#include <memory>
#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/data_dumper_logger_config.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v1/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v2/tf_dialect_to_executor.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/clustering_bridge_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/inference/inference_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/logging_hooks.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/error_logging.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kBridgeComponent[] = "TFXLABridge";

// Run the TF XLA Bridge based on the input pipeline, which can be either TPU
// bridge pipeline or non TPU bridge pipeline.
tensorflow::Status RunTFXLABridge(
    ModuleOp module,
    llvm::function_ref<void(OpPassManager &pm)> pipeline_builder,
    llvm::StringRef module_name = llvm::StringRef()) {
  // Explicitly check that the TensorFlow dialect can constant fold ops.
  // Constant folding is essential for the bridge. Without this check, the
  // bridge may fail with an error that is difficult to understand and not
  // actionable.
  if (!TF::TensorFlowDialect::HasConstantFoldHook()) {
    return tensorflow::errors::Internal(
        "TensorFlow dialect missing constant fold hook in TFXLA bridge phase "
        "1; this could happen if the binary doesn't link the constant fold "
        "hook registration library.");
  }

  PassManager bridge(module.getContext());
  ::tensorflow::applyTensorflowAndCLOptions(bridge);

  // Populate a passmanager with the list of passes that implement the bridge.
  pipeline_builder(bridge);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "tf_xla_bridge_before"),
        module, llvm::StringRef(), &bridge);
  }

  if (VLOG_IS_ON(2) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(),
                                      kDebugGroupBridgePhase1Clustering)) {
    ::tensorflow::tf2xla::internal::EnablePassIRPrinting(
        bridge, kDebugGroupBridgePhase1Clustering, module_name);
  }

  LogicalResult result = bridge.run(module);
  (void)result;

  if (VLOG_IS_ON(1) ||
      DEBUG_DATA_DUMPER()->ShouldDump(module_name.str(), kDebugGroupMain)) {
    ::tensorflow::DumpMlirOpToFile(
        DEBUG_DATA_DUMPER()->GetDumpFilename(module_name.str(), kDebugGroupMain,
                                             "tf_xla_bridge_after"),
        module, llvm::StringRef(), &bridge);
  }

  return diag_handler.ConsumeStatus();
}

}  // namespace

void CreateTPUBridgePipeline(OpPassManager &pm, llvm::StringRef module_name) {
  pm.addPass(CreateTPUValidateInputsPass());
  pm.addNestedPass<func::FuncOp>(
      TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  tensorflow::tf2xla::internal::AddBridgeClusteringPipelinePasses(pm,
                                                                  module_name);
}

tensorflow::Status TPUBridge(ModuleOp module, bool fallback_enabled,
                             llvm::StringRef module_name) {
  VLOG(2)
      << "TPU Bridge called stack trace is "
      << "(NOTE: this is not an error; rather the stack trace for debugging) : "
      << tensorflow::CurrentStackTrace();
  Status bridge_status = RunTFXLABridge(
      module,
      [module_name](OpPassManager &pm) {
        CreateTPUBridgePipeline(pm, module_name);
      },
      module_name);
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      "tpu", "v2", fallback_enabled,
      bridge_status.ok() ? "success" : "failure");
  tsl::OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::MLIR_BRIDGE_PHASE_1,
      bridge_status);
  if (!bridge_status.ok()) {
    tsl::error_logging::Log(TFTPU::kBridgeComponent,
                            "TFXLA_PHASE_ONE_MLIR_TPU_BRIDGE",
                            bridge_status.ToString())
        .IgnoreError();
    return bridge_status;
  }

  Status export_status =
      tensorflow::tf2xla::v2::ExportFromTensorflowDialectToExecutor(
          module, module_name);
  if (!export_status.ok()) {
    tsl::error_logging::Log(TFTPU::kBridgeComponent,
                            "TFXLA_PHASE_ONE_MLIR_TPU_BRIDGE_EXPORT",
                            export_status.ToString())
        .IgnoreError();
  }

  return export_status;
}

}  // namespace TFTPU

namespace TF {

void NoCanonicalization(OpPassManager &pm) {}

void AddGraphExportLoweringPasses(OpPassManager &pm) {
  auto add_pass = [&](std::unique_ptr<Pass> pass) {
    pm.addNestedPass<func::FuncOp>(std::move(pass));
    pm.addPass(CreateBreakUpIslandsPass());
  };

  pm.addPass(TF::CreateTFRegionControlFlowToFunctional());
  add_pass(CreateFunctionalToExecutorDialectConversionPass());
  add_pass(TFDevice::CreateReplicateToIslandPass(/*legacy_graph_export=*/true));
  add_pass(TFDevice::CreateReplicaIDToDeviceOrdinalPass());
  add_pass(TFDevice::CreateParallelExecuteToIslandsPass(
      /*legacy_graph_export=*/true));
  add_pass(TFDevice::CreateLaunchToDeviceAttributePass(
      /*legacy_graph_export=*/true));
  pm.addNestedPass<func::FuncOp>(TFTPU::CreateTPUDevicePropagationPass());
  pm.addNestedPass<func::FuncOp>(TFTPU::CreateTPUColocateSplitsPass());
  pm.addPass(createSymbolDCEPass());
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_convert_control_to_data_outputs_pass) {
    pm.addPass(tf_executor::CreateTFExecutorConvertControlToDataOutputsPass());
  }
  pm.addPass(CreateVerifySuitableForExportPass());
}

tensorflow::Status RunBridgeWithStandardPipeline(ModuleOp module,
                                                 bool enable_logging,
                                                 bool enable_inliner) {
  PassManager bridge(module.getContext());

  StandardPipelineOptions pipeline_options;
  pipeline_options.enable_inliner.setValue(enable_inliner);
  CreateTFStandardPipeline(bridge, pipeline_options);

  mlir::StatusScopedDiagnosticHandler diag_handler(
      module.getContext(), /*propagate=*/false,
      /*filter_stack=*/!VLOG_IS_ON(1));

  if (enable_logging || VLOG_IS_ON(1)) {
    tensorflow::DumpMlirOpToFile(kStandardPipelineBefore, module, "", &bridge);
    if (VLOG_IS_ON(2)) {
      tensorflow::tf2xla::internal::EnablePassIRPrinting(
          bridge, TFTPU::kBridgeComponent);
    }
  }
  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile(kStandardPipelineAfter, module, "", &bridge);
  return diag_handler.ConsumeStatus();
}

void CreateTFXLABridgePipeline(OpPassManager &pm) {
  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this.
  VLOG(2) << "Create TF XLA Bridge pipeline";
  pm.addNestedPass<func::FuncOp>(
      TF::CreateCanonicalizeCompileAndReplicateAttributesPass());
  // This pass expectes unified compilation markers.
  pm.addPass(TFDevice::CreateXlaValidateInputsPass());
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {};
  pm.addNestedPass<func::FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<func::FuncOp>(
      CreateExecutorDialectToFunctionalConversionPass());
  // Guarantee all functions have one use, which enables more exact shape
  // inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  pm.addPass(TF::CreateTFShapeInferencePass());
  // Encapsulate PartitionedCall ops within a cluster so that the composite
  // resource ops can be decomposed.
  pm.addPass(TFDevice::CreateXlaClusterFormationPass());
  // Running canonicalizer before decomposing resource ops in cluster helps the
  // latter pass to converge faster as it does not have to spend time folding
  // away dead ops.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Decompose resource ops.
  pm.addPass(TFDevice::CreateDecomposeResourceOpsInClusterPass());
  // TODO(b/267193636): Remove this flag when outside compilation
  // for generic pipeline is landed.
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_generic_outside_compilation) {
    pm.addPass(TF::CreateTFFunctionalControlFlowToRegions());
  }
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Inline all the function calls. Do not call canonicalizer to prevent it from
  // moving the definition of any constant operand of ops within a cluster to
  // its outside. This may cause the op to fail to verify after the cluster is
  // outlined, as the constant operand is replaced by an argument.
  pm.addPass(mlir::createInlinerPass({}, NoCanonicalization));
  // Lift resource operations out of device computation. This step needs to be
  // done after inlining.
  pm.addPass(TFDevice::CreateResourceOpLiftingPass());
  // TODO(b/267193636): Remove this flag when outside compilation
  // for generic pipeline is landed.
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_generic_outside_compilation) {
    pm.addPass(TFDevice::CreateMarkOpsForOutsideCompilationPass());
    pm.addPass(TFDevice::CreateExtractHeadTailOutsideCompilationPass());
    pm.addPass(TFDevice::CreateExtractOutsideCompilationPass());
  }
  // Outline clusters into cluster functions.
  pm.addPass(TFDevice::CreateClusterOutliningPass());
  // Rewrite cluster functions into XLA  launch ops.
  if (tensorflow::GetMlirCommonFlags()
          ->tf_mlir_enable_generic_outside_compilation) {
    pm.addPass(TFDevice::CreateXlaRewriteV2Pass());
  } else {
    pm.addPass(TFDevice::CreateXlaRewritePass());
  }
  // Re-run the canonicalizer pass as some cleanup during resource op lifting
  // pass opens up some opportunities for canonicalization of cluster ops.
  // Specifically, we want to eliminate pass through results from the cluster
  // op.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

tensorflow::Status RunTFXLABridge(ModuleOp module,
                                  llvm::StringRef module_name) {
  VLOG(2)
      << "CPU/GPU Bridge called stack trace is "
      << "(NOTE: this is not an error; rather the stack trace for debugging) : "
      << tensorflow::CurrentStackTrace();
  Status status = mlir::TFTPU::RunTFXLABridge(
      module,
      [](OpPassManager &pm) {
        CreateTFXLABridgePipeline(pm);
      },
      module_name);
  tensorflow::metrics::UpdateTfMlirBridgeFirstPhaseCounter(
      /*device type*/ "cpu/gpu", /*bridge version*/ "tfxla",
      /*fallback_enabled*/ false,
      /*result*/ status.ok() ? "success" : "failure");
  if (!status.ok()) {
    tsl::error_logging::Log(TFTPU::kBridgeComponent,
                            "TFXLA_PHASE_ONE_MLIR_CPU/GPU_BRIDGE",
                            status.ToString())
        .IgnoreError();
  }

  Status export_status =
      tensorflow::tf2xla::v2::ExportFromTensorflowDialectToExecutor(
          module, module_name);
  if (!export_status.ok()) {
    tsl::error_logging::Log(TFTPU::kBridgeComponent,
                            "TFXLA_PHASE_ONE_MLIR_CPU_BRIDGE_EXPORT",
                            export_status.ToString())
        .IgnoreError();
  }

  return status;
}

}  // namespace TF
}  // namespace mlir
