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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/canonicalize_boundary_value_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/cleanup_optimization_barrier_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_batch_matmul_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_broadcast_like_pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/push_transpose_through_ewise_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/analyze_variables_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/legalize_tensorlist_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tf_legalizations/while_loop_outline_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/split_merged_operands_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/tflite_passes/unfold_large_splat_constants_pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/unfreeze_global_constants.h"

namespace mlir {
namespace quant {
class QuantDialect;
}
namespace quantfork {
class QuantizationForkDialect;
}
namespace mhlo {
class MhloDialect;
}
namespace TF {
class TensorFlowDialect;
}
namespace TFL {
class TFLDialect;
typedef TFLDialect TensorFlowLiteDialect;
}  // namespace TFL
namespace func {
class FuncOp;
}
class ModuleOp;
template <typename T>
class OperationPass;
class Type;

namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////

struct OptimizePassOptions;

////////////////////////////////////////////////////////////////////////////////
// Utilities for backward compatibility
////////////////////////////////////////////////////////////////////////////////

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
// When the given run_tfl_runtime_verification value is true, it will check each
// TFL builtin op towards the TFL runtime capability and the incompatible TF ops
// will be left in the graph without getting legalized. If `preserve_assert_op`
// is true, the TF::AssertOp will not be removed.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass(
    bool run_tfl_runtime_verification, bool preserve_assert_op = false);
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass();

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
inline std::unique_ptr<mlir::Pass> CreateOptimizePass() {
  return Create<OptimizePass>();
}

// Creates an instance of the Tensorflow Lite batch matmul Optimize pass.
inline std::unique_ptr<mlir::Pass> CreateOptimizeBatchMatmulPass() {
  return Create<OptimizeBatchMatmulPass>();
}

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass(
    bool unfold_batch_matmul, bool allow_bf16_and_f16_type_legalization,
    bool use_fake_quant_num_bits = false);
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass();

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass(
    bool allow_tensorlist_pass_through, bool default_to_single_batch,
    bool enable_dynamic_update_slice);

std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass();

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
// Use quant_specs.ops_blocklist and quant_specs.nodes_blocklist if possible
// as they are now structure variables of QuantizationSpecs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    const QuantizationSpecs& quant_specs,
    const absl::flat_hash_set<std::string>& ops_blocklist = {},
    const absl::flat_hash_set<std::string>& nodes_blocklist = {});

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantizePass();

std::unique_ptr<OperationPass<ModuleOp>> CreateLowerQuantAnnotationsPass();

// Overloading of CreateQuantizePass which takes only necessary flags to reduce
// the binary size.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    bool verify_numeric = false, bool whole_model_verify = false,
    bool legacy_float_scale = false,
    const absl::flat_hash_set<std::string>& ops_blocklist = {},
    const absl::flat_hash_set<std::string>& nodes_blocklist = {});

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs);

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass();

// Creates an instance of the TensorFlow Lite dialect
// PrepareDynamicRangeQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>>
CreatePrepareDynamicRangeQuantizePass(const QuantizationSpecs& quant_specs);

std::unique_ptr<OperationPass<func::FuncOp>>
CreatePrepareDynamicRangeQuantizePass();

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops, const CustomOpMap& custom_op_map = {});

// Creates an instance of the TensorFlow Lite dialect QuantizeVariables pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizeVariablesPass();

// Creates an instance of the TensorFlow Lite pass that decomposes hybrid
// quantization patterns to the same dense operation with tfl dequantization
// and quantization patterns.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDecomposeHybridQuantizationPass();

// Creates an instance of the TensorFlow Lite optimize op order pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeOpOrderPass();

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass();

std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    const std::vector<std::string>& trim_funcs_allowlist);

// Creates an instance of the TensorFlow Lite dialect PrepareCompositeFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass();

// Creates an instance of the TensorFlow Lite dialect SplitMergedOperandsPass.
inline std::unique_ptr<mlir::Pass> CreateSplitMergedOperandsPass() {
  return Create<SplitMergedOperandsPass>();
}

// Creates an instance of the TensorFlow Lite dialect OptimizeFunctionalOpsPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeFunctionalOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>> CreateModifyIONodesPass(
    mlir::Type input_type, mlir::Type output_type);

std::unique_ptr<OperationPass<func::FuncOp>> CreateModifyIONodesPass();

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizeRemoveQDQPass();

// Creates an instance of the TensorFlow Lite dialect pass to add default
// quantization parameters.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max, bool is_signed);

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass();

// Creates an instance of the IdentifyDilatedConvPass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateIdentifyDilatedConvPass();

// Creates function pass to legalize TF While to TFL While.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFWhilePass();

// Legalize tflite flex ops to TF ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLiftTfliteFlexOpsPass();

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
inline std::unique_ptr<mlir::Pass> CreateWhileOutlinePass() {
  return Create<WhileOutlinePass>();
}

// Creates an instance of the TensorFlow Lite dialect IfOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateIfOutlinePass();

// Creates a pass to remove operands of TFL WhileOp without changing outcomes.
std::unique_ptr<OperationPass<func::FuncOp>> CreateReduceWhileOperandsPass();

// Verifies runtime constraints.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRuntimeVerifyPass();

// Creates raise custom ops pass, which legalize custom ops to TFL::CustomOp
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass();
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass(
    const std::vector<std::string>& target_ops);

// Creates raise custom ops pass, which legalize custom ops to TFL::CustomOp
std::unique_ptr<OperationPass<func::FuncOp>> CreateLowerCustomOpsPass();

// Inserts a TFL::CallOnce op when the tf_saved_model's session initialzer is
// given.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCallOnceOpFromSessionInitializerPass();

// Replace the tfl wrapped random function body with tfl.customOp.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeJaxRandomPass();

// Creates a pass which is responsible for legalizing TensorFlow variables to
// TensorFlow Lite variables.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeVariablesPass();

// Creates a pass which analyze the model whether it is safe to use
// native TFLite variables or not.
inline std::unique_ptr<mlir::Pass> CreateAnalyzeVariablesPass() {
  return Create<AnalyzeVariablesPass>();
}

// Creates a pass which is responsible for legalizing TensorFlow static hash
// tables to TensorFlow Lite hash tables.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHashTablesPass();

// Creates get arithmetic count pass, which will calculate the arithmetic count
// for each ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateGetArithmeticCountPass();

// Creates unfold large constant pass, which will replace large splat constant
// tensors with fill op.
inline std::unique_ptr<mlir::Pass> CreateUnfoldLargeSplatConstantPass() {
  return Create<UnfoldLargeSplatConstantPass>();
}

// Creates a pass which is responsible for unfreezing mutable global tensors.
inline std::unique_ptr<mlir::Pass> CreateUnfreezeMutableGlobalTensorsPass() {
  return Create<UnfreezeMutableGlobalTensorsPass>();
}

// Creates a pass that adds control dependencies to keep the relative
// execution order of operations with side effects frozen.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePinOpsWithSideEffectsPass();

// Legalize TensorList Ops iff all of them are supported.
inline std::unique_ptr<mlir::Pass> CreateLegalizeTensorListPass() {
  return Create<LegalizeTensorListPass>();
}

// Reduce the type precision of some tensor types if all values within that
// tensor are within the range of the reduced precision.
std::unique_ptr<OperationPass<ModuleOp>> CreateReduceTypePrecisionPass();

// Conservatively pushes transposes through element-wise ops to prepare
// so redundant ones may be grouped and removed.
inline std::unique_ptr<mlir::Pass> CreatePushTransposeThroughEwisePass() {
  return Create<PushTransposeThroughEwisePass>();
}

// Create a pass that canonicalize the boundary values.
inline std::unique_ptr<mlir::Pass> CreateCanonicalizeBoundaryValuePass() {
  return Create<CanonicalizeBoundaryValuePass>();
}

// Creates a pass that brings operations into the same order as graph_info.cc.
std::unique_ptr<OperationPass<func::FuncOp>>
CreatePartitionedTopologicalSortPass();

// Create a pass that cleans up optimization barriers.
inline std::unique_ptr<mlir::Pass> CreateCleanupOptimizationBarrierPass() {
  return Create<CleanupOptimizationBarrierPass>();
}

#define GEN_PASS_DECL_DEFAULTQUANTPARAMSPASS
#define GEN_PASS_DECL_LEGALIZETFPASS
#define GEN_PASS_DECL_LOWERSTATICTENSORLISTPASS
#define GEN_PASS_DECL_MODIFYIONODESPASS
#define GEN_PASS_DECL_POSTQUANTIZEPASS
#define GEN_PASS_DECL_PREPARECOMPOSITEFUNCTIONSPASS
#define GEN_PASS_DECL_PREPAREDYNAMICRANGEQUANTIZEPASS
#define GEN_PASS_DECL_PREPAREQUANTIZEPASS
#define GEN_PASS_DECL_PREPARETFPASS
#define GEN_PASS_DECL_QUANTIZEPASS
#define GEN_PASS_DECL_RAISECUSTOMOPSPASS
#define GEN_PASS_DECL_TRIMFUNCTIONSPASS
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeTFPass(
    const LegalizeTFPassOptions& options);

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareTFPass(
    const PrepareTFPassOptions& options);

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass(
    const LowerStaticTensorListPassOptions& options);

// Creates raise custom ops pass, which legalize custom ops to TFL::CustomOp
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass(
    const RaiseCustomOpsPassOptions& options);

// Creates an instance of the TensorFlow Lite dialect pass to add default
// quantization parameters.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass(
    const DefaultQuantParamsPassOptions& options);

inline void registerTensorFlowLitePasses() {
  registerTensorFlowLiteTdPasses();
  // Register TFLite Converter Passes
  Register<UnfreezeMutableGlobalTensorsPass>();

  // TF Legalization Passes
  Register<AnalyzeVariablesPass>();
  Register<LegalizeTensorListPass>();
  Register<WhileOutlinePass>();

  // TFL Optimization Passes
  Register<OptimizePass, OptimizePassOptions>();
  Register<OptimizeBatchMatmulPass>();
  Register<UnfreezeMutableGlobalTensorsPass>();
  Register<OptimizeBroadcastLikePass, OptimizeBroadcastLikePassOptions>();
  Register<PushTransposeThroughEwisePass>();
  Register<CanonicalizeBoundaryValuePass>();

  // Other TFLite Passes
  Register<UnfoldLargeSplatConstantPass>();
  Register<SplitMergedOperandsPass>();
  Register<CleanupOptimizationBarrierPass>();
}

}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
