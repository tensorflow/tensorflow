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

namespace mlir {
class FuncOp;
class ModuleOp;
template <typename T>
class OperationPass;
class Type;

namespace TFL {
using StringSet = absl::flat_hash_set<std::string>;
class QuantizationSpecs;

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
// When the given run_tfl_runtime_verification value is true, it will check each
// TFL builtin op towards the TFL runtime capability and the incompatible TF ops
// will be left in the graph without getting legalized.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeTFPass(
    bool run_tfl_runtime_verification);

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizePass(
    bool enable_canonicalization);

// Creates an instance of the TensorFlow Lite dialect PrepareTF pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareTFPass(
    bool unfold_batch_matmul, bool allow_bf16_and_f16_type_legalization,
    bool use_fake_quant_num_bits = false);

// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLowerStaticTensorListPass(
    bool allow_tensorlist_pass_through = false,
    bool default_to_single_batch = false);

// Creates an instance of the TensorFlow Lite dialect Quantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    const QuantizationSpecs& quant_specs, const StringSet& ops_blocklist = {},
    const StringSet& nodes_blocklist = {});

// Overloading of CreateQuantizePass which takes only necessary flags to reduce
// the binary size.
std::unique_ptr<OperationPass<FuncOp>> CreateQuantizePass(
    bool verify_numeric = false, bool whole_model_verify = false,
    bool legacy_float_scale = false, const StringSet& ops_blocklist = {},
    const StringSet& nodes_blocklist = {});

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs);

// Creates an instance of the TensorFlow Lite dialect
// PrepareDynamicRangeQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareDynamicRangeQuantizePass(
    const QuantizationSpecs& quant_specs);

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops);

// Creates an instance of the TensorFlow Lite pass that decomposes hybrid
// quantization patterns to the same dense operation with tfl dequantization
// and quantization patterns.
std::unique_ptr<OperationPass<FuncOp>> CreateDecomposeHybridQuantizationPass();

// Creates an instance of the TensorFlow Lite optimize op order pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeOpOrderPass();

// Creates an instance of the TensorFlow Lite dialect TrimFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateTrimFunctionsPass(
    const std::vector<std::string>& trim_funcs_allowlist);

// Creates an instance of the TensorFlow Lite dialect PrepareCompositeFunctions
// pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareCompositeFunctionsPass();

// Creates an instance of the TensorFlow Lite dialect SplitMergedOperandsPass.
std::unique_ptr<OperationPass<FuncOp>> CreateSplitMergedOperandsPass();

// Creates an instance of the TensorFlow Lite dialect OptimizeFunctionalOpsPass.
std::unique_ptr<OperationPass<ModuleOp>> CreateOptimizeFunctionalOpsPass();

std::unique_ptr<OperationPass<FuncOp>> CreateModifyIONodesPass(
    mlir::Type input_type, mlir::Type output_type);

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizeRemoveQDQPass();

// Creates an instance of the TensorFlow Lite dialect pass to add default
// quantization parameters.
std::unique_ptr<OperationPass<FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max, bool is_signed);

// Creates an instance of the TensorFlow Lite dialect pass to convert dense
// tensor to sparse format.
std::unique_ptr<OperationPass<FuncOp>> CreateDenseToSparsePass();

// Creates function pass to legalize TF While to TFL While.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFWhilePass();

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateWhileOutlinePass();

// Creates a pass to remove operands of TFL WhileOp without changing outcomes.
std::unique_ptr<OperationPass<FuncOp>> CreateReduceWhileOperandsPass();

// Verifies runtime constraints.
std::unique_ptr<OperationPass<FuncOp>> CreateRuntimeVerifyPass();

// Creates raise custom ops pass, which legalize custom ops to TFL::CustomOp
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseCustomOpsPass(
    const std::vector<std::string>& target_ops);

// Creates raise custom ops pass, which legalize custom ops to TFL::CustomOp
std::unique_ptr<OperationPass<FuncOp>> CreateLowerCustomOpsPass();

// Inserts an TFL::CallOnce op when the tf_saved_model's session initialzer is
// given.
std::unique_ptr<OperationPass<ModuleOp>>
CreateInsertCallOnceOpFromSessionInitializerPass();

// Replace the tfl wrapped random function body with tfl.customOp.
std::unique_ptr<OperationPass<FuncOp>> CreateLegalizeJaxRandomPass();

// Creates a pass which is responsible for legalizing TensorFlow variables to
// TensorFlow Lite variables.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeVariablesPass();

// Creates a pass which analyze the model whether it is safe to use
// native TFLite variables or not.
std::unique_ptr<OperationPass<ModuleOp>> CreateAnalyzeVariablesPass();

// Creates a pass which is responsible for legalizing TensorFlow static hash
// tables to TensorFlow Lite hash tables.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeHashTablesPass();

// Creates get arithmetic count pass, which will calculate the arithmetic count
// for each ops.
std::unique_ptr<OperationPass<FuncOp>> CreateGetArithmeticCountPass();

// Creates unfold large constant pass, which will replace large splat constant
// tensors with fill op.
std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldLargeSplatConstantPass();
}  // namespace TFL

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASSES_H_
