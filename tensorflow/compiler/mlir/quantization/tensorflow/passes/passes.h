/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir {
namespace quant {

// Creates a main function if it doesn't exist in the module. This is a
// workaround to make ConvertMlirToGraphdef work for multi-signatures graphs.
// TODO(b/204265523): Removes this pass after the exporting MLIR to SavedModel
// path is available.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertMainFunctionPass();

// Converts FakeQuant ops to quant.qcast and quant.dcast (QDQ) pairs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertFakeQuantToQdqPass();

// Lifts the quantizable spots as composite functions.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsPass(
    const tensorflow::quantization::QuantizationOptions& quant_options);

// Apply graph optimizations such as fusing and constant folding to prepare
// lifting.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass(
    tensorflow::quantization::OpSet target_opset);

// Lifts the dynamic range quantizable spots as composite functions.
std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsDRQPass(
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    tensorflow::quantization::OpSet op_set, int min_num_elements_for_weights);

// Replaces tf.CustomAggregator ops with quant.Stats ops for finalizing the
// calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertCustomAggregationOpToQuantStatsPass();

// Issues IDs of custom aggregation ops for preparing the calibration procedure.
std::unique_ptr<OperationPass<ModuleOp>>
CreateIssueIDsOfCustomAggregationOpsPass();

// Inserts quantized function library.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertQuantizedFunctionsPass(
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    tensorflow::quantization::OpSet target_opset);

// Inserts custom aggregation operators for the calibration procedure.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateInsertCustomAggregationOpsPass(
    const ::stablehlo::quantization::CalibrationOptions& calib_opts);

// Replaces composite functions with quantized composite functions. After this
// pass runs, functions in the given graph will be replaced with their quantized
// versions. By doing so, the quantization will be applied to the given input.
// mlir_dump_file_prefix is an optional field that is used for debugging to save
// mlir dump files.
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    tensorflow::quantization::OpSet target_opset,
    bool enable_per_channel_quantization, int min_num_elements_for_weights,
    bool enable_legacy_weight_only = false,
    std::optional<const absl::string_view> mlir_dump_file_prefix =
        std::nullopt);

// Converts dequantize-(quantizable) call-quantize pattern to a single call op
// that has quantized input and output types. It is expected for this pass to
// emit illegal IR with unsupported quantized input and output types. The
// pass following immediately after this one will be responsible for legalizing
// input and output types by unwrapping quantization parameters.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass();

// Overloading of CreateQuantizePass which takes QuantizationSpecs.
std::unique_ptr<OperationPass<func::FuncOp>> CreateQuantizePass(
    QuantizationSpecs quant_specs,
    tensorflow::quantization::OpSet target_opset);

// Creates an instance of the PrepareQuantize pass, which will perform similar
// transformations as TFL::PrepareQuantizePass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs,
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method);

// Creates an instance of the PrepareQuantizeDRQ pass, which will
// perform similar transformations as TFL::PrepareQuantizeDynamicRangePass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizeDRQPass(
    const QuantizationSpecs& quant_specs,
    tensorflow::quantization::OpSet op_set);

// Creates an instance of the PreprocessOp pass, which will perform op
// preprocessing to allow multi-axis quantization, prior to quantization.
std::unique_ptr<OperationPass<ModuleOp>> CreatePreprocessOpPass(
    tensorflow::quantization::OpSet op_set,
    tensorflow::quantization::QuantizationMethod::PresetMethod
        quantization_method,
    bool enable_per_channel_quantization);

// Creates an instance of the PostQuantize pass, which will remove unnecessary
// ops from the final quantized graph.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass();

// Applies optimization patterns after quantization.
std::unique_ptr<OperationPass<mlir::func::FuncOp>> CreateOptimizePass();

// Creates an instance of the ReplaceCastHacksWithTFXLAOpsPass, which will
// replace mixed-type convolution and matmul cast hacks by XLA Conv2DOp and
// MatmulOp.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateReplaceCastHacksWithTFXLAOpsPass();

// Creates a pass that moves & merges initializer function's ops into the @main
// function. This pass should be run on a valid tf_executor dialect. The control
// output of the initializer function for non-variable resource initialization
// will be passed on as a dependency to a new `tf.NoOp`, whose control output
// will be merged into the main function's FetchOp. The initializer functions
// will be removed.
//
// Running this pass essentially has the effect of inlining the initializer
// functions into the main graph. This is beneficial when we wish to find and
// fetch the node that restores resources, after the ModuleOp has been exported
// as GraphDef.
std::unique_ptr<OperationPass<ModuleOp>>
CreateMergeInitializerFunctionOpsToMainPass();

// Creates a pass that moves & merges the "@tf_quant__save" function to "@main"
// function. A new `IdentityOp` will be created. It will have control dependency
// to the save function and returns the file_prefix argument (typed
// `tensor<!tf_type.string>`). The file_prefix argument, which can be identified
// if the "tf_saved_model.index_path" attribute has "__tf_file_prefix", will be
// reused if it already exist in @main. Otherwise a new file prefix argument
// will be created. @tf_quant__save function will be erased.
//
// Running this pass essentially has the effect of inlining the @tf_quant__save
// into the main graph. This is beneficial when we wish to find and fetch
// the node that saves the variables, after the ModuleOp has been exported as
// GraphDef.
std::unique_ptr<OperationPass<ModuleOp>> CreateMergeSaveFunctionOpsToMainPass();

// Creates a pass that "unfreezes" ConstOps into variables. Each ConstOp's use
// will be replaced by a VarHandleOp -> ReadVariableOp pattern. The newly
// created variables will be initialized in the session initializer function via
// AssignVariableOps.
std::unique_ptr<OperationPass<ModuleOp>> CreateUnfreezeConstantsPass();

// Creates a pass that duplicates constants that affect the shape of a tensor
// after some computation.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateDuplicateShapeDeterminingConstantsPass();

// Creates a pass that creates a RestoreV2 op in the initializer function with
// type "restore_op" that initializes variables from the checkpoint. It finds
// tf.AssignVariableOp(tf.VarHandleOp, tf.Const) patterns in the initializer
// function and replaces tf.Consts with the results of RestoreV2.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertRestoreOpPass();

// Creates a pass that creates a new function that wraps the newly created
// SaveV2 op. The new function's name is "tf_quant__save". The function accepts
// a single string tensor as argument, which specifies the path to the
// checkpoint to which the variable's tensor values are saved. It finds
// `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` pattern in the initializer
// function of type "restore_op" to identify the VarHandleOps that should be
// saved using the SaveV2 op.
std::unique_ptr<OperationPass<ModuleOp>> CreateInsertSaveOpPass();

// Creates a pass that marks functions with the attribute `tf._noinline = true`
// to avoid being inlined by the `InlinerPass`. `noinline_functions` is the name
// of the functions to mark.
std::unique_ptr<OperationPass<func::FuncOp>> CreateMarkFunctionsNoinlinePass(
    ArrayRef<std::string> noinline_functions);

// Removes `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns from the
// initializer function (type = "restore_op").
// Note: initializing values (`tf.Const`s) will be removed and this may result
// in an information loss and uninitialized variables eventually. Make sure that
// this effect is desired (e.g. there is a `tf.RestoreV2Op` that restores the
// variables instead).
std::unique_ptr<OperationPass<ModuleOp>>
CreateRemoveVariableInitializationByConstPass();

// Creates a pass that converts Tensorflow Xla ops to non-Xla ops.
std::unique_ptr<OperationPass<func::FuncOp>> CreateConvertTfXlaOpToTfOpPass();

// Creates a pass that converts TPU models for CPU by removing TPU related ops
// such as TPUPartitionedCall, TPUReplicatedOp, etc. The TF quantizer does not
// work with models specifically designed for TPU, so this pass makes the input
// TPU model compatible with the TF quantizer by rewriting the TPU ops. The
// output model of this pass is expected to be ready for the TF quantizer.
std::unique_ptr<OperationPass<ModuleOp>> CreateConvertTpuModelToCpuPass();

// Creates a pass that casts BFloat16 operations to Float32 operations. This
// pass is a part of the ConvertTpuModelToCpu pass to support BF16 optimized TPU
// model quantization.
std::unique_ptr<OperationPass<ModuleOp>> CreateCastBf16OpsToF32Pass();

// Creates a pass that lifts HashTable ops as function arguments. In the graph
// execution mode, resource ops with the same `shared_name` attribute point to
// the same underlying resource. This is not true in the eager execution mode.
// Lifting resource ops as arguments will help unifying them across functions.
std::unique_ptr<OperationPass<ModuleOp>> CreateLiftHashTableOpsAsArgsPass();

// Creates a pass that merges duplicate resource ops in each function. Two
// resource ops are considered duplicated if they have the same `shared_name`.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateMergeDuplicateResourceOpsPass();

// Apply quantization to weights based on the provided schemes.
std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeWeightsPass(
    const tensorflow::quantization::QuantizationOptions& quant_options);

// Propagate quantized type through allowed ops.
std::unique_ptr<OperationPass<ModuleOp>> CreatePropagateQuantizeTypePass();

// Create a pass that inserts dump tensor to quantizable layer's output.
std::unique_ptr<OperationPass<ModuleOp>> CreateAddDumpTensorOpPass(
    ::stablehlo::quantization::DebuggerConfig::DebuggerType debugger_type,
    std::string log_dir_path);

// Creates a pass that add QuantizationUnitLoc to quantizable layers.
std::unique_ptr<OperationPass<func::FuncOp>> CreateAddQuantizationUnitLocPass();

}  // namespace quant
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_PASSES_H_
