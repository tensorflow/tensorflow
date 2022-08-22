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

include "mlir/Pass/PassBase.td"


def AnalyzeVariablesPass : Pass<"tfl-analyze-variables-pass", "mlir::ModuleOp"> {
  let summary = "Analyze variables in the graph";
  let description = [{
      Pass which analyzes the variables in the graph and add an attribute whether
      variables should be legalized to TFLite native ones.
      This pass needs to run post TF->TFL legalization and before variable
      legalization.
  }];
  let constructor = "CreateAnalyzeVariablesPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def DefaultQuantParamsPass : Pass<"tfl-default-quant", "mlir::func::FuncOp"> {
  let summary = "Apply quantization with default quantization parameter";
  let constructor = "CreateDefaultQuantParamsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let options = [
    Option<"default_min_", "default-min", "double", "-1.0", 
               "Default minimum value for TFLite quantization">,
    Option<"default_max_", "default-max", "double", "1.0",
               "Default maximum value for TFLite quantization">,
    Option<"is_signed_", "is-signed", "bool", "false",
               "Is the corresponding integer signed">,
  ];
}
def DecomposeHybridQuantizationPass : Pass<"tfl-decompose-hybrid-quantization", "mlir::func::FuncOp"> {
  let summary = "Decomposes hybridge quantization to explicit quantize / dequantize";
  let description = [{
      Decomposes (with explicit quantize/dequantize ops) selected math 
      operations which exist in the model with hybrid quantization 
      (some arguments/results left in floating point).
  }];
  let constructor = "CreateDecomposeHybridQuantizationPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def DenseToSparsePass : Pass<"tfl-dense-to-sparse", "mlir::func::FuncOp"> {
  let summary = "Convert dense tensor to sparse format.";
  let description = [{
      This pass encodes sparse weights in the model in the proper format, and adds
      Densify() op if necessary. The general algorithm is:
        1. Get list of operands (weights) of an op that can be sparse.
        2. Get list of supported block configurations of the op.
        3. Calculate random sparsity of the weight.
          3.1. If sparsity level is below the encoding threshold, keep in dense.
          3.2. If sparsity level is above the encoding threshold, go to 4.
        4. Try to encode the weight with supported block configurations. If the
           weight was pruned with the same block config, the blocked sparsity level
           should match the random sparsity.
          4.1. Return the matching block config if found.
          4.2. If no matching block config is found, encode the weight with random
               sparsity, and add Densify() op to fall back to dense execution.
  }];
  let constructor = "CreateDenseToSparsePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let options = [
    Option<"default_min_", "default-min", "double", "-1.0", 
               "Default minimum value for TFLite quantization">,
    Option<"default_max_", "default-max", "double", "1.0",
               "Default maximum value for TFLite quantization">,
    Option<"is_signed_", "is-signed", "bool", "false",
               "Is the corresponding integer signed">,
  ];
}

def IdentifyDilatedConvPass : Pass<"tfl-identify-dilated-conv", "mlir::func::FuncOp"> {
  let summary = "Convert dense tensor to sparse format.";
  let constructor = "CreateIdentifyDilatedConvPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def GetArithmeticCountPass : Pass<"tfl-get-arithmetic-count", "mlir::func::FuncOp"> {
  let summary = "Calculate arithmetic count for tfl operations.";
  let constructor = "CreateGetArithmeticCountPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def InsertCallOnceOpFromSessionInitializerPass : Pass<"tfl-insert-call-once-op", "mlir::ModuleOp"> {
  let summary = "Insert CallOnce op when tf_saved_model's session initializer is give.";
  let constructor = "CreateInsertCallOnceOpFromSessionInitializerPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LegalizeHashTablesPass : Pass<"tfl-legalize-hashtables-tf", "mlir::ModuleOp"> {
  let summary = "Legalize TensorFlow hash tables to TensorFlow Lite dialect.";
  let constructor = "CreateLegalizeHashTablesPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LegalizeJaxRandomPass : Pass<"tfl-legalize-random", "mlir::func::FuncOp"> {
  let summary = "Replace jax.random.uniform/normal with tfl.custom.";
  let constructor = "CreateLegalizeJaxRandomPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LegalizeTFPass : Pass<"tfl-legalize-tf", "mlir::func::FuncOp"> {
  let summary = "Legalize from TensorFlow to TensorFlow Lite dialect.";
  let constructor = "CreateLegalizeTFPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect" ,
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
      Option<"run_tfl_runtime_verification_", "run-tfl-runtime-verification",
             "bool", "true", "Allow tfl runtime verification.">,
      Option<"preserve_assert_op_", "preserve-assert-op",
             "bool", "false", "Preserve AssertOp during tfl legalization.">,
  ];
}

def LegalizeWhilePass : Pass<"tfl-legalize-tf-while", "mlir::ModuleOp"> {
  let summary = "Legalize from TensorFlow While to TensorFlow Lite While.";
  let constructor = "CreateLegalizeTFWhilePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LegalizeVariablesPass : Pass<"tfl-legalize-variables-tf", "mlir::ModuleOp"> {
  let summary = "Legalize TensorFlow variables to TensorFlow Lite dialect.";
  let constructor = "CreateLegalizeVariablesPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LiftTfliteFlexOpsPass : Pass<"tfl-lift-tflite-flex-ops", "mlir::func::FuncOp"> {
  let summary = "Lifts TFLite Custom ops into TF dialect operations.";
  let constructor = "CreateLiftTfliteFlexOpsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def LowerStaticTensorListPass : Pass<"tfl-lower-static-tensor-list", "mlir::ModuleOp"> {
  let summary = "Lower TensorList ops within TensorFlow Lite dialect.";
  let constructor = "CreateLowerStaticTensorListPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
      Option<"allow_tensorlist_pass_through_", "allow-tensorlist-pass-through",
             "bool", "false",
             "When specified to true, if the tensorlist ops can't be properly "
             "legalized by this pass, then the IR won't be changed so that "
             "tensorlist ops can pass through (default false).">,
      Option<
        "default_to_single_batch_", "default-to-single-batch", "bool", "false", 
        "When specified to true, if the tensorlist ops has unspecified batch "
        "size, this pass will assume that the batch size is one to proceed "
        "tensorlist op lowering (default true)">,
      Option<"enable_dynamic_update_slice_", "enable-dynamic-update-slice",
             "bool", "false", "When specified to true, lower TensorListSetItem with "
             "DynamicUpdateSlice op (default false).">,
  ];
}

def ModifyIONodesPass : Pass<"tfl-modify-io-nodes", "mlir::func::FuncOp"> {
  let summary = "Modify the type of the model io nodes";
  let description = [{
      This transformation pass modifies the input and output types of the function
      to what are specified. The task was not just adding cast operations, but,
      instead, using tfl.quantize and tfl.dequantize ops to scale the tensors.
  }];
  let constructor = "CreateModifyIONodesPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let options = [
      ListOption<"io_node_types_", "test-io-types", "std::string",
                 "comma separated type strings. Allowed values: 'int8', 'uint8', 'float32']">,
  ];
}

def OptimizePass : Pass<"tfl-optimize", "mlir::func::FuncOp"> {
  let summary = "Optimize within the TensorFlow Lite dialect";
  let constructor = "CreateOptimizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let options = [
      Option<"enable_canonicalization_", "enable-canonicalization",
             "bool", "false",
             "Enable canonicalization during optimization pass.">,
  ];
}

def OptimizeFunctionalOpsPass : Pass<"tfl-optimize-functional-ops", "mlir::ModuleOp"> {
  let summary = "Optimize TensorFlow functional op";
  let constructor = "CreateOptimizeFunctionalOpsPass()";
}

def OptimizeOpOrderPass : Pass<"tfl-optimize-op-order", "mlir::func::FuncOp"> {
  let summary = "Optimize the execution order of the ops.";
  let description = [{
      This transformation pass optimizes the op execution order of the ops in
      the model.
  }];
  let constructor = "CreateOptimizeOpOrderPass()";
}

def PartitionedTopologicalSortPass : Pass<"tfl-partitioned-topological-sort", "mlir::func::FuncOp"> {
  let summary = "Re-sort execution order such that delegated ops stay together";
  let constructor = "CreatePartitionedTopologicalSortPass()";
    let description = [{
      This transformation reorders operations such that operations that will be
      executed by the Flex delegate will be followed by another Flex delegated
      operator, if possible. The reordering uses the same greedy procedure that
      is executed at runtime (in tensorflow/lite/graph_info.cc.)
      This allows us to have an IR of the model that is in the same execution order
      as it will have at runtime (provided Flex is the only delegate present); this
      allows us to do optimizations such as rematerialization.
    }];
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def PinOpsWithSideEffectsPass : Pass<"tfl-pin-ops-with-side-effects", "mlir::func::FuncOp"> {
  let summary = "Pin operators with side effects";
  let description = [{
      This transformation pass wraps operations that have or depend on side effects in
      TFL::ControlNodeOp, which depend on the control token generated by the most recent
      preceding such operation, if any. This copies the logic that is currently executed
      at runtime to enforce that the relative order of side-effectful operations is
      preserved and expresses it in terms of control dependencies.

      For purposes of this pass, an operator is considered to have/depend on side effects if
        - it calls a different function
        - it depends on or generates resource variable handles
  }];
  let constructor = "CreatePinOpsWithSideEffectsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def PostQuantizePass : Pass<"tfl-post-quantize", "mlir::func::FuncOp"> {
  let summary = "Apply post quantization clean up after quantization.";
  let constructor = "CreatePostQuantizePass()";
  let options = [
      Option<"emit_quant_adaptor_ops_", "emit-quant-adaptor-ops",
             "bool", "false",
             "Enable canonicalization during optimization pass.">,
      Option<"enable_custom_op_no_side_effect_", "enable-no-side-effect",
            "std::string", "", "Specifies which custom ops are NoSideEffect.">,
  ];
}

def PostQuantizeRemoveQDQPass : Pass<"tfl-post-quantize-remove-qdq", "mlir::func::FuncOp"> {
  let summary = "Remove qdq from input and output nodes after quantization.";
  let constructor = "CreatePostQuantizeRemoveQDQPass()";
}

def PrepareCompositeFunctionsPass : Pass<"tfl-prepare-composite-funcs-tf", "mlir::ModuleOp"> {
  let summary = "Prepares composite functions in Tensorflow dialect of MLIR.";
  let constructor = "CreatePrepareCompositeFunctionsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
  let description = [{
      This pass uses mechanisms listed in RFC:
      https://github.com/tensorflow/community/pull/113
      It prepares composite functions that are attributed to indicate
      a specific interface (LSTM, SVDF, Embedding lookup etc.) by replacing the
      body with the corresponding fused TFLite op. The replacement need not
      always be a fused op, though that is the primary use case.
  }];
  let options = [
      Option<"tfl_fuse_tftext_", "fuse-tftext", "bool", "false",
             "Fuse TF.Text API ops when it's true">
  ];

}

def PrepareQuantizePass : Pass<"tfl-prepare-quantize", "mlir::func::FuncOp"> {
  let summary = "Remove qdq from input and output nodes after quantization.";
  let constructor = "CreatePrepareQuantizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "quant::QuantizationDialect",
    "quantfork::QuantizationForkDialect"
  ];
  let options = [
      ListOption<"quantize_allowlist_", "quantize-allowlist", "std::string",
                 "comma separated list of allowlisted functions to be quantized. Only used in tests">,
      Option<"quantize_signed_", "quantize-signed", "bool", "false",
             "signed inference type. Only used in tests">,
      Option<"post_training_quantize_", "post-training-quantize", "bool", "false",
             "enable post training quantization. Only used in tests">,
      Option<"legacy_float_scale_", "legacy-float-scale", "bool", "false",
             "calculate quantization scales in float instead of double">,
      Option<"disable_per_channel_", "disable-per-channel", "bool", "false",
             "Whether disable per-channel quantized weights.">,
      Option<"disable_set_input_nodes_quantization_params_",
             "disable-set-input-nodes-quantization-params",
             "bool", "false",
             "Whether disable set input nodes quantization parameters.">,
  ];
}

def PrepareDynamicRangeQuantizePass : Pass<"tfl-prepare-quantize-dynamic-range", "mlir::func::FuncOp"> {
  let summary = "Prepare TFL dialect for dynamic range quantization.";
  let constructor = "CreatePrepareDynamicRangeQuantizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect"
  ];
  let options = [
      Option<"enable_dynamic_range_per_channel_quantization_",
              "enable-dynamic-range-per-channel-quantization", "bool",
              "true", "Whether enable per-channel quantized weights.">,
      Option<"min_elements_for_weights_",
              "min-elements-for-weights", "int64_t", "1024",
              "The minimum number of elements in a weights array required to apply quantization.">,
      Option<"enable_float16_quantization_",
              "enable-float16-quantization", "bool",
              "false", "Whether apply float16 quantization. If false, int8 quantization is applied.">,
      Option<"enable_custom_op_quantization_",
              "enable-custom-op-quantization", "std::string", "",
              "Specifies which pairs of a custom op and indices are quantizable where the indices are separated with a space.">,
  ];
}

def PrepareTFPass : Pass<"tfl-prepare-tf", "mlir::func::FuncOp"> {
  let summary = "Prepare TF for legalization to TensorFlow Lite dialect.";
  let constructor = "CreatePrepareTFPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect",
    "mhlo::MhloDialect"
  ];
  let options = [
      Option<"unfold_batch_matmul_", "emit-quant-adaptor-ops",
             "bool", "true",
             "Unfold BatchMatMul into individual MatMul ops.">,
      Option<"allow_bf16_and_f16_type_legalization_", "allow-bf16-and-f16-type-legalization",
             "bool", "false",
             "Allow bf16 type legalization.">,
      Option<"use_fake_quant_num_bits_", "use-fake-quant-num-bits",
             "bool", "false",
             "Use quantization calculated from fake quant attributes.">,
  ];
}

def QuantizePass : Pass<"tfl-quantize", "mlir::func::FuncOp"> {
  let summary = "Apply quantization on models in TensorFlow Lite dialect.";
  let constructor = "CreateDefaultQuantizePass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect",
    "mlir::quant::QuantizationDialect",
    "mlir::quantfork::QuantizationForkDialect"
  ];

  let options = [
      Option<"enable_numeric_verify_", "numeric-verify",
             "bool", "false",
             "Whether verify numericals at runtime.">,
      Option<"error_tolerance_", "error-tolerance",
             "float", "5.0f",
             "Error tolerance for numeric verify. Valid when `-numeric-verify` is set.">,
      Option<"enable_whole_model_verify_", "whole-model-verify",
             "bool", "false",
             "Whether verify numericals layer by layer or whole model. Valid when `-numeric-verify` is set.">,
      Option<"enable_log_if_failed_", "log-if-failed",
             "bool", "false",
             "Whether verify numericals with thresholding tolerance. Valid when `-numeric-verify` is set.">,
      Option<"enable_dynamic_range_quantization_", "enable-dynamic-range-quantization",
             "bool", "false",
             "Whether run post-training dynamic range quantization pass">,
      Option<"enable_weight_only_quantization_", "enable-weight-only-quantization",
             "bool", "false",
             "Whether to run weight-only for post-training dynamic range quantization pass">,
      Option<"enable_legacy_quantize_", "legacy-quantize",
             "bool", "false",
             "Use legacy quantize mode in test. Valid when `-legacy-quantize` is set.">,
      ListOption<"ops_blocklist_flag_", "ops-blocklist",
             "std::string", "Names of ops to blocklist from quantization">,
      ListOption<"nodes_blocklist_flag_", "locs-blocklist",
             "std::string", "Names of location to blocklist from quantization">,
      Option<"enable_custom_op_weight_only_", "enable-custom-op-weight-only",
             "std::string", "", "Specifies which custom ops are weight-only.">,
  ];
}

def RaiseCustomOpsPass : Pass<"tfl-raise-custom-ops", "mlir::func::FuncOp"> {
  let summary = "Raise custom ops into tflite dialect.";
  let constructor = "CreateRaiseCustomOpsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];

  let options = [
      ListOption<"target_ops_", "test-raise-tf-targets", "std::string",
             "comma separated list of target op names to be wrapped. Only used in tests">,
  ];
}

def ReduceWhileOperandsPass : Pass<"tfl-reduce-while", "mlir::func::FuncOp"> {
  let summary = "Reduce the number of operands and results of a whlieOp..";
  let constructor = "CreateReduceWhileOperandsPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect", "TF::TensorFlowDialect"];
}

def RuntimeVerifyPass : Pass<"tfl-runtime-verify", "mlir::func::FuncOp"> {
  let summary = "TFLite runtime verification";
  let constructor = "CreateRuntimeVerifyPass()";
}

def SplitMergedOperandsPass : Pass<"tfl-split-merged-operands", "mlir::func::FuncOp"> {
  let summary = "Split merged stateful operands for tfl operations.";
  let constructor = "CreateSplitMergedOperandsPass()";
}

def TrimFunctionsPass : Pass<"tfl-trim-funcs-tf", "mlir::ModuleOp"> {
  let summary = "Trim functions to restrict them to a specified allowlist prior to legalization to TensorFlow lite dialect";
  let constructor = "CreateTrimFunctionsPass()";
  let options = [
      ListOption<"trim_funcs_allowlist_", "trim-funcs-allowlist", "std::string",
                 "comma separated list of allowlisted functions. The first "
                 "function specified will be used as main">
  ];
}

def UnfoldLargeSplatConstantPass : Pass<"unfold-large-splat-constant", "mlir::ModuleOp"> {
  let summary = "Unfold large splat constant tensors.";
  let constructor = "CreateUnfoldLargeSplatConstantPass()";
  let dependentDialects = ["TFL::TensorFlowLiteDialect"];
}

def WhileOutlinePass : Pass<"tfl-while-loop-outline", "mlir::ModuleOp"> {
  let summary = "Hoist while op regions into functions";
  let constructor = "CreateWhileOutlinePass()";
  let dependentDialects = ["TF::TensorFlowDialect"];
}
