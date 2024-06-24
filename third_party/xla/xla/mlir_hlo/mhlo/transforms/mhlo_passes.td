/* Copyright 2020 The OpenXLA Authors.

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

def ChloLegalizeToHighLevelMhloPass : Pass<"chlo-legalize-to-high-level-mhlo", "func::FuncOp"> {
  let summary = "Legalize CHLO's with XLA counterparts, like TopK and Erf.";
  let description = [{
    Performs direct legalization of CHLO->MHLO only for high-level (non-basis)
    ops with XLA support. These are MHLO ops that directly model the CHLO op,
    such as TopK and Erf.
  }];
  let dependentDialects = ["mhlo::MhloDialect"];
}

def ChloLegalizeToHloPass : Pass<"chlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "Legalize CHLO to MHLO with XLA-supported ops.";
  let description = [{
    Performs legalization of CHLO->StableHLO->MHLO, while also preserving MHLO
    high level operations when possible (see ChloLegalizeToHighLevelMhloPass).
  }];
  let dependentDialects = [
    "mhlo::MhloDialect",
    "mlir::shape::ShapeDialect",
    "mlir::stablehlo::StablehloDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def HloCanonicalizeScatterPass : Pass<"hlo-canonicalize-scatter", "func::FuncOp"> {
  let summary = "Rewrites scatter into transposes, reshapes and a simple scatter.";
  let constructor = "createHloCanonicalizeScatterPass()";
  let dependentDialects = ["mhlo::MhloDialect", "tensor::TensorDialect"];
}

def HloCanonicalizeGatherPass : Pass<"hlo-canonicalize-gather", "func::FuncOp"> {
  let summary = "Rewrites gather into transposes, reshapes and a simple gather.";
  let constructor = "createHloCanonicalizeGatherPass()";
  let dependentDialects = ["mhlo::MhloDialect", "tensor::TensorDialect"];
}

def HloCanonicalizeReductionPass : Pass<"hlo-canonicalize-reduction", "func::FuncOp"> {
  let summary = "Canonicalize reduction ops to be suitable for codegen.";
  let constructor = "createHloCanonicalizeReductionPass()";
}

def HloCanonicalizeDotPass : Pass<"hlo-canonicalize-dot", "func::FuncOp"> {
  let summary = "Rewrites dot operands that contain unit dimension.";
  let constructor = "createHloCanonicalizeDotPass()";
  let dependentDialects = ["mhlo::MhloDialect", "tensor::TensorDialect"];
}

def HloLegalizeToMemrefPass :Pass<"hlo-legalize-to-memref", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to memref dialect.";
  let constructor = "createLegalizeToMemrefPass()";
}

def HloLegalizeToArithmeticPass :Pass<"hlo-legalize-to-arithmetic", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to arithmetic dialect.";
  let constructor = "createLegalizeToArithmeticPass()";
}

def HloLegalizeSortPass : Pass<"hlo-legalize-sort", "func::FuncOp"> {
  let summary = "Legalize from MHLO sort to SCF control flow.";
  let constructor = "createLegalizeSortPass()";
  let dependentDialects = ["arith::ArithDialect",
                           "bufferization::BufferizationDialect",
                           "scf::SCFDialect", "tensor::TensorDialect"];
}

def LegalizeBroadcastToBroadcastInDimPass : Pass<"mhlo-legalize-broadcast-to-broadcast-in-dim", "func::FuncOp"> {
  let summary = "Legalizes broadcast ops to broadcast_in_dim ops.";
  let constructor = "createLegalizeBroadcastToBroadcastInDimPass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def LegalizeControlFlowPass : Pass<"mhlo-legalize-control-flow", "func::FuncOp"> {
  let summary = "Legalize from MHLO control flow to SCF control flow.";
  let constructor = "createLegalizeControlFlowPass()";
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

def LegalizeCreateTokenToAfterAllPass : Pass<"mhlo-legalize-create-token-to-after-all", "func::FuncOp"> {
  let summary = "Legalize create_token ops to after_all ops.";
  let constructor = "createLegalizeCreateTokenToAfterAllPass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def LegalizeCrossReplicaSumToAllReducePass : Pass<"mhlo-legalize-cross-replica-sum-to-all-reduce", "func::FuncOp"> {
  let summary = "Legalizes cross-replica-sum ops to all_reduce ops.";
  let constructor = "createLegalizeCrossReplicaSumToAllReducePass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def LegalizeDotGeneralToDotPass : Pass<"mhlo-legalize-dot-general-to-dot", "func::FuncOp"> {
  let summary = "Legalizes dot_general ops to dot ops.";
  let constructor = "createLegalizeDotGeneralToDotPass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def LegalizeDotToDotGeneralPass : Pass<"mhlo-legalize-dot-to-dot-general", "func::FuncOp"> {
  let summary = "Legalizes dot ops to dot_general ops.";
  let constructor = "createLegalizeDotToDotGeneralPass()";
}

def LegalizeEinsumToDotGeneralPass : Pass<"mhlo-legalize-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "Legalizes einsum ops to dot_general ops.";
  let constructor = "createLegalizeEinsumToDotGeneralPass()";
}

def LegalizeGatherToTorchIndexSelectPass : Pass<"mhlo-legalize-gather-to-torch-index-select", "func::FuncOp"> {
  let summary = "Legalizes gathers to a torch index select.";
  let constructor = "createLegalizeGatherToTorchIndexSelectPass()";
}

def LegalizeTorchIndexSelectToGatherPass : Pass<"mhlo-legalize-torch-index-select-to-gather", "func::FuncOp"> {
  let summary = "Legalizes torch index select to a gather.";
  let constructor = "createLegalizeTorchIndexSelectToGatherPass()";
}


def LegalizeTanhToApproximationPass : Pass<"mhlo-legalize-trigonometric-to-approximation", "func::FuncOp"> {
  let summary = "Legalize trigonometric operations from standard dialect to an approximation.";
  let constructor = "createLegalizeTrigonometricToApproximationPass()";
}

def HloLegalizeToLinalgPass : Pass<"hlo-legalize-to-linalg", "func::FuncOp"> {
  let summary = "Legalize from HLO dialect to Linalg dialect.";
  let constructor = "createLegalizeHloToLinalgPass()";
  let options = [Option<"enablePrimitiveOps", "enable-primitive-ops", "bool",
                        /*default=*/"false",
                        "Lower to primitive Linalg ops (map, reduce and "
                        "transpose) when possible, instead of linalg.generic">];
}

def HloLegalizeShapeComputationsPass : Pass<"hlo-legalize-shape-computations", "func::FuncOp"> {
  let summary = "Legalize HLOs shape operations to core-mlir operations.";
  let constructor = "createLegalizeShapeComputationsPass()";
}

def LegalizeToStandardPass : Pass<"mhlo-legalize-to-std", "func::FuncOp"> {
  let summary = "Legalize from MHLO dialect to standard dialect.";
  let constructor = "createLegalizeToStdPass()";
}

def LowerComplexPass : Pass<"mhlo-test-lower-complex", "func::FuncOp"> {
  let summary = "Lower complex operations into non-complex operations.";
  let constructor = "createLowerComplexPass()";
}

def LegalizeGeneralDotPass : Pass<"mhlo-test-lower-general-dot", "func::FuncOp"> {
  let summary = "Tests lowering general dot to a non-batched dot when possible.";
  let constructor = "createLegalizeGeneralDotPass()";
}


def TestMaterializeBroadcastsPass : Pass<"mhlo-test-materialize-broadcasts", "func::FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestMaterializeBroadcastsPass()";
}

def OptimizeMhloPass : Pass<"mhlo-test-optimize", "func::FuncOp"> {
  let summary = "Run optional HLO optimizations.";
  let constructor = "createOptimizeMhloPass()";
}

def SinkConstantsToControlFlowPass : Pass<"mhlo-sink-constants-to-control-flow", "func::FuncOp"> {
  let summary = "Sink constants implicitly captured in control flow regions. This "
    "is necessary to export to XLA.";
  let constructor = "createSinkConstantsToControlFlowPass()";
  let description = [{
    A pass that sinks constants implicitly captured in control flow regions. This
    is necessary to export to XLA, because XLA's representation of control flow
    doesn't have the notion of implicit capture.

    For example given this function:

    ```mlir
      func @sink_const_to_sort(%arg0: tensor<16xf32>) {
        %c0 = arith.constant dense<1.0> : tensor<f32>
        %0 = "mhlo.sort"(%arg0) ( {
        ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
          %1 = "mhlo.divide"(%arg1, %c0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          %2 = "mhlo.divide"(%arg2, %c0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
          %3 = "mhlo.compare"(%1, %2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
          "mhlo.return"(%3) : (tensor<i1>) -> ()
        }) {is_stable = true} : (tensor<16xf32>) -> tensor<16xi32>
        return
      }
    ```

    Observe how the arith.constant is moved into the region it's used in:

    ```mlir
      module  {
        func @sink_const_to_sort(%arg0: tensor<16xf32>) {
          %0 = "mhlo.sort"(%arg0) ( {
          ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
            %cst = arith.constant dense<1.000000e+00> : tensor<f32>
            %1 = mhlo.divide %arg1, %cst : tensor<f32>
            %2 = mhlo.divide %arg2, %cst : tensor<f32>
            %3 = "mhlo.compare"(%1, %2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
            "mhlo.return"(%3) : (tensor<i1>) -> ()
          }) {is_stable = true} : (tensor<16xf32>) -> tensor<16xi32>
          return
        }
      }
    ```
  }];
}

def TestInferShapedTypeMethodsPass : Pass<"mhlo-test-infer-shaped-type-methods", "func::FuncOp"> {
  let summary = "Uses test ops to invoke InferShapedTypeOpInterface methods.";
  let constructor = "createTestInferShapedTypeMethodsPass()";
}

def BroadcastPropagationPass : Pass<"mhlo-broadcast-propagation", "func::FuncOp"> {
  let summary = "Move dynamic broadcasts up over element-wise operations and "
    "broadcast the operands rather than the result. This will eventually allow "
    "for larger fusions.";
  let constructor = "createBroadcastPropagationPass()";
}

def RestrictMaxRankPass : Pass<"mhlo-restrict-max-rank", "func::FuncOp"> {
  let summary = "Restrict maximum rank of any of the intermediate tensors";
  let description = "Transform operations in the module so that the maximum "
  "rank is restricted. This is done by doing transformations that could "
  "potentially increase overhead but helps in reducing dimensionality. This "
  "could be useful in backends that don't support higher ranked tensors.";
  let constructor = "createRestrictMaxRankPass()";
}

def MergeAssumingOpsPass : Pass<"mhlo-merge-assuming-ops", "func::FuncOp"> {
  let summary = "Prepare moving dynamic broadcasts up over element-wise "
    "operations and broadcast the operands rather than the result. This will "
    "eventually allow for larger fusions.";
  let constructor = "createMergeAssumingOpsPass()";
}

def ShapeReificationPass : Pass<"shape-reification", "func::FuncOp"> {
  let summary = "Iteratively reify all shape computations.";
  let constructor = "createShapeReificationPass()";
}

def SymbolicShapeOptimization : Pass<"symbolic-shape-optimization", "func::FuncOp"> {
  let summary = "Analyzes shapes and performs shape-related optimizations";
  let constructor = "createSymbolicShapeOptimizationPass()";
}

def ShapeSimplification
    : Pass<"shape-simplification", "mlir::func::FuncOp"> {
  let summary = "Simplify shape ops";
  let constructor = "createShapeSimplification()";
}

def ConstraintFusionPass : Pass<"constraint-fusion", "func::FuncOp"> {
  let summary = "Fuse shape constraints and merge all assuming regions.";
  let constructor = "createConstraintFusionPass()";
}

def GroupReductionDimensionsPass
    : Pass<"group-reduction-dimensions", "func::FuncOp"> {
  let summary = "Group dimensions of reduction operations";
  let description = "Group reduction and parallel dimensions of reduction "
    "operations and realize them through equivalent 1D or 2D reductions, if "
    "possible.";
  let constructor = "createGroupReductionDimensionsPass()";
  let options = [
    Option<"prefer_columns_reductions_", "prefer-columns-reductions", "bool",
           /*default=*/"true", "When simplifying reductions, prefer to use "
           "column reductions over row reductions.">,
  ];
  let dependentDialects = ["tensor::TensorDialect"];
}

def TestUnfuseBatchNormPass : Pass<"mhlo-test-unfuse-batch-norm", "func::FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestUnfuseBatchNormPass()";

  let dependentDialects = ["arith::ArithDialect", "shape::ShapeDialect", "tensor::TensorDialect"];
}

def ExpandHloTuplesPass : Pass<"expand-hlo-tuples", "ModuleOp"> {
  let summary = "Expand HLO tuple for the entry function of the module.";
  let constructor = "createExpandHloTuplesPass()";
  let options = [
    Option<"entry_function_name_", "entry-function", "std::string",
           /*default=*/"", "the name of entry function of the module">,
  ];

  let dependentDialects = ["mhlo::MhloDialect"];
}

def FlattenTuplePass : Pass<"mhlo-flatten-tuple", "func::FuncOp"> {
  let summary = "Flatten tuples in operands and results of operators that "
    "support both tuple and variadic type.";
  let constructor = "createFlattenTuplePass()";
}

def ConvertToSignlessPass : Pass<"convert-to-signless", "ModuleOp"> {
  let summary = "Pass to transform the IR to be on signless integers.";
  let constructor = "createConvertToSignlessPass()";
}

def MhloExpandOpsSimplifierPass
    : Pass<"mhlo-expand-ops-simplifier", "func::FuncOp"> {
  let summary = "Expand feature rich mhlo ops into a set of simpler mhlo ops.";
  let constructor = "createMhloExpandOpsSimplifierPass()";
}

def CollapseElementwiseMapPass
    : Pass<"mhlo-collapse-elementwise-map", "func::FuncOp"> {
  let summary = "Collapse the mhlo.map if the map only has elementwise ops.";
  let constructor = "createCollapseElementwiseMapPass()";
}

def HloLegalizeToStablehloPass : Pass<"hlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize HLO to StableHLO.";
  let constructor = "createHloLegalizeToStablehloPass()";
  let dependentDialects = ["stablehlo::StablehloDialect"];
  let options = [
    Option<"allow_experimental_features_", "allow-experimental-features",
           "bool", /*default=*/"false",
           "Allow legalization of experimental MHLO features via StableHLO "
           "custom_call">
  ];
}

def StablehloLegalizeToHloPass : Pass<"stablehlo-legalize-to-hlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to HLO.";
  let constructor = "createStablehloLegalizeToHloPass()";
  let dependentDialects = ["mhlo::MhloDialect"];
}

def PrepareForExportPass : Pass<"xla-prepare-for-export", "mlir::func::FuncOp"> {
  let summary = "Prepare for XLA export";

  let description = [{
    This pass transforms functions in preparation for exporting to XLA. This

    * converts splat constants to constants and broadcasts to reduce size of
      and speedup the creation of the generated proto during export.

    Note: The result of this pass need not be a module in canonical form and
    canonicalization may undo transformations.
  }];
}

def ShapeLegalizeToHloPass : Pass<"shape-legalize-to-hlo", "func::FuncOp"> {
  let summary = "Legalize shape-related ops to HLO.";
  let constructor = "createShapeLegalizeToHloPass()";
  let description = [{
    An experimental pass that legalizes shape-related ops to MHLO ops.

    Bringing shape and data computations together via an optional pass will
    make it possible for the MHLO ecosystem to potentially leverage the
    compilation pipelines that use HLO operations to model dynamism.
  }];
  let dependentDialects = ["mhlo::MhloDialect"];
  let options = [
    Option<"legalize_constraints_", "legalize-constraints", "bool",
           /*default=*/"false", "Whether to legalize Cstr Ops to shape_assertion custom_call">
  ];
}
