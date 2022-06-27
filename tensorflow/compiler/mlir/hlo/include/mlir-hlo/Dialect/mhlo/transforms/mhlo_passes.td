/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

def ChloLegalizeToHloPass : Pass<"chlo-legalize-to-hlo", "func::FuncOp"> {
  let summary = "Legalize CHLO to HLO.";
  let constructor = "createChloLegalizeToHloPass()";
  let options = [
    Option<"legalize_broadcasts_", "legalize-broadcasts", "bool",
           /*default=*/"true", "Legalize implicit broadcasts to explicit HLO broadcasting forms">,
    Option<"expand_compositions_", "expand-compositions", "bool",
           /*default=*/"true", "Expands client-centric compositions to HLO primitives">,
  ];
}

def HloCanonicalizeReductionPass : Pass<"hlo-canonicalize-reduction", "func::FuncOp"> {
  let summary = "canonicalize reduction ops to be suitable for codegen.";
  let constructor = "createHloCanonicalizeReductionPass()";
}

def HloLegalizeToLhloPass : Pass<"hlo-legalize-to-lhlo", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to LHLO dialect.";
  let constructor = "createLegalizeToLhloPass()";
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
  let dependentDialects = ["arith::ArithmeticDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

def LegalizeControlFlowPass : Pass<"mhlo-legalize-control-flow", "func::FuncOp"> {
  let summary = "Legalize from MHLO control flow to SCF control flow.";
  let constructor = "createLegalizeControlFlowPass()";
  let dependentDialects = ["scf::SCFDialect", "tensor::TensorDialect"];
}

def LegalizeEinsumToDotGeneralPass : Pass<"mhlo-legalize-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "Legalizes einsum ops to dot_general ops.";
  let constructor = "createLegalizeEinsumToDotGeneralPass()";
}

def LegalizeGatherToTorchIndexSelectPass : Pass<"mhlo-legalize-gather-to-torch-index-select", "func::FuncOp"> {
  let summary = "Legalizes gathers to a torch index select.";
  let constructor = "createLegalizeGatherToTorchIndexSelectPass()";
}


def LegalizeTanhToApproximationPass : Pass<"mhlo-legalize-trigonometric-to-approximation", "func::FuncOp"> {
  let summary = "Legalize trigonometric operations from standard dialect to an approximation.";
  let constructor = "createLegalizeTrigonometricToApproximationPass()";
}

def HloLegalizeShapeOpsToStandardPass : Pass<"hlo-legalize-shapeops-to-standard", "func::FuncOp"> {
  let summary = "Legalize shape operations from HLO dialect to standard dialect.";
  let constructor = "createLegalizeHloShapeOpsToStandardPass()";
}

def HloLegalizeToLinalgPass : Pass<"hlo-legalize-to-linalg", "func::FuncOp"> {
  let summary = "Legalize from HLO dialect to Linalg dialect.";
  let constructor = "createLegalizeHloToLinalgPass()";
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
}

def TestUnfuseBatchNormPass : Pass<"mhlo-test-unfuse-batch-norm", "func::FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestUnfuseBatchNormPass()";
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

def SparseRewritingPass : Pass<"mhlo-sparse-rewriting", "func::FuncOp"> {
  let summary = "Pass to rewrite mhlo sparse tensor types.";
  let constructor = "createSparseRewritingPass()";
}

/// Rank specialization passes.

def RankSpecializationClusterPass
    : Pass<"mhlo-rank-specialization-cluster", "func::FuncOp"> {
  let constructor = "createRankSpecializationClusterPass()";
}

def RankSpecializationToSCFPass
    : Pass<"mhlo-rank-specialization-to-scf", "func::FuncOp"> {
  let constructor = "createRankSpecializationToSCFPass()";
  let options = [
    Option<"max_target_rank_", "max-target-rank", "int", /*default=*/"8",
           "The maximum supported rank after rank specialization. Any argument "
           "of greater rank may result in a runtime failure.">,
  ];
}

def CollapseElementwiseMapPass
    : Pass<"mhlo-collapse-elementwise-map", "func::FuncOp"> {
  let summary = "Collapse the mhlo.map if the map only has elementwise ops.";
  let constructor = "createCollapseElementwiseMapPass()";
}
