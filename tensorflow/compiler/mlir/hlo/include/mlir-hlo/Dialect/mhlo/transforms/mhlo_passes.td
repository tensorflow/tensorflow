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

def ChloLegalizeToHloPass : FunctionPass<"chlo-legalize-to-hlo"> {
  let summary = "Legalize CHLO to HLO.";
  let constructor = "createChloLegalizeToHloPass()";
  let options = [
    Option<"legalize_broadcasts_", "legalize-broadcasts", "bool",
           /*default=*/"true", "Legalize implicit broadcasts to explicit HLO broadcasting forms">,
    Option<"expand_compositions_", "expand-compositions", "bool",
           /*default=*/"true", "Expands client-centric compositions to HLO primitives">,
  ];
}

def HloCanonicalizeReductionPass : FunctionPass<"hlo-canonicalize-reduction"> {
  let summary = "canonicalize reduction ops to be suitable for codegen.";
  let constructor = "createHloCanonicalizeReductionPass()";
}

def HloLegalizeToLhloPass : Pass<"hlo-legalize-to-lhlo", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to LHLO dialect.";
  let constructor = "createLegalizeToLhloPass()";
}

def HloLegalizeToMemrefPass : FunctionPass<"hlo-legalize-to-memref"> {
  let summary = "Legalize from HLO dialect to memref dialect.";
  let constructor = "createLegalizeToMemrefPass()";
}

def LegalizeControlFlowPass : FunctionPass<"mhlo-legalize-control-flow"> {
  let summary = "Legalize from MHLO control flow to CFG control flow.";
  let constructor = "createLegalizeControlFlowPass()";
}

def LegalizeControlFlowToScfPass : FunctionPass<"mhlo-control-flow-to-scf"> {
  let summary = "Legalize from MHLO control flow to SCF control flow.";
  let constructor = "createControlFlowToScfPass()";
}

def LegalizeEinsumToDotGeneralPass : FunctionPass<"mhlo-legalize-einsum-to-dot-general"> {
  let summary = "Legalizes einsum ops to dot_general ops.";
  let constructor = "createLegalizeEinsumToDotGeneralPass()";
}

def LegalizeGatherToTorchIndexSelectPass : FunctionPass<"mhlo-legalize-gather-to-torch-index-select"> {
  let summary = "Legalizes gathers to a torch index select.";
  let constructor = "createLegalizeGatherToTorchIndexSelectPass()";
}


def LegalizeTanhToApproximationPass : FunctionPass<"mhlo-legalize-trigonometric-to-approximation"> {
  let summary = "Legalize trigonometric operations from standard dialect to an approximation.";
  let constructor = "createLegalizeTrigonometricToApproximationPass()";
}

def HloLegalizeToLinalgPass : FunctionPass<"hlo-legalize-to-linalg"> {
  let summary = "Legalize from HLO dialect to Linalg dialect.";
  let constructor = "createLegalizeHloToLinalgPass()";
}

def LegalizeToStandardPass : FunctionPass<"mhlo-legalize-to-std"> {
  let summary = "Legalize from MHLO dialect to standard dialect.";
  let constructor = "createLegalizeToStdPass()";
}

def MarkShapeCalculationPass : Pass<"mhlo-mark-shape-calc", "ModuleOp"> {
  let summary = "Identify and mark with an attribute operations performing shape calculation";
  let constructor = "createMarkShapeCalcOpPass()";
}

def LowerComplexPass : FunctionPass<"mhlo-test-lower-complex"> {
  let summary = "Lower complex operations into non-complex operations.";
  let constructor = "createLowerComplexPass()";
}


def LegalizeGeneralDotPass : FunctionPass<"mhlo-test-lower-general-dot"> {
  let summary = "Tests lowering general dot to a non-batched dot when possible.";
  let constructor = "createLegalizeGeneralDotPass()";
}


def TestMaterializeBroadcastsPass : FunctionPass<"mhlo-test-materialize-broadcasts"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestMaterializeBroadcastsPass()";
}


def MhloFusionPass : FunctionPass<"mhlo-fusion"> {
  let summary = "Fuse mhlo ops to kLoop/kInput fusion patterns.";
  let constructor = "createMhloFusionPass()";
}


def OptimizeMhloPass : FunctionPass<"mhlo-test-optimize"> {
  let summary = "Run optional HLO optimizations.";
  let constructor = "createOptimizeMhloPass()";
}


def SinkConstantsToControlFlowPass : FunctionPass<"mhlo-sink-constants-to-control-flow"> {
  let summary = "Sink constants implicitly captured in control flow regions. This "
    "is necessary to export to XLA.";
  let constructor = "createSinkConstantsToControlFlowPass()";
}


def TestInferShapedTypeMethodsPass : FunctionPass<"mhlo-test-infer-shaped-type-methods"> {
  let summary = "Uses test ops to invoke InferShapedTypeOpInterface methods.";
  let constructor = "createTestInferShapedTypeMethodsPass()";
}

def BroadcastPropagationPass : FunctionPass<"mhlo-broadcast-propagation"> {
  let summary = "Move dynamic broadcasts up over element-wise operations and "
    "broadcast the operands rather than the result. This will eventually allow "
    "for larger fusions.";
  let constructor = "createBroadcastPropagationPass()";
}

def TestUnfuseBatchNormPass : Pass<"mhlo-test-unfuse-batch-norm", "FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestUnfuseBatchNormPass()";
}

/// Rank specialization passes.

def RankSpecializationClusterPass
    : FunctionPass<"mhlo-rank-specialization-cluster"> {
  let constructor = "createRankSpecializationClusterPass()";
}

def RankSpecializationToSCFPass
    : FunctionPass<"mhlo-rank-specialization-to-scf"> {
  let constructor = "createRankSpecializationToSCFPass()";
  let options = [
    Option<"max_target_rank_", "max-target-rank", "int", /*default=*/"8",
           "The maximum supported rank after rank specialization. Any argument "
           "of greater rank may result in a runtime failure.">,
  ];
}
