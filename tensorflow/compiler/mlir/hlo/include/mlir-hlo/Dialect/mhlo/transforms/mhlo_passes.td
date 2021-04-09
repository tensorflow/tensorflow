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
    Option<"broadcast_only_", "broadcast-only", "bool",
           /*default=*/"false", "Only lower broadcasting chlo to non-broadcasting equivalents">,
  ];
}

def HloLegalizeToLhloPass : Pass<"hlo-legalize-to-lhlo", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to LHLO dialect.";
  let constructor = "createLegalizeToLhloPass()";
}

def LegalizeControlFlowPass : Pass<"mhlo-legalize-control-flow", "FuncOp"> {
  let summary = "Legalize from MHLO control flow to CFG control flow.";
  let constructor = "createLegalizeControlFlowPass()";
}

def LegalizeControlFlowToScfPass : Pass<"mhlo-control-flow-to-scf", "FuncOp"> {
  let summary = "Legalize from MHLO control flow to SCF control flow.";
  let constructor = "createControlFlowToScfPass()";
}

def LegalizeGatherToTorchIndexSelectPass : Pass<"mhlo-legalize-gather-to-torch-index-select", "FuncOp"> {
  let summary = "Legalizes gathers to a torch index select.";
  let constructor = "createLegalizeGatherToTorchIndexSelectPass()";
}


def LegalizeTanhToApproximationPass : Pass<"mhlo-legalize-trigonometric-to-approximation", "FuncOp"> {
  let summary = "Legalize trigonometric operations from standard dialect to an approximation.";
  let constructor = "createLegalizeTrigonometricToApproximationPass()";
}


def HloLegalizeToLinalgPass : Pass<"hlo-legalize-to-linalg", "FuncOp"> {
  let summary = "Legalize from HLO dialect to Linalg dialect.";
  let constructor = "createLegalizeHloToLinalgPass()";
}


def LegalizeToStandardPass : Pass<"mhlo-legalize-to-std", "FuncOp"> {
  let summary = "Legalize from MHLO dialect to standard dialect.";
  let constructor = "createLegalizeToStdPass()";
}

def LowerComplexPass : Pass<"mhlo-test-lower-complex", "FuncOp"> {
  let summary = "Lower complex operations into non-complex operations.";
  let constructor = "createLowerComplexPass()";
}


def LegalizeGeneralDotPass : Pass<"mhlo-test-lower-general-dot", "FuncOp"> {
  let summary = "Tests lowering general dot to a non-batched dot when possible.";
  let constructor = "createLegalizeGeneralDotPass()";
}


def TestMaterializeBroadcastsPass : Pass<"mhlo-test-materialize-broadcasts", "FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestMaterializeBroadcastsPass()";
}


def MhloFusionPass : Pass<"mhlo-fusion", "FuncOp"> {
  let summary = "Fuse mhlo ops to kLoop/kInput fusion patterns.";
  let constructor = "createMhloFusionPass()";
}


def OptimizeMhloPass : Pass<"mhlo-test-optimize", "FuncOp"> {
  let summary = "Run optional HLO optimizations.";
  let constructor = "createOptimizeMhloPass()";
}


def SinkConstantsToControlFlowPass : FunctionPass<"mhlo-sink-constants-to-control-flow"> {
  let summary = "Sink constants implicitly captured in control flow regions. This "
    "is necessary to export to XLA.";
  let constructor = "createSinkConstantsToControlFlowPass()";
}


def TestInferShapedTypeMethodsPass : Pass<"mhlo-test-infer-shaped-type-methods", "FuncOp"> {
  let summary = "Uses test ops to invoke InferShapedTypeOpInterface methods.";
  let constructor = "createTestInferShapedTypeMethodsPass()";
}


def TransformUnrankedHloPass : Pass<"mhlo-transform-unranked-hlo", "FuncOp"> {
  let summary = "Realize element-wise operations on ranked tensors where possible.";
  let constructor = "createTransformUnrankedHloPass()";
}

def MoveUpDynamicBroadcastsForFusionPass :
    Pass<"mhlo-move-up-dynamic-broadcasts-for-fusion", "FuncOp"> {
  let summary = "Move up dynamic broadcasts and shape computations to allow "
      "for fusion across broadcasts.";
  let constructor = "createMoveUpDynamicBroadcastsForFusionPass()";
}

def TestUnfuseBatchNormPass : Pass<"mhlo-test-unfuse-batch-norm", "FuncOp"> {
  let summary = "Test pass for materializing 'broadcast_dimensions' attributes.";
  let constructor = "createTestUnfuseBatchNormPass()";
}
