/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// TF SavedModel dialect test passes.

def FreezeVariablesTestPass : Pass<"tf-freeze-variables-test-pass",
  "ModuleOp"> {
  let summary = "Freezes read only variables";

  let description = [{
    Test pass for freezing read only variables in the graph.
  }];

  let constructor = "::mlir::tf_test::CreateFreezeVariableTestPass()";
}

def InitTextFileToImportTestPass : Pass<"tf-init-text-file-to-import-test",
  "ModuleOp"> {
  let summary = "generate a temporary file and invoke InitTextFileToImportPass";
  let constructor = "::mlir::tf_test::CreateInitTextFileToImportTestPass()";
}

def InitTextFileToImportSavedModelTestPass : Pass<"tf-init-text-file-to-import-saved-model-test",
  "ModuleOp"> {
  let summary = "mimick a saved model and invoke InitTextFileToImportPass";
  let constructor = "::mlir::tf_test::CreateInitTextFileToImportSavedModelTestPass()";
}

def TestTensorFlowLowerTFPass : Pass<"test-tf-lower-tf", "FuncOp"> {
  let summary = "Test pass for TF->TF lowerings";
  let constructor = "::mlir::tf_test::CreateTestTFLowerTFPass()";

  let options = [
    Option<"default_patterns_", "default-patterns", "bool", /*default=*/"true",
           "Activate the default TF->TF patterns">,
    Option<"pre_hlo_patterns_", "pre-hlo-patterns", "bool", /*default=*/"true",
           "Activate the pre-HLO TF->TF patterns">,
  ];
}

def TestResourceAliasAnalysis : Pass<"tf-test-resource-alias-analysis", "FuncOp"> {
  let summary = "Add remarks based on resource alias analysis result, for "
    "testing purpose.";
  let constructor = "::mlir::tf_test::CreateTestResourceAliasAnalysisPass()";
}

def TestTensorFlowVisitorUtil : FunctionPass<"tf-test-visitor-util"> {
  let summary = "Test pass for TF visitor utils";
  let description = [{
    Add remarks that trace order of visiting operations using TF visitor
    utilities.
  }];
  let constructor = "::mlir::tf_test::CreateTestVisitorUtilPass()";
}

def TestTensorFlowVisitorUtilInterrupt :
  FunctionPass<"tf-test-visitor-util-interrupt"> {
  let summary = "Test pass for TF visitor utils with interruption";
  let description = [{
    Add remarks that trace order of visiting operations using TF visitor
    utilities, interrupt version.
  }];
  let constructor = "::mlir::tf_test::CreateTestVisitorUtilInterruptPass()";
}

def TestClusteringPolicyPass : Pass<"tf-test-clustering-policy", "FuncOp"> {
  let summary = "Test pass for clustering based on the user defined policy";
  let constructor = "::mlir::tf_test::CreateTestClusteringPolicyPass()";
}

def TestSideEffectAnalysisPass : Pass<"tf-test-side-effect-analysis", "ModuleOp"> {
  let summary = "Test pass for analyzing side-effect analysis result";
  let description = [{
    Adds "Predecessors" and "Successors" remarks for each op based on
    the SideEffectAnalysis result. For testing purpose only.
  }];
  let constructor = "::mlir::tf_test::CreateTestSideEffectAnalysisPass()";
}

def LiftVariablesInvalidSessionTestPass : Pass<"tf-saved-model-lift-variables-invalid-session-test", "ModuleOp"> {
  let summary = "Lift variables and save them as global tensors with an invalid "
           "session";
  let constructor = "mlir::tf_test::CreateLiftVariablesInvalidSessionTestPass()";
}

def LiftVariablesTestPass : Pass<"tf-saved-model-lift-variables-test", "ModuleOp"> {
  let summary = "Lift variables and save them as global tensors";
  let constructor = "mlir::tf_test::CreateLiftVariablesTestPass()";
}

def InitializeVariablesInSessionInitializerPass : Pass<"tf-saved-model-initialize-variables-in-session-init", "ModuleOp"> {
  let summary = "Initialize variables in session initializer function.";

  let constructor = "mlir::tf_test::CreateInitializeVariablesInSessionInitializerTestPass()";
}

