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
