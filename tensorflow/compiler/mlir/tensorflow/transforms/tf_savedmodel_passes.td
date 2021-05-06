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

// TF SavedModel dialect passes.

def SavedModelLiftVariablePass : Pass<"tf-savedmodel-lift-variable-pass",
    "ModuleOp"> {
  let summary = "Convert function arguments to global tensors";

  let description = [{
    This pass takes care of finding all variables from the function arguments
    and converting them to the corresponding global tensors, that will be
    located out of function. Also it converts resource arguments from function
    types to the corresponding saved model arguments accordingly.
  }];

  let constructor = "::mlir::tf_saved_model::CreateLiftVariablesPass()";
}
