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

def FreezeGlobalTensorsPass : Pass<"tf-saved-model-freeze-global-tensors", "ModuleOp"> {
  let summary = "Freeze tf_saved_model.global_tensor's in func bodies.";

  let description = [{
    This pass will replace a func's bound inputs which are bound to
    tf.ReadVariable ops global tensors with tf.Const ops inside the func's body.
    If this pass runs successfully, the resultant IR will be guaranteed to:

    1. Not contain any tf_saved_model.global_tensor ops
    2. Not contain any tf_saved_model.bound_input arg attrs on tf_saved_model
       exported functions
    Else, the pass fails.

    The reason this pass has this contract is so that once this succeeds, we know
    the IR is in correct form for inference backends (like lite) that do not
    support resources/variables. Further, this contract also ensures that this
    pass lowers from saved model to pure TF. Hence it fails, if it cannot lower.
  }];

  let constructor = "::mlir::tf_saved_model::CreateFreezeGlobalTensorsPass()";

  let options = [
    Option<"allow_mutable_tensors", "allow-mutable-tensors", "bool", /*default=*/"false",
           "Allows mutable tensors to be in the graph.">
  ];
}

def OptimizeGlobalTensorsPass : Pass<"tf-saved-model-optimize-global-tensors", "ModuleOp"> {
  let summary = "Optimize tf_saved_model.global_tensor's.";

  let constructor = "mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass()";
}

def RemoveVariablesInSessionInitializerPass : Pass<"tf-saved-model-remove-vars-in-session-initializer", "ModuleOp"> {
  let summary = "Remove variables in tf saved model's session initializer.";

  let constructor = "mlir::tf_saved_model::CreateRemoveVariablesInSessionInitializerPass()";
}

def FreezeAssetsPass : Pass<"tf-saved-model-freeze-assets", "ModuleOp"> {
  let summary = "Freeze tf_saved_model.asset's in func bodies.";
  let constructor = "mlir::tf_saved_model::CreateFreezeAssetsPass()";
}

def DedupBoundInputBindingPass : Pass<"tf-saved-model-dedup-bound-input-binding-pass", "mlir::func::FuncOp"> {
  let summary = "Remove duplicate 'tf_saved_model.bound_input' bindings.";
  let constructor = "mlir::tf_saved_model::CreateDedupBoundInputBindingPass()";
}

def LowerGlobalsToMlProgramPass: Pass<"tf-saved-model-lower-globals-to-mlprogram", "mlir::ModuleOp"> {
  let summary = "Remove (and remap) function arguments that map to global tensors.";
  let description = [{
    This pass will lower AssignVariableOp and ReadVariableOp to their ml_program
    equivalents, for those resources which stem from a "bound input" function argument
    that maps to a GlobalTensorOp. The arguments in question will be removed.

    So this will lower e.g.

      func @f(%arg0 {tf_saved_model.bound_input = @v}) {
        %0 = "tf.ReadVariableOp"(%arg0)
        ...
      }

    to

      func @f() {
        %0 = ml_program.global_load @v
        ...
      }
  }];
  let constructor = "mlir::tf_saved_model::CreateLowerGlobalsToMlProgramPass()";
}

def LowerVariableOpsToMlProgramPass: Pass<"tf-saved-model-lower-variable-ops-to-mlprogram", "mlir::ModuleOp"> {
  let summary = "Lower tf.ReadVariable and tf.AssignVariable";

  let description = [{
    This pass will lower AssignVariableOp and ReadVariableOp to their ml_program
    equivalents, for those resources which come from a VarHandle op, but also
    for those which come from a "bound input" parameter. For the latter,
    said parameter will not be removed, unlike LowerGlobalsToMlProgramPass.

    So this will lower e.g.

      %0 = "tf.VarHandleOp" {name = "foo"}
      %1 = "tf.ReadVariableOp"(%0)

    to

      %0 = ml_program.global_load @vars.foo
  }];

  let constructor = "::mlir::tf_saved_model::CreateLowerVariableOpsToMlProgramPass()";
}
