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

def StripSavedModuleMetadataPass : Pass<"tf-strip-saved-module-metadata", "ModuleOp"> {
  let summary = "Removes saved_model attributes.";
  let description = [{
      Removes all saved_model attributes from a module and its functions.
      This only removes attributes, saved_model ops will stay in place.

      This pass is useful for pass pipelines that have already consumed
      all the tf_saved_model information, and now want to do further
      transformations without tf_saved_model's verifiers kicking in.

      For example, this would remove the semantics attribute from module:

        module attributes {tf_saved_model.semantics} [...]

      so the result becomes just

        module [...]
  }];
  let constructor = "::mlir::tf_saved_model::CreateStripSavedModuleMetadataPass()";
}

def ConvertSessionInitializerToFunctionPass : Pass<"tf-saved-model-convert-session-initializer-to-function", "ModuleOp"> {
  let summary = "Converts the session initializer to a function.";
  let description = [{
      This converts
        "tf_saved_model.session_initializer"() {initializers = [@a, @b, @c]} : () -> ()
      to
	func.func @session_initializer() {
	  call @a() : () -> ()
	  call @b() : () -> ()
	  call @c() : () -> ()
	  return
	}
  }];
  let constructor = "::mlir::tf_saved_model::CreateConvertSessionInitializerToFunctionPass()";
}

def AddFunctionsForExportedNamesPass : Pass<"tf-saved-model-add-functions-for-exported-names", "ModuleOp"> {
  let summary = "Create a trampoline function for each exported name.";
  let description = [{
      This converts
        func.func @foo() attributes {tf_saved_model.exported_names = ["a", "b", "c"]}
      to
        func.func @a() {return @foo_internal()}
        func.func @b() {return @foo_internal()}
        func.func @c() {return @foo_internal()}
        func.func private @foo_internal()
  }];
  let constructor = "::mlir::tf_saved_model::CreateAddFunctionsForExportedNamesPass()";
}

def AssetSinkingPass : Pass<"tf-saved-model-asset-sinking", "mlir::ModuleOp"> {
  let summary = "Sinks SavedModel asset filenames to constants";

  let description = [{
    This pass sinks arguments of SavedModel methods that are bounded to
    `tf_saved_model.asset` into constants in the methods. After the pass, unused
    asset ops are removed from the module.

    This is to convert initialization methods with bound inputs into the same
    methods without any arguments, so that program invocation doesn't need to
    track and explicitly pass asset filenames.

    This pass accepts an option `saved-model-dir`, which specifies the directory
    where SavedModel is stored. This is a required option because all asset
    filenames are relative to this directory.
  }];

  let constructor = "::mlir::tf_saved_model::CreateAssetSinkingPass(\"\")";

  let options = [
    Option<"saved_model_dir_", "saved-model-dir", "std::string", "",
           "SavedModel directory, which is prepended to asset file names.">,
  ];
}
