/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the definition file for the TensorFlow SavedModel dialect.

#ifdef SAVED_MODEL_DIALECT
#else
#define SAVED_MODEL_DIALECT

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def TfSavedModel_Dialect : Dialect {
  let name = "tf_saved_model";

  let summary = "Dialect used to represent TensorFlow SavedModel in MLIR.";

  let description = [{
    This dialect's main responsibility is to represent the additional
    information present in the SavedModel's SavedObjectGraph, which
    describes the public interface of this SavedModel.

    Module's using this dialect should have the module attribute
    `tf_saved_model.semantics`.

    This dialect provides an attribute
    `tf_saved_model.exported_names = <array of strings>` which indicates which
    objects in the module are exported, and under what names. The names
    are derived from the object's path in the SavedObjectGraph. For example,
    a Python `tf.Module` with an exported function "foo" will result in an MLIR
    module where the function has an exported name "foo". If the `tf.Module`
    contains a subobject "self.bar" which itself has an exported function
    "baz", then the function will have an exported name "bar.baz".
    If an object in the object graph is reachable via multiple paths
    from the root of the object graph, then this array can have multiple
    entries.

    This dialect provides a function arg attribute
    `tf_saved_model.bound_input = @some_symbol_name` which can reference a
    `tf_saved_model.global_tensor` that is bound to that function argument.
  }];

  let cppNamespace = "tf_saved_model";
}


//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

class TfSavedModel_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TfSavedModel_Dialect, mnemonic, traits> {
}

def TfSavedModel_GlobalTensorOp : TfSavedModel_Op<"global_tensor"> {
  let summary = "Represents a global tensor value.";
  let description = [{
    Represents a tensor that is not bound to the lifetime of any particular
    function. Such tensors can be marked as mutable via the `is_mutable`
    attribute.

    These tensors are bound to the arguments of func ops via the
    `tf_saved_model.bound_input` argument attr.

    The `sym_name` represents the symbol table name used for internal IR
    references. The externally visible names, if any, are represented via
    a `tf_saved_model.exported_names` attribute.

    The `value` attribute contains the tensor's value (or initial value, in the
    case it is mutable).
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    ElementsAttr:$value,
    UnitAttr:$is_mutable
  );
}

#endif // SAVED_MODEL_DIALECT
