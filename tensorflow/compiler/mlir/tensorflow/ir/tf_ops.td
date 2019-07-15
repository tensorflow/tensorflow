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

// This is the operation definition file for TensorFlow.
//
// This file contains TensorFlow ops whose definitions are amended to fix
// issues or provide more information. In this file you have full control
// of the op definition; all changes will be retained with subsequent
// refreshes.
//
// This file includes another file, `tf_generated_ops.td`, which contains
// all ops whose definitions are generated from TensorFlow codebase.
// Changes made there are not respected.

#ifdef TF_OPS
#else
#define TF_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_generated_ops.td"

// In MLIR, the TensorFlow tensor value is represented as an ElementsAttr, with
// its type encoding the tensor's shape and data type.
def TF_ConstOp : TF_Op<"Const", [NoSideEffect]> {
  let summary = "Constant tensor op";

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let builders = [
    OpBuilder<
      "Builder *builder, OperationState *result, Attribute value">,
    OpBuilder<
      "Builder *builder, OperationState *result, Type type, Attribute value">,
  ];

  let hasFolder = 1;
}

// TODO(fengliuai): The tf.Identity is side-effect free and it doesn't change
// the status of the system during the execution. However it shouldn't be folded
// in general if it used to serve for caching and some other invariant checks,
// so we removed the side-effect free property in the op definition. This is a
// hack, and we should fix it if we have a better way to model it.
def TF_IdentityOp : TF_Op<"Identity", [SameOperandsAndResultType]> {
  let summary = "Identity op";

  let description = [{
Returns a tensor with the same shape and contents as input.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MeanOp : TF_Op<"Mean", [NoSideEffect]> {
  let summary = "Computes the mean of elements across dimensions of a tensor.";

  let description = [{
Reduces `input` along the dimensions given in `axis`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.
  }];

  let arguments = (ins
    TF_NumberTensor:$input,
    TF_I32OrI64Tensor:$reduction_indices,

    DefaultValuedAttr<BoolAttr, "false">:$keep_dims
  );

  let results = (outs
    TF_NumberTensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr Tidx = TF_DerivedOperandTypeAttr<1>;
}

// In MLIR, the 'tf.Placeholder.input' instruction is used to capture attributes
// of function arguments.
// Note: NoSideEffect trait is not added intentionally to preserve the captured
// attributes even if the input is unused.
def TF_PlaceholderInputOp : TF_Op<"Placeholder.input",
    [SameOperandsAndResultType]> {
  let summary = "PlaceholderInput op";

  let description = [{
Inserts a placeholder for a tensor that will be always fed.
  }];

  let arguments = (ins
    TF_Tensor:$arg,

    OptionalAttr<F32Attr>:$min,
    OptionalAttr<F32Attr>:$max,
    OptionalAttr<TF_IntTypeAttr>:$type
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_PlaceholderOp : TF_Op<"Placeholder", [NoSideEffect]> {
  let summary = "Placeholder op";

  let description = [{
Inserts a placeholder for a tensor that will be always fed.
  }];

  let arguments = (ins
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_TensorListReserveOp : TF_Op<"TensorListReserve", [NoSideEffect]> {
  let summary = "List of the given size with empty elements.";

  let description = [{
element_shape: the shape of the future elements of the list
num_elements: the number of elements to reserve
handle: the output list
element_dtype: the desired type of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$num_elements
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    if (handle_dtype().getSubtypes().size() != 1) {
      return emitOpError(
          "must have exactly one subtype in the result variant type");
    }

    return Verify(*this);
  }];

  DerivedTypeAttr element_dtype = DerivedTypeAttr<
      "return getElementTypeOrSelf(element_type());">;

  let extraClassDeclaration = [{
    // Returns type of the TensorList element produced by this op.
    TensorType element_type() { return handle_dtype().getSubtypes()[0]; }

    // Returns data type of the result handle. Returned type contains type of
    // the TensorList element as a subtype.
    VariantType handle_dtype() {
      return getElementTypeOrSelf(handle()->getType()).cast<TF::VariantType>();
    }
  }];
}

#endif // TF_OPS
