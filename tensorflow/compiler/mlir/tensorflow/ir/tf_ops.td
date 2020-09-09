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

#ifndef TF_OPS
#define TF_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_generated_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"

class TF_TensorListInitOp<string mnemonic> : TF_Op<mnemonic, [NoSideEffect]> {
  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<0>;

  let verifier = [{
    // This is required to populate derived attributes during export in a
    // meaningful way. Else during export to GraphDef element_type() query
    // will result in out of bounds access/assert.
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
      return getElementTypeOrSelf(handle().getType()).cast<TF::VariantType>();
    }
  }];
}

def TF_CaseOp : TF_Op<"Case", []> {
  let summary = [{
An n-way switch statement which calls a single branch function.
  }];

  let description = [{
An n-way switch statement, implementing the following:
    ```
    switch (branch_index) {
      case 0:
        output = branches[0](input);
        break;
      case 1:
        output = branches[1](input);
        break;
      ...
      case [[nbranches-1]]:
      default:
        output = branches[nbranches-1](input);
        break;
    }
    ```
  }];

  let arguments = (ins
    I32Tensor:$branch_index,
    Variadic<TF_Tensor>:$input,

    Confined<SymbolRefArrayAttr, [ArrayMinCount<1>]>:$branches,

    // Used to map StatelessCase and Case op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let hasCanonicalizer = 1;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_CaseRegionOp : TF_Op<"CaseRegion",
      [SingleBlockImplicitTerminator<"YieldOp">, NoRegionArguments]> {
  let summary = [{
An n-way switch statement which calls a single branch function.
  }];

  let description = [{
An n-way switch statement, implementing the following:
    ```
    switch (branch_index) {
      case 0:
        output = branches[0](input);
        break;
      case 1:
        output = branches[1](input);
        break;
      ...
      case [[nbranches-1]]:
      default:
        output = branches[nbranches-1](input);
        break;
    }
    ```
  }];

  let arguments = (ins
    I32Tensor:$branch_index,

    // Used to map StatelessCase and Case op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let verifier = [{
    return Verify(*this);
  }];
}

// In MLIR, the TensorFlow tensor value is represented as an ElementsAttr, with
// its type encoding the tensor's shape and data type.
def TF_ConstOp : TF_Op<"Const", [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
      "OpBuilder &builder, OperationState &result, Attribute value">,
    OpBuilder<
      "OpBuilder &builder, OperationState &result, Type type, Attribute value">,
  ];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(ArrayRef<Type> l, ArrayRef<Type> r) {
      return BroadcastCompatible(l, r);
    }
  }];
}

def TF_CollectivePermuteOp : TF_Op<"CollectivePermute", []> {
  let summary = "An Op to permute tensors across replicated TPU instances.";

  let description = [{
Each instance supplies its own input.

For example, suppose there are 4 TPU instances: `[A, B, C, D]`. Passing
source_target_pairs=`[[0,1],[1,2],[2,3],[3,0]]` gets the outputs:
`[D, A, B, C]`.
  }];

  let arguments = (ins
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$input,
    I32Tensor:$source_target_pairs
  );

  let results = (outs
    TensorOf<[BF16, F16, F32, F64, I16, I32, I64, I8, TF_Complex128, TF_Complex64, TF_Qint32, TF_Qint8, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_EmptyTensorListOp : TF_TensorListInitOp<"EmptyTensorList"> {
  let summary = "Creates and returns an empty tensor list.";

  let description = [{
All list elements must be tensors of dtype element_dtype and shape compatible
with element_shape.

handle: an empty tensor list.
element_dtype: the type of elements in the list.
element_shape: a shape compatible with that of elements in the list.
  }];

  let arguments = (ins
    TF_I32OrI64Tensor:$element_shape,
    I32Tensor:$max_num_elements
  );
}

// TODO(fengliuai): The tf.Identity is side-effect free and it doesn't change
// the status of the system during the execution. However it shouldn't be folded
// in general if it used to serve for caching and some other invariant checks,
// so we removed the side-effect free property in the op definition. This is a
// hack, and we should fix it if we have a better way to model it.
def TF_IdentityOp : TF_Op<"Identity", [TF_OperandsSameAsResultsTypeOrRef]> {
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

  TF_DerivedResultTypeAttr T = TF_DerivedResultTypeAttr<0>;
}

def TF_IfOp : TF_Op<"If", []> {
  let summary = "output = cond ? then_branch(input) : else_branch(input)";

  let description = [{
output = cond ? then_branch(input) : else_branch(input)

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
input: A list of input tensors.
then_branch: A function that takes 'inputs' and returns a list of
    tensors, whose types are the same as what else_branch returns.
else_branch: A function that takes 'inputs' and returns a list of
    tensors.  whose types are the same as what then_branch returns.
  }];

  let arguments = (ins
    TF_Tensor:$cond,
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$then_branch,
    FlatSymbolRefAttr:$else_branch,

    // Used to map StatelessIf and If op defined in TensorFlow to a common op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr Tcond = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Get the then branch function.
    FuncOp then_func() {
     return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, then_branch());
    }

    // Get the else branch function.
    FuncOp else_func() {
     return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, else_branch());
    }
  }];
}

def TF_YieldOp : TF_Op<"Yield",
      [Terminator, ParentOneOf<["CaseRegionOp", "IfRegionOp", "WhileRegionOp"]>]> {
  let summary = "Yield operation";

  let description = [{
    The "yield" operation represents a return operation within the conditional
    and body of structured control flow (e.g., if and while). The operation
    takes a variable number of operands and produces no results. The number and
    types of inputs must match the signature of the operation that contains the
    region.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

def TF_IfRegionOp : TF_Op<"IfRegion",
      [SingleBlockImplicitTerminator<"YieldOp">, NoRegionArguments]> {
  let summary = "output = cond ? then_branch output : else_branch output";

  let description = [{
"output = cond ? then_branch output : else_branch output"

cond: A Tensor. If the tensor is a scalar of non-boolean type, the
    scalar is converted to a boolean according to the
    following rule: if the scalar is a numerical value, non-zero means
    True and zero means False; if the scalar is a string, non-empty
    means True and empty means False. If the tensor is not a scalar,
    being empty means False and being non-empty means True.
then_branch: A region that computes the outputs of the op if cond = true.
    It returns a list of tensors using tf.yield (as the terminator). The
    types of these returned tensors is same as that of the else_branch
else_branch: A region that computes the outputs of the op if cond = false.
    It returns a list of tensors using tf.yield (as the terminator). The
    types of these returned tensors is same as that of the then_branch
  }];

  let arguments = (ins
    0DTensorOf<[I1]>:$cond,

    // Used to map StatelessIf and If op defined in TensorFlow to a common op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let regions = (region SizedRegion<1>:$then_branch, SizedRegion<1>:$else_branch);

  let verifier = [{
    return Verify(*this);
  }];

  let hasCanonicalizer = 1;
}

def TF_LegacyCallOp : TF_Op<"LegacyCall",
                            [CallOpInterface, NoSideEffect]> {
  let summary =
    "returns `f(inputs)`, where `f` is a function.";

  let description = [{
    The LegacyCall operation represents a direct call to a function that is
    within the same symbol scope as the call and is mapped to a GraphDef node
    with the function name as the op name. Unlike a PartitionedCall which
    represents asynchronously executing a function across multiple devices, a
    LegacyCall ignores specification for ops in the attached function and
    instead executes it on the device assigned to this op.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    FlatSymbolRefAttr:$f,
    DefaultValuedAttr<BoolAttr, "false">:$_disable_call_shape_inference
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return fAttr(); }

    // returns the callee of this operation.
    FuncOp func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, f());
    }
  }];
}

def TF_ParseExampleOp : TF_Op<"ParseExample",
                               [NoSideEffect,
                                AttrSizedResultSegments,
                                AttrSizedOperandSegments]> {

  let summary =
    "Transforms a vector of tf.Example protos (as strings) into typed tensors.";

  let arguments = (ins
    TF_StrTensor:$serialized,
    TF_StrTensor:$names,
    Variadic<TF_StrTensor>:$sparse_keys,
    Variadic<TF_StrTensor>:$dense_keys,
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_defaults,

    TF_ShapeAttrArray:$dense_shapes,
    I32ElementsAttr:$result_segment_sizes,
    I32ElementsAttr:$operand_segment_sizes
  );

  let results = (outs
    Variadic<I64Tensor>:$sparse_indices,                    // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$sparse_values,  // len(sparse_types)
    Variadic<I64Tensor>:$sparse_shapes,                     // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_values    // len(Tdense)
  );

  TF_DerivedOperandSizeAttr Nsparse = TF_DerivedOperandSizeAttr<2>;
  TF_DerivedOperandSizeAttr Ndense = TF_DerivedOperandSizeAttr<3>;
  TF_DerivedOperandTypeListAttr Tdense = TF_DerivedOperandTypeListAttr<4>;
  TF_DerivedResultTypeListAttr sparse_types = TF_DerivedResultTypeListAttr<1>;

  let verifier = ?;
}

def TF_ParseExampleV2Op : TF_Op<"ParseExampleV2",
                                [NoSideEffect,
                                 AttrSizedResultSegments]> {

  let summary =
    "Transforms a vector of tf.Example protos (as strings) into typed tensors.";

  let arguments = (ins
    TF_StrTensor:$serialized,
    TF_StrTensor:$names,
    TF_StrTensor:$sparse_keys,
    TF_StrTensor:$dense_keys,
    TF_StrTensor:$ragged_keys,
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_defaults,

    Confined<I64Attr, [IntMinValue<0>]>:$num_sparse,
    TF_ShapeAttrArray:$dense_shapes,
    I32ElementsAttr:$result_segment_sizes
  );

  let results = (outs
    Variadic<I64Tensor>:$sparse_indices,                    // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$sparse_values,  // len(sparse_types)
    Variadic<I64Tensor>:$sparse_shapes,                     // len(sparse_types)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$dense_values,   // len(Tdense)
    Variadic<TensorOf<[F32, I64, TF_Str]>>:$ragged_values,  // len(ragged_value_types)
                                                            //     = len(ragged_split_types)
    Variadic<TensorOf<[I32, I64]>>:$ragged_row_splits       // len(ragged_split_types)
                                                            //     = len(ragged_value_types)
  );

  // The Verify(ParseExampleV2Op) function validates that the lengths and types
  // of these attrs are compatible.
  TF_DerivedOperandTypeListAttr Tdense = TF_DerivedOperandTypeListAttr<5>;
  TF_DerivedResultTypeListAttr sparse_types = TF_DerivedResultTypeListAttr<1>;
  TF_DerivedResultTypeListAttr ragged_value_types =
    TF_DerivedResultTypeListAttr<4>;
  TF_DerivedResultTypeListAttr ragged_split_types =
    TF_DerivedResultTypeListAttr<5>;

  let verifier = [{
    return Verify(*this);
  }];
}

def TF_PartitionedCallOp : TF_Op<"PartitionedCall",
                                 [CallOpInterface, NoSideEffect]> {
  let summary =
    "returns `f(inputs)`, where `f`'s body is placed and partitioned.";

  let description = [{
Asynchronously executes a function, potentially across multiple devices but
within a single process. The kernel places and partitions a given function's
underlying graph, and executes each of the partitioned subgraphs as a function.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    SymbolRefAttr:$f,
    StrAttr:$config,
    StrAttr:$config_proto,
    StrAttr:$executor_type
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return fAttr(); }

    // returns the callee of this operation.
    FuncOp func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, f());
    }
  }];

  let verifier = [{ return VerifyPartitionedCall(*this); }];
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

def TF_PlaceholderWithDefaultOp : TF_Op<"PlaceholderWithDefault", [NoSideEffect]> {
  let summary = "Placeholder op";

  let description = [{
    A placeholder op that passes through input when its output is not fed.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
  DerivedAttr shape = TF_DerivedResultShapeAttr;
}

def TF_StatefulPartitionedCallOp : TF_Op<"StatefulPartitionedCall",
                                         [CallOpInterface]> {
  let summary =
    "returns `f(inputs)`, where `f`'s body is placed and partitioned.";

  let description = [{
Asynchronously executes a function, potentially across multiple devices but
within a single process. The kernel places and partitions a given function's
underlying graph, and executes each of the partitioned subgraphs as a function.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,

    FlatSymbolRefAttr:$f,
    StrAttr:$config,
    StrAttr:$config_proto,
    StrAttr:$executor_type
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return fAttr(); }

    // returns the callee of this operation.
    FuncOp func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, f());
    }
  }];

  let verifier = [{ return VerifyPartitionedCall(*this); }];
}

def TF_WhileOp : TF_Op<"While", []> {
  let summary = [{
output = input; While (Cond(output)) { output = Body(output) }
  }];

  let description = [{
output = input; While (Cond(output)) { output = Body(output) }

input: A list of input tensors whose types are T.
output: A list of output tensors whose types are T.
cond: A function that takes 'input' and returns a tensor.  If the tensor is
    a scalar of non-boolean, the scalar is converted to a boolean
    according to the following rule: if the scalar is a numerical
    value, non-zero means True and zero means False; if the scalar is
    a string, non-empty means True and empty means False. If the
    tensor is not a scalar, non-emptiness means True and False
    otherwise.
body: A function that takes a list of tensors and returns another
      list of tensors. Both lists have the same types as specified
      by T.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$input,

    FlatSymbolRefAttr:$cond,
    FlatSymbolRefAttr:$body,
    DefaultValuedAttr<I64Attr, "10">:$parallel_iterations,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let verifier = [{
    return Verify(*this);
  }];
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Get the condition function.
    FuncOp cond_func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, cond());
    }

    // Get the body function.
    FuncOp body_func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, body());
    }
  }];
}

def TL_WhileRegionOp : TF_Op<"WhileRegion",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "while operation";
  let description = [{
  The tf.WhileRegion op represents a while loop using 2 regions and a set of
  iteration variables. The iteration variables maintained by this Op have the
  same types as the inputs. The Op executes a while loop described by the
  following pseudo code:

  ```
     func WhileRegionOp(inputs) {
       iteration_vars = inputs;
       while (cond(iteration_vars)) {
           iteration_vars = body(iteration_vars);
       }
       return iteration_vars;
     }
  ```

  `cond` is the condition region and `body` is the body region. Both these
  regions accept the current value of the iteration variables as inputs. The
  condition region returns a tensor<i1> which, if false, will exit the loop.
  The body region computes new values of the iteration variables. The iteration
  variables are initialized to the Op input, and the results of the
  tf.WhileRegion op are the final values of the iteration variables.

  This implies that the operand and result types for tf.WhileRegion should be
  the same. Note that the condition and body regions can implicitly capture
  loop invariant values directly. In canonical form, iteration variables that
  pass through the loop body unmodified are converted to implicitly captured
  references to their values outside the loop.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$input,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    DefaultValuedAttr<BoolAttr, "false">:$is_stateless,
    DefaultValuedAttr<I64Attr, "10">:$parallel_iterations
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let verifier = [{ return Verify(*this); }];

  let hasCanonicalizer = 1;
}

def TF_TensorListReserveOp : TF_TensorListInitOp<"TensorListReserve"> {
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
}

// This operation when auto-generated is marked as NoSideEffect because it isn't
// stateful in TensorFlow. However it is kept alive through control dependency,
// and does not have any output. When placed in an island it wouldn't be kept
// alive in any way and the canonicalizer would just always fold it away.
def TF_TPUReplicateMetadataOp : TF_Op<"TPUReplicateMetadata", []> {
  let summary = [{
Metadata indicating how the TPU computation should be replicated.
  }];

  let description = [{
This operation holds the metadata common to operations of a `tpu.replicate()` computation subgraph.
  }];

  let arguments = (ins
    Confined<I64Attr, [IntMinValue<0>]>:$num_replicas,
    DefaultValuedAttr<I64Attr, "1">:$num_cores_per_replica,
    StrAttr:$topology,
    DefaultValuedAttr<BoolAttr, "true">:$use_tpu,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$device_assignment,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$computation_shape,
    DefaultValuedAttr<StrArrayAttr, "{}">:$host_compute_core,
    DefaultValuedAttr<StrArrayAttr, "{}">:$padding_map,
    DefaultValuedAttr<StrAttr, "STEP_MARK_AT_ENTRY">:$step_marker_location,
    DefaultValuedAttr<BoolAttr, "false">:$allow_soft_placement,
    DefaultValuedAttr<BoolAttr, "false">:$use_spmd_for_xla_partitioning
  );

  let results = (outs);
}

def TF_VarHandleOp : TF_Op<"VarHandleOp", []> {
  let summary = "Creates a handle to a Variable resource from its name.";

  let description = [{
container: the container this variable is placed in.
shared_name: the name by which this variable is referred to.
dtype and shape: attributes representing the data type and shape held in the
  variable.

Example:
    resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[8, 16], container="foo", shared_name="bar")
  returns a handle for a variable with name "bar" in container "foo", and the
  variable holds a tensor of shape [8, 16] and dtype int32.
  }];

  let arguments = (ins
    DefaultValuedAttr<StrAttr, "">:$container,
    DefaultValuedAttr<StrAttr, "">:$shared_name
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_VariableAlloc]>:$resource
  );

  TF_DerivedOperandOrResultHandleTypeAttr dtype =
    TF_DerivedOperandOrResultHandleTypeAttr<"resource">;
  TF_DerivedOperandOrResultHandleShapeAttr shape =
    TF_DerivedOperandOrResultHandleShapeAttr<"resource">;
}

// Multiple variadic operands with different sizes are not supported by the
// dialect generator, so we manually added the op.
def TF_SendTPUEmbeddingGradientsOp : TF_Op<"SendTPUEmbeddingGradients", [AttrSizedOperandSegments]> {
  let summary = "Performs gradient updates of embedding tables.";

  let description = [{
inputs: A TensorList of gradients with which to update embedding tables.
    This argument has the same length and shapes as the return value of
    RecvTPUEmbeddingActivations, but contains gradients of the model's loss
    with respect to the embedding activations. The embedding tables are updated
    from these gradients via the optimizer specified in the TPU embedding
    configuration given to tpu.initialize_system.
learning_rates: A TensorList of float32 scalars, one for each dynamic learning
    rate tag: see the comments in
    //third_party/tensorflow/core/protobuf/tpu/optimization_parameters.proto.
    Multiple tables can share the same dynamic learning rate tag as specified
    in the configuration. If the learning rates for all tables are constant,
    this list should be empty.
config: Serialized TPUEmbeddingConfiguration proto.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,
    Variadic<TF_Tensor>:$learning_rates,
    StrAttr:$config
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandSizeAttr NN = TF_DerivedOperandSizeAttr<1>;
}

// Multiple variadic operands with different sizes are not supported by the
// dialect generator, so we manually added the op.
def TF__SendTPUEmbeddingGradientsOp : TF_Op<"_SendTPUEmbeddingGradients", [AttrSizedOperandSegments]> {
  let summary = "Performs gradient updates of embedding tables.";

  let description = [{
The gradients argument is a TensorList having the same length and shapes as the
return value of _RecvTPUEmbeddingActivations, but contains gradients of the
model's loss with respect to the embedding activations. The embedding tables are
updated from these gradients via the optimizer specified in the
TPUEmbeddingConfiguration proto given to tpu.initialize_system.

gradients: A TensorList of gradients with which to update embedding tables.
learning_rates: A TensorList of learning rates used for updating the embedding
    tables via the optimizer. The length of the TensorList must be equal to the
    number of dynamic learning rate tags specified in the
    TPUEmbeddingConfiguration proto.
deduplication_data: A Tensor with type=DT_VARIANT containing the deduplication
    data. The tensor is an XLA nested tuple containing N elements. Each
    element of the nested tuple is a tuple of rank 1 tensors. Each tensor either
    contains indices (DT_INT32) for embedding lookup or weights (DT_FLOAT) to
    apply to the output of the embedding lookup operation.
config: Serialized TPUEmbeddingConfiguration proto.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$gradients,
    Variadic<TF_Tensor>:$learning_rates,
    TF_VariantTensor:$deduplication_data,
    StrAttr:$config
  );

  TF_DerivedOperandSizeAttr NumTables = TF_DerivedOperandSizeAttr<0>;
  TF_DerivedOperandSizeAttr NumLearningRateTags = TF_DerivedOperandSizeAttr<1>;
}

// Updated the op description text from the auto-generated op definition.
def TF__RecvTPUEmbeddingDeduplicationDataOp : TF_Op<"_RecvTPUEmbeddingDeduplicationData", []> {
  let summary = [{
Receives deduplication data (indices and weights).
  }];

  let description = [{
The deduplication data is a Tensor with type=DT_VARIANT. The tensor itself is an
XLA nested tuple containing N elements. Each element of the nested tuple is a
tuple of rank 1 tensors. Each tensor either contains indices (DT_INT32) for
embedding lookup or weights (DT_FLOAT) to apply to the output of the embedding
lookup operation.
  }];

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    TF_VariantTensor:$output
  );
}

def TF_XlaShardingOp : TF_Op<"XlaSharding", [NoSideEffect]> {
  let summary = [{
An op which shards the input based on the given sharding attribute.
  }];

  let arguments = (ins
    TF_Tensor:$input,

    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_InfeedDequeueTupleOp : TF_Op<"InfeedDequeueTuple", []> {
  let summary = "Fetches multiple values from infeed as an XLA tuple.";

  let arguments = (ins
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedResultShapeListAttr shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr dtypes = TF_DerivedResultTypeListAttr<0>;
}

def TF_StringFormatOp : TF_Op<"StringFormat", [NoSideEffect]> {
  let summary = "Formats a string template using a list of tensors.";

  let description = [{
Formats a string template using a list of tensors, pretty-printing tensor summaries.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    DefaultValuedAttr<StrAttr, "%s">:$strtemplate,
    DefaultValuedAttr<StrAttr, "%s">:$placeholder,
    DefaultValuedAttr<I64Attr, "3">:$summarize
  );

  let results = (outs
    TF_StrTensor:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

//===----------------------------------------------------------------------===//
// tf.data ops
//===----------------------------------------------------------------------===//

def TF_BatchDatasetV2Op : TF_Op<"BatchDatasetV2", [NoSideEffect]> {
  let summary = [{
Creates a dataset that batches `batch_size` elements from `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    I64Tensor:$batch_size,
    I1Tensor:$drop_remainder,

    DefaultValuedAttr<BoolAttr, "false">:$parallel_copy,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );
}

def TF_MapDatasetOp : TF_Op<"MapDataset", [NoSideEffect]> {
  let summary = [{
    Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_MapAndBatchDatasetOp : TF_Op<"MapAndBatchDataset", [NoSideEffect]> {
  let summary = "Creates a dataset that fuses mapping with batching.";

  let description = [{
Creates a dataset that applies `f` to the outputs of `input_dataset` and then
batches `batch_size` of them.

Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
to `batch_size * num_parallel_batches` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    I64Tensor:$batch_size,
    I64Tensor:$num_parallel_calls,
    I1Tensor:$drop_remainder,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_ParallelMapDatasetOp : TF_Op<"ParallelMapDataset", [NoSideEffect]> {
  let summary = [{
    Creates a dataset that applies `f` to the outputs of `input_dataset`.
  }];

  let description = [{
    Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes
    up to `num_parallel_calls` copies of `f` in parallel.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$other_arguments,
    I32Tensor:$num_parallel_calls,

    SymbolRefAttr:$f,
    Confined<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedAttr<BoolAttr, "true">:$use_inter_op_parallelism,
    DefaultValuedAttr<BoolAttr, "false">:$sloppy,
    DefaultValuedAttr<BoolAttr, "false">:$preserve_cardinality
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Targuments = TF_DerivedOperandTypeListAttr<1>;
}

def TF_TensorSliceDatasetOp : TF_Op<"TensorSliceDataset", []> {
  let summary = [{
    Creates a dataset that emits each dim-0 slice of `components` once.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$components,
    Confined<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes
  );

  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeListAttr Toutput_types = TF_DerivedOperandTypeListAttr<0>;
}

def TF_ToBoolOp : TF_Op<"ToBool", [NoSideEffect]> {
  let summary = "Converts a tensor to a scalar predicate.";

  let description = [{
Converts a tensor to a scalar predicate with the following rules:

- For 0D tensors, truthiness is determined by comparing against a "zero"
  value. For numerical types it is the obvious zero. For strings it is the
  empty string.

- For >0D tensors, truthiness is determined by looking at the number of
  elements. If has zero elements, then the result is false. Otherwise the
  result is true.

This matches the behavior of If and While for determining if a tensor counts
as true/false for a branch condition.
  }];

  let arguments = (ins
    TF_Tensor:$input
  );

  let results = (outs
    0DTensorOf<[I1]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value", [{
      build(builder, result, RankedTensorType::get({}, builder.getI1Type()),
            value);
    }]>];

  let hasCanonicalizer = 1;
}

def TF_BesselI0eOp : TF_Op<"BesselI0e", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the Bessel i0e function of `x` element-wise.";

  let description = [{
Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.

This function is faster and numerically stabler than `bessel_i0(x)`.
  }];

  let arguments = (ins
    TF_FpTensor:$x
  );

  let results = (outs
    TF_FpTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_BesselI1eOp : TF_Op<"BesselI1e", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the Bessel i1e function of `x` element-wise.";

  let description = [{
Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.

This function is faster and numerically stabler than `bessel_i1(x)`.
  }];

  let arguments = (ins
    TF_FpTensor:$x
  );

  let results = (outs
    TF_FpTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_TPUPartitionedCallOp : TF_Op<"TPUPartitionedCall", [CallOpInterface]> {
  let summary = "Calls a function placed on a specified TPU device.";

  let arguments = (ins
    Variadic<TF_Tensor>:$args,
    I32Tensor:$device_ordinal,

    SymbolRefAttr:$f,
    DefaultValuedAttr<I64Attr, "0">:$autotuner_thresh
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return args(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return fAttr(); }

    // returns the callee of this operation.
    FuncOp func() {
      return SymbolTable::lookupNearestSymbolFrom<FuncOp>(*this, f());
    }
  }];

  let verifier = [{ return VerifyPartitionedCall(*this); }];
}

def TF_BatchFunctionOp : TF_Op<"BatchFunction", [AttrSizedOperandSegments]> {
  let summary = [{
Batches all the inputs tensors to the computation done by the function.
  }];

  let description = [{
So, for example, in the following code

  ```python

  # This input will be captured.
  y = tf.placeholder_with_default(1.0, shape=[])

  @tf.Defun(tf.float32)
  def computation(a):
    return tf.matmul(a, a) + y

  b = gen_batch_ops.batch_function(
          f=computation
          in_tensors=[a],
          captured_tensors=computation.captured_inputs,
          Tout=[o.type for o in computation.definition.signature.output_arg],
          num_batch_threads=1,
          max_batch_size=10,
          batch_timeout_micros=100000,  # 100ms
          allowed_batch_sizes=[3, 10],
          batching_queue="")
  ```

If more than one session.run call is simultaneously trying to compute `b`
the values of `a` will be gathered, non-deterministically concatenated
along the first axis, and only one thread will run the computation.

Assumes that all arguments of the function are Tensors which will be batched
along their first dimension.

Arguments that are captured, are not batched. The session.run call which does
the concatenation, will use the values of the captured tensors available to it.
Therefore, typical uses of captured tensors should involve values which remain
unchanged across session.run calls. Inference is a good example of this.

SparseTensor is not supported. The return value of the decorated function
must be a Tensor or a list/tuple of Tensors.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$in_tensors,
    Variadic<TF_Tensor>:$captured_tensors,

    SymbolRefAttr:$f,
    I64Attr:$num_batch_threads,
    I64Attr:$max_batch_size,
    I64Attr:$batch_timeout_micros,
    DefaultValuedAttr<I64Attr, "10">:$max_enqueued_batches,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$allowed_batch_sizes,
    StrAttr:$container,
    StrAttr:$shared_name,
    StrAttr:$batching_queue,
    DefaultValuedAttr<BoolAttr, "false">:$enable_large_batch_splitting,
    I32ElementsAttr:$operand_segment_sizes
  );

  let results = (outs
    Variadic<TF_Tensor>:$out_tensors
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedOperandTypeListAttr Tcaptured = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;
}

def TF_AddV2Op : TF_Op<"AddV2", [Commutative, NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
                 WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint8Ref]>:$x,
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint8Ref]>:$y
  );

  let results = (outs
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint8Ref]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_DivNoNanOp : TF_Op<"DivNoNan", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                    WithBroadcastableBinOpBuilder {
  let summary = "Returns 0 if the denominator is zero.";

  let description = [{
*NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[F16, F32, F64, TF_Complex, TF_F16Ref, TF_F32Ref, TF_F64Ref, TF_ComplexRef]>:$x,
    TensorOf<[F16, F32, F64, TF_Complex, TF_F16Ref, TF_F32Ref, TF_F64Ref, TF_ComplexRef]>:$y
  );

  let results = (outs
    TensorOf<[F16, F32, F64, TF_Complex, TF_F16Ref, TF_F32Ref, TF_F64Ref, TF_ComplexRef]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_MaximumOp : TF_Op<"Maximum", [NoSideEffect, ResultsBroadcastableShape, TF_SameOperandsAndResultElementTypeResolveRef]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns the max of x and y (i.e. x > y ? x : y) element-wise.";

  let description = [{
*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Float, I16, I32, I64, TF_Uint8, TF_FloatRef, TF_Int16Ref, TF_Int32Ref, TF_Int64Ref, TF_Uint8Ref]>:$x,
    TensorOf<[TF_Float, I16, I32, I64, TF_Uint8, TF_FloatRef, TF_Int16Ref, TF_Int32Ref, TF_Int64Ref, TF_Uint8Ref]>:$y
  );

  let results = (outs
    TensorOf<[TF_Float, I16, I32, I64, TF_Uint8, TF_FloatRef, TF_Int16Ref, TF_Int32Ref, TF_Int64Ref, TF_Uint8Ref]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_RealDivOp : TF_Op<"RealDiv", [NoSideEffect, ResultsBroadcastableShape, TF_CwiseBinary]>,
                   WithBroadcastableBinOpBuilder {
  let summary = "Returns x / y element-wise for real types.";

  let description = [{
If `x` and `y` are reals, this will return the floating-point division.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  }];

  let arguments = (ins
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint16, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint16Ref, TF_Uint8Ref]>:$x,
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint16, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint16Ref, TF_Uint8Ref]>:$y
  );

  let results = (outs
    TensorOf<[TF_Float, TF_SInt, TF_Complex, TF_Uint16, TF_Uint8, TF_FloatRef, TF_SIntRef, TF_ComplexRef, TF_Uint16Ref, TF_Uint8Ref]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let hasFolder = 1;
}

def TF_AddOp : TF_Op<"Add", [NoSideEffect, ResultsBroadcastableShape, TF_LayoutAgnostic, TF_SameOperandsAndResultElementTypeResolveRef]>,
               WithBroadcastableBinOpBuilder {
  let summary = "Returns x + y element-wise.";

  let description = [{
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

Given two input tensors, the `tf.add` operation computes the sum for every element in the tensor.

Both input and output have a range `(-inf, inf)`.
  }];

  let arguments = (ins
    TensorOf<[TF_NumberNotQuantizedOrStr, TF_NumberNotQuantizedOrStrRef]>:$x,
    TensorOf<[TF_NumberNotQuantizedOrStr, TF_NumberNotQuantizedOrStrRef]>:$y
  );

  let results = (outs
    TensorOf<[TF_NumberNotQuantizedOrStr, TF_NumberNotQuantizedOrStrRef]>:$z
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;
}

def TF_MlirLocalVarOp : TF_Op<"MlirLocalVarOp", []> {
  let summary = "Creates a handle to an in-scope variable.";

  let description = [{
Used by internal passes for temporary representation of local state, which will
be eventually removed.
  }];

  let arguments = (ins);

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_VariableAlloc]>:$resource
  );
}

def TF_TPUReshardVariablesOp : TF_Op<"TPUReshardVariables", []> {
  let summary = [{
Op that reshards on-device TPU variables to specified state. Internal use only.
  }];

  let description = [{
The sharding state is represented as the key of the compilation that generated
the sharding/unsharding programs along with the main program. new_format_key
specifies the desired state, and format_state_var is the current state of the
variables.
  }];

  let arguments = (ins
    Arg<Variadic<TF_ResourceTensor>, "", [TF_VariableRead,TF_VariableWrite]>:$vars,
    TF_StrTensor:$new_format_key,
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$format_state_var
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_StatefulStandardNormalV2Op : TF_Op<"StatefulStandardNormalV2", []> {
  let summary = "Outputs random values from a normal distribution.";

  let description = [{
The generated values will have mean 0 and standard deviation 1.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    I64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatefulTruncatedNormalOp : TF_Op<"StatefulTruncatedNormal", []> {
  let summary = "Outputs random values from a truncated normal distribution.";

  let description = [{
The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    I64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatefulUniformOp : TF_Op<"StatefulUniform", []> {
  let summary = "Outputs random values from a uniform distribution.";

  let description = [{
The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    I64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TF_FpTensor:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

def TF_StatefulUniformFullIntOp : TF_Op<"StatefulUniformFullInt", []> {
  let summary = "Outputs random integers from a uniform distribution.";

  let description = [{
The generated values are uniform integers covering the whole range of `dtype`.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    I64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TensorOf<[I32, I64, TF_Uint32, TF_Uint64]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

// TODO(lyandy): Investigate supported dtypes (`minval`, `maxval`, `output`) for
// `tf.StatefulUniformInt`. tf2xla kernels support i32, i64, ui32, and ui64
// while TensorFlow CPU/GPU kernels only support i32 and i64.
def TF_StatefulUniformIntOp : TF_Op<"StatefulUniformInt", []> {
  let summary = "Outputs random integers from a uniform distribution.";

  let description = [{
The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    I64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape,
    TensorOf<[I32, I64, TF_Uint32, TF_Uint64]>:$minval,
    TensorOf<[I32, I64, TF_Uint32, TF_Uint64]>:$maxval
  );

  let results = (outs
    TensorOf<[I32, I64, TF_Uint32, TF_Uint64]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<3>;
}

#endif // TF_OPS
