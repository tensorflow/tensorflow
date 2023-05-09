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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class TF_TensorListInitOp<string mnemonic> : TF_Op<mnemonic, [Pure]> {
  let results = (outs
    TF_VariantTensor:$handle
  );

  TF_DerivedOperandTypeAttr shape_type = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;

  DerivedTypeAttr element_dtype = DerivedTypeAttr<
      "return getElementTypeOrSelf(element_type());">;

  let extraClassDeclaration = [{
    // Returns type of the TensorList element produced by this op.
    TensorType element_type() { return handle_dtype().getSubtypes()[0]; }

    // Returns data type of the result handle. Returned type contains type of
    // the TensorList element as a subtype.
    VariantType handle_dtype() {
      return getElementTypeOrSelf(getHandle().getType()).cast<TF::VariantType>();
    }
  }];
}

def TF_CaseOp : TF_Op<"Case",
      [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

    ConfinedAttr<SymbolRefArrayAttr, [ArrayMinCount<1>]>:$branches,

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

  let hasVerifier = 1;


 let extraClassDeclaration = [{
    int num_branches() { return getBranches().size(); }

    // Gets function corresponding branch # `index`.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveBranchFunction(::mlir::SymbolTableCollection* table, int index) {
      auto flat_sym_ref = getBranches()[index].cast<FlatSymbolRefAttr>();
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, flat_sym_ref);
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, flat_sym_ref);
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp branch_function(int index) { return ResolveBranchFunction(nullptr, index); }

    // Resolve all branch functions.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    void ResolveBranchFunctions(::mlir::SymbolTableCollection* table,
        SmallVectorImpl<func::FuncOp> &functions) {
      functions.reserve(num_branches());
      for (int idx : llvm::seq<int>(0, num_branches()))
        functions.push_back(ResolveBranchFunction(table, idx));
    }
    // TODO(b/204997177): Deprecate and remove.
    void get_branch_functions(SmallVectorImpl<func::FuncOp> &functions) {
      return ResolveBranchFunctions(nullptr, functions);
    }
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

  let hasVerifier = 1;

  let hasCanonicalizer = 1;

}

// In MLIR, the TensorFlow tensor value is represented as an ElementsAttr, with
// its type encoding the tensor's shape and data type.
def TF_ConstOp : TF_Op<"Const", [ConstantLike, Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Constant tensor op";

  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;

  let builders = [
    OpBuilder<(ins "Attribute":$value)>,
    OpBuilder<(ins "Type":$type, "Attribute":$value)>,
  ];

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return BroadcastCompatible(l, r);
    }
  }];
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
    TF_Int32Tensor:$max_num_elements
  );
}

def TF_IfOp : TF_Op<"If", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Resolve the then branch function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveThenFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getThenBranchAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        *this, getThenBranchAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp then_function(::mlir::SymbolTableCollection* table = nullptr) {
      return ResolveThenFunction(table);
    }

    // Resolve the else branch function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveElseFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getElseBranchAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        *this, getElseBranchAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp else_function(::mlir::SymbolTableCollection* table = nullptr) {
      return ResolveElseFunction(table);
    }
  }];
}

def TF_YieldOp : TF_Op<"Yield",
      [Pure, ReturnLike, Terminator,
       ParentOneOf<["CaseRegionOp", "IfRegionOp", "WhileRegionOp"]>]> {
  let summary = "Yield operation";

  let description = [{
    The "yield" operation represents a return operation within the conditional
    and body of structured control flow (e.g., if and while). The operation
    takes a variable number of operands and produces no results. The number and
    types of inputs must match the signature of the operation that contains the
    region.
  }];

  let arguments = (ins Variadic<AnyType>);
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
    BoolAttr:$is_stateless,
    // Used to maintain function name when round-tripping
    // between functional and regional control flow.  This can be removed if
    // the runtime does not require globally unique then/else branch function names.
    OptionalAttr<StrAttr>:$_then_func_name,
    OptionalAttr<StrAttr>:$_else_func_name
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let regions = (region SizedRegion<1>:$then_branch, SizedRegion<1>:$else_branch);

  let hasRegionVerifier = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
      "llvm::ArrayRef<::mlir::NamedAttribute>":$attributes,
      "unsigned":$numRegions),
    [{
      assert(numRegions == 2u && "mismatched number of regions");
      build($_builder, $_state, resultTypes, operands, attributes);
    }]>];

  let hasCanonicalizer = 1;
}

def TF_LegacyCallOp : TF_Op<"LegacyCall",
      [CallOpInterface,
       DeclareOpInterfaceMethods<SymbolUserOpInterface>, Pure]> {
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
    DefaultValuedOptionalAttr<BoolAttr, "false">:$_disable_call_shape_inference
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return getArgs(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getFAttr(); }
    // Sets the callee from the callable
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee);

    // Returns the resolved callee function of this operation.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveFunc(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp func() {  return ResolveFunc(nullptr); }
  }];
}

def TF_ParseExampleOp : TF_Op<"ParseExample",
                               [Pure,
                                AttrSizedResultSegments,
                                AttrSizedOperandSegments]> {

  let summary =
    "Transforms a vector of tf.Example protos (as strings) into typed tensors.";

  let arguments = (ins
    TF_StrTensor:$serialized,
    TF_StrTensor:$names,
    Variadic<TF_StrTensor>:$sparse_keys,
    Variadic<TF_StrTensor>:$dense_keys,
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$dense_defaults,

    TF_ShapeAttrArray:$dense_shapes,
    DenseI32ArrayAttr:$result_segment_sizes,
    DenseI32ArrayAttr:$operand_segment_sizes
  );

  let results = (outs
    Variadic<TF_Int64Tensor>:$sparse_indices,                           // len(sparse_types)
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$sparse_values,  // len(sparse_types)
    Variadic<TF_Int64Tensor>:$sparse_shapes,                            // len(sparse_types)
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$dense_values    // len(Tdense)
  );

  TF_DerivedOperandSizeAttr Nsparse = TF_DerivedOperandSizeAttr<2>;
  TF_DerivedOperandSizeAttr Ndense = TF_DerivedOperandSizeAttr<3>;
  TF_DerivedOperandTypeListAttr Tdense = TF_DerivedOperandTypeListAttr<4>;
  TF_DerivedResultTypeListAttr sparse_types = TF_DerivedResultTypeListAttr<1>;

  let hasVerifier = 0;
}

def TF_ParseExampleV2Op : TF_Op<"ParseExampleV2",
                                [Pure,
                                 AttrSizedResultSegments]> {

  let summary =
    "Transforms a vector of tf.Example protos (as strings) into typed tensors.";

  let arguments = (ins
    TF_StrTensor:$serialized,
    TF_StrTensor:$names,
    TF_StrTensor:$sparse_keys,
    TF_StrTensor:$dense_keys,
    TF_StrTensor:$ragged_keys,
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$dense_defaults,

    ConfinedAttr<I64Attr, [IntMinValue<0>]>:$num_sparse,
    TF_ShapeAttrArray:$dense_shapes,
    DenseI32ArrayAttr:$result_segment_sizes
  );

  let results = (outs
    Variadic<TF_Int64Tensor>:$sparse_indices,                           // len(sparse_types)
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$sparse_values,  // len(sparse_types)
    Variadic<TF_Int64Tensor>:$sparse_shapes,                            // len(sparse_types)
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$dense_values,   // len(Tdense)
    Variadic<TensorOf<[TF_Float32, TF_Int64, TF_Str]>>:$ragged_values,  // len(ragged_value_types)
                                                            //     = len(ragged_split_types)
    Variadic<TensorOf<[TF_Int32, TF_Int64]>>:$ragged_row_splits         // len(ragged_split_types)
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

  let hasVerifier = 1;
}

def TF_PlaceholderOp : TF_Op<"Placeholder", [Pure]> {
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

def TF_PlaceholderWithDefaultOp : TF_Op<"PlaceholderWithDefault", [Pure]> {
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
      [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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
    operand_range getArgOperands() { return getArgs(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getFAttr(); }
    // Sets the callee from the callable
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee);

    // Returns the resolved callee function of this operation.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveFunc(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp func() {  return ResolveFunc(nullptr); }
  }];
}

def TF_WhileOp : TF_Op<"While",
      [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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
    ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "10">, [IntMinValue<1>]>:$parallel_iterations,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless,

    // In TensorFlow, While has a special behavior where if `output_shapes`
    // attribute is not empty, those shapes are used in its shape function
    // as result shapes instead of propagating operand shapes as result shapes.
    // This allows for different result shapes from operand shapes. While these
    // shapes are imported and set as a part of the result type, there is no
    // indicator differentiating between having no output shapes compared to
    // having all unranked shapes. Thus this attribute is set to determine
    // which shape function behavior to use for this op, specifically
    // propagating operand shapes as result shapes when this attribute is not
    // set, or preserving result shapes as is when this attribute is set.
    UnitAttr:$shape_invariant
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultShapeListAttr output_shapes = TF_DerivedResultShapeListAttr<0>;

  let extraClassDeclaration = [{
    // Get the condition function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveCondFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getCondAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getCondAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp cond_function() { return ResolveCondFunction(nullptr); }

    // Get the body function.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveBodyFunction(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getBodyAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getBodyAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp body_function() { return ResolveBodyFunction(nullptr); }
  }];
}

def TF_WhileRegionOp : TF_Op<"WhileRegion",
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

    ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "10">, [IntMinValue<1>]>:$parallel_iterations,

    // Used to map StatelessWhile and While op defined in TensorFlow to a common
    // op.
    BoolAttr:$is_stateless,

    // In TensorFlow, While has a special behavior where if `output_shapes`
    // attribute is not empty, those shapes are used in its shape function
    // as result shapes instead of propagating operand shapes as result shapes.
    // This allows for different result shapes from operand shapes. While these
    // shapes are imported and set as a part of the result type, there is no
    // indicator differentiating between having no output shapes compared to
    // having all unranked shapes. Thus this attribute is set to determine
    // which shape function behavior to use for this op, specifically
    // propagating operand shapes as result shapes when this attribute is not
    // set, or preserving result shapes as is when this attribute is set.
    UnitAttr:$shape_invariant
  );
  let results = (outs Variadic<AnyTensor>:$output);

  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let hasVerifier = 1;

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
    TF_Int32Tensor:$num_elements
  );
}

def TF_VarHandleOp : TF_Op<"VarHandleOp", [DeclareOpInterfaceMethods<TF_ResourceHandleAllocatorInterface>]> {
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
    DefaultValuedStrAttr<StrAttr, "">:$container,
    DefaultValuedStrAttr<StrAttr, "">:$shared_name
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_VariableAlloc]>:$resource
  );

  DerivedTypeAttr dtype = DerivedTypeAttr<
      "return getElementTypeOrSelf(resource_subtype());">;
  DerivedAttr shape = DerivedAttr<
      "ShapedType",
      "return resource_subtype().cast<ShapedType>();",
      [{ mlir::TF::ShapeAttr::get($_ctxt, $_self) }]>;

  let extraClassDeclaration = [{
    TensorType resource_subtype() { return resource_type().getSubtypes()[0]; }

    ResourceType resource_type() {
      return getElementTypeOrSelf(getResource()).cast<TF::ResourceType>();
    }
  }];

  let hasVerifier = 1;
}

def TF_XlaShardingOp : TF_Op<"XlaSharding", [Pure, TF_NoConstantFold]> {
  let summary = [{
An op which shards the input based on the given sharding attribute.
  }];

  let arguments = (ins
    TF_Tensor:$input,

    DefaultValuedStrAttr<StrAttr, "">:$sharding,
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
    OptionalAttr<StrAttr>:$_XlaSharding,
    OptionalAttr<ArrayAttr>:$layouts
  );

  let results = (outs
    Variadic<TF_Tensor>:$outputs
  );

  TF_DerivedResultShapeListAttr shapes = TF_DerivedResultShapeListAttr<0>;
  TF_DerivedResultTypeListAttr dtypes = TF_DerivedResultTypeListAttr<0>;
}

// TODO(b/177675373): Make dtypes and shapes derived attributes,
// use more general solution.
def TF_InfeedEnqueueTupleOp : TF_Op<"InfeedEnqueueTuple", []> {
  let summary = [{
Feeds multiple Tensor values into the computation as an XLA tuple.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be provided using the infeed mechanism.}]>:$inputs,

    ConfinedAttr<TypeArrayAttr, [ArrayMinCount<1>]>:$dtypes,
    TF_ShapeAttrArray:$shapes,
    DefaultValuedOptionalAttr<I64ArrayAttr, "{}">:$layouts,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$device_ordinal
  );

  let results = (outs);
}

// This op is manually defined because the attribute name `template` (which is
// a keyword) is changed to `strtemplate`.
def TF_StringFormatOp : TF_Op<"StringFormat", [Pure]> {
  let summary = "Formats a string template using a list of tensors.";

  let description = [{
Formats a string template using a list of tensors, pretty-printing tensor summaries.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    DefaultValuedStrAttr<StrAttr, "%s">:$strtemplate,
    DefaultValuedStrAttr<StrAttr, "%s">:$placeholder,
    DefaultValuedOptionalAttr<I64Attr, "3">:$summarize
  );

  let results = (outs
    TF_StrTensor:$output
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

//===----------------------------------------------------------------------===//
// tf.data ops
//===----------------------------------------------------------------------===//

def TF_ReduceDatasetOp : TF_Op<"ReduceDataset", [SameVariadicOperandSize]> {
  let summary = [{
    Reduces the input dataset to a singleton using a reduce function.
  }];

  let arguments = (ins
    TF_VariantTensor:$input_dataset,
    Variadic<TF_Tensor>:$initial_state,
    Variadic<TF_Tensor>:$other_arguments,

    SymbolRefAttr:$f,
    ConfinedAttr<TypeArrayAttr, [ArrayMinCount<1>]>:$Tstate,
    ConfinedAttr<TypeArrayAttr, [ArrayMinCount<0>]>:$Targuments,
    ConfinedAttr<TypeArrayAttr, [ArrayMinCount<1>]>:$output_types,
    ConfinedAttr<TF_ShapeAttrArray, [ArrayMinCount<1>]>:$output_shapes,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$use_inter_op_parallelism
  );

  let results = (outs
    Variadic<TF_Tensor>:$components
  );
}

// Manually defined to restrict result type to `I1Tensor`.
def TF_ToBoolOp : TF_Op<"ToBool", [DeclareOpInterfaceMethods<InferTypeOpInterface>, Pure]> {
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
    I1Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return ArraysAreCastCompatible(l, r);
    }
  }];
}

def TF_BesselI0eOp : TF_Op<"BesselI0e", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the Bessel i0e function of `x` element-wise.";

  let description = [{
Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.

This function is faster and numerically stabler than `bessel_i0(x)`.
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_BesselI1eOp : TF_Op<"BesselI1e", [Pure, TF_SameOperandsAndResultTypeResolveRef]> {
  let summary = "Computes the Bessel i1e function of `x` element-wise.";

  let description = [{
Exponentially scaled modified Bessel function of order 0 defined as
`bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.

This function is faster and numerically stabler than `bessel_i1(x)`.
  }];

  let arguments = (ins
    TF_FloatTensor:$x
  );

  let results = (outs
    TF_FloatTensor:$y
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange inferred, TypeRange actual) {
      return ArraysAreCastCompatible(inferred, actual);
    }
  }];
}

def TF_TPUPartitionedCallOp : TF_Op<"TPUPartitionedCall",
      [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Calls a function placed on a specified TPU device.";

  let arguments = (ins
    Variadic<TF_Tensor>:$args,
    TF_Int32Tensor:$device_ordinal,

    SymbolRefAttr:$f,
    DefaultValuedOptionalAttr<I64Attr, "0">:$autotuner_thresh
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeListAttr Tin = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tout = TF_DerivedResultTypeListAttr<0>;

  let extraClassDeclaration = [{
    // Gets the argument operands to the called function.
    operand_range getArgOperands() { return getArgs(); }

    // Returns the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getFAttr(); }
    // Sets the callee from the callable.
    void setCalleeFromCallable(CallInterfaceCallable callee);

    // Returns the resolved callee function of this operation.
    // Prefer passing in SymbolTableCollection to reduce lookup costs by
    // enabling reusing cached symbol table lookup.
    func::FuncOp ResolveFunc(::mlir::SymbolTableCollection* table) {
      if (table)
        return table->lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
      return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(*this, getFAttr());
    }
    // TODO(b/204997177): Deprecate and remove.
    func::FuncOp func() {  return ResolveFunc(nullptr); }
  }];
}

def TF_StatefulUniformFullIntOp : TF_Op<"StatefulUniformFullInt", []> {
  let summary = "Outputs random integers from a uniform distribution.";

  let description = [{
The generated values are uniform integers covering the whole range of `dtype`.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead,TF_VariableWrite]>:$resource,
    TF_Int64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape
  );

  let results = (outs
    TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>:$output
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
    TF_Int64Tensor:$algorithm,
    TF_I32OrI64Tensor:$shape,
    TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>:$minval,
    TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>:$maxval
  );

  let results = (outs
    TensorOf<[TF_Int32, TF_Int64, TF_Uint32, TF_Uint64]>:$output
  );

  TF_DerivedOperandTypeAttr shape_dtype = TF_DerivedOperandTypeAttr<2>;
  TF_DerivedOperandTypeAttr dtype = TF_DerivedOperandTypeAttr<3>;
}

def TF_CloseSummaryWriterOp : TF_Op<"CloseSummaryWriter", []> {
  let summary = "Flushes and closes the summary writer.";

  let description = [{
Also removes it from the resource manager. To reopen, use another
CreateSummaryFileWriter op.

writer: A handle to the summary writer resource.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryFree]>:$writer
  );

  let results = (outs);
}

// TODO(b/168035831): Model db_uri read/write.
def TF_CreateSummaryDbWriterOp : TF_Op<"CreateSummaryDbWriter", []> {
  let summary = "Creates summary database writer accessible by given resource handle.";

  let description = [{
This can be used to write tensors from the execution graph directly
to a database. Only SQLite is supported right now. This function
will create the schema if it doesn't exist. Entries in the Users,
Experiments, and Runs tables will be created automatically if they
don't already exist.

writer: Handle to SummaryWriter resource to overwrite.
db_uri: For example "file:/tmp/foo.sqlite".
experiment_name: Can't contain ASCII control characters or <>. Case
  sensitive. If empty, then the Run will not be associated with any
  Experiment.
run_name: Can't contain ASCII control characters or <>. Case sensitive.
  If empty, then each Tag will not be associated with any Run.
user_name: Must be valid as both a DNS label and Linux username. If
  empty, then the Experiment will not be associated with any User.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_StrTensor:$db_uri,
    TF_StrTensor:$experiment_name,
    TF_StrTensor:$run_name,
    TF_StrTensor:$user_name
  );

  let results = (outs);
}

// TODO(b/168035831): Model logdir read/write.
def TF_CreateSummaryFileWriterOp : TF_Op<"CreateSummaryFileWriter", []> {
  let summary = "Creates a summary file writer accessible by the given resource handle.";

  let description = [{
writer: A handle to the summary writer resource
logdir: Directory where the event file will be written.
max_queue: Size of the queue of pending events and summaries.
flush_millis: How often, in milliseconds, to flush the pending events and
  summaries to disk.
filename_suffix: Every event file's name is suffixed with this suffix.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_StrTensor:$logdir,
    TF_Int32Tensor:$max_queue,
    TF_Int32Tensor:$flush_millis,
    TF_StrTensor:$filename_suffix
  );

  let results = (outs);
}

def TF_FlushSummaryWriterOp : TF_Op<"FlushSummaryWriter", []> {
  let summary = "Flushes the writer's unwritten events.";

  let description = [{
writer: A handle to the summary writer resource.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer
  );

  let results = (outs);
}

def TF_ImportEventOp : TF_Op<"ImportEvent", []> {
  let summary = "Outputs a `tf.Event` protocol buffer.";

  let description = [{
When CreateSummaryDbWriter is being used, this op can be useful for
importing data from event logs.

writer: A handle to a summary writer.
event: A string containing a binary-encoded tf.Event proto.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_StrTensor:$event
  );

  let results = (outs);
}

def TF_SummaryWriterOp : TF_Op<"SummaryWriter", [DeclareOpInterfaceMethods<TF_ResourceHandleAllocatorInterface>]> {
  let summary = "Returns a handle to be used to access a summary writer.";

  let description = [{
The summary writer is an in-graph resource which can be used by ops to write
summaries to event files.

writer: the summary writer resource. Scalar handle.
  }];

  let arguments = (ins
    StrAttr:$shared_name,
    StrAttr:$container
  );

  let results = (outs
    Res<TF_ResourceTensor, "", [TF_SummaryAlloc]>:$writer
  );
}

def TF_WriteAudioSummaryOp : TF_Op<"WriteAudioSummary", []> {
  let summary = "Writes a `Summary` protocol buffer with audio.";

  let description = [{
The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

writer: A handle to a summary writer.
step: The step to write the summary for.
tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 2-D of shape `[batch_size, frames]`.
sample_rate: The sample rate of the signal in hertz.
max_outputs: Max number of batch elements to generate audio for.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tag,
    TF_Float32Tensor:$tensor,
    TF_Float32Tensor:$sample_rate,

    ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "3">, [IntMinValue<1>]>:$max_outputs
  );

  let results = (outs);
}

def TF_WriteGraphSummaryOp : TF_Op<"WriteGraphSummary", []> {
  let summary = "Writes a `GraphDef` protocol buffer to a `SummaryWriter`.";

  let description = [{
writer: Handle of `SummaryWriter`.
step: The step to write the summary for.
tensor: A scalar string of the serialized tf.GraphDef proto.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tensor
  );

  let results = (outs);
}

def TF_WriteHistogramSummaryOp : TF_Op<"WriteHistogramSummary", []> {
  let summary = "Writes a histogram summary.";

  let description = [{
The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

writer: A handle to a summary writer.
step: The step to write the summary for.
tag: Scalar.  Tag to use for the `Summary.Value`.
values: Any shape. Values to use to build the histogram.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tag,
    TF_IntOrFpTensor:$values
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_WriteImageSummaryOp : TF_Op<"WriteImageSummary", []> {
  let summary = "Writes a `Summary` protocol buffer with images.";

  let description = [{
The summary has up to `max_images` summary values containing images. The
images are built from `tensor` which must be 4-D with shape `[batch_size,
height, width, channels]` and where `channels` can be:

*  1: `tensor` is interpreted as Grayscale.
*  3: `tensor` is interpreted as RGB.
*  4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float
input, the values are normalized one image at a time to fit in the range
`[0, 255]`.  `uint8` values are unchanged.  The op uses two different
normalization algorithms:

*  If the input values are all positive, they are rescaled so the largest one
   is 255.

*  If any input value is negative, the values are shifted so input value 0.0
   is at 127.  They are then rescaled so that either the smallest value is 0,
   or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

The `bad_color` argument is the color to use in the generated images for
non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
Each element must be in the range `[0, 255]` (It represents the value of a
pixel in the output image).  Non-finite values in the input tensor are
replaced by this tensor in the output image.  The default value is the color
red.

writer: A handle to a summary writer.
step: The step to write the summary for.
tag: Scalar. Used to build the `tag` attribute of the summary values.
tensor: 4-D of shape `[batch_size, height, width, channels]` where
  `channels` is 1, 3, or 4.
max_images: Max number of batch elements to generate images for.
bad_color: Color to use for pixels with non-finite values.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tag,
    TensorOf<[TF_Float16, TF_Float32, TF_Uint8]>:$tensor,
    TF_Uint8Tensor:$bad_color,

    ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "3">, [IntMinValue<1>]>:$max_images
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_WriteRawProtoSummaryOp : TF_Op<"WriteRawProtoSummary", []> {
  let summary = "Writes a `Summary` protocol buffer with serialized string `Summary` protocol buffers.";

  let description = [{
writer: A handle to a summary writer.
step: The step to write the summary for.
tensor: A tensor holding one or more serialized `Summary` protobufs to write.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tensor
  );

  let results = (outs);
}

def TF_WriteScalarSummaryOp : TF_Op<"WriteScalarSummary", []> {
  let summary = "Writes a `Summary` protocol buffer with scalar values.";

  let description = [{
The input `tag` and `value` must have the scalars.

writer: A handle to a summary writer.
step: The step to write the summary for.
tag: Tag for the summary.
value: Value for the summary.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_StrTensor:$tag,
    TF_IntOrFpTensor:$value
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<3>;
}

def TF_WriteSummaryOp : TF_Op<"WriteSummary", []> {
  let summary = "Outputs a `Summary` protocol buffer with a tensor.";

  let description = [{
writer: A handle to a summary writer.
step: The step to write the summary for.
tensor: A tensor to serialize.
tag: The summary's tag.
summary_metadata: Serialized SummaryMetadata protocol buffer containing
 plugin-related metadata for this summary.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_SummaryWrite]>:$writer,
    TF_Int64Tensor:$step,
    TF_Tensor:$tensor,
    TF_StrTensor:$tag,
    TF_StrTensor:$summary_metadata
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<2>;
}

def TF__TPUDeviceOrdinalPlaceholderOp : TF_Op<"_TPUDeviceOrdinalPlaceholder", [Pure]> {
  let summary = [{
Placeholder device ordinal that represents device ordinal of a replicated op.
  }];

  let description = [{
This op can be used when certain rewrite passes materialize ops that require a
device ordinal of a replicated op but replication logic has been abstracted away
using tf_device.replicate op. Subsequent rewrite passes must replace this op with
a constant output that represents the correct device ordinal of the replicated
operations inside a TPU host.
  }];

  let arguments = (ins);

  let results = (outs
    TF_Int64Tensor:$device_ordinal
  );
}

// We must manually define TPUPartitionedInput, TPUPartitionedInputV2,
// TPUPartitionedOutput, and TPUPartitionedOutputV2 since they have an
// optional attribute, _XlaSharding, unlike the TensorFlow definition.
def TF_TPUPartitionedInputOp : TF_Op<"TPUPartitionedInput", [Pure]> {
  let summary = [{
An op that groups a list of partitioned inputs together. This op
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,

    DefaultValuedOptionalAttr<I64Attr, "0">:$partition_dim,
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}

def TF_TPUPartitionedInputV2Op : TF_Op<"TPUPartitionedInputV2", [Pure]> {
  let summary = [{
An op that groups a list of partitioned inputs together. Supports ND sharding.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$inputs,
    I64ArrayAttr:$partition_dims,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_packed,
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;

  let hasVerifier = 1;
}

def TF_TPUPartitionedOutputOp : TF_Op<"TPUPartitionedOutput", [Pure]> {
  let summary = [{
An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned
  }];

  let description = [{
outputs outside the XLA computation.
  }];

  let arguments = (ins
    TF_Tensor:$inputs,

    DefaultValuedOptionalAttr<I64Attr, "0">:$partition_dim,
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultSizeAttr num_splits = TF_DerivedResultSizeAttr<0>;
}

def TF_TPUPartitionedOutputV2Op : TF_Op<"TPUPartitionedOutputV2", [Pure]> {
  let summary = [{
An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned
  }];

  let description = [{
outputs outside the XLA computation. Supports ND sharding.
  }];

  let arguments = (ins
    TF_Tensor:$inputs,

    I64ArrayAttr:$partition_dims,
    OptionalAttr<StrAttr>:$_XlaSharding
  );

  let results = (outs
    Variadic<TF_Tensor>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedResultSizeAttr num_splits = TF_DerivedResultSizeAttr<0>;
}

// Declares symbol reference attribute `shape_inference_graph` to be optional
// unlike the TensorFlow definition. This is required to support ops that use
// empty string value for the attribute to signify missing.
def TF_XlaHostComputeOp : TF_Op<"XlaHostCompute", [TF_SendSideEffect, TF_RecvSideEffect, TF_XlaHostComputeSideEffect]> {
  let summary = [{
A pseudo-op to represent host-side computation in an XLA program.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Tensor>, [{A list of tensors that will be sent to the host.}]>:$inputs,

    StrArrayAttr:$ancestors,
    TF_ShapeAttrArray:$shapes,
    OptionalAttr<SymbolRefAttr>:$shape_inference_graph,
    StrAttr:$key,
    DefaultValuedStrAttr<StrAttr, "">:$send_key,
    DefaultValuedStrAttr<StrAttr, "">:$recv_key,
    DefaultValuedOptionalAttr<I64Attr, "1000000">:$cost_estimate_ns,
    DefaultValuedOptionalAttr<I64Attr, "0">:$tpu_core
  );

  let results = (outs
    Res<Variadic<TF_Tensor>, [{A list of tensors that will be returned to the device.}]>:$outputs
  );

  TF_DerivedOperandTypeListAttr Tinputs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF_ConfigureAndInitializeGlobalTPUOp : TF_Op<"ConfigureAndInitializeGlobalTPU", []> {
  let summary = [{
An op that initialize the TPU system in a multi-client set up.
  }];

  let description = [{
Initializes global TPU system for mutli-client execution.

This op does the work of both ConfigureDistributedTpuOp and
InitializeHostForDistributedTpuOp, and outputs the latter's result.
  }];

  let arguments = (ins);

  let results = (outs
    Res<TF_Int32Tensor, [{A vector containing the global TPU id of each TPU on the host.}]>:$output
  );
}

def TF_ShutdownTPUSystemOp : TF_Op<"ShutdownTPUSystem", []> {
  let summary = [{
An op that shuts down the TPU system.
  }];

  let arguments = (ins);
  let results = (outs
    TF_BoolTensor:$success
  );
}

// Internal op for testing value-based side-effects for non-resource values.
// TODO(mgester) We should have an extension of TF dialect only for testing so
// TF dialect is not polluted with test ops.
def TF__InternalTestNonResourceValueSideEffects_ : TF_Op<"_InternalTestNonResourceValueSideEffects_", []> {
  let summary = "Internal op for testing only";

  let arguments = (ins
    Arg<TF_StrTensor,"", [TF_DatasetIteratorRead, TF_DatasetIteratorWrite]>:$key
  );
  let results = (outs);
}

def TF__InternalTestMustExecuteTrait_ : TF_Op<"_InternalTestMustExecuteTrait_", [TF_MustExecute]> {
  let summary = "Internal op for testing only";

  let arguments = (ins);
  let results = (outs);
}

def TF_SetStaticDimensionBoundsOp : TF_Op<"SetStaticDimensionBounds", []> {
  let summary = "Op used to indicate to the compiler and runtime the static bounds of a tensor.";
  let description = [{
The information passed through this op can possibly be used by the compiler and
runtime to perform certain optimizations such as more efficient DMAs. The
bounds passed via this op should be considered advisory only, and depending on
the implementation, might do nothing and simply be an identity

`input`: The tensor that has dynamic dimensions.
`static_shape`: The static shape of the tensor, corresponds to the maximum bounds of each dimension.
`output` is the input tensor with no changes done to it.

Example usage:

def tpu_call(args):
  def model_fn(args):
    # do something with dynamic tensor

  @function.Defun(capture_resource_var_by_value=False)
  def tpu_subgraph():
      return tf.tpu.rewrite(model_fn, args)

  return tf.raw_ops.TPUPartitionedCall(
      args=tpu_subgraph.captured_inputs,
      Tout=[o.type for o in tpu_subgraph.definition.signature.output_arg],
      f=tpu_subgraph,
      device_ordinal=[0])

static_shape = tf.placeholder(tf.int32, shape=([3]), name='static_size')

w = tf.Variable(tf.constant([[1.0], [2.0], [3.0]]), name='w')

w_dyn = tf.SetDynamicDimensionBounds(w, static_size])
tpu_call([w_dyn])
}];
  let arguments = (ins
    TF_Tensor:$input,
    TF_I32OrI64Tensor:$static_shape
  );

  let hasVerifier = 1;

  let results = (outs
    TF_Tensor:$output
  );
}

def TF_TPUCompileMlirAndExecuteOp : TF_Op<"TPUCompileMlirAndExecute", [AttrSizedOperandSegments]> {
  let summary = "Op that compiles a computation in MLIR into a TPU program, and loads and executes it on a TPU device.";

  let description = [{
For the internal use of the TPU compiler.

'static_shapes' are tensors specifying the maximum dimension sizes for the tensors specified in `dynamic_operands`.
'args' are inputs to the TPU computation.
'operands_with_static_shape' are the indices of the operands that have a maximal static shape specified.
'mlir_module' is a serialized MLIR module with a `main` function that contains
target computation.
'metadata' is a serialized TPUCompileMetadataProto describing the shapes and
types of the inputs to the computation, as well as a mapping onto the TPU pod
topology.
'producer_name' is a string describing the name of the framework that add support for running this portion of the model on TPUs.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$args,
    Variadic<TF_Int64Tensor>:$static_shapes,
    OptionalAttr<I32ArrayAttr>:$operands_with_static_shape,
    DefaultValuedStrAttr<StrAttr, "">:$mlir_module,
    StrAttr:$metadata,
    StrAttr:$producer_name
  );

  let results = (outs
    TF_Tensor:$rendezvous_key_base,
    Variadic<TF_Tensor>:$results
  );

  TF_DerivedOperandTypeListAttr Targs = TF_DerivedOperandTypeListAttr<0>;
  TF_DerivedResultTypeListAttr Tresults = TF_DerivedResultTypeListAttr<0>;
}

def TF_CollateTPUEmbeddingMemoryOp : TF_Op<"CollateTPUEmbeddingMemory", []> {
  let summary = [{
An op that merges the string-encoded memory config protos from all hosts.
  }];

  let arguments = (ins
    Arg<Variadic<TF_StrTensor>, [{String-encoded memory config protos containing metadata about
the memory allocations reserved for TPUEmbedding across all hosts.}]>:$memory_configs
  );

  let results = (outs
    TF_StrTensor:$merged_memory_config
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<0>;
}


// Following ops are manually defined as the ops are using inputs with name `t`
// that will lead to conflict with getter for attribute `T` as both becomes
// getT().
def TF_BatchNormWithGlobalNormalizationOp : TF_Op<"BatchNormWithGlobalNormalization", [Pure]> {
  let summary = "Batch normalization.";

  let description = [{
This op is deprecated. Prefer `tf.nn.batch_normalization`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 4D input Tensor.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D mean Tensor with size matching the last dimension of x.
This is the first output from tf.nn.moments,
or a saved moving average thereof.}]>:$m,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D variance Tensor with size matching the last dimension of x.
This is the second output from tf.nn.moments,
or a saved moving average thereof.}]>:$v,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D beta Tensor with size matching the last dimension of x.
An offset to be added to the normalized tensor.}]>:$beta,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 1D gamma Tensor with size matching the last dimension of x.
If "scale_after_normalization" is true, this tensor will be multiplied
with the normalized tensor.}]>:$gamma,

    F32Attr:$variance_epsilon,
    BoolAttr:$scale_after_normalization
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>:$result
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_ClipByValueOp : TF_Op<"ClipByValue", [Pure, TF_SameOperandsAndResultElementTypeResolveRef]> {
  let summary = "Clips tensor values to a specified min and max.";

  let description = [{
Given a tensor `x`, this operation returns a tensor of the same type and
shape as `x` with its values clipped to `clip_value_min` and `clip_value_max`.
Any values less than `clip_value_min` are set to `clip_value_min`. Any values
greater than `clip_value_max` are set to `clip_value_max`.
  }];

  let arguments = (ins
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A `Tensor`.}]>:$x,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
as `t`. The minimum value to clip by.}]>:$clip_value_min,
    Arg<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
as `x`. The maximum value to clip by.}]>:$clip_value_max
  );

  let results = (outs
    Res<TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64, TF_Float16, TF_Float32, TF_Float64, TF_Int16, TF_Int32, TF_Int64, TF_Int8, TF_Qint16, TF_Qint32, TF_Qint8, TF_Quint16, TF_Quint8, TF_Uint16, TF_Uint32, TF_Uint64, TF_Uint8]>, [{A clipped `Tensor` with the same shape as input 't'.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}


def TF_L2LossOp : TF_Op<"L2Loss", [Pure]> {
  let summary = "L2 Loss.";

  let description = [{
Computes half the L2 norm of a tensor without the `sqrt`:

    output = sum(t ** 2) / 2
  }];

  let arguments = (ins
    Arg<TF_FloatTensor, [{Typically 2-D, but may have any dimensions.}]>:$x
  );

  let results = (outs
    Res<TF_FloatTensor, [{0-D.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SelectOp : TF_Op<"Select", [Pure]> {
  let summary = "Selects elements from `x` or `y`, depending on `condition`.";

  let description = [{
The `x`, and `y` tensors must all have the same shape, and the
output will also have that shape.

The `condition` tensor must be a scalar if `x` and `y` are scalars.
If `x` and `y` are vectors or higher rank, then `condition` must be either a
scalar, a vector with size matching the first dimension of `x`, or must have
the same shape as `x`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `x` (if true) or `y` (if false).

If `condition` is a vector and `x` and `y` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `x` and `y`.
If `condition` has the same shape as `x` and `y`, then it chooses which
element to copy from `x` and `y`.

For example:

```python
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e)  # => [[1, 6], [7, 4]]


# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]

```
  }];

  let arguments = (ins
    TF_BoolTensor:$condition,
    Arg<TF_Tensor, [{= A `Tensor` which may have the same shape as `condition`.
If `condition` is rank 1, `x` may have higher rank,
but its first dimension must match the size of `condition`.}]>:$then_value,
    Arg<TF_Tensor, [{= A `Tensor` with the same type and shape as `x`.}]>:$else_value
  );

  let results = (outs
    Res<TF_Tensor, [{= A `Tensor` with the same type and shape as `x` and `y`.}]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let hasVerifier = 1;
}

def TF_SelectV2Op : TF_Op<"SelectV2", [Pure, ResultsBroadcastableShape]> {
  let summary = "";

  let arguments = (ins
    TF_BoolTensor:$condition,
    TF_Tensor:$then_value,
    TF_Tensor:$else_value
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<1>;

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$e, "Value":$t)>
  ];
}

def TF_TPUCopyWithDynamicShapeOp : TF_Op<"TPUCopyWithDynamicShape", [Pure, AttrSizedOperandSegments]> {
  let summary = [{
Op that copies host tensors to device with bounded dynamic shape support.
  }];

  let description = [{
This op copies the padded tensor on cpu to TPU without the padded data. `tensors` 
is a list of cpu tensors with padded data. `unpadded_sizes` is a list of shape
tensors which describes unpadded size of each dimension for each cpu tensor. 
The size of the `unpadded_sizes` should be the same as `tensors`. They are both
on host. `tpu_tensors` are list of tpu device tensors without the padded data.
`tpu_tensors` also has the same size of the `tensors` and the shapes of 
`tpu_tensors` are determined by the `unpadded_sizes`.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$tensors,
    Variadic<TF_Int32Tensor>:$unpadded_sizes
  );

  let results = (outs
    Variadic<TF_Tensor>:$tpu_tensors
  );

  TF_DerivedOperandSizeAttr N = TF_DerivedOperandSizeAttr<1>;
  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

def TF_TPUAnnotateTensorsWithDynamicShapeOp : TF_Op<"TPUAnnotateTensorsWithDynamicShape", [Pure]> {
  let summary = [{
Placeholder op which takes the output of TPUCopyWithDynamicShapeOp and pass
them to the following tpu ops.
  }];

  let description = [{
This op serves as an annotation for the dynamic shaped tensor and will be
removed during the bridge rewrite.
  }];

  let arguments = (ins
    Variadic<TF_Tensor>:$tensors
  );

  let results = (outs
    Variadic<TF_Tensor>:$tpu_tensors
  );

  TF_DerivedOperandTypeListAttr T = TF_DerivedOperandTypeListAttr<0>;
}

def TF_ConvertToCooTensorOp : TF_Op<"ConvertToCooTensor", [Pure]> {
  let summary = [{
Op that converts tensors into coo format.
  }];

  let description = [{
This op coverts the dense, sparse and ragged tensor into standard coo tensor
format which contains three 1D tensors.
  }];

  let arguments = (ins
    TF_Int32Tensor:$indices_or_row_splits,
    TF_Int32Tensor:$values,
    TF_Float32Tensor:$weights,

    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$sample_count,
    StrAttr:$combiner
  );

  let results = (outs
    TF_Int32Tensor:$row_ids,
    TF_Int32Tensor:$col_ids,
    TF_Float32Tensor:$gains
  );
}

def TF_ResourceGatherNdOp : TF_Op<"ResourceGatherNd", []> {
  let summary = "GatherNd on a resource.";

  let description = [{
This op reads the variable referenced by the first argument, and
then performs a GatherNd operation on it.
  }];

  let arguments = (ins
    Arg<TF_ResourceTensor, "", [TF_VariableRead]>:$resource,
    TF_I32OrI64Tensor:$indices
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr Tindices = TF_DerivedOperandTypeAttr<1>;
  TF_DerivedResultTypeAttr dtype = TF_DerivedResultTypeAttr<0>;
}

#endif // TF_OPS
