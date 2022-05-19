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

#ifndef TFG_OPS
#define TFG_OPS

include "tensorflow/core/ir/dialect.td"
include "tensorflow/core/ir/interfaces.td"
include "tensorflow/core/ir/types/attributes.td"
include "tensorflow/core/ir/types/types.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TFGraph op definitions
//===----------------------------------------------------------------------===//

// Base class for intrinsic TFG operations. Intrinsic operations exist only in
// the TFG dialect in MLIR.
class TFG_IntrinsicOp<string mnemonic, list<Trait> traits = []> :
    Op<TFGraphDialect, mnemonic, [TraitList<traits>, IntrinsicOperation]>;

// Base class for concrete TFG operations. Concrete operations are registered
// TFG dialect operations that are imported from TensorFlow.
class TFGraph_ConcreteOp<string mnemonic, list<Trait> traits = []> :
    Op<TFGraphDialect, mnemonic, traits>;

def TFGraph_GraphOp : TFG_IntrinsicOp<"graph",
    [HasOnlyGraphRegion, SingleBlock, IsolatedFromAbove,
     OpAsmOpInterface, TraitList<GraphRegionNoTerminator.traits>]> {
  let summary = "Define a flat graph of operations";
  let description = [{
    This operation is modeling a unordered list of nodes forming a TensorFlow
    graph. This corresponds to the GraphDef proto definition minus the
    FunctionDefLibrary.
  }];
  let arguments = (ins TFType_VersionAttr:$version);
  let regions = (region SizedRegion<1>:$nodes);
  let extraClassDeclaration = [{
    TFGraphDialect *getDialect() {
      return cast<TFGraphDialect>((*this)->getDialect());
    }

    // Override the getOps() inherited from the OneRegion trait to return a
    // range of TFOp instead of the generic mlir::Operation.
    auto getOps() { return getRegion().template getOps<TFOp>(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    // This will filter the `tfg.` prefix in front of operations inside the
    // graph body.
    static StringRef getDefaultDialect() {
      return "tfg";
    }
  }];

  let assemblyFormat = "qualified($version) attr-dict-with-keyword $nodes";
  let hasVerifier = 1;
}


def TFGraph_GraphFuncOp : TFG_IntrinsicOp<"func", [
    RegionKindInterface, HasOnlyGraphRegion, SingleBlock, CallableOpInterface,
    FunctionOpInterface, IsolatedFromAbove, Symbol, OpAsmOpInterface,
    DeclareOpInterfaceMethods<ControlArgumentInterface>]> {
  let summary = "Define a function as modeled by FunctionDef.";
  let description = [{
    This operation is modeling a function definition, corresponding to the
    FunctionDef proto. It defines a region `body` which contains an unordered
    list of nodes forming a TensorFlow graph.
    The body still has a return operation as terminator which must always be
    the last operation.

    A function can be generic if it specifies the `generic` attribute. Such
    function is akin to a C++ template: it need "instantiation attributes" to be
    specified at each call site in order to make it into a concrete function,
    see `tensorflow::InstantiateFunction()` for more info.
  }];

  let arguments = (ins
    // This attribute is set if the function is generic/parameterized.
    UnitAttr:$generic,
    // Attributes required by FunctionOpInterface. These are declared here for
    // ease of access and to cache the attribute names in the op.
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<ArrayAttr>:$arg_attrs,
    OptionalAttr<ArrayAttr>:$res_attrs,
    // Optional function attributes.
    OptionalAttr<StrAttr>:$description,
    UnitAttr:$is_stateful,
    OptionalAttr<FlatSymbolRefAttr>:$gradient,
    OptionalAttr<I32ElementsAttr>:$resource_arg_unique_ids_keys,
    OptionalAttr<I32ElementsAttr>:$resource_arg_unique_ids_values,
    // Optional attributes that only appear on generic functions. On generic
    // functions, the control output names are stored on the function instead of
    // on the terminator like in non-generic functions.
    OptionalAttr<StrArrayAttr>:$control_output
  );

  // Allow construction of an empty function with no attributes.
  let builders = [
    // Construct an empty function with no attributes.
    OpBuilder<(ins), [{ (void)$_state.addRegion(); }]>,
    // Construct a function with a name, a function type, and a generic flag.
    OpBuilder<(ins "const Twine &":$sym_name, "FunctionType":$type,
                   "bool":$generic), [{
      build($_builder, $_state);
      $_state.addAttribute(GraphFuncOp::sym_nameAttrName($_state.name),
                           $_builder.getStringAttr(sym_name));
      $_state.addAttribute(GraphFuncOp::function_typeAttrName($_state.name),
                           TypeAttr::get(type));
      if (generic) {
        $_state.addAttribute(GraphFuncOp::genericAttrName($_state.name),
                             $_builder.getUnitAttr());
      }
    }]>
  ];

  let regions = (region AnyRegion:$body);
  let extraClassDeclaration = [{
    using SingleBlock::getBody;

    FlatSymbolRefAttr getGradient() {
      return (*this)->getAttrOfType<FlatSymbolRefAttr>("gradient");
    }

    TFGraphDialect *getDialect() {
      return cast<TFGraphDialect>((*this)->getDialect());
    }

    /// Returns the type of this function.
    /// FIXME: We should drive this via the ODS `type` param.
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<FunctionType>();
    }

    // Returns true if this function is marked for JIT compilation.
    bool isMarkedForCompilation();

    // Return the function called by this operation if this operation is a call
    // and the function exists in the symbol_table, otherwise return null.
    static GraphFuncOp getCalledFunction(Operation *op,
                                         SymbolTable &symbol_table);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    // This will filter the `tfg.` prefix in front of operations inside the
    // func body.
    static StringRef getDefaultDialect() {
      return "tfg";
    }

    /// Get a special name to use when printing block arguments.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn set_name_fn);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    // Returns the region on the current operation that is callable. This may
    // return null in the case of an external callable object, e.g. an external
    // function.
    Region *getCallableRegion() {
      return isExternal() ? nullptr : &this->getRegion();
    }

    // Returns the results types that the callable region produces when
    // executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    LogicalResult verifyType();
    LogicalResult verifyBody();

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TFGraph_DictionaryArrayAttr
    : TypedArrayAttrBase<DictionaryAttr, "an array of dictionaries">;

def TFGraph_ReturnOp : TFG_IntrinsicOp<"return",
  [NoSideEffect, HasParent<"GraphFuncOp">, ReturnLike, Terminator]> {
  let summary = "Return values from a Function.";
  let description = [{
    The `return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number must match the signature of the function that contains
    the operation. The types of the input to the call must be compatible with
    the function signature in terms of shape and matching the element type, and
    similarly for the function results,
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       TFGraph_DictionaryArrayAttr:$control_ret_attrs);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ValueRange":$control_operands)>];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}


def TFGraph_GetResultOp : TFG_IntrinsicOp<"get_result",
  [NoSideEffect]> {
  let summary = "Extract a value out of an opaque tensor in a generic Function.";
  let description = [{
    The `get_result` operation exists in generic functions before instantiation,
    when the number of output per node isn't known yet. In generic function
    nodes define a single SSA value which is then "unpacked" by name using this
    operation.
    Example:

    ```mlir
    %Switch, %ctl_1 = tfg.Switch(%G_z, %G_z) name = "cond/Switch"
    %s_true = tfg.get_result %Switch "output_true” : 0
    %s_false = tfg.get_result %Switch "output_false” : 0
    ```
  }];

  let arguments = (ins
    OpaqueTensorType:$value,
    StrAttr:$name,
    I32Attr:$number
  );
  let results = (outs
    OpaqueTensorType:$result
  );

  let assemblyFormat = "`(` $value `)` $name `:` $number attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// TFGraph Concrete Ops
//===----------------------------------------------------------------------===//

// TODO(jeffniu): Much of the TF ops' ODS definitions and verifiers should be
// auto-generated from the TF op registry, with a mix of hand-written code for
// op-specific cases.
// TODO(jeffniu): Also, many of these should be derived attributes (e.g. types).

// Base class for TFGraph if-like operations.
class TFGraph_IfLikeOp<string mnemonic>
    : TFGraph_ConcreteOp<mnemonic,
                         [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    // Op operands.
    TFGraph_OpaqueTensorOr<I1Tensor>:$cond,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TFType_FuncAttr:$then_branch,
    TFType_FuncAttr:$else_branch,
    // Optional attributes. These attributes are present on import and are
    // required for export, but can be dropped if they contain redundant type
    // information.
    OptionalAttr<TypeAttr>:$Tcond,
    OptionalAttr<TypeArrayAttr>:$Tin,
    OptionalAttr<TypeArrayAttr>:$Tout,
    OptionalAttr<TFGraph_ShapesAttr>:$output_shapes
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$cond, "ValueRange":$args,
                   "FuncAttr":$then_branch, "FuncAttr":$else_branch), [{
      build($_builder, $_state, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    }]>
  ];

  let extraClassDefinition = [{
    LogicalResult $cppClass::verifySymbolUses(
        SymbolTableCollection &symbol_table) {
      return VerifyIfLikeOp(*this, symbol_table);
    }
  }];
}

def TFGraph_IfOp : TFGraph_IfLikeOp<"If"> {
  let summary = "An if operation, in functional form.";
}
def TFGraph_StatelessIfOp : TFGraph_IfLikeOp<"StatelessIf"> {
  let summary = "A stateless if operation, in functional form.";
}
def TFGraph_StatefulIfOp : TFGraph_IfLikeOp<"StatefulIf"> {
  let summary = "A stateful if operation, in functional form.";
}

// Base class for TFGraph case-like operations.
class TFGraph_CaseLikeOp<string mnemonic>
    : TFGraph_ConcreteOp<mnemonic,
                         [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    // Op operands.
    TFGraph_OpaqueTensorOr<I32Tensor>:$branch_index,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TypedArrayAttrBase<TFType_FuncAttr, "an array of functions">:$branches,
    // Optional attributes. These attributes are present on import and are
    // required for export, but can be dropped if they contain redundant type
    // information.
    OptionalAttr<TypeArrayAttr>:$Tin,
    OptionalAttr<TypeArrayAttr>:$Tout,
    OptionalAttr<TFGraph_ShapesAttr>:$output_shapes
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$branch_index,
                   "ValueRange":$args, "ArrayAttr":$branches), [{
      build($_builder, $_state, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    }]>
  ];

  let extraClassDefinition = [{
    LogicalResult $cppClass::verifySymbolUses(
        SymbolTableCollection &symbol_table) {
      return VerifyCaseLikeOp(*this, symbol_table);
    }
  }];
}

def TFGraph_CaseOp : TFGraph_CaseLikeOp<"Case"> {
  let summary = "A case operation, in functional form";
}
def TFGraph_StatelessCaseOp : TFGraph_CaseLikeOp<"StatelessCase"> {
  let summary = "A stateless case operation, in functional form";
}
def TFGraph_StatefulCaseOp : TFGraph_CaseLikeOp<"StatefulCase"> {
  let summary = "A stateful case operation, in functional form";
}

// Base class for TFGraph while-like operations.
class TFGraph_WhileLikeOp<string mnemonic>
    : TFGraph_ConcreteOp<mnemonic,
                         [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    // Op operands.
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TFType_FuncAttr:$cond,
    TFType_FuncAttr:$body,
    I64Attr:$parallel_iterations,
    // Optional attributes. These attributes are present on import and are
    // required for export, but can be dropped if they contain redundant type
    // information.
    OptionalAttr<TypeArrayAttr>:$T,
    OptionalAttr<TFGraph_ShapesAttr>:$output_shapes
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "ValueRange":$args, "FuncAttr":$cond,
                   "FuncAttr":$body, "IntegerAttr":$parallel_iterations), [{
      build($_builder, $_state, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    }]>
  ];

  let extraClassDefinition = [{
    LogicalResult $cppClass::verifySymbolUses(
        SymbolTableCollection &symbol_table) {
      return VerifyWhileLikeOp(*this, symbol_table);
    }
  }];
}

def TFGraph_WhileOp : TFGraph_WhileLikeOp<"While"> {
  let summary = "A functional while loop operation.";
}
def TFGraph_StatelessWhileOp : TFGraph_WhileLikeOp<"StatelessWhile"> {
  let summary = "A functional stateless while loop operation.";
}
def TFGraph_StatefulWhileOp : TFGraph_WhileLikeOp<"StatefulWhile"> {
  let summary = "A functional stateful while loop operation.";
}

// A functional for loop operation.
def TFGraph_ForOp
    : TFGraph_ConcreteOp<"For",
                         [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "A functional for-loop operation.";

  let arguments = (ins
    // Op operands.
    TFGraph_OpaqueTensorOr<I32Tensor>:$start,
    TFGraph_OpaqueTensorOr<I32Tensor>:$limit,
    TFGraph_OpaqueTensorOr<I32Tensor>:$delta,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TFType_FuncAttr:$body,
    // Optional attributes. These attributes are present on import and are
    // required for export, but can be dropped if they contain redundant type
    // information.
    OptionalAttr<TypeArrayAttr>:$T
  );
  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$start, "Value":$limit,
                   "Value":$delta, "ValueRange":$args, "FuncAttr":$body), [{
      build($_builder, $_state, results, start, limit, delta, args, body,
            /*T=*/{});
    }]>
  ];

  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

  let builders = [
    OpBuilder<(ins "TypeRange":$results, "Value":$start, "Value":$limit,
                   "Value":$delta, "ValueRange":$args, "FuncAttr":$body), [{
      build($_builder, $_state, results, start, limit, delta, args, body,
            /*T=*/{});
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// TFGraph Region Op Definitions
//===----------------------------------------------------------------------===//

// TODO(jeffniu): Explore dropping all arguments/result attributes.
def TFGraph_RegionAttrs : AttrDef<TFGraphDialect, "Region"> {
  let description = [{
    Region attributes on control-flow operations represent argument, result,
    and general attributes on a region, analogous to function attributes. These
    arise when converting control-flow operations from functional form to region
    form; the control-flow functions' attributes are preserved as region
    attributes. These may also be added by transformations that create region
    control-flow operations.

    The region attributes are required to have as many argument attributes as
    there are data block arguments and result attributes as data results. If
    operands or results are added or removed from the owning operation, the
    region attributes, if present, must be updated.
  }];

  let mnemonic = "region_attrs";
  let summary = "Region attributes, argument and result attributes.";
  let assemblyFormat = "`<` $attrs $arg_attrs $res_attrs `>`";
  let parameters = (ins
    "DictionaryAttr":$attrs,
    "ArrayAttr":$arg_attrs,
    "ArrayAttr":$res_attrs
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "DictionaryAttr":$attrs,
                                        "ArrayAttr":$arg_attrs,
                                        "ArrayAttr":$res_attrs), [{
      return $_get(attrs.getContext(), attrs, arg_attrs, res_attrs);
    }]>
  ];
}

// Base class for TFGraph region ops. Region ops contain only single-block graph
// regions.
class TFGraph_RegionOp<string mnemonic, list<Trait> traits = []> 
    : TFG_IntrinsicOp<mnemonic, [
        HasOnlyGraphRegion, SingleBlock, RegionKindInterface,
        DeclareOpInterfaceMethods<OpAsmOpInterface, 
          ["getDefaultDialect", "getAsmResultNames"]>, TraitList<traits>]>;

def TFGraph_YieldOp : TFG_IntrinsicOp<"yield",
    [NoSideEffect, ReturnLike, Terminator, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let description = [{
    The `yield` operation is the terminator for control-flow regions. The
    operation takes data operands and control results. All data operands are
    inputs for successor regions. Typically, this means that the number of
    data operands matches the number of results of the parent operation.

    The semantics of function calls in TensorFlow mean that the region "returns"
    data operands as they become avaiable so that users can start executing
    immediately. All functional control-flow operations behave as multi-device
    function calls, which means that the control result of the parent operation
    depends on the control operands of the `yield`.
  }];

  let summary = "TF graph terminator for control-flow regions.";

  let arguments = (ins Variadic<TFGraph_Tensor>:$args,
                       Variadic<ControlType>:$ctls);
  let assemblyFormat = [{ (`(` $args^ `)`)? (`[` $ctls^ `]`)? attr-dict
                          (`:` type($args)^)? }];
}

// Base class for TFGraph if-like operations with regions.
class TFGraph_IfLikeRegionOp<string mnemonic> : TFGraph_RegionOp<
    mnemonic, [PreservedAttributesInterface,
               DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getRegionInvocationBounds"]>,
               DeclareOpInterfaceMethods<ControlArgumentInterface,
                                         ["getControlToken"]>]> {
  let description = [{
    An `If` operation in region form. `If` takes on data operand, which if true,
    executes the `then` region, otherwise the `else` region. The execution of
    either region is blocked on the operation's control operands.

    The regions have no block arguments; everything that escapes the scope is an
    implicit capture. Both regions return values of the same types which are
    compatible with the result types of the `If` operation.

    For each region, the operation contains optional "call" attributes and
    region attributes. There are never region argument attributes, and the
    number of result attributes must match the number of data results.
  }];

  let arguments = (ins
    // Op operands.
    I1Tensor:$cond,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$then_attrs,
    OptionalAttr<DictionaryAttr>:$else_attrs,
    OptionalAttr<TFGraph_RegionAttrs>:$then_region_attrs,
    OptionalAttr<TFGraph_RegionAttrs>:$else_region_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region SizedRegion<1>:$then_region,
                        SizedRegion<1>:$else_region);

  let extraClassDeclaration = [{
    Block &then_block() { return then_region().front(); }
    Block &else_block() { return else_region().front(); }
    YieldOp then_yield();
    YieldOp else_yield();

    bool areTypesCompatible(Type lhs, Type rhs) {
      return tf_type::HasCompatibleElementTypes(lhs, rhs);
    }

    RegionAttr getPreservedAttrs(unsigned index) {
      if (index == 0) return then_region_attrsAttr();
      return else_region_attrsAttr();
    }
    void setPreservedAttrs(unsigned index, RegionAttr attrs) {
      if (index == 0) then_region_attrsAttr(attrs);
      else else_region_attrsAttr(attrs);
    }
  }];

  let extraClassDefinition = [{
    StringRef $cppClass::getDefaultDialect() { return "tfg"; }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      GenericGetAsmResultNames(*this, setNameFn);
    }

    YieldOp $cppClass::then_yield() {
      return cast<YieldOp>(then_block().getTerminator());
    }
    YieldOp $cppClass::else_yield() {
      return cast<YieldOp>(else_block().getTerminator());
    }

    void $cppClass::getSuccessorRegions(
        Optional<unsigned> index, ArrayRef<Attribute> operands,
        SmallVectorImpl<RegionSuccessor> &regions) {
      GetIfLikeRegionOpSuccessorRegions(*this, index, operands, regions);
    }
    void $cppClass::getRegionInvocationBounds(
        ArrayRef<Attribute> operands,
        SmallVectorImpl<InvocationBounds> &invocationBounds) {
      invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
    }

    LogicalResult $cppClass::verify() { return VerifyIfLikeRegionOp(*this); }

    BlockArgument $cppClass::getDataValueOf(BlockArgument ctl) {
      return GetLoopRegionDataOf(ctl);
    }
    BlockArgument $cppClass::getControlTokenOf(BlockArgument data) {
      return GetLoopRegionControlOf(data);
    }
    BlockArgument $cppClass::getDataValue(Region &region, unsigned idx) {
      return GetLoopRegionDataArgs(region)[idx];
    }
    BlockArgument $cppClass::getControlToken(Region &region, unsigned idx) {
      return GetLoopRegionControlTokens(region)[idx];
    }
  }];

  let assemblyFormat = [{
    $cond (` ` `[` $ctls^ `]`)? `then`
      $then_region
    `else`
      $else_region
    attr-dict `:` functional-type($cond, $outs)
  }];

  let hasVerifier = 1;
}

def TFGraph_IfRegionOp : TFGraph_IfLikeRegionOp<"IfRegion"> {
  let summary = "An if-then-else op with graph regions.";
}
def TFGraph_StatelessIfRegionOp : TFGraph_IfLikeRegionOp<"StatelessIfRegion"> {
  let summary = "A stateless if-then-else op with graph regions.";
}
def TFGraph_StatefulIfRegionOp : TFGraph_IfLikeRegionOp<"StatefulIfRegion"> {
  let summary = "A stateful if-then-else op with graph regions.";
}

// Base calss for TFGraph case-like operations with regions.
class TFGraph_CaseLikeRegionOp<string mnemonic> : TFGraph_RegionOp<
    mnemonic, [PreservedAttributesInterface,
               DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getRegionInvocationBounds"]>,
               DeclareOpInterfaceMethods<ControlArgumentInterface,
                                         ["getControlToken"]>]> {
  let description = [{
    A `Case` operation in region form. `Case` takes one data operand, an index,
    and executes the region corresponding to that index. The execution of any
    of its regions is blocked on the operation's control operands.

    The regions have no block arguments; everything that escapes the scope is an
    implicit capture. All regions return values of the same types which are
    compatible with the result types of the `Case` operation.

    For each region, the operation contains optional "call" attributes and
    region attributes in arrays. There are never region argument attributes, and
    the number of result attributes must match the number of data results.
  }];

  let arguments = (ins
    // Op operands.
    I32Tensor:$branch_index,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<TFGraph_DictionaryArrayAttr>:$branch_attrs,
    OptionalAttr<
        TypedArrayAttrBase<TFGraph_RegionAttrs,
                           "case region preserved attributes">
    >:$region_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let extraClassDeclaration = [{
    Block &branch_block(unsigned idx) { return branches()[idx].front(); }
    YieldOp branch_yield(unsigned idx);

    bool areTypesCompatible(Type lhs, Type rhs) {
      return tf_type::HasCompatibleElementTypes(lhs, rhs);
    }

    RegionAttr getPreservedAttrs(unsigned index) {
      if (auto attrs = region_attrsAttr())
        return attrs[index].cast<RegionAttr>();
      return {};
    }
    void setPreservedAttrs(unsigned index, RegionAttr attrs) {
      SmallVector<Attribute> array = llvm::to_vector(region_attrsAttr());
      array[index] = attrs;
      region_attrsAttr(ArrayAttr::get(getContext(), array));
    }
  }];

  let extraClassDefinition = [{
    StringRef $cppClass::getDefaultDialect() { return "tfg"; }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      GenericGetAsmResultNames(*this, setNameFn);
    }

    YieldOp $cppClass::branch_yield(unsigned idx) {
      return cast<YieldOp>(branch_block(idx).getTerminator());
    }

    void $cppClass::getSuccessorRegions(
        Optional<unsigned> index, ArrayRef<Attribute> operands,
        SmallVectorImpl<RegionSuccessor> &regions) {
      GetCaseLikeRegionOpSuccessorRegions(*this, index, operands, regions);
    }
    void $cppClass::getRegionInvocationBounds(
        ArrayRef<Attribute> operands,
        SmallVectorImpl<InvocationBounds> &invocationBounds) {
      invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
    }

    LogicalResult $cppClass::verify() { return VerifyCaseLikeRegionOp(*this); }

    BlockArgument $cppClass::getDataValueOf(BlockArgument ctl) {
      return GetLoopRegionDataOf(ctl);
    }
    BlockArgument $cppClass::getControlTokenOf(BlockArgument data) {
      return GetLoopRegionControlOf(data);
    }
    BlockArgument $cppClass::getDataValue(Region &region, unsigned idx) {
      return GetLoopRegionDataArgs(region)[idx];
    }
    BlockArgument $cppClass::getControlToken(Region &region, unsigned idx) {
      return GetLoopRegionControlTokens(region)[idx];
    }
  }];

  let assemblyFormat = [{
    $branch_index (` ` `[` $ctls^ `]`)?
      $branches
    attr-dict `:` functional-type($branch_index, $outs)
  }];

  let hasVerifier = 1;
}

def TFGraph_CaseRegionOp : TFGraph_CaseLikeRegionOp<"CaseRegion"> {
  let summary = "A case op with graph regions.";
}
def TFGraph_StatelessCaseRegionOp :
    TFGraph_CaseLikeRegionOp<"StatelessCaseRegion"> {
  let summary = "A stateless case op with graph regions.";
}
def TFGraph_StatefulCaseRegionOp :
    TFGraph_CaseLikeRegionOp<"StatefulCaseRegion"> {
  let summary = "A stateful case op with graph regions.";
}

// Special terminator op for while op condition regions.
def TFGraph_ConditionOp : TFG_IntrinsicOp<"condition",
    [NoSideEffect, ReturnLike, Terminator, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let description = [{
    The `condition` operation is a special terminator for the condition region
    of while loops. Its only purpose is to support reasoning about region
    control-flow by forwarding the arguments to the condition region to the body
    region or back to the parent operation's results.

    The `condition` operation accepts one `cond` data input, which if true, will
    cause the execution to branch to the loop body, and if false, will cause the
    execution to return to the parent operation.

    There must be the same number of arguments `args` as there are operands and
    results in the parent `While` loop.
  }];

  let summary = "TF graph terminator op for while-loop condition regions.";

  let arguments = (ins
    I1Tensor:$cond,
    Variadic<TFGraph_Tensor>:$args,
    Variadic<ControlType>:$ctls
  );
  let assemblyFormat = [{ $cond `:` type($cond) (` ` `(` $args^ `)`)?
                          (`[` $ctls^ `]`)? attr-dict (`:` type($args)^)? }];
}

// TF graph while loop op with regions.
class TFGraph_WhileLikeRegionOp<string mnemonic> : TFGraph_RegionOp<
    mnemonic, [AttrSizedOperandSegments, PreservedAttributesInterface,
               DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getSuccessorEntryOperands"]>,
               DeclareOpInterfaceMethods<ControlArgumentInterface,
                                         ["getControlToken"]>]> {
  let description = [{
    A `While` operation in region form. The operation takes N operands and
    produces N results of the same types. Each region has N data block arguments
    and N corresponding control tokens.

    The entry into the condition region is blocked on the operation's control
    operands. The operation always executes the condition region at least once
    with the arguments as the operation's data operands. If the condition region
    returns `true`, the body region is executed with the arguments provided to
    the condition region. The body region will branch back to the condition
    region with N results. If `false`, the condition region will branch back to
    the parent `While` operation with the argument values as results.

    For each region, the operation contains optional "call" attributes and
    region attributes. The number of argument attributes must match the number
    of data operands and the number of result attributes must match the number
    of data results.

    Values can be propagated through the loop bodies as iteration variables,
    but the regions allow implicitly captured values as well.
  }];

  let arguments = (ins
    // Op operands.
    Variadic<TFGraph_Tensor>:$init,
    Variadic<ControlType>:$ctls,
    I64Attr:$parallel_iterations,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$cond_attrs,
    OptionalAttr<DictionaryAttr>:$body_attrs,
    OptionalAttr<TFGraph_RegionAttrs>:$cond_region_attrs,
    OptionalAttr<TFGraph_RegionAttrs>:$body_region_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region SizedRegion<1>:$cond_region,
                        SizedRegion<1>:$body_region);

  let extraClassDeclaration = [{
    Block &cond_block() { return cond_region().front(); }
    Block &body_block() { return body_region().front(); }
    ConditionOp cond_condition();
    YieldOp body_yield();

    bool areTypesCompatible(Type lhs, Type rhs) {
      return tf_type::HasCompatibleElementTypes(lhs, rhs);
    }

    RegionAttr getPreservedAttrs(unsigned index) {
      if (index == 0) return cond_region_attrsAttr();
      return body_region_attrsAttr();
    }
    void setPreservedAttrs(unsigned index, RegionAttr attrs) {
      if (index == 0) cond_region_attrsAttr(attrs);
      else body_region_attrsAttr(attrs);
    }
  }];

  let extraClassDefinition = [{
    StringRef $cppClass::getDefaultDialect() { return "tfg"; }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      GenericGetAsmResultNames(*this, setNameFn);
    }

    ConditionOp $cppClass::cond_condition() {
      return cast<ConditionOp>(cond_block().getTerminator());
    }
    YieldOp $cppClass::body_yield() {
      return cast<YieldOp>(body_block().getTerminator());
    }

    OperandRange $cppClass::getSuccessorEntryOperands(unsigned index) {
      return init();
    }
    void $cppClass::getSuccessorRegions(
        Optional<unsigned> index, ArrayRef<Attribute> operands,
        SmallVectorImpl<RegionSuccessor> &regions) {
      return GetWhileLikeRegionOpSuccessorRegions(*this, index, operands,
                                                  regions);
    }

    LogicalResult $cppClass::verify() { return VerifyWhileLikeRegionOp(*this); }

    BlockArgument $cppClass::getDataValueOf(BlockArgument ctl) {
      return GetLoopRegionDataOf(ctl);
    }
    BlockArgument $cppClass::getControlTokenOf(BlockArgument data) {
      return GetLoopRegionControlOf(data);
    }
    BlockArgument $cppClass::getDataValue(Region &region, unsigned idx) {
      return GetLoopRegionDataArgs(region)[idx];
    }
    BlockArgument $cppClass::getControlToken(Region &region, unsigned idx) {
      return GetLoopRegionControlTokens(region)[idx];
    }
  }];

  let assemblyFormat = [{
    (`(` $init^ `)`)? (`[` $ctls^ `]`)?
      $cond_region
    `do`
      $body_region
    attr-dict (`:` functional-type($init, $outs)^)?
  }];

  let hasVerifier = 1;
}

def TFGraph_WhileRegionOp : TFGraph_WhileLikeRegionOp<"WhileRegion"> {
  let summary = "A region-based while loop operation.";
}
def TFGraph_StatelessWhileRegionOp : TFGraph_WhileLikeRegionOp<"StatelessWhileRegion"> {
  let summary = "A stateless region-based while loop operation.";
}
def TFGraph_StatefulWhileRegionOp : TFGraph_WhileLikeRegionOp<"StatefulWhileRegion"> {
  let summary = "A stateful region-based while loop operation.";
}

// TF graph for loop op with region.
def TFGraph_ForRegionOp : TFGraph_RegionOp<
    "ForRegion", [AttrSizedOperandSegments, PreservedAttributesInterface,
                  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                            ["getSuccessorEntryOperands"]>,
                  DeclareOpInterfaceMethods<ControlArgumentInterface,
                                            ["getControlToken"]>]> {
  let description = [{
    A `For` operation in region form. The operation takes three scalar `i32`
    operands: a lower bound, an upper bound, and a step. As well, it takes N
    operands and produces N results of the same types. The loop body region has
    a scalar `i32` block argument and its corresponding control token and N data
    block arguments of the same types as the N data operands and N results. The
    scalar `i32` block argument is the current loop index.

    The operation is blocked on its control operands. The execution enters the
    loop body, which must return N data results. The values are either fed into
    the next iteration of the body or taken as the operation's results.

    For the body region, the operation contains optional "call" attributes and
    region attributes. The number of argument attributes must match the number
    of data operands and the number of result attributes must match the number
    of data results.

    Values can be propagated through the loop bodies as iteration variables,
    but the regions allow implicitly captured values as well.
  }];

  let arguments = (ins
    // Op Operands.
    I32Tensor:$start,
    I32Tensor:$limit,
    I32Tensor:$delta,
    Variadic<TFGraph_Tensor>:$init,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$body_attrs,
    OptionalAttr<TFGraph_RegionAttrs>:$region_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region SizedRegion<1>:$body_region);

  let extraClassDeclaration = [{
    Block &body_block() { return body_region().front(); }
    YieldOp body_yield();

    bool areTypesCompatible(Type lhs, Type rhs) {
      return tf_type::HasCompatibleElementTypes(lhs, rhs);
    }

    RegionAttr getPreservedAttrs(unsigned index) {
      return region_attrsAttr();
    }
    void setPreservedAttrs(unsigned index, RegionAttr attrs) {
      region_attrsAttr(attrs);
    }
  }];

  let extraClassDefinition = [{
    StringRef $cppClass::getDefaultDialect() { return "tfg"; }
    void $cppClass::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
      GenericGetAsmResultNames(*this, setNameFn);
    }

    YieldOp $cppClass::body_yield() {
      return cast<YieldOp>(body_block().getTerminator());
    }
  }];

  let assemblyFormat = [{
    (`(` $init^ `)`)? (`[` $ctls^ `]`)?
    `from` $start `to` $limit `by` $delta
      $body_region
    attr-dict `:` `(` type($start) `,` type($limit) `,` type($delta)
    (`,` type($init)^)? `)` (`->` `(` type($outs)^ `)`)?
  }];

  let hasVerifier = 1;
}

#endif // TFG_OPS
