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

def TFGraph_GraphOp : TFGraph_Op<"graph",
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


def TFGraph_GraphFuncOp : TFGraph_Op<"func", [
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
    TypeAttr:$type,
    ArrayAttr:$arg_attrs,
    ArrayAttr:$res_attrs,
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
      $_state.addAttribute(GraphFuncOp::typeAttrName($_state.name),
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
    FunctionType getType() {
      return getTypeAttr().getValue().cast<FunctionType>();
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
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

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

def TFGraph_ReturnOp : TFGraph_Op<"return",
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


def TFGraph_GetResultOp : TFGraph_Op<"get_result",
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

// Base class for TFGraph concrete ops.
class TFGraph_ConcreteOp<string mnemonic, list<Trait> traits = []> :
    TFGraph_Op<mnemonic, traits>;

// Base class for TFGraph if-like operations.
class TFGraph_IfLikeOp<string mnemonic>
    : TFGraph_ConcreteOp<mnemonic,
                         [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    // Op operands.
    I1Tensor:$cond,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TypeAttr:$Tcond,
    TypeArrayAttr:$Tin,
    TypeArrayAttr:$Tout,
    TFType_FuncAttr:$then_branch,
    TFType_FuncAttr:$else_branch,
    TFGraph_ShapesAttr:$output_shapes
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

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
    I32Tensor:$branch_index,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TypeArrayAttr:$Tin,
    TypeArrayAttr:$Tout,
    TypedArrayAttrBase<TFType_FuncAttr, "an array of functions">:$branches,
    TFGraph_ShapesAttr:$output_shapes
  );

  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

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
    TypeArrayAttr:$T,
    TFType_FuncAttr:$cond,
    TFType_FuncAttr:$body,
    TFGraph_ShapesAttr:$output_shapes,
    I64Attr:$parallel_iterations
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);

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
    I32Tensor:$start,
    I32Tensor:$limit,
    I32Tensor:$delta,
    Variadic<TFGraph_TensorOrControlType>:$args,
    // Required attributes.
    TypeArrayAttr:$T,
    TFType_FuncAttr:$body
  );

  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
}

//===----------------------------------------------------------------------===//
// TFGraph Region Op Definitions
//===----------------------------------------------------------------------===//

// TODO(jeffniu): Add detailed op descriptions.

// Base class for TFGraph region ops.
class TFGraph_RegionOp<string mnemonic, list<Trait> traits = []> :
    TFGraph_Op<mnemonic, [HasOnlyGraphRegion, SingleBlock,
                          DeclareOpInterfaceMethods<OpAsmOpInterface,
                              ["getDefaultDialect", "getAsmResultNames"]>,
                          TraitList<traits>]>;

// Yield terminator op for control-flow ops with regions.
def TFGraph_YieldOp : TFGraph_Op<"yield",
    [NoSideEffect, ReturnLike, Terminator, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "TF graph terminator for control-flow regions.";

  let arguments = (ins Variadic<TFGraph_Tensor>:$args,
                       Variadic<ControlType>:$ctls);
  let assemblyFormat = "(`(` $args^ `)`)? (`[` $ctls^ `]`)? attr-dict (`:` type($args)^)?";
}

// Base class for TFGraph if-like operations with regions.
class TFGraph_IfLikeRegionOp<string mnemonic> : TFGraph_RegionOp<
    mnemonic, [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getRegionInvocationBounds"]>]> {
  let arguments = (ins
    // Op operands.
    I1Tensor:$cond,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$then_attrs,
    OptionalAttr<DictionaryAttr>:$else_attrs
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
      return getElementTypeOrSelf(lhs) == getElementTypeOrSelf(rhs);
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
    mnemonic, [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getRegionInvocationBounds"]>]> {
  let arguments = (ins
    // Op operands.
    I32Tensor:$branch_index,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<TFGraph_DictionaryArrayAttr>:$branch_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);

  let extraClassDeclaration = [{
    Block &branch_block(unsigned idx) { return branches()[idx].front(); }
    YieldOp branch_yield(unsigned idx);

    bool areTypesCompatible(Type lhs, Type rhs) {
      return getElementTypeOrSelf(lhs) == getElementTypeOrSelf(rhs);
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
def TFGraph_ConditionOp : TFGraph_Op<"condition",
    [NoSideEffect, ReturnLike, Terminator, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
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
    mnemonic, [AttrSizedOperandSegments,
               DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                         ["getSuccessorEntryOperands"]>,
               DeclareOpInterfaceMethods<ControlArgumentInterface,
                                         ["getControlToken"]>]> {
  let arguments = (ins
    // Op operands.
    Variadic<TFGraph_Tensor>:$init,
    Variadic<ControlType>:$ctls,
    I64Attr:$parallel_iterations,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$cond_attrs,
    OptionalAttr<DictionaryAttr>:$body_attrs
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
      return getElementTypeOrSelf(lhs) == getElementTypeOrSelf(rhs);
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
    "ForRegion", [AttrSizedOperandSegments,
                  DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                            ["getSuccessorEntryOperands"]>,
                  DeclareOpInterfaceMethods<ControlArgumentInterface,
                                            ["getControlToken"]>]> {
  let arguments = (ins
    // Op Operands.
    I32Tensor:$start,
    I32Tensor:$limit,
    I32Tensor:$delta,
    Variadic<TFGraph_Tensor>:$init,
    Variadic<ControlType>:$ctls,
    // Optional attributes.
    OptionalAttr<DictionaryAttr>:$body_attrs
  );
  let results = (outs Variadic<TFGraph_Tensor>:$outs, ControlType:$ctl);
  let regions = (region SizedRegion<1>:$body_region);

  let extraClassDeclaration = [{
    Block &body_block() { return body_region().front(); }
    YieldOp body_yield();

    bool areTypesCompatible(Type lhs, Type rhs) {
      return getElementTypeOrSelf(lhs) == getElementTypeOrSelf(rhs);
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
