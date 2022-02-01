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

#ifdef TFG_OPS
#else
#define TFG_OPS

include "tensorflow/core/ir/dialect.td"
include "tensorflow/core/ir/interfaces.td"
include "tensorflow/core/ir/types/attributes.td"
include "tensorflow/core/ir/types/types.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
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

  // Make the assembly pretty!
  let printer = [{ PrintGraphOp(p, *this); }];
  let parser = [{ return ParseGraphOp(parser, result); }];
  let verifier = [{ return VerifyGraph(*this); }];
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
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$generic);

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

  let parser = [{ return ParseGraphFunc(parser, result); }];
  let printer = [{ return PrintGraphFunc(*this, p); }];
  let verifier = [{ return VerifyGraphFunc(*this); }];
  let hasCanonicalizeMethod = 1;
}


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

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  // Use the dialect printer/parser like other unregistered ops in this dialect.
  let parser = [{ return ParseCustomTfOp(parser, result); }];
  let printer = [{ PrintCustomTfOp(*this, p); }];
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

// A cast operation.
def TFGraph_CastOp : TFGraph_Op<"Cast", [NoSideEffect]> {
  let summary = "Cast between two compatible tensor types.";

  let arguments = (ins
    // Op operands.
    TFGraph_Tensor:$x,
    Variadic<ControlType>:$ctls,
    // Required attributes.
    TFGraph_TypeOrPlaceholder:$SrcT,
    TFGraph_TypeOrPlaceholder:$DstT,
    DefaultValuedAttr<BoolAttr, "false">:$Truncate
  );
  let results = (outs AnyType:$y, ControlType:$ctl);

  let hasFolder = 1;
}

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

#endif // TFG_OPS
