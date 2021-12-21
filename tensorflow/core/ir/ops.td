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
include "tensorflow/core/ir/types/attributes.td"

include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// TFGraph op definitions
//===----------------------------------------------------------------------===//

def TFGraph_GraphOp : TFGraph_Op<"graph",
    [HasOnlyGraphRegion, SingleBlock, IsolatedFromAbove,
     OpAsmOpInterface, OpTraitList<GraphRegionNoTerminator.traits>]> {
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
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verifyGraph(*this); }];
}


def TFGraph_GraphFuncOp : TFGraph_Op<"func", [
    RegionKindInterface, HasOnlyGraphRegion, SingleBlock, CallableOpInterface,
    FunctionLike, IsolatedFromAbove, Symbol, OpAsmOpInterface]> {
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
      return isExternal() ? nullptr : &FunctionLike::getBody();
    }

    // Returns the results types that the callable region produces when
    // executed.
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<GraphFuncOp>;

    // Returns the number of arguments. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getNumInputs(); }

    // Returns the number of results. Hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getNumResults(); }

    // Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    // attribute is present and checks if it holds a function type. Ensures
    // getType, getNumFuncArguments, and getNumFuncResults can be called safely
    LogicalResult verifyType();

    // Hook for OpTrait::FunctionLike, called after verifying the function
    // type and the presence of the (potentially empty) function body.
    LogicalResult verifyBody();
  }];

  let parser = [{ return parseGraphFunc(parser, result); }];
  let printer = [{ return printGraphFunc(*this, p); }];
  let verifier = [{ return verifyGraphFunc(*this); }];
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

  // Use the dialect printer/parser like other unregistered op in this dialect.
  let parser = [{ return parseCustomTfOp(parser, result); }];
  let printer = [{ printCustomTfOp(*this, p); }];
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

#endif // TFG_OPS
