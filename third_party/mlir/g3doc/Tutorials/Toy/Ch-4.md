# Chapter 4: Enabling Generic Transformation with Interfaces

[TOC]

## Background: Grappling with an Extensible IR

Through dialects, MLIR allows for the representation of many different levels of
abstraction; the Toy dialect that we have previously defined is one such
example. Though these different dialects may represent different abstractions,
there is often a set of common transformations and analyses that we would like
to perform. The problem that arises is that naively implementing each
transformation for each dialect leads to large amounts of code duplication, as
the internal algorithms are generally very similar, if not the same. We would
like to provide the ability for transformations to opaquely hook into dialects
like Toy to get the information they need.

MLIR provides a set of always available-hooks for certain core transformations,
as seen in the [previous chapter](Ch-3.md), where we registered some
canonicalizations via a hook on our operations (`getCanonicalizationPatterns`).
However, these types of hooks don't really scale well. Therefore, a more generic
solution was designed, in the form of [interfaces](../../Interfaces.md), to make
the MLIR infrastructure as extensible as the representation. Interfaces provide
a generic mechanism for dialects and operations to provide information to a
transformation or analysis.

## Shape Inference: Preparing for Code Generation

Our Toy IR currently operates on generic tensors, meaning that we don't know the
shape of tensors other than during the initialization of constants. This
complicates optimizations, as well as code generation. Fortunately, we can
simply propagate the shapes through the computation until they are all known.
The issue is how to handle calls to user-defined generic functions: every call
site could deduce different shapes. One possibility would be to perform symbolic
inference based on the argument types, but this would be hard to generalize if
we were to introduce more control flow in the language. Another approach would
be function specialization, where every call site with new argument shapes
duplicates the called function and specializes it. The approach we take for Toy
is to inline all of the function calls, then perform intraprocedural shape
propagation.

### Inlining

Here we could write an inlining algorithm specifically designed for the Toy
dialect, but that can become quite complicated depending on the level of
complexity that we want. Disregarding cost modeling, the pure structural
transformation is already complex to implement from scratch. Thankfully, MLIR
provides a generic inliner algorithm that dialects can plug into. All we need to
do in Toy is to provide the [interfaces](../../Interfaces.md) for the inliner to
hook into.

The first thing we need to do is to define the constraints on inlining
operations in the Toy dialect. This information is provided through a
[dialect interface](../../Interfaces.md#dialect-interfaces). This is essentially
a class containing a set of virtual hooks for which a dialect may provide a
specialization. In this case, the interface is `DialectInlinerInterface`.

```c++
/// This class defines the interface for handling inlining with Toy operations.
/// We simplify inherit from the base interface class and provide a
/// specialization of the necessary methods.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region. For Toy this hook can simply return true, as all Toy
  /// operations are inlinable.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  /// This hook is called when a terminator operation has been inlined. The only
  /// terminator that we have in the Toy dialect is the return
  /// operation(toy.return). We handle the return by replacing the values
  /// previously returned by the call operation with the operands of the
  /// return.
  void handleTerminator(Operation *op,
                        ArrayRef<ValuePtr> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()]->replaceAllUsesWith(it.value());
  }
};
```

We then register our dialect interface directly on the Toy dialect, similarly to
how we did for operations.

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
  addInterfaces<ToyInlinerInterface>();
}
```

Next, we need to provide a way for the inliner to know that `toy.generic_call`
represents a call to a function. MLIR provides an
[operation interface](../../Interfaces.md#operation-interfaces) that can be used
to mark an operation as being "call-like". Unlike dialect interfaces, operation
interfaces provide a more refined granularity of information that is specific
and core to a single operation. The interface that we will be adding here is the
`CallOpInterface`.

To add this interface we just need to include the definition into our operation
specification file (`Ops.td`):

```tablegen
#ifdef MLIR_CALLINTERFACES
#else
include "mlir/Analysis/CallInterfaces.td"
#endif // MLIR_CALLINTERFACES
```

and add it to the traits list of `GenericCallOp`:

```tablegen
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

In the above we also use the `DeclareOpInterfaceMethods` directive to
auto-declare all of the interface methods in the class declaration of
GenericCallOp. This means that we just need to provide a definition:

```c++
/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }
```

Now that the inliner has been informed about the Toy dialect, we can add the
inliner pass to the pass manager for Toy:

```c++
  pm.addPass(mlir::createInlinerPass());
```

Now let's look at a working example:

```mlir
func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64>
  %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  "toy.return"(%2) : (tensor<*xf64>) -> ()
}
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64>
  %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64>
  %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  "toy.print"(%5) : (tensor<*xf64>) -> ()
  "toy.return"() : () -> ()
}
```

We have two calls to multiple_transpose that we would like to inline into main,
but if we look at the output nothing has changed. We are missing one last subtle
piece: there is a hidden type conversion on the edge of the call. If we look at
the above, the operands to the generic_call are of type `tensor<2x3xf64>`, while
the inputs to the function expect `tensor<*xf64>`. To resolve this difference,
the inliner expects an explicit cast operation to be inserted. For this, we need
to add a new operation to the Toy dialect, `ToyCastOp`(toy.cast), to represent
casts between two different shapes.

```tablegen
def CastOp : Toy_Op<"cast", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  // Set the folder bit so that we can fold redundant cast operations.
  let hasFolder = 1;
}
```

We can then override the necessary hook on the ToyInlinerInterface to insert
this for us when necessary:

```c++
struct ToyInlinerInterface : public DialectInlinerInterface {
  ...

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, ValuePtr input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

If we run the working example through the pipeline again, we get the expected:

```mlir
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %2 = "toy.cast"(%1) : (tensor<2x3xf64>) -> tensor<*xf64>
  %3 = "toy.cast"(%0) : (tensor<2x3xf64>) -> tensor<*xf64>
  %4 = "toy.transpose"(%2) : (tensor<*xf64>) -> tensor<*xf64>
  %5 = "toy.transpose"(%3) : (tensor<*xf64>) -> tensor<*xf64>
  %6 = "toy.mul"(%4, %5) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  "toy.print"(%6) : (tensor<*xf64>) -> ()
  "toy.return"() : () -> ()
}
```

NOTE: The generic inliner will also perform simplifications, so the output may
be a bit cleaner than expected.

### Intraprocedural Shape Inference

Now that we have inlined all of the functions, we are left with a main function
containing a mix of static and dynamically shaped operations. We can now write a
simple shape inference pass to propagate shapes intraprocedurally (within a
single function). We could write this as a pass that directly encodes the
constraints of the operations within the Toy dialect, but this seems like a good
candidate for a transformation that could be written generically. As a good rule
of thumb, it is best to express a transformation as generically as possible,
such that it can be extended to other dialects in the future. There is no
telling how many other dialects may have similar needs or encounter the same
problems.

For shape inference, if we break down the problem to its core, we really just
want operations to tell us the expected outputs given a set of statically known
inputs. (We can definitely get more complex than that, but for our needs we can
keep it simple.) Given that this property is core to a specific operation, we
can define an operation interface that can be specified on operations that need
to have their result shapes inferred.

Similarly to operations, we can also
[define operation interfaces](../../OpDefinitions.md#operation-interfaces) using
the operation definition specification (ODS) framework.

The interface is defined by inheriting from `OpInterface`, which takes the name
to be given to the generated C++ interface class as a template argument. For our
purposes, we will name the generated class a simpler `ShapeInference`. We also
provide a description for the interface.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];
}
```

Next, we define the interface methods that the operations will need to provide.
An interface method is comprised of: a description; a C++ return type in string
form; a method name in string form; and a few optional components, depending on
the need. See the
[ODS documentation](../../OpDefinitions.md#operation-interfaces) for more
information.

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

Now that the interface is defined, we can add it to the necessary Toy operations
in a similar way to how we added the `CallOpInterface` to the GenericCallOp:

```
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

Each of these operations will then need to provide a definition for the
`inferShapes()` method. As an example, for the mul op, the result shape is
inferred as the shape of the inputs.

```c++
/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.
void MulOp::inferShapes() { getResult()->setType(getOperand(0)->getType()); }
```

At this point, each of the necessary Toy operations provide a mechanism by which
to infer their output shapes. The ShapeInferencePass is a FunctionPass: it will
runs on each Function in isolation. MLIR also supports general
[OperationPasses](../../WritingAPass.md#operation-pass) that run on any isolated
operation (i.e. other function-like operations), but here our module only
contains functions, so there is no need to generalize to all operations.

Implementing such a pass is done by creating a class inheriting from
`mlir::FunctionPass` and overriding the `runOnFunction()` method:

```c++
class ShapeInferencePass : public mlir::FunctionPass<ShapeInferencePass> {
  void runOnFunction() override {
    FuncOp function = getFunction();
    ...
  }
};
```

The algorithm operates as follows:

1.  Build a worklist containing all the operations that return a dynamically
    shaped tensor: these are the operations that need shape inference.
2.  Iterate on the worklist:
    -   find an operation to process: the next ready operation in the worklist
        has all of its arguments non-generic,
    -   if no operation is found, break out of the loop,
    -   remove the operation from the worklist,
    -   infer the shape of its output from the argument types.
3.  If the worklist is empty, the algorithm succeeded.

When processing an operation, we query if it registered the `ShapeInference`
interface.

```c++
  // Ask the operation to infer its output shapes.
  LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

  /// We check if an operation has a particular interface by casting.
  if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
    shapeOp.inferShapes();
  } else {
    op->emitError("unable to infer shape of operation without shape "
                  "inference interface");
    return signalPassFailure();
  }
```

We can then add our pass to the pass manager:

```c++
  pm.addPass(mlir::createShapeInferencePass());
```

If we rerun our original example, we now get the following:

```mlir
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "toy.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  %2 = "toy.mul"(%1, %1) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  "toy.print"(%2) : (tensor<3x2xf64>) -> ()
  "toy.return"() : () -> ()
}
```

You can build `toyc-ch4` and try yourself: `toyc-ch4
test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt`.

In the [next chapter](Ch-5.md), we will start the process of code generation by
targeting a lower level dialect for optimizing some of the more compute-heavy
Toy operations.
