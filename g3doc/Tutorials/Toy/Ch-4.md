# Chapter 4: Using Interfaces

[Interfaces](../../Interfaces.md) provide a generic method for applying
transformations across dialects. We first describe how to leverage an existing
MLIR interface, and then walk through writing your own interface.

## Function Inlining

In order to apply function inlining in the Toy dialect, we override the
DialectInlinerInterface in Toy, enable inlining and add special handling for the
return operation:

```Toy(.cpp)
//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy
/// operations.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All operations within toy can be inlined.
  bool isLegalToInline(Operation *, Region *,
                       BlockAndValueMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(toy.return) by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value *> valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()]->replaceAllUsesWith(it.value());
  }
};
```

Next, we call into the interface by adding an inliner pass to the pass manager
for toy:

```Toy(.cpp)
    pm.addPass(mlir::createInlinerPass());
```

** Insert example here **

## Shape Inference

The Toy language allows for implicit shapes and hence requires shape inference.
We implement shape inference as a generic
[Operation Interface](../../Interfaces.md#operation-interfaces).

1.  We first create the ShapeInferenceOpInterface by specializing the
    OpInterface class using [ODS](../../OpDefinitions.md#operation-interfaces).
    This class defines interface methods that Toy operations must override for
    shape inference.

```Toy(.cpp)
def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let methods = [
    InterfaceMethod<
      "bool", "returnsGenericArray", (ins), [{
      if (getNumResults() == 1) {
        auto arrayTy = op.getResult()->getType().cast<RankedTensorType>();
        return arrayTy.getShape().empty();
      }
      return false;
    }]>,
    InterfaceMethod<"void", "inferShapes", (ins), [{}]>
  ];
}
```

1.  Next, we override the inferShapes() method within Toy operations. As an
    example, for the transpose op, the result shape is inferred by swapping the
    dimensions of the input tensor.

```Toy(.cpp)
  void inferShapes() {
    SmallVector<int64_t, 2> dims;
    auto arrayTy = getOperand()->getType().cast<RankedTensorType>();
    dims.insert(dims.end(), arrayTy.getShape().begin(),
                arrayTy.getShape().end());
    if (dims.size() == 2)
      std::swap(dims[0], dims[1]);
    getResult()->setType(RankedTensorType::get(dims, arrayTy.getElementType()));
    return;
  }
```

1.  We then create a generic ShapeInference Function pass that uses operation
    casting to access the inferShapes() method. This is an intraprocedural shape
    inference pass that executes after function inlining and iterates over
    operations in a worklist calling inferShapes for each operation with unknown
    result shapes.

2.  Finally, we call into shape inference pass by adding it to the pass manager
    for toy:

```Toy(.cpp)
    pm.addPass(mlir::createShapeInferencePass());
```

** Insert example here **
