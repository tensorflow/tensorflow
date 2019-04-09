# Linalg Part 2: Compute Operations

We now describe the main compute operations `linalg.dot`, `linalg.matvec` and
`linalg.matmul`. These operations are a subset of a more general tensor
contraction
[class](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg2/include/linalg2/TensorOps.h)
of operations. In this tutorial, we define a tensor contraction as a generic
operation which:

1.  Reads a `getNumInputs()` number of input ssa-values of ViewType.
2.  Writes into a `getNumOutputs()` number of input ssa-values of ViewType.
3.  Can be written in scalar loop form as a perfect loop nest with
    `getNumParallelDims()` outermost loops with parallel semantics and
    `getNumReductionDims()` innermost dimensions with reduction semantics.
4.  Has a scalar form that is specific to each particular specialization.

## Operation Definition

In this section we do not discuss the specific properties of tensor contractions
but only define the `linalg.dot`, `linalg.matvec` and `linalg.matmul` operations
as opaque operations with side-effects (reads and writes into input and output
views).

These operations take input and output views of the proper rank as operands. For
the purpose of illustration, assume all the elemental types are fixed to `f32`.
The invariants checked by the op-specific
[verify](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg2/lib/TensorOps.cpp)
functions are:

1.  `linalg.dot` reads two one-dimensional `view<?xf32>` and writes a
    zero-dimensional `view<f32>` (i.e. a scalar).
2.  `linalg.matvec` reads a two-dimensional `view<?x?xf32>` matrix and a one
    dimensional `view<?xf32>` vector and writes a one-dimensional `view<?xf32>`.
3.  `linalg.matmul` reads two two-dimensional `view<?x?xf32>` matrices and
    writes a two-dimensional `view<?x?xf32>` matrix.

Other operations on higher-order tensors can be defined and would behave
similarly with respect to IR verification and interactions with ViewType
operands. The generic form of verification and pretty-printing is defined on the
`TensorContractionBase`
[class](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg2/include/linalg2/TensorOps.h).

Note that in order to give TensorContractionBase access to the mlir::Op in a
generic fashion, we use a CRTP pattern where:

```
template <class ConcreteOp> class TensorContractionBase { ... };

class DotOp : public TensorContractionBase<DotOp>,
              public mlir::Op<DotOp, mlir::OpTrait::VariadicOperands,
                              mlir::OpTrait::ZeroResult> { ... }
```

In turn, this allows the generic behavior of TensorContractionBase to be
implemented once and reused across ops. The generic verify method is:

```
template <class ConcreteOp>
mlir::LogicalResult linalg::TensorContractionBase<ConcreteOp>::verify() {
  auto *concreteOp = static_cast<ConcreteOp *>(this)->getOperation();
  if (getNumInputs() <= 0)
    concreteOp->emitOpError("expected at least one input");
  ...
}
```

Each specialized operation then calls into the generic verification method
before applying its own verification steps.

```
LogicalResult linalg::MatmulOp::verify() {
  if (failed(TensorContractionBaseType::verify()))
    return failure();
  auto *A = getOperand(0), *B = getOperand(1), *C = getOperand(2);
  unsigned index = 0;
  for (auto *v : {A, B, C}) {
    if (getViewRank(v) != 2)
      return emitOpError("operand " + Twine(index++) + " must be of rank 2");
  }
  return success();
}
```

Note that in a more future-proof design, it is considered a best practice for
operations which share similarity in their behavior to be defined with Tablegen.

All TensorContractionBase ops pretty-print similarly. In the case of
`linalg.matmul` the pretty-printed form is: `linalg.matmul(%A, %B, %C) :
view<?x?xf32>`

## Putting it all together

The
[example](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg2/Example.cpp)
demonstrates how to construct some simple IR snippets that pass through the
verifier checks. The example demonstrate how to allocate three memref buffers
from `index` function arguments and use those buffers as backing data structures
for views that get passed to
