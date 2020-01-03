# Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization

[TOC]

At this point, we are eager to generate actual code and see our Toy language
take life. We will use LLVM to generate code, but just showing the LLVM builder
interface here wouldn't be very exciting. Instead, we will show how to perform
progressive lowering through a mix of dialects coexisting in the same function.

To make it more interesting, in this chapter we will consider that we want to
reuse existing optimizations implemented in a dialect optimizing affine
transformations: `Affine`. This dialect is tailored to the computation-heavy
part of the program and is limited: it doesn't support representing our
`toy.print` builtin, for instance, neither should it! Instead, we can target
`Affine` for the computation heavy part of Toy, and in the
[next chapter](Ch-6.md) directly the `LLVM IR` dialect for lowering `print`. As
part of this lowering, we will be lowering from the
[TensorType](../../LangRef.md#tensor-type) that `Toy` operates on to the
[MemRefType](../../LangRef.md#memref-type) that is indexed via an affine
loop-nest. Tensors represent an abstract value-typed sequence of data, meaning
that they don't live in any memory. MemRefs, on the other hand, represent lower
level buffer access, as they are concrete references to a region of memory.

# Dialect Conversions

MLIR has many different dialects, so it is important to have a unified framework
for [converting](../../Glossary.md#conversion) between them. This is where the
`DialectConversion` framework comes into play. This framework allows for
transforming a set of `illegal` operations to a set of `legal` ones. To use this
framework, we need to provide two things (and an optional third):

*   A [Conversion Target](../../DialectConversion.md#conversion-target)

    -   This is the formal specification of what operations or dialects are
        legal for the conversion. Operations that aren't legal will require
        rewrite patterns to perform
        [legalization](./../../Glossary.md#legalization).

*   A set of
    [Rewrite Patterns](../../DialectConversion.md#rewrite-pattern-specification)

    -   These are the set of [patterns](../../QuickstartRewrites.md) used to
        convert `illegal` operations into a set of zero or more `legal` ones.

*   Optionally, a [Type Converter](../../DialectConversion.md#type-conversion).

    -   If provided, this is used to convert the types of block arguments. We
        won't be needing this for our conversion.

## Conversion Target

For our purposes, we want to convert the compute-intensive `Toy` operations into
a combination of operations from the `Affine` `Standard` dialects for further
optimization. To start off the lowering, we first define our conversion target:

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  target.addIllegalDialect<ToyDialect>();
  target.addLegalOp<PrintOp>();
  ...
}
```

## Conversion Patterns

After the conversion target has been defined, we can define how to convert the
`illegal` operations into `legal` ones. Similarly to the canonicalization
framework introduced in [chapter 3](Ch-3.md), the
[`DialectConversion` framework](../../DialectConversion.md) also uses
[RewritePatterns](../../QuickstartRewrites.md) to perform the conversion logic.
These patterns may be the `RewritePatterns` seen before or a new type of pattern
specific to the conversion framework `ConversionPattern`. `ConversionPatterns`
are different from traditional `RewritePatterns` in that they accept an
additional `operands` parameter containing operands that have been
remapped/replaced. This is used when dealing with type conversions, as the
pattern will want to operate on values of the new type but match against the
old. For our lowering, this invariant will be useful as it translates from the
[TensorType](../../LangRef.md#tensor-type) currently being operated on to the
[MemRefType](../../LangRef.md#memref-type). Let's look at a snippet of lowering
the `toy.transpose` operation:

```c++
/// Lower the `toy.transpose` operation to an affine loop nest.
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  /// Match and rewrite the given `toy.transpose` operation, with the given
  /// operands that have been remapped from `tensor<...>` to `memref<...>`.
  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Call to a helper function that will lower the current operation to a set
    // of affine loops. We provide a functor that operates on the remapped
    // operands, as well as the loop induction variables for the inner most
    // loop body.
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS. This adaptor is automatically provided by the ODS
          // framework.
          TransposeOpOperandAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return matchSuccess();
  }
};
```

Now we can prepare the list of patterns to use during the lowering process:

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  ...

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::OwningRewritePatternList patterns;
  patterns.insert<..., TransposeOpLowering>(&getContext());

  ...
```

## Partial Lowering

Once the patterns have been defined, we can perform the actual lowering. The
`DialectConversion` framework provides several different modes of lowering, but,
for our purposes, we will perform a partial lowering, as we will not convert
`toy.print` at this time.

```c++
void ToyToAffineLoweringPass::runOnFunction() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  target.addIllegalDialect<ToyDialect>();
  target.addLegalOp<PrintOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::OwningRewritePatternList patterns;
  patterns.insert<..., TransposeOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  auto function = getFunction();
  if (mlir::failed(mlir::applyPartialConversion(function, target, patterns)))
    signalPassFailure();
}
```

### Design Considerations With Partial Lowering

Before diving into the result of our lowering, this is a good time to discuss
potential design considerations when it comes to partial lowering. In our
lowering, we transform from a value-type, TensorType, to an allocated
(buffer-like) type, MemRefType. However, given that we do not lower the
`toy.print` operation, we need to temporarily bridge these two worlds. There are
many ways to go about this, each with their own tradeoffs:

*   Generate `load` operations from the buffer

One option is to generate `load` operations from the buffer type to materialize
an instance of the value type. This allows for the definition of the `toy.print`
operation to remain unchanged. The downside to this approach is that the
optimizations on the `affine` dialect are limited, because the `load` will
actually involve a full copy that is only visible *after* our optimizations have
been performed.

*   Generate a new version of `toy.print` that operates on the lowered type

Another option would be to have another, lowered, variant of `toy.print` that
operates on the lowered type. The benefit of this option is that there is no
hidden, unnecessary copy to the optimizer. The downside is that another
operation definition is needed that may duplicate many aspects of the first.
Defining a base class in [ODS](../../OpDefinitions.md) may simplify this, but
you still need to treat these operations separately.

*   Update `toy.print` to allow for operating on the lowered type

A third option is to update the current definition of `toy.print` to allow for
operating the on the lowered type. The benefit of this approach is that it is
simple, does not introduce an additional hidden copy, and does not require
another operation definition. The downside to this option is that it requires
mixing abstraction levels in the `Toy` dialect.

For the sake of simplicity, we will use the third option for this lowering. This
involves updating the type constraints on the PrintOp in the operation
definition file:

```tablegen
def PrintOp : Toy_Op<"print"> {
  ...

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
}
```

## Complete Toy Example

Looking back at our current working example:

```mlir
func @main() {
  %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %2 = "toy.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  %3 = "toy.mul"(%2, %2) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  "toy.print"(%3) : (tensor<3x2xf64>) -> ()
  "toy.return"() : () -> ()
}
```

With affine lowering added to our pipeline, we can now generate:

```mlir
func @main() {
  %cst = constant 1.000000e+00 : f64
  %cst_0 = constant 2.000000e+00 : f64
  %cst_1 = constant 3.000000e+00 : f64
  %cst_2 = constant 4.000000e+00 : f64
  %cst_3 = constant 5.000000e+00 : f64
  %cst_4 = constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = alloc() : memref<3x2xf64>
  %1 = alloc() : memref<3x2xf64>
  %2 = alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // Load the transpose value from the input buffer and store it into the
  // next input buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Multiply and store into the output buffer.
  affine.for %arg0 = 0 to 2 {
    affine.for %arg1 = 0 to 3 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  "toy.print"(%0) : (memref<3x2xf64>) -> ()
  dealloc %2 : memref<2x3xf64>
  dealloc %1 : memref<3x2xf64>
  dealloc %0 : memref<3x2xf64>
  return
}
```

## Taking Advantage of Affine Optimization

Our naive lowering is correct, but it leaves a lot to be desired with regards to
efficiency. For example, the lowering of `toy.mul` has generated some redundant
loads. Let's look at how adding a few existing optimizations to the pipeline can
help clean this up. Adding the `LoopFusion` and `MemRefDataFlowOpt` passes to
the pipeline gives the following result:

```mlir
func @main() {
  %cst = constant 1.000000e+00 : f64
  %cst_0 = constant 2.000000e+00 : f64
  %cst_1 = constant 3.000000e+00 : f64
  %cst_2 = constant 4.000000e+00 : f64
  %cst_3 = constant 5.000000e+00 : f64
  %cst_4 = constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = alloc() : memref<3x2xf64>
  %1 = alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // Load the transpose value from the input buffer.
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // Multiply and store into the output buffer.
      %3 = mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  "toy.print"(%0) : (memref<3x2xf64>) -> ()
  dealloc %1 : memref<2x3xf64>
  dealloc %0 : memref<3x2xf64>
  return
}
```

Here, we can see that a redundant allocation was removed, the two loop nests
were fused, and some unnecessary `load`s were removed. You can build `toyc-ch5`
and try yourself: `toyc-ch5 test/lowering.toy -emit=mlir-affine`. We can also
check our optimizations by adding `-opt`.

In this chapter we explored some aspects of partial lowering, with the intent to
optimize. In the [next chapter](Ch-6.md) we will continue the discussion about
dialect conversion by targeting LLVM for code generation.
