# Chapter 3: High-level Language-Specific Analysis and Transformation

[TOC]

Creating a dialect that closely represents the semantics of an input language
enables analyses, transformations and optimizations in MLIR that require
high-level language information and are generally performed on the language AST.
For example, `clang` has a fairly
[heavy mechanism](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)
for performing template instantiation in C++.

We divide compiler transformations into two categories: local and global. In
this chapter, we focus on how to leverage the Toy Dialect and its high-level
semantics to perform local pattern-match transformations that would be difficult
in LLVM. For this, we use MLIR's
[Generic DAG Rewriter](../../GenericDAGRewriter.md).

There are two methods that can be used to implement pattern-match
transformations: 1. Imperative, C++ pattern-match and rewrite 2. Declarative,
rule-based pattern-match and rewrite using table-driven
[Declarative Rewrite Rules](../../DeclarativeRewrites.md) (DRR). Note that the
use of DRR requires that the operations be defined using ODS, as described in
[Chapter 2](Ch-2.md).

# Optimize Transpose using C++ style pattern-match and rewrite

Let's start with a simple pattern and try to eliminate a sequence of two
transpose that cancel out: `transpose(transpose(X)) -> X`. Here is the
corresponding Toy example:

```Toy(.toy)
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Which corresponds to the following IR:

```MLIR(.mlir)
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "toy.transpose"(%0) : (tensor<*xf64>) -> tensor<*xf64>
  "toy.return"(%1) : (tensor<*xf64>) -> ()
}
```

This is a good example of a transformation that is trivial to match on the Toy
IR but that would be quite hard for LLVM to figure. For example, today Clang
can't optimize away the temporary array, and the computation with the naive
transpose is expressed with these loops:

```c++
#define N 100
#define M 100

void sink(void *);
void double_transpose(int A[N][M]) {
  int B[M][N];
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       B[j][i] = A[i][j];
    }
  }
  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < M; ++j) {
       A[i][j] = B[j][i];
    }
  }
  sink(A);
}
```

For a simple C++ approach to rewrite involving matching a tree-like pattern in
the IR and replacing it with a different set of operations, we can plug into the
MLIR `Canonicalizer` pass by implementing a `RewritePattern`:

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::PatternMatchResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value *transposeInput = op.getOperand();
    TransposeOp transposeInputOp =
        llvm::dyn_cast_or_null<TransposeOp>(transposeInput->getDefiningOp());
    // If the input is defined by another Transpose, bingo!
    if (!transposeInputOp)
      return matchFailure();

    // Use the rewriter to perform the replacement
    rewriter.replaceOp(op, {transposeInputOp.getOperand()}, {transposeInputOp});
    return matchSuccess();
  }
};
```

The implementation of this rewriter is in `ToyCombine.cpp`. The
[canonicalization pass](../../Canonicalization.md) applies transformations
defined by operations in a greedy, iterative manner. To ensure that the
canonicalization pass applies our new transform, we set
[hasCanonicalizer = 1](../../OpDefinitions.md#hascanonicalizer) and register the
pattern with the canonicalization framework.

```c++
// Register our patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyRedundantTranspose>(context);
}
```

We also need to update our main file, `toyc.cpp`, to add an optimization
pipeline. In MLIR, the optimizations are run through a `PassManager` in a
similar way to LLVM:

```c++
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
```

Finally, we can run `toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt` and
observe our pattern in action:

```MLIR(.mlir)
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
  "toy.return"(%arg0) : (tensor<*xf64>) -> ()
}
```

As expected, we now directly return the function argument, bypassing any
transpose operation. However, one of the transposes still hasn't been
eliminated. That is not ideal! What happened is that our pattern replaced the
last transform with the function input and left behind the now dead transpose
input. The Canonicalizer knows to clean up dead operations; however, MLIR
conservatively assumes that operations may have side-effects. We can fix this by
adding a new trait, `NoSideEffect`, to our `TransposeOp`:

```TableGen(.td):
def TransposeOp : Toy_Op<"transpose", [NoSideEffect]> {...}
```

Let's retry now `toyc-ch3 test/transpose_transpose.toy -emit=mlir -opt`:

```MLIR(.mlir)
func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  "toy.return"(%arg0) : (tensor<*xf64>) -> ()
}
```

Perfect! No `transpose` operation is left - the code is optimal.

In the next section, we use DRR for pattern match optimizations associated with
the Reshape op.

# Optimize Reshapes using DRR

Declarative, rule-based pattern-match and rewrite (DRR) is an operation
DAG-based declarative rewriter that provides a table-based syntax for
pattern-match and rewrite rules:

```TableGen(.td):
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
```

A redundant reshape optimization similar to SimplifyRedundantTranspose can be
expressed more simply using DRR as follows:

```TableGen(.td):
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

The automatically generated C++ code corresponding to each of the DRR patterns
can be found under path/to/BUILD/projects/mlir/examples/toy/Ch3/ToyCombine.inc.

DRR also provides a method for adding argument constraints when the
transformation is conditional on some properties of the arguments and results.
An example is a transformation that eliminates reshapes when they are redundant,
i.e. when the input and output shapes are identical.

```TableGen(.td):
def TypesAreIdentical : Constraint<CPred<"$0->getType() == $1->getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

Some optimizations may require additional transformations on instruction
arguments. This is achieved using NativeCodeCall, which allows for more complex
transformations either by calling into a C++ helper function or by using inline
C++. An example of such an optimization is FoldConstantReshape, where we
optimize Reshape of a constant value by reshaping the constant in place and
eliminating the reshape operation.

```TableGen(.td):
def ReshapeConstant : NativeCodeCall<"$0.reshape(($1->getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```

We demonstrate these reshape optimizations using the following
trivialReshape.toy program:

```c++
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```MLIR(.mlir)
module {
  func @main() {
    %0 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}
                           : () -> tensor<2xf64>
    %1 = "toy.reshape"(%0) : (tensor<2xf64>) -> tensor<2x1xf64>
    %2 = "toy.reshape"(%1) : (tensor<2x1xf64>) -> tensor<2x1xf64>
    %3 = "toy.reshape"(%2) : (tensor<2x1xf64>) -> tensor<2x1xf64>
    "toy.print"(%3) : (tensor<2x1xf64>) -> ()
    "toy.return"() : () -> ()
  }
}
```

We can try to run `toyc-ch3 test/trivialReshape.toy -emit=mlir -opt` and observe
our pattern in action:

```MLIR(.mlir)
module {
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00], [2.000000e+00]]> \
                           : tensor<2x1xf64>} : () -> tensor<2x1xf64>
    "toy.print"(%0) : (tensor<2x1xf64>) -> ()
    "toy.return"() : () -> ()
  }
}
```

As expected, no reshape operations remain after canonicalization.

Further details on the declarative rewrite method can be found at
[Table-driven Declarative Rewrite Rule (DRR)](../../DeclarativeRewrites.md).

In this chapter, we saw how to use certain core transformations through always
available hooks. In the [next chapter](Ch-4.md), we will see how to use generic
solutions that scale better through Interfaces.
