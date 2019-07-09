# Chapter 4: High-level Language-Specific Analysis and Transformation

Creating a dialect that closely represents the semantics of an input language
enables analyses and transformations in MLIR that are generally performed on the
language AST. For example, `clang` has a fairly
[heavy mechanism](https://clang.llvm.org/doxygen/classclang_1_1TreeTransform.html)
for performing template instantiation in C++.

Another aspect is optimization. While some previous language specific
optimizations have been implemented in LLVM (like the
[ARC optimizer](http://llvm.org/doxygen/ObjCARCOpts_8cpp_source.html#l00468)),
it has been at the cost of relying on either adding enough concepts in LLVM, to
be able to embed the high-level semantics of the input, or using fragile
"best-effort" metadata to decorate the IR with the information needed for these
custom optimizations.

We show in this chapter how to leverage the Toy Dialect and its high-level
semantics to perform transformations that would be difficult in LLVM: first a
simple combine of two redundant operations, and second a full interprocedural
shape inference with function specialization.

# Basic Optimization: Eliminate Redundant Transpose

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
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  %0 = "toy.transpose"(%arg0) : (!toy<"array">) -> !toy<"array">
  %1 = "toy.transpose"(%0) : (!toy<"array">) -> !toy<"array">
  "toy.return"(%1) : (!toy<"array">) -> ()
}
```

This is a good example of a transformation that is trivial to match on the Toy
IR but that would be quite hard for LLVM to figure. For example today clang
can't optimize away the temporary array and the computation with the naive
transpose expressed with these loops:

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

For simple rewrite involving matching a tree-like pattern in the IR and
replacing it with a different set of operations, we can plug into the MLIR
`Canonicalizer` pass by implementing a `RewritePattern`:

```c++
/// Fold transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::RewritePattern {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : RewritePattern(TransposeOp::getOperationName(), /* benefit = */ 1, context) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::PatternMatchResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    // We can directly cast the current operation as this will only get invoked
    // on TransposeOp.
    TransposeOp transpose = op->cast<TransposeOp>();
    // look through the input to the current transpose
    mlir::Value *transposeInput = transpose.getOperand();
    // If the input is defined by another Transpose, bingo!
    if (!matchPattern(transposeInput, mlir::m_Op<TransposeOp>()))
      return matchFailure();

    auto transposeInputOp =
        transposeInput->getDefiningOp()->cast<TransposeOp>();
    // Use the rewriter to perform the replacement
    rewriter.replaceOp(op, {transposeInputOp.getOperand()}, {transposeInputOp});
    return matchSuccess();
  }
};
```

Let's see how to improve our `TransposeOp` by extending it with a new static
method:

```c++
  /// This hook returns any canonicalization pattern rewrites that the operation
  /// supports, for use by the canonicalization pass.
  static void getCanonicalizationPatterns(mlir::OwningRewritePatternList &results,
                                          mlir::MLIRContext *context) {
    results.push_back(llvm::make_unique<SimplifyRedundantTranspose>(context));
  }
```

The implementation of this rewriter is in `ToyCombine.cpp`. We also need to
update our main file, `toyc.cpp`, to add an optimization pipeline. In MLIR, the
optimizations are ran through a `PassManager` in a similar way to LLVM:

```c++
mlir::PassManager pm;
pm.addPass(mlir::createCanonicalizerPass());
pm.run(&module);
```

Finally, we can try to run `toyc test/transpose_transpose.toy -emit=mlir -opt`
and observe our pattern in action:

```MLIR(.mlir)
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  %0 = "toy.transpose"(%arg0) : (!toy<"array">) -> !toy<"array">
  "toy.return"(%arg0) : (!toy<"array">) -> ()
}
```

As expected we now directly return the function argument, bypassing any
transpose operation. However one of the transpose hasn't been eliminated. That
is not ideal! What happened is that our pattern replaced the last transform with
the function input and left behind the now dead transpose input. The
Canonicalizer knows to cleanup dead operations, however MLIR conservatively
assumes that operations may have side-effects. We can fix it by adding a new
trait, `HasNoSideEffect`, to our `TransposeOp`:

```c++
class TransposeOp : public mlir::Op<TransposeOp, mlir::OpTrait::OneOperand,
                                    mlir::OpTrait::OneResult,
                                    mlir::OpTrait::HasNoSideEffect> {
```

Let's retry now `toyc test/transpose_transpose.toy -emit=mlir -opt`:

```MLIR(.mlir)
func @transpose_transpose(%arg0: !toy<"array">)
  attributes  {toy.generic: true} {
  "toy.return"(%arg0) : (!toy<"array">) -> ()
}
```

Perfect! No `transpose` operation is left, the code is optimal.

The code in `mlir/ToyCombine.cpp` implements a few more patterns that eliminate
trivial reshapes, or fold them into constants.

# Shape Inference and Generic Function Specialization

Our IR operates on generic arrays, we don't know the shape of the arrays other
than during initialization of constants. However we can propagate the shapes
through the computation until they are all known. The issue is how to handle
calls to user-defined generic functions: every call site could deduce different
shapes. One possibility would be to perform symbolic inference based on the
argument types, but this would be hard to generalize if we were to introduce
more control flow in the language. Instead we will proceed by function
specialization: for every call site with new argument shapes we duplicate the
function and specialize it. This is akin to C++ template instantiation:

```
template<int M1, int N1, int M2, int N2>
auto multiply_add(array<M1, N1> a, array<M1, N1> b) {
  auto prod = mul(a, b);
  auto sum = add(prod, a);
  return sum;
}
```

Every new call to `multiply_add` would instantiate the template and emit code
for the specific shape and deduce the return type. Clang implements this
transformation on its AST, but we will implement it in an MLIR pass here.

The ShapeInferencePass is a `ModulePass`: it will run on the Module as a whole.
MLIR also supports `FunctionPass`es which are restricted to modify a single
function at a time. This pass couldn't be a function pass due the nature of its
interprocedural transformations.

Implementing such a pass is done by creating a class inheriting from
`mlir::ModulePass` and overriding the `runOnModule()` method:

```
class ShapeInferencePass : public mlir::ModulePass<ShapeInferencePass> {

  void runOnModule() override {
    auto &module = getModule();
    ...
```

The algorithm has two levels, first intra-procedurally:

1.  Build a worklist containing all the operations that are returning a generic
    Toy array: these are the operations that need shape inference.
2.  Iterate on the worklist:
    -   find an operation to process: the next ready operation in the worklist
        has all of its arguments non-generic,
    -   if no operation is found, break out of the loop,
    -   remove the operation from the worklist,
    -   infer the shape of its output from the arguments type.
3.  If the worklist is empty, the algorithm succeeded and we infer the return
    type for the function from the return operation.

There is a twist though: when a call to a generic function is encountered, shape
inference requires the return type of the callee to be inferred first. At this
point we need to specialize the callee by cloning it. Here is the
inter-procedural flow that wraps the intra-procedural inference:

1.  Keep a worklist of function to process. Start with function "main".
2.  While the worklist isn't empty:
    -   Take the last inserted function in the worklist.
    -   Run the intra-procedural shape inference on this function.
    -   If the intra-procedural shape inference can't complete, it returns a
        FuncOp that needs to be inferred first. In this case, queue this new
        function and continue. Otherwise the inference succeeded and we can pop
        from the queue.

The full code is in `mlir/ShapeInferencePass.cpp`.

# Future Work: Optimizing Buffer Allocation?

Toy is value-based. Naively this is a lot of allocation, what if we want to
statically optimize placement? What is the right abstraction level to perform
buffer assignment?
