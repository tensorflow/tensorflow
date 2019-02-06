# Quickstart tutorial to adding MLIR graph rewrite

This document will present a quickstart to adding graph rewrites. We shall start
by defining an operation, showing multiple ways to define the rewrite using
patterns, as well as defining the rewrite using a graph walker (note: using
patterns and the rewrite engine is preferred, showing the walker is for
demonstration purposes).

See [MLIR specification](LangRef.md) for more information about MLIR, the
structure of the IR, operations, etc.

## Adding operation

An operation in MLIR is specified using a definition in
[TableGen](https://llvm.org/docs/TableGen/LangIntro.html) file. TableGen is a
modeling tool to specify the ops and the C++ code to interact with these
operations are generated from. To define an operation one needs to specify:

*   The operation name. This name is a unique identifier of the operation within
    MLIR. Most operations are within a dialect, so for example one could have
    `tfl.add` to represent the add operation in the TensorFlow Lite dialect.
    Instead of repeating the dialect in the op definition, a base class for the
    op dialect is commonly created that prepends the dialect namespace given an
    op name.
*   The traits of the operation. These allow you to specify traits of the
    operation, such as whether it has side effects or whether it should be
    verified that the operands and result types are the same. These are backed
    by C++ traits that perform the verification.
*   The arguments of the operation. These are the input operands (values at
    runtime produced by other ops) and attributes (compile time known constant
    values that affect the behavior of the op) that are the inputs of/define the
    behavior of the operation. The input operands may be named, the attributes
    must be named.
*   The result(s) of the operation. These may again named or not.
*   Documentation of the operation. This includes a one-line summary as well as
    a longer human-readable description of the operation.
*   Derived attributes. These are accessors used to compute attributes from
    already known information. For example, the shape attribute for reshape
    where that information is already captured in the type of the operation.
*   Dialect specific information. Additional information could be added to the
    operation definition that are only used by dialect specific drivers. These
    are ignored by the main op and doc generators, but could be used in, say,
    the translation from a dialect to another representation.

```td {.td}
def TFL_LeakyReluOp: TFL_Op<"leaky_relu", [NoSideEffect, SameValueType]>,
                     Results<(outs Tensor)> {
  let arguments = (
    ins F32Tensor:$x,
    // Slope of the activation function at x < 0.
    F32Attr:$alpha
  );

  let summary = "Leaky ReLU operator";
  let description = [{
    Element-wise Leaky ReLU operator
      x -> x >= 0 ? x : (alpha * x)
  }];

  // TFLite specific attribute that is used when generating the output
  // flatbuffer.
  let hasOptions = 0b1;
}
```

Note in the above the result types and inputs are specified in different ways,
one by way of trait and the other by way of let. It is possible to specify both
in either way.

<!-- TODO: Define a style convention. -->

Operations can also have custom parser, printer, builder, verifier, constant
folder, or canonicalizer. These require specifying additional C++ methods to
invoke for additional functionality. For example, if an operation is marked to
have a constant folder, the constant folder also needs to be added, e.g.,:

```c++
Attribute SpecificOp::constantFold(ArrayRef<Attribute> operands,
                                   MLIRContext *context) const {
  if (unable_to_fold)
    return {};
  ....
  return val;
}
```

## Adding patterns

There are multiple forms of graph rewrite that can be performed in MLIR. One of
the most common is DAG tile to DAG tile rewrite. Patterns provide a concise way
to express this transformation as a pair of source pattern to match and
resultant pattern. There is both the C++ classes to represent this
transformation, as well as the patterns in TableGen from which these can be
generated.

### TableGen patterns

Let us continue with LeakyRelu. To map from TensorFlow's `LeakyRelu` to
TensorFlow Lite's `LeakyRelu`:

```td {.td}
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>
```

The pattern is specified by instantiating a `Pat` with a source and result DAG.
The arguments in the from pattern is captured and can be used in the to pattern.
This is a simple pattern as we have a 1:1 mapping and the attribute does not
need to be transformed (e.g., both have a floating point attribute for alpha).
The names of the attributes specified in the pattern is for matching/referencing
and need not match the original attribute name in the op definition but the
order of arguments of the dags do need to match.

To specify a pattern, both the source and results ops need to be defined using
TableGen. For the above case the TensorFlow LeakyRelu was not defined yet in
TableGen and instead a shortened definition was added in the legalize patterns
file:

```td {.td}
def TF_LeakyReluOp : Op<"tf.LeakyRelu">,
                     Arguments<(ins Tensor:$arg, F32Attr:$alpha)>;
```

If this were a more advance pattern that the current framework could not express
as destination then one could use a general native code fallback method. This
consists of defining a pattern as well as adding a C++ function to perform the
replacement:

```td {.td}
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a),
          (cOp<"createTFLLeakyRelu"> $arg, $a)>;
```

```c++
void createTFLLeakyRelu(OperationInst *op, ArrayRef<Value *> operands,
                        ArrayRef<Attribute> attrs, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mlir::TFL::LeakyReluOp>(
      op, operands[0]->getType(), /*arg=*/operands[0],
      /*alpha=*/attrs[0].cast<FloatAttr>());
}
```

This allows for arbitrarily complex builders. Input pattern side one can express
multi-op patterns with constraints on input operands and attributes. But input
patterns cannot yet express constraints across multiple operands/attributes.

### C++ rewrite specification

In case patterns are not sufficient there is also the fully C++ way of
expressing a rewrite:

```c++
struct ConvertTFLeakyRelu : public RewritePattern {
  ConvertTFLeakyRelu(MLIRContext *context)
      : RewritePattern("tf.LeakyRelu", 1, context) {}
  PatternMatchResult match(OperationInst *op) const override {
    return matchSuccess();
  }

  void rewrite(OperationInst *op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFL::LeakyReluOp>(
        op, op->getResult(0)->getType(), op->getOperand(0),
        /*alpha=*/op->getAttrOfType<FloatAttr>("alpha"));
  }
};
```

In the C++ rewrite the static benefit of the rewrite pattern is specified at
construction. While in the pattern generator a simple heuristic is currently
employed based around the number of ops matched and replaced.

The above rule did not capture the matching operands/attributes, but in general
`match` function may populate and return a `PatternState` (or class derived from
one) to pass information extracted during matching to the rewrite.

## Testing

MLIR uses [lit](https://llvm.org/docs/CommandGuide/lit.html) (LLVM Integrated
Testing) tool for performing testing. Testing is performed by way of creating
the input IR file, running a transformation and then verifying the output IR.
C++ unit tests are the exception, with the IR transformation serving as the core
testing mechanism. This results in fewer binaries that need to be built (and
linked) and forces to focus on the representation as an important piece.

For the legalization transform above we would have a test (probably as part of
the legalization pass test in TensorFlow Lite) such as:

```mlir
// RUN: mlir-opt -tfl-legalize-tf %s | FileCheck %s

func @LeakyRelu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.LeakyRelu"(%arg0) {alpha: 0.1} : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: LeakyRelu
// CHECK:  %0 = "tfl.leaky_relu"(%arg0) {alpha: 1.000000e-01} : (tensor<1xf32>) -> tensor<1xf32>
}
```

The RUN command at the top results in running the `mlir-opt` binary (which is
compiler writer tool to exercise different registered passes) to invoke the
optimization pass this transform was added as part of on the current file and to
verify its output using `FileCheck`. `FileCheck` is textual output verifier. In
particular it uses the CHECK expressions to verify the given output is produced.

There can be multiple RUN commands with different corresponding CHECK prefixes.
And in addition multiple independent tests separated by `// -----` and
`mlir-opt` invoked with `-split-input-file` flag. This is especially useful for
error testing.

This results in very simple, directed testing without need to work around
constant propagation or other, unrelated, optimization passes.

## Adding optimization pass

Optimization passes that do not fit/difficult to specify in the above structure
can be specified as general iterations across modules/functions. They have
general structure like:

```c++
namespace {
struct TestPass : public FunctionPass {
  TestPass() : FunctionPass(&TestPass::passID) {}
  PassResult runOnFunction(Function *f) override;

  static char passID;
};
} // end anonymous namespace

char TestPass::passID = 0;

PassResult TestPass::runOnFunction(Function *f) {
  f->walk([](OperationInst *op) {
    ....
  });
  return success();
}

static PassRegistration<TestPass> pass("flag-name-to-invoke-pass-via-mlir-opt",
                                       "Pass description here");
```

TODO: Create an example here.
