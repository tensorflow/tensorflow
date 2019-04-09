# Chapter 5: CodeGen via Lowering to Lower-Level Dialects

At this point, we are eager to generate actual code and see our Toy language
taking life. We will obviously use LLVM to generate code, but just showing the
LLVM builder interface wouldn't be very exciting here. Instead, we will show how
to perform progressive lowering through a mix of dialects coexisting in the same
function.

To make it more interesting, we will consider that we want to reuse existing
optimizations implemented in a dialect optimizing linear algebra: `Linalg`. This
dialect is tailored to the computation heavy part of the program, and is
limited: it doesn't support representing our `toy.print` builtin for instance,
neither should it! Instead we can target `Linalg` for the computation heavy part
of Toy (mostly matmul), we will target the `Affine` dialect for other
well-formed loop nest, and directly the `LLVM IR` dialect for lowering `print`.

# The `DialectConversion` Framework

Similarly to the canonicalization patterns introduced in the previous section,
the `DialectConversion` framework involves its own set of patterns. This
framework operates a bit differently from the canonicalizer: a new function is
created and the pattern matching operation in the original function are expected
to emit the IR in the new function.

Dialect conversion requires three components, implemented by overriding virtual
methods defined in `DialectConversion`:

-   Type Conversion: for things like block arguments' type.
-   Function signature conversion: for every function it is invoked with the
    function type and the conversion generates a new prototype for the converted
    function. The default implementation will call into the type conversion for
    the returned values and for each of the parameters.
-   Operations convertions: each pattern is expected to generate new results
    matching the current operations' in the new function. This may involve
    generating one or multiple new operations, or possibly just remapping
    existing operands (folding).

A typical starting point for implementing our lowering would be:

```c++
class Lowering : public DialectConversion {
public:
  // This gets called for block and region arguments, and attributes.
  Type convertType(Type t) override { /*...*/ }

  // This gets called for functions.
  FunctionType convertFunctionSignatureType(FunctionType type,
      ArrayRef<NamedAttributeList> argAttrs,
      SmallVectorImpl<NamedAttributeList> &convertedArgAttrs) { /*...*/ }

  // This gets called once to set up operation converters.
  llvm::DenseSet<DialectOpConversion *>
  initConverters(MLIRContext *context) override {
    return ConversionListBuilder<MulOpConversion,
                                 PrintOpConversion,
                                 TransposeOpConversion>::build(allocator, context);
  }

private:
  llvm::BumpPtrAllocator allocator;
};
```

Individual operation converters are following this pattern:

```c++
/// Lower a toy.add to an affine loop nest.
///
/// This class inherit from `DialectOpConversion` and override `rewrite`,
/// similarly to the PatternRewriter introduced in the previous chapter.
/// It will be called by the DialectConversion framework (see `LateLowering`
/// class below).
class AddOpConversion : public DialectOpConversion {
public:
  explicit AddOpConversion(MLIRContext *context)
      : DialectOpConversion(toy::AddOp::getOperationName(), 1, context) {}

  /// Lower the `op` by generating IR using the `rewriter` builder. The builder
  /// is setup with a new function, the `operands` array has been populated with
  /// the rewritten operands for `op` in the new function.
  /// The results created by the new IR with the builder are returned, and their
  /// number must match the number of result of `op`.
  SmallVector<Value *, 4> rewrite(Operation *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    ...

    // Return the newly allocated buffer, it will be used as an operand when
    // converting the operations corresponding to the users of this `toy.add`.
    return result;
  }
```

## Linalg

Linalg is an advanced dialect for dense algebra optimizations. It is implemented
as [a separate tutorial](../Linalg/Ch-1.md) in parallel with Toy. We are acting
as a user of this dialect by lowering Toy matrix multiplications to
`linalg.matmul`.

To support this, we will split our lowering in two parts: an *early lowering*
that emits operations in the `Linalg` dialect for a subset of the Toy IR, and a
*late lowering* that materializes buffers and converts all operations and type
to the LLVM dialect. We will then be able to run specific optimizations in
between the two lowering.

Let's look again at our example `multiply_transpose`:

```mlir
func @multiply_transpose(%arg0: !toy.array, %arg1: !toy.array)
  attributes  {toy.generic: true} {
  %0 = "toy.transpose"(%arg1) : (!toy.array) -> !toy.array
  %1 = "toy.mul"(%arg0, %0) : (!toy.array, !toy.array) -> !toy.array
  "toy.return"(%1) : (!toy.array) -> ()
}
```

After shape inference, and lowering to `Linalg`, here is what our IR will look
like:

```mlir
func @multiply_transpose_2x3_2x3(%arg0: !toy.array<2, 3>, %arg1: !toy.array<2, 3>) -> !toy.array<2, 2>
  attributes  {toy.generic: false} {
  %c3 = constant 3 : index
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c1 = constant 1 : index
  %0 = "toy.transpose"(%arg1) : (!toy.array<2, 3>) -> !toy.array<3, 2>
  %1 = "toy.alloc"() : () -> !toy.array<2, 2>
  %2 = "toy.cast"(%1) : (!toy.array<2, 2>) -> memref<2x2xf64>
  %3 = "toy.cast"(%arg0) : (!toy.array<2, 3>) -> memref<2x3xf64>
  %4 = "toy.cast"(%0) : (!toy.array<3, 2>) -> memref<3x2xf64>
  %5 = linalg.range %c0:%c2:%c1 : !linalg.range
  %6 = linalg.range %c0:%c3:%c1 : !linalg.range
  %7 = linalg.view %3[%5, %6] : !linalg<"view<?x?xf64>">
  %8 = linalg.view %4[%6, %5] : !linalg<"view<?x?xf64>">
  %9 = linalg.view %2[%5, %5] : !linalg<"view<?x?xf64>">
  linalg.matmul(%7, %8, %9) : !linalg<"view<?x?xf64>">
  "toy.return"(%1) : (!toy.array<2, 2>) -> ()
}
```

Note how the operations from multiple dialects are coexisting in this function.

You can reproduce this result with `bin/toyc-ch5
test/Examples/Toy/Ch5/lowering.toy -emit=mlir-linalg`

## Emitting LLVM

The availability of various dialects allows for a smooth lowering by reducing
the impedance mismatch between dialects. For example we don't need to lower our
`toy.print` over array directly to LLVM IR, we can use the well structured loop
from the `Affine` dialect for convenience when scanning the array and insert a
call to `llvm.printf` in the body. We will rely on MLIR lowering to LLVM for the
`Affine` dialect, we get it for free. Here is a simplified version of the code
in this chapter for lowering `toy.print`:

```c++
    // Create our loop nest now
    using namespace edsc;
    using llvmCall = intrinsics::ValueBuilder<LLVM::CallOp>;
    ScopedContext scope(rewriter, loc);
    ValueHandle zero = intrinsics::constant_index(0);
    ValueHandle fmtCst(getConstantCharBuffer(rewriter, loc, "%f "));
    ValueHandle fmtEol(getConstantCharBuffer(rewriter, loc, "\n"));
    MemRefView vOp(operand);
    IndexedValue iOp(operand);
    IndexHandle i, j, M(vOp.ub(0)), N(vOp.ub(1));
    LoopBuilder(&i, zero, M, 1)({
      LoopBuilder(&j, zero, N, 1)({
        llvmCall(retTy,
                 rewriter.getFunctionAttr(printfFunc),
                 {fmtCst, iOp(i, j)})
      }),
      llvmCall(retTy, rewriter.getFunctionAttr(printfFunc), {fmtEol})
    });
```

For instance the Toy IR may contain:

```
  "toy.print"(%0) : (!toy.array<2, 2>) -> ()
```

which the converter above will turn into this sequence:

```mlir
  affine.for %i0 = 0 to 2 {
    affine.for %i1 = 0 to 2 {
      %3 = load %0[%i0, %i1] : memref<2x2xf64>
      %4 = llvm.call @printf(%1, %3) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
    }
    %5 = llvm.call @printf(%2, %cst_21) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
  }
```

Note the mix of a loop nest in the `Affine` dialect, with an operation
`llvm.call` in the body. MLIR knows already how to lower this to:

```mlir
  llvm.br ^bb1(%87 : !llvm.i64)
^bb1(%89: !llvm.i64):   // 2 preds: ^bb0, ^bb5
  %90 = llvm.icmp "slt" %89, %88 : !llvm.i64
  llvm.cond_br %90, ^bb2, ^bb6
^bb2:   // pred: ^bb1
  %91 = llvm.constant(0 : index) : !llvm.i64
  %92 = llvm.constant(2 : index) : !llvm.i64
  llvm.br ^bb3(%91 : !llvm.i64)
^bb3(%93: !llvm.i64):   // 2 preds: ^bb2, ^bb4
  %94 = llvm.icmp "slt" %93, %92 : !llvm.i64
  llvm.cond_br %94, ^bb4, ^bb5
^bb4:   // pred: ^bb3
  %95 = llvm.constant(2 : index) : !llvm.i64
  %96 = llvm.constant(2 : index) : !llvm.i64
  %97 = llvm.mul %89, %96 : !llvm.i64
  %98 = llvm.add %97, %93 : !llvm.i64
  %99 = llvm.getelementptr %6[%98] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
  %100 = llvm.load %99 : !llvm<"double*">
  %101 = llvm.call @printf(%48, %100) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
  %102 = llvm.constant(1 : index) : !llvm.i64
  %103 = llvm.add %93, %102 : !llvm.i64
  llvm.br ^bb3(%103 : !llvm.i64)
^bb5:   // pred: ^bb3
  %104 = llvm.call @printf(%76, %71) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
  %105 = llvm.constant(1 : index) : !llvm.i64
  %106 = llvm.add %89, %105 : !llvm.i64
  llvm.br ^bb1(%106 : !llvm.i64)
```

We appreciate the ease to generate the former, as well as the readability!

You may reproduce these results with `echo "def main() { print([[1,2],[3,4]]); }
" | bin/toyc-ch5 -x toy - -emit=llvm-dialect` and `echo "def main() {
print([[1,2],[3,4]]); } " | bin/toyc-ch5 -x toy - -emit=llvm-ir`.

# CodeGen: Getting Out of MLIR

At this point, all the IR is expressed in the LLVM dialect, MLIR can perform a
straight conversion to an LLVM module. You may look into
[`Ch5/toyc.cpp`](../../../examples/toy/Ch5/toyc.cpp) for the `dumpLLVM()`
function:

```c++
int dumpLLVM() {
  mlir::MLIRContext context;
  auto module = loadFileAndProcessModule(context, /* EnableLowering=*/ true);
  auto llvmModule = translateModuleToLLVMIR(*module);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

Adding a JIT isn't much more involved either:

```c++
int runJit() {
  mlir::MLIRContext context;
  auto module = loadFileAndProcessModule(context, /* EnableLowering=*/ true);

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create an MLIR execution engine.  Note that it takes a null pass manager
  // to make sure it won't run "default" passes on the MLIR that would trigger
  // a second conversion to LLVM IR.  The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine =
      mlir::ExecutionEngine::create(module.get(), /*pm=*/nullptr);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function with the arguments.  Note that, for API
  // uniformity reasons, it takes a list of type-erased pointers to arguments.
  auto invocationResult = engine->invoke("main");
  if(invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

You can play with it, from the build directory:

```bash
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch5 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

You can also play with `-emit=mlir`, `-emit=mlir-linalg`, `-emit=llvm-dialect`,
and `-emit=llvm-ir` to compare the various level of IR involved. Try also
options like `--print-ir-after-all` to track the evolution of the IR throughout
the pipeline.
