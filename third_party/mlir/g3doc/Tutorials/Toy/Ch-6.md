# Chapter 6: Lowering to LLVM and CodeGeneration

[TOC]

In the [previous chapter](Ch-5.md), we introduced the
[dialect conversion](../../DialectConversion.md) framework and partially lowered
many of the `Toy` operations to affine loop nests for optimization. In this
chapter, we will finally lower to LLVM for code generation.

# Lowering to LLVM

For this lowering, we will again use the dialect conversion framework to perform
the heavy lifting. However, this time, we will be performing a full conversion
to the [LLVM dialect](../../Dialects/LLVM.md). Thankfully, we have already
lowered all but one of the `toy` operations, with the last being `toy.print`.
Before going over the conversion to LLVM, let's lower the `toy.print` operation.
We will lower this operation to a non-affine loop nest that invokes `printf` for
each element. Note that, because the dialect conversion framework supports
[transitive lowering](Glossary.md#transitive-lowering), we don't need to
directly emit operations in the LLVM dialect. By transitive lowering, we mean
that the conversion framework may apply multiple patterns to fully legalize an
operation. In this example, we are generating a structured loop nest instead of
the branch-form in the LLVM dialect. As long as we then have a lowering from the
loop operations to LLVM, the lowering will still succeed.

During lowering we can get, or build, the declaration for printf as so:

```c++
/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get("printf", context);

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
  auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get("printf", context);
}
```

Now that the lowering for the printf operation has been defined, we can specify
the components necessary for the lowering. These are largely the same as the
components defined in the [previous chapter](Ch-5.md).

## Conversion Target

For this conversion, aside from the top-level module, we will be lowering
everything to the LLVM dialect.

```c++
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
```

## Type Converter

This lowering will also transform the MemRef types which are currently being
operated on into a representation in LLVM. To perform this conversion, we use a
TypeConverter as part of the lowering. This converter specifies how one type
maps to another. This is necessary now that we are performing more complicated
lowerings involving block arguments. Given that we don't have any
Toy-dialect-specific types that need to be lowered, the default converter is
enough for our use case.

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

## Conversion Patterns

Now that the conversion target has been defined, we need to provide the patterns
used for lowering. At this point in the compilation process, we have a
combination of `toy`, `affine`, and `std` operations. Luckily, the `std` and
`affine` dialects already provide the set of patterns needed to transform them
into LLVM dialect. These patterns allow for lowering the IR in multiple stages
by relying on [transitive lowering](Glossary.md#transitive-lowering).

```c++
  mlir::OwningRewritePatternList patterns;
  mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
  mlir::populateLoopToStdConversionPatterns(patterns, &getContext());
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());
```

## Full Lowering

We want to completely lower to LLVM, so we use a `FullConversion`. This ensures
that only legal operations will remain after the conversion.

```c++
  mlir::ModuleOp module = getModule();
  if (mlir::failed(mlir::applyFullConversion(module, target, patterns,
                                             &typeConverter)))
    signalPassFailure();
```

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

We can now lower down to the LLVM dialect, which produces the following code:

```mlir
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> !llvm.i32
llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : !llvm.double
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : !llvm.double

  ...

^bb16:
  %221 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : !llvm.i64
  %223 = llvm.mlir.constant(2 : index) : !llvm.i64
  %224 = llvm.mul %214, %223 : !llvm.i64
  %225 = llvm.add %222, %224 : !llvm.i64
  %226 = llvm.mlir.constant(1 : index) : !llvm.i64
  %227 = llvm.mul %219, %226 : !llvm.i64
  %228 = llvm.add %225, %227 : !llvm.i64
  %229 = llvm.getelementptr %221[%228] : (!llvm<"double*">, !llvm.i64) -> !llvm<"double*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, !llvm.double) -> !llvm.i32
  %232 = llvm.add %219, %218 : !llvm.i64
  llvm.br ^bb15(%232 : !llvm.i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

See [Conversion to the LLVM IR Dialect](../../ConversionToLLVMDialect.md) for
more in-depth details on lowering to the LLVM dialect.

# CodeGen: Getting Out of MLIR

At this point we are right at the cusp of code generation. We can generate code
in the LLVM dialect, so now we just need to export to LLVM IR and setup a JIT to
run it.

## Emitting LLVM IR

Now that our module is comprised only of operations in the LLVM dialect, we can
export to LLVM IR. To do this programmatically, we can invoke the following
utility:

```c++
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule)
    /* ... an error was encountered ... */
```

Exporting our module to LLVM IR generates:

```.llvm
define void @main() {
  ...

102:
  %103 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %104 = mul i64 %96, 2
  %105 = add i64 0, %104
  %106 = mul i64 %100, 1
  %107 = add i64 %105, %106
  %108 = getelementptr double, double* %103, i64 %107
  %109 = load double, double* %108
  %110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double %109)
  %111 = add i64 %100, 1
  br label %99

  ...

115:
  %116 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %24, 0
  %117 = bitcast double* %116 to i8*
  call void @free(i8* %117)
  %118 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %16, 0
  %119 = bitcast double* %118 to i8*
  call void @free(i8* %119)
  %120 = extractvalue { double*, i64, [2 x i64], [2 x i64] } %8, 0
  %121 = bitcast double* %120 to i8*
  call void @free(i8* %121)
  ret void
}
```

If we enable optimization on the generated LLVM IR, we can trim this down quite
a bit:

```.llvm
define void @main()
  %0 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.000000e+00)
  %1 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 4.000000e+00)
  %3 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 9.000000e+00)
  %5 = tail call i32 (i8*, ...) @printf(i8* nonnull dereferenceable(1) getelementptr inbounds ([4 x i8], [4 x i8]* @frmt_spec, i64 0, i64 0), double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}

```

The full code listing for dumping LLVM IR can be found in `Ch6/toy.cpp` in the
`dumpLLVMIR()` function:

```c++

int dumpLLVMIR(mlir::ModuleOp module) {
  // Translate the module, that contains the LLVM dialect, to LLVM IR.
  auto llvmModule = mlir::translateModuleToLLVMIR(module);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

## Setting up a JIT

Setting up a JIT to run the module containing the LLVM dialect can be done using
the `mlir::ExecutionEngine` infrastructure. This is a utility wrapper around
LLVM's JIT that accepts `.mlir` as input. The full code listing for setting up
the JIT can be found in `Ch6/toy.cpp` in the `runJit()` function:

```c++
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/EnableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

You can play around with it from the build directory:

```sh
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

You can also play with `-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, and
`-emit=llvm` to compare the various levels of IR involved. Also try options like
[`--print-ir-after-all`](../../WritingAPass.md#ir-printing) to track the
evolution of the IR throughout the pipeline.

So far, we have worked with primitive data types. In the
[next chapter](Ch-7.md), we will add a composite `struct` type.
