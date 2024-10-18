/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_OPS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "xla/service/gpu/fusions/ir/xla_gpu_dialect.td"
include "xla/service/gpu/fusions/ir/xla_gpu_attrs.td"
include "xla/service/gpu/fusions/ir/xla_gpu_types.td"

class XLAGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaGpuDialect, mnemonic, traits> {
}

def XLAGPU_AllocateSharedOp : XLAGPU_Op<"allocate_shared", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Allocates a shared memory tile.";

  let description = [{
    Allocates a shared memory tensor. The tensor is shared among all threads in
    a block.

    ```mlir
    %shared = xla_gpu.allocate_shared : tensor<32x32xf32>
    ```
  }];

  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def XLAGPU_SyncThreadsOp : XLAGPU_Op<"sync_threads", [
   TypesMatchWith<"result type matches type of dest",
                   "operands", "results", "$_self">,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Synchronizes threads.";

  let description = [{
    Synchronizes threads, taking any number of distributed tensors and returning
    the synchronized state.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "operands attr-dict `:` type($operands)";
}

def XLAGPU_AtomicRMWOp : XLAGPU_Op<"atomic_rmw",
    [Pure,
     TypesMatchWith<"result type matches type of dest",
                    "input", "result", "$_self">,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Atomically updates an element of a tensor.";

  let description = [{
    Reads an element from a tensor, computes the updated value for it, and
    writes back the result.
  }];

  let arguments = (ins AnyRankedTensor:$input, Variadic<Index>:$indices);
  let results = (outs AnyRankedTensor:$result);
  // The region takes the current value in the tensor as an argument and yields
  // the updated value.
  let regions = (region SizedRegion<1>:$computation);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::Value":$memref, "mlir::ValueRange":$ivs)>];

  let extraClassDeclaration = [{
    mlir::Block* getBody() { return &getComputation().front(); }
    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    // The value stored in tensor[ivs].
    mlir::Value getCurrentValue() {
      return getRegion().getArgument(0);
    }
  }];
  let hasFolder = 1;

  let assemblyFormat = [{
    $input `[` $indices `]` `:` type($input) $computation attr-dict
  }];
}

def XLAGPU_YieldOp : XLAGPU_Op<"yield", [
    ParentOneOf<["::xla::gpu::AtomicRMWOp", "::xla::gpu::LoopOp"]>,
    ReturnLike, Terminator]> {
  let summary = "Terminator for atomic_rmw ops.";
  let arguments = (ins Variadic<AnyType>:$result);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

def XLAGPU_PureCallOp : XLAGPU_Op<"pure_call",
    [Pure, CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Function call without side effects.";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let builders = [
    OpBuilder<(ins "mlir::func::FuncOp":$callee, CArg<"mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "mlir::FlatSymbolRefAttr":$callee, CArg<"mlir::ValueRange", "{}">:$operands, CArg<"llvm::ArrayRef<mlir::Type>", "{}">:$result_types), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(result_types);
    }]>,
    ];
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getOperands();
    }

    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

def XLAGPU_ShuffleReduceOp : XLAGPU_Op<"shuffle_reduce",
    [Pure,  CallOpInterface,
     TypesMatchWith<"result type matches type of operands",
                     "operands", "results", "$_self">]> {
  let summary = "Performs a full warp shuffle and reduces the values";
  let description = [{
    This op performs a full warp shuffle and reduces the results using the given
    function. The function is invoked with the operands from the low lanes,
    followed by the operands from the high lanes. For example:

    ```
    %result:2 = xla_gpu.shuffle_reduce (%in0, %in1) to 16 combiner=@argmax
    ```

    Will perform shuffles with distance 16, 8, 4, 2 and 1, and will invoke
    @argmax five times. The first invocations will be

    ```
    @argmax(%in0[i], %in1[i], %in0[16+i], %in1[16+i])
    ```
  }];
  let builders = [
    OpBuilder<(ins "mlir::func::FuncOp":$combiner, "mlir::ValueRange":$operands, "int64_t":$max_distance), [{
      $_state.addOperands(operands);
      $_state.addAttribute("combiner", mlir::SymbolRefAttr::get(combiner));
      $_state.addAttribute("max_distance",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(combiner.getContext(), 64),
              max_distance));
      $_state.addTypes(combiner.getFunctionType().getResults());
    }]>];
  let arguments = (ins FlatSymbolRefAttr:$combiner,
                   Variadic<AnyType>:$operands,
                   I64Attr:$max_distance);
  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    operand_range getArgOperands() {
      return getOperands();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("reducer", callee.get<mlir::SymbolRefAttr>());
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def XLAGPU_PredicatedInsertOp : XLAGPU_Op<"predicated_insert",
    [Pure,
     TypesMatchWith<"result type matches type of operands",
                    "dest", "result", "$_self">,
     TypesMatchWith<"value type matches element type of dest",
                    "dest", "value",
                    "::llvm::cast<mlir::TensorType>($_self).getElementType()">]> {
  let summary = "Inserts a value into a tensor if a condition holds";
  let arguments = (ins I1:$condition, AnyType:$value,
    AnyStaticShapeTensor:$dest, Variadic<Index>:$indices);
  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = [{
     $value `into` $dest `[` $indices `]` `if` $condition attr-dict `:` type($dest)
  }];
}

def XLAGPU_PredicatedExtractOp : XLAGPU_Op<"predicated_extract",
    [Pure,
     TypesMatchWith<"fallback type matches element type of src",
                    "src", "fallback",
                    "::llvm::cast<mlir::TensorType>($_self).getElementType()">,
     TypesMatchWith<"result type matches element type of src",
                    "src", "result",
                    "::llvm::cast<mlir::TensorType>($_self).getElementType()">]> {
  let summary = "Inserts a value into a tensor if a condition holds";
  let arguments = (ins I1:$condition, AnyType:$fallback,
    AnyStaticShapeTensor:$src, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $src `[` $indices `]` `if` $condition `else` $fallback attr-dict `:` type($src)
  }];
}

def ApplyIndexingOp : XLAGPU_Op<"apply_indexing", [Pure]> {
  let summary = "Applies indexing map to a list of SSA values";
  let description = [{
    The `apply_indexing` operation applies an indexing_map to a list
    of SSA values, yielding a single SSA value. The number of dimension and
    symbol arguments must be equal to the respective number of dimensional and
    symbolic inputs in the indexing_map. The index mapping can be
    multi-dimensional, and so the `apply_indexing` operation always returns one
    value. The operands and results must all have ‘index’ type.

    Example:

    ```mlir
    #map = #xla_gpu.indexing_map<(d0, d1)[s0] -> (d0 floordiv 8 + d1 floordiv 128, s0)>
    %results:2 = xla_gpu_ops.apply_indexing #map (%0 in [0, 10], %1 in [0, 11])[%2 in [11, 32]]
    ```
  }];
  let arguments = (ins Variadic<Index>:$operands,
                   XLAGPU_IndexingMapAttr:$indexing_map_attr);
  let results = (outs Variadic<Index>);

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$dims, "mlir::ValueRange":$symbols,
                   "const IndexingMap&":$indexing_map)>,
    OpBuilder<(ins "mlir::ValueRange":$operands,
                   "const IndexingMap&":$indexing_map)>,
    OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::AffineMap":$affine_map,
               "llvm::ArrayRef<IndexingMap::Variable>":$dim_vars,
               "llvm::ArrayRef<IndexingMap::Variable>":$range_vars)>,
  ];
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    xla::gpu::IndexingMap getIndexingMap();
    // Extracts the affine map from the attribute.
    mlir::AffineMap getAffineMap() { return getIndexingMapAttr().getMap(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def LoopOp : XLAGPU_Op<"loop", [
    AttrSizedOperandSegments, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface,
      ["getAsmBlockArgumentNames", "getAsmResultNames"]>,
    SingleBlockImplicitTerminator<"xla::gpu::YieldOp">
  ]> {
  let summary = "Loop nest that iterates over all feasible values of RangeVars.";
  let description = [{

    ```mlir
      #map = #xla_gpu.indexing_map<(d0)[s0, s1] -> (d0 + s0, s1),
        domain:
        d0 in [0, 3],
        s0 in [0, 1024],
        s1 in [0, 32]
      >
      // Initial sum set to 0.
      %sum_0 = arith.constant 0.0 : f32
      %dim = arith.constant 1 : index
      // iter_args binds initial values to the loop's region arguments.
      %sum = xla_gpu.loop (%dim)[%i, %j] -> (%r0, %r1)
          in #map iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = tensor.extract %buffer[%i, %j] : tensor<1024x32xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        xla_gpu.yield %sum_next : f32
      }
    ```
  }];
  let arguments = (ins XLAGPU_IndexingMapAttr:$indexing_map_attr,
                       Variadic<Index>:$dims,
                       Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "IndexingMapAttr":$indexing_map_attr,
      "mlir::ValueRange":$dims, "mlir::ValueRange":$inits,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
           "/*ivs=*/mlir::ValueRange, /*map_results=*/mlir::ValueRange, "
           "/*init=*/mlir::ValueRange)>", "nullptr">)>,
    OpBuilder<(ins "const IndexingMap&":$indexing_map,
      "mlir::ValueRange":$dims, "mlir::ValueRange":$inits,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
           "/*ivs=*/mlir::ValueRange, /*map_results=*/mlir::ValueRange, "
           "/*init=*/mlir::ValueRange)>", "nullptr">)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn = llvm::function_ref<
        void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange,
             mlir::ValueRange, mlir::ValueRange)>;

    // Returns the indexing map constructed from IndexingMapAttr.
    xla::gpu::IndexingMap getIndexingMap();
    int64_t getNumInductionVars() {
      return getBody()->getNumArguments() - getIndexingMapAttr().getNumResults()
        - getNumResults();
    }
    mlir::BlockArgument getInductionVar(int64_t index) {
      return getBody()->getArgument(index);
    }
    mlir::Block::BlockArgListType getInductionVars() {
      return getBody()->getArguments().take_front(getNumInductionVars());
    }
    mlir::Block::BlockArgListType getIndexingMapResults() {
      return getBody()->getArguments().drop_front(getNumInductionVars())
        .drop_back(getNumResults());
    }
    mlir::Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().take_back(getNumResults());
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def XLAGPU_MaterializeOp : XLAGPU_Op<"materialize",
    [AttrSizedOperandSegments, CallOpInterface]> {
  let summary = "Reads a tensor into registers";
  let arguments = (ins Variadic<AnyType>:$input,
                       Variadic<Index>:$indices,
                       FlatSymbolRefAttr:$callee,
                       XLAGPU_IndexingMapAttr:$map);
  let results = (outs XLAGPU_IndexedVectorType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $callee `(` $input `)` `at` $map `(` $indices `)` attr-dict `:` functional-type($input, results)
  }];
  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
    operand_range getArgOperands() {
      return getInput();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputMutable();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

def XLAGPU_InsertOp : XLAGPU_Op<"insert", [TypesMatchWith<
                   "expected result type to match dest type",
                   "result", "dest", "$_self">]> {
  let summary = "Inserts an indexed vector into a tensor";
  let arguments = (ins XLAGPU_IndexedVectorType:$source,
                       Variadic<Index>:$indices,
                       AnyRankedTensor:$dest,
                       XLAGPU_IndexingMapAttr:$map);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `(` $indices `)` `into` $dest `at` $map attr-dict `:` type($source) `->` type($result)
  }];
}

def XLAGPU_ReduceOp : XLAGPU_Op<"reduce", [
    Pure,  CallOpInterface, SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Performs a reduction";
  let description = [{
    The `xla_gpu.reduce` op performs a variadic reduction of the provided
    operands using the list of dimensions and a symbol for a combiner function.

    ```mlir
    func.func @add(%a_acc: f32, %b_acc: i32, %a: f32, %b: i32)
        -> (f32, i32) {
      %0 = arith.addf %a_acc, %a : f32
      %1 = arith.addi %b_acc, %b : i32
      func.return %0, %1 : f32, i32
    }
    %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
      combiner=@add : tensor<16x8x4xf32>, tensor<16x8x4xi32>
        to tensor<8xf32>, tensor<8xi32>
    ```
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyType>:$inits,
    ConfinedAttr<DenseI64ArrayAttr,
      [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions,
    FlatSymbolRefAttr:$combiner);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("combiner", callee.get<mlir::SymbolRefAttr>());
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def XLAGPU_ReindexOp : XLAGPU_Op<"reindex", [
    Pure,
    OptionalTypesMatchWith<
        "scalar padding type matches element type operand",
        "operand", "padding",
        "::llvm::cast<::mlir::ShapedType>($_self).getElementType()">
  ]> {
  let summary = "Reindexes a tensor.";
  let description = [{
    The `xla_gpu.reindex` op models a general index transformation on a tensor.
    Examples are transposes, broadcasts and reshapes. However, any index
    transformation that is expressible as an indexing map is allowed.
  }];
  let arguments = (ins
    AnyRankedTensor:$operand,
    Optional<AnyTypeOf<[AnySignlessInteger, AnyFloat, AnyComplex]>>:$padding,
    XLAGPU_IndexingMapAttr:$map);
  let results = (outs AnyRankedTensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Type":$type, "mlir::Value":$operand,
                   "mlir::Value":$padding,
                   "const IndexingMap&":$indexing_map)>,
  ];

  // TODO(jreiffers): Add a verifier.
  let assemblyFormat = [{
    $operand `at` $map (`default` $padding^)? attr-dict
      `:` type($operand) `->` type($result)
  }];
}

#endif // XLA_SERVICE_GPU_FUSIONS_MLIR_OPS
