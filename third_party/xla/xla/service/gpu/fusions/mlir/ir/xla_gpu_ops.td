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
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def XlaGpuDialect : Dialect {
  let name = "xla_gpu";

  let description = [{
    This dialect contains ops required for lowering HLO to LLVM.
  }];

  let cppNamespace = "::xla::gpu";
}

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

  let assemblyFormat = [{
    $input `[` $indices `]` `:` type($input) $computation attr-dict
  }];
}

def XLAGPU_YieldOp : XLAGPU_Op<"yield",
    [HasParent<"::xla::gpu::AtomicRMWOp">, Terminator]> {
  let summary = "Terminator for atomic_rmw ops.";
  let arguments = (ins AnyType:$result);

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
    }]>];
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
    [Pure,
     TypesMatchWith<"result type matches type of operands",
                     "operands", "results", "$_self">]> {
  let summary = "Performs a full warp shuffle and reduces the values";
  let description = [{
    This op performs a full warp shuffle and reduces the results using the given
    function. The function is invoked with the operands from the low lanes,
    followed by the operands from the high lanes. For example:

    ```
    shuffle_reduce @argmax(%value, %idx) : (f32, index)
    ```

    Will perform shuffles with distance 16, 8, 4, 2 and 1, and will invoke
    @argmax five times. The first invocations will be

    ```
    @argmax(%value[i], %idx[i], %value[16+i], %idx[16+i])
    ```
  }];
  let builders = [
    OpBuilder<(ins "mlir::func::FuncOp":$reducer, "mlir::ValueRange":$operands, "int64_t":$max_distance), [{
      $_state.addOperands(operands);
      $_state.addAttribute("reducer", mlir::SymbolRefAttr::get(reducer));
      $_state.addAttribute("max_distance",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(reducer.getContext(), 64),
              max_distance));
      $_state.addTypes(reducer.getFunctionType().getResults());
    }]>];
  let arguments = (ins FlatSymbolRefAttr:$reducer,
                   Variadic<AnyType>:$operands,
                   I64Attr:$max_distance);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $reducer `(` $operands `)` `to` $max_distance attr-dict `:` type($operands)
  }];
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
    The `apply_indexing` operation applies an affine map to a list
    of SSA values, yielding a single SSA value. The number of dimension and
    symbol arguments must be equal to the respective number of dimensional and
    symbolic inputs in the affine map. The affine mapping can be
    multi-dimensional, and so the `apply_indexing` operation always returns one
    value. The  operands and results must all have ‘index’ type.

    Example:

    ```mlir
    #map = affine_map<(d0, d1)[s0] -> (d0 floordiv 8 + d1 floordiv 128, s0)>
    %results:2 = xla_gpu_ops.apply_indexing #map (%0 in [0, 10], %1 in [0, 11])[%2 in [11, 32]]
    ```
  }];
  let arguments = (ins Variadic<Index>:$operands,
                       AffineMapAttr:$map,
                       DenseI64ArrayAttr:$lower_bounds,
                       DenseI64ArrayAttr:$upper_bounds);
  let results = (outs Variadic<Index>);

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$operands,
                   "const IndexingMap&":$indexing_map)>,
    OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::AffineMap":$affine_map,
               "llvm::ArrayRef<DimVar>":$dim_vars,
               "llvm::ArrayRef<RangeVar>":$range_vars)>,
    OpBuilder<(ins "mlir::ValueRange":$operands,
                   "mlir::AffineMap":$affine_map,
                   "llvm::ArrayRef<int64_t>":$lower_bounds,
                   "llvm::ArrayRef<int64_t>":$upper_bounds)>,
  ];
  let extraClassDeclaration = [{
    // Return an indexing map constructed from affine_map and the bounds.
    xla::gpu::IndexingMap getIndexingMap();
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS
