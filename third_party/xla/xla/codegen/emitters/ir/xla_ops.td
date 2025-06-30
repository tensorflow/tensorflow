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

#ifndef XLA_CODEGEN_EMITTERS_IR_XLA_OPS
#define XLA_CODEGEN_EMITTERS_IR_XLA_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "xla/codegen/emitters/ir/xla_dialect.td"
include "xla/codegen/emitters/ir/xla_attrs.td"

class XLA_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaDialect, mnemonic, traits> {
}

def XLA_AtomicRMWOp : XLA_Op<"atomic_rmw",
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
  let builders = [
    OpBuilder<(ins "mlir::Value":$memref, "mlir::ValueRange":$ivs)>,
    OpBuilder<(ins "mlir::Value":$memref, "mlir::ValueRange":$ivs,
                   "mlir::Type":$block_arg_type)>,
  ];

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
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `[` $indices `]` `:` type($input) $computation attr-dict
  }];
}

def XLA_YieldOp : XLA_Op<"yield", [
    ParentOneOf<["::xla::AtomicRMWOp", "::xla::LoopOp"]>,
    ReturnLike, Terminator]> {
  let summary = "Terminator for atomic_rmw ops.";
  let arguments = (ins Variadic<AnyType>:$result);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

def XLA_PureCallOp : XLA_Op<"pure_call",
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
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
}

def XLA_PredicatedInsertOp : XLA_Op<"predicated_insert",
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


def XLA_PredicatedExtractOp : XLA_Op<"predicated_extract",
    [Pure,
     TypesMatchWith<"fallback type matches element type of src",
                    "src", "fallback",
                    "::llvm::cast<mlir::TensorType>($_self).getElementType()">,
     TypesMatchWith<"result type matches element type of src",
                    "src", "result",
                    "::llvm::cast<mlir::TensorType>($_self).getElementType()">]> {
  let summary = "Extracts a value from a tensor if a condition holds";
  let arguments = (ins I1:$condition, AnyType:$fallback,
    AnyStaticShapeTensor:$src, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $src `[` $indices `]` `if` $condition `else` $fallback attr-dict `:` type($src)
  }];
}

def ApplyIndexingOp : XLA_Op<"apply_indexing", [Pure]> {
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
    #map = #xla.indexing_map<(d0, d1)[s0] -> (d0 floordiv 8 + d1 floordiv 128, s0)>
    %results:2 = xla_ops.apply_indexing #map (%0 in [0, 10], %1 in [0, 11])[%2 in [11, 32]]
    ```
  }];
  let arguments = (ins Variadic<Index>:$operands,
                   XLA_IndexingMapAttr:$indexing_map_attr);
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
    xla::IndexingMap getIndexingMap();
    // Extracts the affine map from the attribute.
    mlir::AffineMap getAffineMap() { return getIndexingMapAttr().getMap(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def LoopOp : XLA_Op<"loop", [
    AttrSizedOperandSegments, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface,
      ["getAsmBlockArgumentNames", "getAsmResultNames"]>,
    SingleBlockImplicitTerminator<"xla::YieldOp">
  ]> {
  let summary = "Loop nest that iterates over all feasible values of RangeVars.";
  let description = [{

    ```mlir
      #map = #xla.indexing_map<(d0)[s0, s1] -> (d0 + s0, s1),
        domain:
        d0 in [0, 3],
        s0 in [0, 1024],
        s1 in [0, 32]
      >
      // Initial sum set to 0.
      %sum_0 = arith.constant 0.0 : f32
      %dim = arith.constant 1 : index
      // iter_args binds initial values to the loop's region arguments.
      %sum = xla.loop (%dim)[%i, %j] -> (%r0, %r1)
          in #map iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = tensor.extract %buffer[%i, %j] : tensor<1024x32xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        xla.yield %sum_next : f32
      }
    ```
  }];
  let arguments = (ins XLA_IndexingMapAttr:$indexing_map_attr,
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
    xla::IndexingMap getIndexingMap();
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

def WorkGroupIdOp : XLA_Op<"workgroup_id", [
    Pure,
  ]> {
  let assemblyFormat = "$dimension attr-dict";
  let arguments =(ins WorkGroupDimensionAttr:$dimension);
  let results = (outs Index);
}


#endif // XLA_CODEGEN_EMITTERS_IR_XLA_OPS

