/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for ST ops.

#ifndef GML_ST_OPS
#define GML_ST_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_subset_ops.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_extension_ops.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_legacy_ops.td"

def GMLST_MaterializeOp : GMLST_Op<"materialize",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyShaped:$source, AnySubset:$subset);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source `at` $subset attr-dict `:` type($source) `at` type($subset)
  }];
  let hasVerifier = 0;
}

def GMLST_DimOp : GMLST_Op<"dim", []> {
  let summary = [{Returns the size of a tile in a given dimension}];
  let arguments = (ins GMLST_TileType:$tile, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tile `[` $dim `]` attr-dict `:` qualified(type($tile))
  }];
  let hasVerifier = 0;
}

def GMLST_OffsetOp : GMLST_Op<"offset", []> {
  let summary = [{Returns the offset of a tile in a given dimension}];
  let arguments = (ins GMLST_TileType:$tile, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tile `[` $dim `]` attr-dict `:` qualified(type($tile))
  }];
  let hasVerifier = 0;
}

def GMLST_SizeOp : GMLST_Op<"size", []> {
  let summary = [{Returns the size of a tile in a given dimension}];
  let arguments = (ins GMLST_TileType:$tile, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tile `[` $dim `]` attr-dict `:` qualified(type($tile))
  }];
  let hasVerifier = 0;
}

class GMLST_LoopLikeOp<string mnemonic, list<Trait> traits = []>
    : GMLST_Op<mnemonic, !listconcat(traits, [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      RecursiveSideEffects,
      SingleBlockImplicitTerminator<"gml_st::SubsetYieldOp">
    ])> {
  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyShaped>:$outputs);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBounds,
      "ValueRange":$upperBounds, "ValueRange":$steps, "ValueRange":$outputs,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*outputs=*/ValueRange)>", "nullptr">:$bodyBuilderFn)>,
  ];

  code extraBaseClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return step().size(); }

    /// Number of output operands
    unsigned getNumOutputs() { return outputs().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }

    /// Return whether the op has no output tensors.
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0;
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def GMLST_ParallelOp : GMLST_LoopLikeOp<"parallel", []> {
  let summary = "Loop-like operation for parallel loops";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the output tensors or memrefs.

    Tensor-based version:

    The body region of the loop contains subset operations applied to
    every output tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.subset_yield` which yields a tensor into a subset of outs.

    Example:

    ```mlir
    %space = gml_st.space [8, 16] : !gml_st.tile<8x16>

    %result = gml_st.parallel (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4)
        outs(%output: tensor<8x16xf32>) {
      %tile = gml_st.tile %space [%i, %j] [4, 4] [1, 1]
        : ! gml_st.tile<8x16> to !gml_st.tile<4x4>
      %lhs_sub = gml_st.materialize %lhs_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>
      %rhs_sub = gml_st.materialize %rhs_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>
      %out_sub = gml_st.materialize %out_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>

      %result_sub = linalg.generic (%lhs_sub, %rhs_sub, %out_sub) ...

      gml_st.subset_yield %result_sub for %tile
        : tensor<4x4xf32> at !gml_st.tile<4x4>
    }
    ```
  }];
  let extraClassDeclaration = extraBaseClassDeclaration;
}

def GMLST_ForOp : GMLST_LoopLikeOp<"for", []> {
  let summary = "Loop-like operation for sequential loops";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the output tensors or memrefs.

    Tensor-based version:

    The body region of the loop contains subset operations applied to
    every output tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.subset_yield` which yields a tensor into a subset of outs.

    Example:

    ```mlir
    %space = gml_st.space [8, 16] : !gml_st.tile<8x16>

    %result = gml_st.for (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4)
        outs(%out_ = %output: tensor<8x16xf32>)
      %tile = gml_st.tile %in_space [%i, %j] [4, 4] [1, 1]
        : ! gml_st.tile<8x16> to !gml_st.tile<4x4>
      %lhs_sub = gml_st.materialize %lhs_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>
      %rhs_sub = gml_st.materialize %rhs_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>
      %out_sub = gml_st.materialize %out_ at %tile
        : tensor<8x16xf32> at !gml_st.tile<4x4>

      %result_sub = linalg.generic (%lhs_sub, %rhs_sub, %out_sub) ...

      gml_st.subset_yield %result_sub for %tile
        : tensor<4x4xf32> at !gml_st.tile<4x4>
    }
    ```
  }];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    ValueRange getRegionOutputArgs() {
      return getBody()->getArguments().take_back(outputs().size());
    }
  }];
}

def GMLST_SubsetYieldOp : GMLST_Op<"subset_yield", [NoSideEffect, ReturnLike,
      Terminator, SameVariadicOperandSize]> {
  let summary = "Subset Yield operation";
  let description = [{
    `gml_st.subset_yield` is a special terminator operation for
    `gml_st.parallel` or `gml_st.for` body.

    Example:

    ```mlir
    gml_st.subset_yield %result_sub for %tile
      : tensor<4x4xf32> for !gml_st.tile<4x4>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$values,
                       Variadic<AnyType>:$subsets);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `at` $subsets `:` type($values) `at` type($subsets))?";
}

#endif // GML_ST_OPS
