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
include "mlir-hlo/Dialect/gml_st/IR/gml_st_set_ops.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_extension_ops.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_legacy_ops.td"

def GMLST_MaterializeOp : GMLST_Op<"materialize",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyShaped:$source, AnySet:$set);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $source`[` $set `]` attr-dict `:` type($source) `[` type($set) `]`
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
  let hasFolder = 1;
}

def GMLST_SizeOp : GMLST_Op<"size", []> {
  let summary = [{Returns the size of a tile in a given dimension}];
  let arguments = (ins GMLST_TileType:$tile, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tile `[` $dim `]` attr-dict `:` qualified(type($tile))
  }];
  let hasVerifier = 0;
  let hasFolder = 1;
}

def GMLST_StrideOp : GMLST_Op<"stride", []> {
  let summary = [{Returns the stride of a tile in a given dimension}];
  let arguments = (ins GMLST_TileType:$tile, Index:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tile `[` $dim `]` attr-dict `:` qualified(type($tile))
  }];
  let hasVerifier = 0;
  let hasFolder = 1;
}

class GMLST_LoopLikeOp<string mnemonic, list<Trait> traits = []>
    : GMLST_Op<mnemonic, !listconcat(traits, [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      RecursiveSideEffects,
      SingleBlockImplicitTerminator<"gml_st::SetYieldOp">
    ])> {
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  code extraBaseClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return step().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }

    /// Return whether the op has no output tensors.
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0;
    }

    /// Return terminator of the loop body.
    SetYieldOp getTerminator();
  }];

  let hasCustomAssemblyFormat = 1;
}

def GMLST_ParallelOp : GMLST_LoopLikeOp<"parallel", []> {
  let summary = "Loop-like operation for parallel loops";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the output tensors or memrefs.

    Tensor-based version:

    The body region of the loop contains set operations applied to
    every output tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.set_yield` which yields a tensor into a subset of outs.

    Example:

    ```mlir
    %space = gml_st.space [8, 16] : !gml_st.tile<8x16>

    %result = gml_st.parallel (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4) {
      %tile = gml_st.tile %space [%i, %j] [4, 4] [1, 1]
        : ! gml_st.tile<8x16> to !gml_st.tile<4x4>

      %lhs_sub = gml_st.materialize %lhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %rhs_sub = gml_st.materialize %rhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %out_sub = gml_st.materialize %out_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]

      %result_sub = linalg.generic (%lhs_sub, %rhs_sub, %out_sub) ...

      gml_st.set_yield %result_sub into %out[%tile]
        : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]
    }
    ```
  }];
  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step);
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBounds,
      "ValueRange":$upperBounds, "ValueRange":$steps,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange)>",
      "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def GMLST_ForOp : GMLST_LoopLikeOp<"for", []> {
  let summary = "Loop-like operation for sequential loops";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the output tensors or memrefs.

    Tensor-based version:

    The body region of the loop contains set operations applied to
    every output tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.set_yield` which yields a tensor into a subset of outs.

    Example:

    ```mlir
    %space = gml_st.space [8, 16] : !gml_st.tile<8x16>

    %result = gml_st.for (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4)
        outs(%out_ = %output: tensor<8x16xf32>) {
      %tile = gml_st.tile %in_space [%i, %j] [4, 4] [1, 1]
        : ! gml_st.tile<8x16> to !gml_st.tile<4x4>

      %lhs_sub = gml_st.materialize %lhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %rhs_sub = gml_st.materialize %rhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %out_sub = gml_st.materialize %out_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]

      %result_sub = linalg.generic (%lhs_sub, %rhs_sub, %out_sub) ...

      gml_st.set_yield %result_sub into %out_[%tile]
        : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyShaped>:$outputs);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBounds,
      "ValueRange":$upperBounds, "ValueRange":$steps, "ValueRange":$outputs,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*outputs=*/ValueRange)>", "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Number of output operands
    unsigned getNumOutputs() { return outputs().size(); }

    /// Get the region output args.
    Block::BlockArgListType getRegionOutputArgs() {
      return getBody()->getArguments().take_back(getNumOutputs());
    }

    /// Get the region output arg that corresponds to an OpOperand.
    BlockArgument getRegionOutputArgForOpOperand(OpOperand &opOperand) {
      assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
             "expected an output args operand");
      assert(opOperand.getOwner() == getOperation() &&
             "opOperand does not belong to this gml_st::ForOp operation");
      return getBody()->getArgument(opOperand.getOperandNumber() -
                                    getNumControlOperands() + getNumLoops());
    }

    /// Get the OpOperand& that corresponds to a region output arg.
    OpOperand &getOpOperandForRegionOutputArg(BlockArgument bbArg) {
      assert(bbArg.getArgNumber() >= getNumLoops() &&
             "expected a bbArg that is not an induction variable");
      assert(bbArg.getOwner()->getParentOp() == getOperation() &&
             "bbArg does not belong to the gml_st::ForOp body");
      return getOperation()->getOpOperand(
        getNumControlOperands() + bbArg.getArgNumber() - getNumLoops());
    }

    /// Get the OpResult that corresponds to an OpOperand.
    OpResult getResultForOpOperand(OpOperand &opOperand) {
      assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
             "expected an output args operand");
      assert(opOperand.getOwner() == getOperation() &&
             "opOperand does not belong to this gml_st::ForOp operation");
      return getOperation()->getResult(
        opOperand.getOperandNumber() - getNumControlOperands());
    }

    /// Get the OpOperand& that corresponds to an OpResultOpOperand.
    OpOperand &getOpOperandForResult(OpResult opResult) {
      assert(opResult.getDefiningOp() == getOperation() &&
             "opResult does not belong to the gml_st::ForOp operation");
      return getOperation()->getOpOperand(
        getNumControlOperands() + opResult.getResultNumber());
    }
  }];
}

def GMLST_SetYieldOp : GMLST_Op<"set_yield", [NoSideEffect, ReturnLike,
      Terminator, SameVariadicOperandSize]> {
  let summary = "Set yield operation";
  let description = [{
    `gml_st.set_yield` is a special terminator operation for
    `gml_st.parallel` or `gml_st.for` body.

    Example:

    ```mlir
    gml_st.set_yield %result_sub at %tile into %dst
      : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$srcs,
                       Variadic<AnyType>:$dsts,
                       Variadic<AnyType>:$sets);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];


  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{

    unsigned getNumUpdates() { return srcs().size(); }

    // Methods for `srcs` arguments.
    OpOperand* getSrcOperand(unsigned i) {
      return &getOperation()->getOpOperand(3 * i);
    }

    SmallVector<OpOperand*> getSrcOperands() {
      SmallVector<OpOperand*> operands;
      for (unsigned i = 0, e = getNumOperands(); i < e; i += 3)
        operands.push_back(&getOperation()->getOpOperand(i));
      return operands;
    }

    bool isSrcOperand(OpOperand& operand) {
      return operand.getOperandNumber() % 3 == 0;
    }

    // Methods for `dst` arguments.
    OpOperand* getDstOperand(unsigned i) {
      return &getOperation()->getOpOperand(3 * i + 1);
    }

    SmallVector<OpOperand*> getDstOperands() {
      SmallVector<OpOperand*> operands;
      for (unsigned i = 1, e = getNumOperands(); i < e; i += 3)
        operands.push_back(&getOperation()->getOpOperand(i));
      return operands;
    }

    FailureOr<OpResult> getTiedOpResult(OpOperand &opOperand) {
      if (!isDstOperand(opOperand)) return failure();

      auto parent = getOperation()->getBlock()->getParentOp();
      if (isa<ParallelOp>(parent) || isa<ForOp>(parent)) {
        return parent->getResult(0);
      }
      return failure();
    }

    bool isDstOperand(OpOperand& operand) {
      return operand.getOperandNumber() % 3 == 1;
    }

    // Methods for `set` arguments.
    OpOperand* getSetOperand(unsigned i) {
      return &getOperation()->getOpOperand(3 * i + 2);
    }

    SmallVector<OpOperand*> getSetOperands() {
      SmallVector<OpOperand*> operands;
      for (unsigned i = 2, e = getNumOperands(); i < e; i += 3)
        operands.push_back(&getOperation()->getOpOperand(i));
      return operands;
    }

    bool isSetOperand(OpOperand& operand) {
      return operand.getOperandNumber() % 3 == 2;
    }
  }];
}

#endif // GML_ST_OPS
