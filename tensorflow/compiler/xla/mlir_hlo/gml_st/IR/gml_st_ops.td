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

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "gml_st/IR/gml_st_ops_base.td"
include "gml_st/IR/gml_st_legacy_ops.td"

///////////////////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////////////////

// Base class of all subset types.
class GMLST_Set<string name> : TypeDef<GmlSt_Dialect, name> { }

def GMLST_TileType : GMLST_Set<"Tile"> {
  let mnemonic = "tile";

  let summary = "Type that represents a tile in an N-d iteration space.";
  let description = [{
    Values with the tile type represent an N-dimensional rectangle. Unlike a
    TensorType, there is no element type specified for the tile. Each dimension
    may be a static non-negative decimal constant or be dynamically determined
    (indicated by `?`).

    Examples:

    ```mlir
    // Full dynamic dimensions.
    !gml_st.tile<? x ? x f32>

    // Partially dynamic dimensions.
    !gml_st.tile<? x 3 x ? x f32>

    // Full static shape.
    !gml_st.tile<16 x 8 x f32>

    // Tile with rank zero.
    !gml_st.tile<f32>

    // Zero-element dimensions are allowed.
    !gml_st.tile<0 x 42 x f32>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<ShapeTypeDimensionsList>($shape) `>`";

  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumElements() const {
      return ShapedType::getNumElements(getShape());
    }
  }];
}

def AnySet : Type<Or<[GMLST_TileType.predicate]>, "subset type">;

def RankedTensorOrVector : AnyTypeOf<[
  AnyRankedTensor, AnyVectorOfAnyRank
], "", "::mlir::ShapedType">;
def RankedTensorOrVectorOrScalar : AnyTypeOf<[
  AnyRankedTensor, AnyVectorOfAnyRank, AnyFloat, AnyInteger, AnyComplex, Index
]>;

///////////////////////////////////////////////////////////////////////////////
// Ops
///////////////////////////////////////////////////////////////////////////////

def GMLST_TileOp : GMLST_Op<"tile", [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<InferTypeOpInterface>,
      OffsetSizeAndStrideOpInterface,
      Pure
    ]> {
  let summary = "Operation to generate a TileType from offsets, sizes, strides";
  let description = [{
    This is a metadata operation that encapsulates mixed dynamic and static
    offsets, sizes and strides.

    Examples:

    ```mlir
    // Tile with partially dynamic sizes.
    %tile_2d = gml_st.tile [0, 0] [%x, 16] [1, 1] : !gml_st.tile<?x16>

    // Tile with rank zero.
    %tile_0d = gml_st.tile [] [] [] : !gml_st.tile<>
    ```
  }];
  let arguments = (ins Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs GMLST_TileType:$result);

  let builders = [
   OpBuilder<(ins "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
   OpBuilder<(ins "ArrayRef<OpFoldResult>":$offsets,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let assemblyFormat = [{
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getType().cast<TileType>().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 0; }
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def GMLST_MaterializeOp : GMLST_Op<"materialize", [Pure]> {
  let summary = "Operation to extract values from tensor or vector";
  let description = [{
    The "materialize" operation extracts a tensor or a vector from another
    tensor or a vector as specified by the value of a provided set.

    MaterializeOp accepts `!gml_st.tile` type to extract tensors from tensors.

    ```
    %subtensor = gml_st.materialize %tensor[%tile]
      : tensor<?x?xf32>[!gml_st.tile<42x16>] to tensor<42x16xf32>
    ```

    If `!gml_st.tile` has a single element or rank-0, it is also possible to
    extract an element from a tensor.

    ```
    %element = gml_st.materialize %tensor[%tile_with_single_element]
      : tensor<3x1xindex>[!gml_st.tile<1x1>] to index

    %element = gml_st.materialize %tensor[%tile_0d]
      : tensor<f32>[!gml_st.tile<>] to f32

    %subtensor = gml_st.materialize %tensor[%tile_with_single_element]
      : tensor<3x1xindex>[!gml_st.tile<1x1>] to tensor<1x1xindex>
    ```

    MaterializeOp can also extract a vector from a vector.

    ```
    %subvector = gml_st.materialize %vector[%tile]
      : vector<64x32xf32>[!gml_st.tile<42x16>] to vector<42x16xf32>
    ```

    The operation is designed to be polymorphic to support non-rectangular
    subsets. It will accept `set` arguments of types other than `!gml_st.tile`.
  }];
  let arguments = (ins RankedTensorOrVector:$source, AnySet:$set);
  let results = (outs RankedTensorOrVectorOrScalar:$result);

  let builders = [OpBuilder<(ins "Value":$source, "Value":$set)>];

  let assemblyFormat = [{
    $source`[` $set `]` attr-dict `:` type($source) `[` type($set) `]`
      `to` type($result)
  }];

  let hasCanonicalizer = 1;
}

class GMLST_LoopLikeOp<string mnemonic, list<Trait> traits = []>
    : GMLST_Op<mnemonic, !listconcat(traits, [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"gml_st::SetYieldOp">
    ])> {
  let results = (outs Variadic<RankedTensorOrVector>:$results);
  let regions = (region SizedRegion<1>:$region);

  code extraBaseClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return getStep().size(); }

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
    This is a multi-dimensional loop-like operation to support distribution on
    tensors. The loop can have variadic number of results.

    The body region contains exactly one block that terminates with
    `gml_st.set_yield` which specifies how to combine partial results computed
    in every iteration of the loop.

    Example:

    ```mlir
    %add = gml_st.parallel (%i, %j)
        = (%c0, %c0) to (%c8, %c16) step (%c4, %c4) {

      %tile = gml_st.tile [%i, %j] [4, 4] [1, 1]
        : !gml_st.tile<8x16> to !gml_st.tile<4x4>

      %lhs_sub = gml_st.materialize %lhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>] to tensor<4x4xf32>
      %rhs_sub = gml_st.materialize %rhs_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>] to tensor<4x4xf32>
      %out_sub = gml_st.materialize %out_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>] to tensor<4x4xf32>

      %add_sub = linalg.map
        ins(%lhs_sub: tensor<4x4xf32>, %rhs_sub: tensor<4x4xf32>)
        outs(%out_sub: tensor<4x4xf32>)
        (%lhs_elem: f32, %rhs_elem: f32) {
          %0 = arith.addf %lhs_elem, %rhs_elem : f32
          linalg.yield %0 : f32
        }

      gml_st.set_yield %add_sub into %out[%tile]
        : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]
    } : tensor<16x64xf32>
    ```

    The terminator specifies, that the partial result `%add_sub` is to become a
    part of the final result initialized by `%out` at the position defined by
    `%tile`.

    Example with concurrent updates:

    %sum = gml_st.parallel (%i) = (%c0) to (%c8) step (%c1) {
      %tile = gml_st.tile [%i] [1] [1] : !gml_st.tile<1>
      %in_sub = gml_st.materialize %in[%tile]
        : tensor<8xf32>[!gml_st.tile<1>] to tensor<1xf32>

      gml_st.set_yield %in_sub into %out[%tile_0d]
        acc (%new, %old: tensor<1xf32>) {
          %combined = mhlo.add %new, %old : tensor<1xf32>
          gml_st.yield %combined : tensor<1xf32>
        } : tensor<1xf32> into tensor<8xf32>[!gml_st.tile<1>]
    } : tensor<8xf32>

    Every iteration of this loop extracts an element of the input and combines
    it with the overlapping subset of the output. In that case,
    `gml_st.set_yield` has an optional 'accumulator' region, that models the
    concurrent update. The code in the accumulator can be lowered to atomic RMW
    or to some other synchronization primitive.

    After bufferization the loop does not produce any results.

    ```mlir
    gml_st.parallel (%i, %j) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4) {
      %lhs_sub = memref.subview %lhs[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>
      %rhs_sub = memref.subview %rhs[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>
      %out_sub = memref.subview %out[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>

      linalg.map
        ins(%lhs_sub: memref<4x4xf32>, %rhs_sub: memref<4x4xf32>)
        outs(%out_sub: memref<4x4xf32>)
        (%lhs_elem: f32, %rhs_elem: f32) {
          %0 = arith.addf %lhs_elem, %rhs_elem : f32
          linalg.yield %0 : f32
        }

      gml_st.set_yield
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       OptionalAttr<StrAttr>:$distributionType);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBounds,
      "ValueRange":$upperBounds, "ValueRange":$steps,
      CArg<"Optional<StringAttr>", "llvm::None">:$distributionType,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange)>",
      "nullptr">:$bodyBuilderFn)>,
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Return the destinations for a gml_st.parallel op.
    ValueRange getLoopLikeOpInits();
  }];

  let hasCanonicalizer = 1;
}

def GMLST_ForOp : GMLST_LoopLikeOp<"for", []> {
  let summary = "Loop-like operation for sequential loops";
  let description = [{
    This is a multi-dimensional loop-like operation with sequential semantics.
    The arguments also include the loop-carried tensor or vector variables.
    The loop can have variadic number of results.

    The body region contains exactly one block that terminates with
    `gml_st.set_yield` which specifies how to update a subset of the
    loop-carried variable on every iteration.

    Example:

    ```mlir
    %add = gml_st.for (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4)
        outs(%out_ = %out: tensor<8x16xf32>) {
      %tile = gml_st.tile [%i, %j] [4, 4] [1, 1]
        : !gml_st.tile<8x16> to !gml_st.tile<4x4>

      %lhs_sub = gml_st.materialize %lhs[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %rhs_sub = gml_st.materialize %rhs[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]
      %out_sub = gml_st.materialize %out_[%tile]
        : tensor<8x16xf32>[!gml_st.tile<4x4>]

      %add_sub = linalg.map
        ins(%lhs_sub: tensor<4x4xf32>, %rhs_sub: tensor<4x4xf32>)
        outs(%out_sub: tensor<4x4xf32>)
        (%lhs_elem: f32, %rhs_elem: f32) {
          %0 = arith.addf %lhs_elem, %rhs_elem : f32
          linalg.yield %0 : f32
        }

      gml_st.set_yield %add_sub into %out_[%tile]
        : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]
    }
    ```

    After bufferization the `outs` argument list becomes empty and the loop
    does not produce any results.

    ```mlir
    gml_st.for (%i, %j) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4) {
      %lhs_sub = memref.subview %lhs[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>
      %rhs_sub = memref.subview %rhs[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>
      %out_sub = memref.subview %out[%tile]
        : memref<8x16xf32> to memref<4x4xf32, #map>

      linalg.map
        ins(%lhs_sub: memref<4x4xf32>, %rhs_sub: memref<4x4xf32>)
        outs(%out_sub: memref<4x4xf32>)
        (%lhs_elem: f32, %rhs_elem: f32) {
          %0 = arith.addf %lhs_elem, %rhs_elem : f32
          linalg.yield %0 : f32
        }

      gml_st.set_yield
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
    unsigned getNumOutputs() { return getOutputs().size(); }

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

    /// Return the destinations for a gml_st.for op.
    ValueRange getLoopLikeOpInits() {
      return getOutputs();
    }
  }];

  let hasCanonicalizer = 1;
}

def GMLST_SetYieldOp : GMLST_Op<"set_yield", [
      Pure,
      ReturnLike,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"YieldOp">,
      Terminator
    ]> {
  let summary = "Set yield operation";
  let description = [{
    `gml_st.set_yield` is a special terminator operation for `gml_st.parallel`
    or `gml_st.for` body. It specifies how to combine a source tensor, vector
    or scalar with the destination tensor or vector.

    Example:

    ```mlir
    // `src` is a tensor.
    gml_st.set_yield %src at %tile into %dst
      : tensor<4x4xf32> into tensor<16x64xf32>[!gml_st.tile<4x4>]

    // `src` is a scalar.
    gml_st.set_yield %src at %tile into %dst
      : f32 into tensor<16x64xf32>[!gml_st.tile<1x1>]

    // `src` and `dst` are vectors.
    gml_st.set_yield %src at %tile into %dst
      : vector<4x4xf32> into vector<16x64xf32>[!gml_st.tile<4x4>]

    ```

    The operation is designed to be polymorphic to support non-rectangular
    subsets. It will accept `set` arguments of types other than `!gml_st.tile`.
  }];

  let arguments = (ins Variadic<RankedTensorOrVectorOrScalar>:$srcs,
                       Variadic<RankedTensorOrVector>:$dsts,
                       Variadic<AnySet>:$sets,
                       BoolArrayAttr:$accumulatorFlags);
  let regions = (region VariadicRegion<SizedRegion<1>>:$accumulators);

  let builders = [
    OpBuilder<(ins)>,

    // Builder with default update behaviour, i.e. overriding output.
    OpBuilder<(ins "ValueRange":$srcs, "ValueRange":$dsts, "ValueRange":$sets)>,

    // Builder with custom update behaviour.
    OpBuilder<(ins "ValueRange":$srcs, "ValueRange":$dsts, "ValueRange":$sets,
      "ArrayAttr":$accumulatorFlags,
      "ArrayRef<function_ref<void(OpBuilder &, Location, Value, Value)>>"
      :$combiners)>
  ];
  let skipDefaultBuilders = 1;

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    unsigned getNumUpdates() { return getSrcs().size(); }

    // Methods for `dst` arguments.
    OpOperand* getDstOperand(unsigned i) {
      return &getOperation()->getOpOperand(getNumUpdates() + i);
    }

    FailureOr<OpResult> getTiedOpResult(OpOperand &opOperand) {
      if (!isDstOperand(opOperand)) return failure();

      auto parent = getOperation()->getBlock()->getParentOp();
      if (isa<ParallelOp>(parent) || isa<ForOp>(parent)) {
        return parent->getResult(opOperand.getOperandNumber() -
                                 getNumUpdates());
      }
      return failure();
    }

    bool isDstOperand(OpOperand& operand) {
      return operand.getOperandNumber() >= getNumUpdates() &&
             operand.getOperandNumber() < getNumUpdates() * 2;
    }

    unsigned getNumDstOperands() { return getNumOperands() - getNumUpdates(); }
  }];

  let hasCanonicalizer = 1;
}

// TODO(b/253560795): Figure out where this operation shoud live, and how to
// model it properly.
def GMLST_DistributeOp : GMLST_Op<"distribute", [
      Pure,
      AllElementTypesMatch<["source", "result"]>
    ]> {
  let summary = "Tile combining operation";
  let description = [{
    `gml_st.distribute` is in a sense an inverse operation to
    `gml_st.materialize`, and a non-terminator version of `gml_st.set_yield`.
    It takes one tile of a vector per parallel iteration, and returns a larger
    vector composed of these tiles.

    It is only needed as an intermediate step when flattening gml_st.parallel
    operations during SIMTfication for GPU, where one `gml_st.distribute`
    replaces each output of `gml_st.set_yield`. It only works on vectors, since
    tensor arguments have been bufferized into memrefs by that point in the
    pipeline. This is also the reason why it does not need a destination
    argument.
  }];

  let arguments = (ins AnyVectorOfAnyRank:$source, AnySet:$set);
  let results = (outs AnyVectorOfAnyRank:$result);

  let assemblyFormat = [{
    $source `into` `[` $set `]` attr-dict `:` type($source)
      `into` type($result) `[` type($set) `]`
  }];

  let hasVerifier = 0;
}

#endif // GML_ST_OPS
