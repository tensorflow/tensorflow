/* Copyright 2023 The JAX Authors.

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

#ifndef TPU_ATTRS
#define TPU_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TPU_Dialect : Dialect {
  let name = "tpu";
  let cppNamespace = "::mlir::tpu";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    static StringRef GetCoreTypeKey() { return "tpu.core_type"; }

    static std::optional<CoreType> GetCoreTypeAttr(Operation *op);
  }];
}

class TPU_Attr<string name, string mnemonic_, list<Trait> traits = []>
    : AttrDef<TPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

// TODO(b/369418606): Find out the way to verify vreg size.
def TPU_Vreg : Type<IsVectorOfNonZeroRankTypePred, "native-sized vreg", "::mlir::VectorType">;

class TPU_Type<string name, string mnemonic_, list<Trait> traits = []>
    : TypeDef<TPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def TPU_CoreType : I32EnumAttr<"CoreType", "Core type", [
  I32EnumAttrCase<"kTc", 0, "tc">,
  I32EnumAttrCase<"kScScalarSubcore", 1, "sc_scalar_subcore">,
  I32EnumAttrCase<"kScVectorSubcore", 2, "sc_vector_subcore">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_CoreTypeEnum : EnumAttr<TPU_Dialect, TPU_CoreType, "core_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_PipelineMode : I32EnumAttr<"PipelineMode", "Pipeline mode", [
  I32EnumAttrCase<"kSynchronous", 1, "synchronous">,
  I32EnumAttrCase<"kDoubleBuffered", 2, "double_buffered">
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PipelineModeEnum : EnumAttr<TPU_Dialect, TPU_PipelineMode, "pipeline_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_SemaphoreType : TPU_Type<"Semaphore", "semaphore", [MemRefElementTypeInterface]>;
def TPU_DMASemaphoreType : TPU_Type<"DMASemaphore", "dma_semaphore", [MemRefElementTypeInterface]>;
def TPU_SomeSemaphoreType : AnyTypeOf<[TPU_SemaphoreType, TPU_DMASemaphoreType]>;

def TPU_DimensionSemantics : I32EnumAttr<"DimensionSemantics", "Dimension semantics", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"arbitrary", 1>,
  I32EnumAttrCase<"core_parallel", 2>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_DimensionSemanticsEnum
    : EnumAttr<TPU_Dialect, TPU_DimensionSemantics, "dimension_semantics"> {
    let assemblyFormat = "`<` $value `>`";
}

// All indices/sizes are in element-space.
// Note that the implementation will require statically provable tile alignment.
def TPU_ElementWindowAttr : TPU_Attr<"ElementWindow", "element_window"> {
  // Including low padding, to avoid backwards-incompatible changes once we add it.
  let parameters = (ins
    ArrayRefParameter<"int64_t", "">:$pad_low,
    ArrayRefParameter<"int64_t", "">:$pad_high
  );
  let assemblyFormat = "`<` `[` $pad_low `]` `,` `[` $pad_high `]` `>`";
}

def TPU_ContractPrecision : I32EnumAttr<"ContractPrecision", "Contraction precision", [
  I32EnumAttrCase<"kBF16", 0, "bf16">,
  I32EnumAttrCase<"kFP32", 1, "fp32">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ContractPrecisionEnum
    : EnumAttr<TPU_Dialect, TPU_ContractPrecision, "contract_precision"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_PackFormat : I32EnumAttr<"PackFormat", "Pack format", [
  I32EnumAttrCase<"kCompressed", 0, "compressed">,
  I32EnumAttrCase<"kInterleaved", 1, "interleaved">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_PackFormatEnum : EnumAttr<TPU_Dialect, TPU_PackFormat, "pack_format"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_TiledCase   : I32EnumAttrCase<"tiled", 0>;
def TPU_LaneCase    : I32EnumAttrCase<"lanes", 1>;
def TPU_SublaneCase : I32EnumAttrCase<"sublanes", 2>;
def TPU_VectorLayoutDim : I32EnumAttr<
  "VectorLayoutDim", "", [TPU_TiledCase, TPU_LaneCase, TPU_SublaneCase]>;

def TPU_VectorLayoutAttr : TPU_Attr<"VectorLayout", "vpad"> {
  let description = [{TODO}];

  let parameters = (ins "Layout":$layout);
  let hasCustomAssemblyFormat = 1;
}

def TPU_TiledLayoutAttr
  : TPU_Attr<"TiledLayout", "tiled",
             [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface>]> {
  let description = [{TODO}];
  let parameters = (ins
    ArrayRefParameter<"::xla::Tile", "">:$tiles,
    ArrayRefParameter<"int64_t", "">:$tile_strides
  );

  let hasCustomAssemblyFormat = 1;
}

def TPU_MemorySpace : I32EnumAttr<"MemorySpace", "Memory space", [
  I32EnumAttrCase<"kAny", 4294967295, "any">,
  // TODO(apaszke): Rename to kXYZ in C++
  I32EnumAttrCase<"vmem", 0, "vmem">,
  I32EnumAttrCase<"smem", 1, "smem">,
  I32EnumAttrCase<"kHbm", 2, "hbm">,
  I32EnumAttrCase<"kCmem", 3, "cmem">,
  I32EnumAttrCase<"kSemaphoreMem", 4, "semaphore_mem">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_MemorySpaceEnum
    : EnumAttr<TPU_Dialect, TPU_MemorySpace, "memory_space"> {
    let assemblyFormat = "`<` $value `>`";
}

class TPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits> {
}

def DefaultMemWrite : MemoryEffects<[MemWrite<DefaultResource>]>;
def DefaultMemRead : MemoryEffects<[MemRead<DefaultResource>]>;

def TPU_ReductionKind : I32EnumAttr<"ReductionKind", "Reduction kind", [
  I32EnumAttrCase<"SUM", 0, "sum">,
  I32EnumAttrCase<"MAX", 1, "max">,
  I32EnumAttrCase<"MIN", 2, "min">
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_ReductionKindAttr
    : EnumAttr<TPU_Dialect, TPU_ReductionKind, "reduction_kind"> {
    let assemblyFormat = "`<` $value `>`";
}

def TPU_AllReduceOp : TPU_Op<"all_reduce", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$input, I64Attr:$dim, TPU_ReductionKindAttr:$kind);
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def TPU_StoreOp : TPU_Op<"store", [DefaultMemWrite, AttrSizedOperandSegments]> {
  let arguments = (ins
    TPU_Vreg:$valueToStore,
    AnyType:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    Optional<AnyType>:$mask,
    OptionalAttr<I32Attr>:$sublane_stride  // In sublane-sized units
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? `sublanes` $sublane_mask  (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($valueToStore) `,` type($mask)
  }];
}

def TPU_LoadOp : TPU_Op<"load", [DefaultMemRead]> {
  let arguments = (ins
    AnyType:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    OptionalAttr<I32Attr>:$sublane_stride  // In sublane-sized units
  );
  let results = (outs TPU_Vreg:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` `sublanes` $sublane_mask (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($result)
  }];
}

// TODO(jevinjiang): migrate tpu.strided_store to general vector store op.
def TPU_VectorStoreOp :TPU_Op<"vector_store", [DefaultMemWrite, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides,
    Optional<AnyVectorOfNonZeroRank>:$mask   // Elementwise mask.
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? attr-dict `:` type($base) `,` type($valueToStore) `,` type($mask)
  }];
  let hasVerifier = 1;
}

// tpu.vector_load loads a vector from memory into a register.
//
//  base   : Memref to load from.
//  indices: Scalar indices into base. indices must be of the same rank as the
//           base memref shape.
//  strides: The stride to use for calculating the address of subsequent
//           elements. If left unspecified, the stride is implicitly 1 along
//           each dimension. Otherwise the stride must match the rank of the
//           memref shape.
//  mask   : Elementwise vector mask. Must be broadcastable to the shape of the
//           result vector. Depending on the core type, this may be a dynamic
//           (lane) mask consumed from a register or a static (sublane) mask
//           that must be the result of arith.constant.
def TPU_VectorLoadOp :TPU_Op<"vector_load", [DefaultMemRead, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides,
    Optional<AnyVectorOfNonZeroRank>:$mask   // Elementwise mask.
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` (`masked` $mask^)? attr-dict `:` type($base) `,` type($result) `,` type($mask)
  }];
  let hasVerifier = 1;
}

def TPU_StridedLoadOp : TPU_Op<"strided_load", [DefaultMemRead]> {
  let arguments = (ins
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) `,` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_StridedStoreOp : TPU_Op<"strided_store", [DefaultMemWrite]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore attr-dict `:` type($base) `,` type($valueToStore)
  }];
  let hasVerifier = 1;
}

def TPU_ShuffledLoadOp : TPU_Op<"shuffled_load", [DefaultMemRead]> {
  let arguments = (ins
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    DenseI32ArrayAttr:$sublane_offsets
  );
  let results = (outs TPU_Vreg:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` type($base) `,` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_ShuffledStoreOp : TPU_Op<"shuffled_store", [DefaultMemWrite]> {
  let arguments = (ins
    TPU_Vreg:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    DenseI32ArrayAttr:$sublane_offsets
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore attr-dict `:` type($base) `,` type($valueToStore)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// TODO(jevinjiang): deprecate to use dynamic_rotate.
def TPU_RotateOp : TPU_Op<"rotate", [Pure, SameOperandsAndResultType]> {
  let description = [{
    Rotates the given vector by the given amount in the given dimension, i.e.,
    for a 2D vector of shape (m, n), rotating dim 0 by `amount` will shift a row
    at index `i` to index `(i + amount) % m`
  }];
  let arguments = (ins
    AnyVectorOfNonZeroRank:$value,
    SI32Attr:$amount,
    SI32Attr:$dimension,
    // When the stride is specified, the rotation amount for each index on the
    // stride dimension will be (amount + stride * index).
    OptionalAttr<SI32Attr>:$stride,
    OptionalAttr<SI32Attr>:$stride_dimension
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $value `by` $amount `dim` $dimension (`stride` $stride `stride_dim` $stride_dimension^)? attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def TPU_DynamicRotateOp : TPU_Op<"dynamic_rotate", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$value,
    I32:$amount,
    SI32Attr:$dimension,
    // When the stride is specified, the rotation amount for each index on the
    // stride dimension will be (amount + stride * index).
    OptionalAttr<SI32Attr>:$stride,
    OptionalAttr<SI32Attr>:$stride_dimension
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $value `by` $amount `dim` $dimension attr-dict `:` type($value) `,` type($amount)  `->` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_IotaOp : TPU_Op<"iota", [Pure]> {
  let arguments = (ins OptionalAttr<I32Attr>:$dimension);
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ attr-dict `:` type($output) }];
}

// TODO(mvoz): deprecated - use concat. Canonicalization will do so automatically.
// b/376295711
def TPU_RepeatOp : TPU_Op<"repeat", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    I32Attr:$dimension,
    I32Attr:$times
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $source `,` $dimension `x` $times attr-dict `:` type($source) `->` type($output) }];
}

def TPU_BroadcastInSublanesOp : TPU_Op<"broadcast_in_sublanes", [Pure]> {
  let description = [{
    For each sublane `i`, broadcasts the value in lane `lane + i` along the entire
    sublane. If `lane + i` is not in [0, lane_count), then the value in sublane `i`
    is not defined (can be anything).
  }];
  let arguments = (ins
    TPU_Vreg:$source,  // All sublanes should be equal.
    I32Attr:$lane  // Coordinates of the first element to take.
  );
  // Output shape should be the same, except for position dim which contains
  // the newly inserted dimension.
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $source `,` $lane attr-dict `:` type($source) `->` type($output)
  }];
}

// Integer unpacks are always signed at the moment.
def TPU_UnpackSubelementsOp : TPU_Op<"unpack_subelements", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    I32Attr:$index,
    TPU_PackFormatEnum:$pack_format
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $source `,` $index attr-dict `:` type($source) `->` type($output) }];
}

// Integer packs are always signed at the moment.
// Float to integer packing rounds to nearest even.
def TPU_PackSubelementsOp : TPU_Op<"pack_subelements", [Pure, SameTypeOperands]> {
  let arguments = (ins
    Variadic<TPU_Vreg>:$sources,
    DenseI32ArrayAttr:$positions,
    TPU_PackFormatEnum:$pack_format
  );
  let results = (outs TPU_Vreg:$output);
  let assemblyFormat = [{ $sources attr-dict `:` type($sources) `->` type($output) }];
  let builders = [
    OpBuilder<(ins "::mlir::VectorType":$output_type, "::mlir::ArrayRef<::mlir::Value>":$padded_sources, "::mlir::tpu::PackFormat":$pack_format)>,
  ];
  let extraClassDeclaration = [{
    static ::mlir::SmallVector<::mlir::Value> getPaddedSources(::mlir::ValueRange sources, ::mlir::ArrayRef<int32_t> positions, int packing_factor);
  }];
  let hasVerifier = 1;
}

def TPU_RelayoutOp : TPU_Op<"relayout", [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_PackMaskOp : TPU_Op<"pack_vmsk", [Pure, SameTypeOperands]> {
  let arguments = (ins
    VectorOfNonZeroRankOf<[I1]>: $low,
    VectorOfNonZeroRankOf<[I1]>: $high
  );
  let results = (outs VectorOfNonZeroRankOf<[I1]>:$output);
  let assemblyFormat = [{ $low `,` $high `,` attr-dict `:` type($low) `,` type($high) `->` type($output) }];
}

def TPU_GatherOp : TPU_Op<"gather", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    DenseI32ArrayAttr:$indices,
    I32Attr:$dimension
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $source `[` $indices `]` `in` $dimension attr-dict
    `:` type($source) `->` type($output)
  }];
}

def TPU_DynamicGatherOp : TPU_Op<"dynamic_gather", [Pure, SameOperandsAndResultShape, AllTypesMatch<["source", "output"]>]> {
  let description = [{
    Gathers elements from `source` using `indices`.

    Given a shape `N0 x N1 x ...`, `output[i0, i1, ...]` is given by
    `input[j0, j1, ...]` where `jn = indices[i0, i1, ...] mod Ni` for
    `n = dimension` and `jn = in` otherwise.

    Similar to `np.take_along_axis`, except that OOB indices wrap.
  }];
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    VectorOfNonZeroRankOf<[AnyInteger]>:$indices,
    I32Attr:$dimension
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $source `[` $indices `]` `in` $dimension attr-dict
    `:` type($source) `,` type($indices) `->` type($output)
  }];
  let hasVerifier = 1;
}

def TPU_RoundingMode : I32EnumAttr<"RoundingMode", "Rounding mode", [
  I32EnumAttrCase<"kTowardsZero", 0, "towards_zero">,
  I32EnumAttrCase<"kToNearestEven", 1, "to_nearest_even">,
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::tpu";
}

def TPU_RoundingModeEnum : EnumAttr<TPU_Dialect, TPU_RoundingMode, "rounding_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

// Internal operation. All arith.fptosi operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_FPToSIOp : TPU_Op<"fptosi", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyVectorOfAnyRank:$input, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnyVectorOfAnyRank:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasCanonicalizeMethod = 1;
}

// Internal operation. All arith.sitofp operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_SIToFPOp : TPU_Op<"sitofp", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyType:$in, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($output) }];
}

def TPU_DotDimensionNumbersAttr : TPU_Attr<"DotDimensionNumbers", "dot_dimension_numbers"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t", "">:$lhs_contracting_dims,
    ArrayRefParameter<"int64_t", "">:$rhs_contracting_dims,
    ArrayRefParameter<"int64_t", "">:$lhs_non_contracting_dims,
    ArrayRefParameter<"int64_t", "">:$rhs_non_contracting_dims,
    // The contract is a flattened structure, wherein, each element is half of a
    // pair of indices. The first element is always 0 (lhs) or 1 (rhs) and the
    // second index is the index from the lhs or rhs.
    ArrayRefParameter<"int64_t", "">:$output_dim_order,
    OptionalArrayRefParameter<"int64_t", "">:$lhs_batch_dims,
    OptionalArrayRefParameter<"int64_t", "">:$rhs_batch_dims
    );
    let assemblyFormat = "`<` `[` $lhs_contracting_dims `]` `,` `[` $rhs_contracting_dims `]` `,` "
                     "`[` $lhs_non_contracting_dims `]` `,` `[` $rhs_non_contracting_dims `]` `,` "
                     "`[` $output_dim_order `]` `,` "
                     "`[` (`]`):($lhs_batch_dims^ `]`)? `,` "
                     "`[` (`]`):($rhs_batch_dims^ `]`)? `>`";
}

// TODO(apaszke): Think hard about precision
def TPU_MatmulOp : TPU_Op<"matmul", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$lhs,
    AnyVectorOfNonZeroRank:$rhs,
    AnyVectorOfNonZeroRank:$acc,
    // These flags are deprecated - if dimension_numbers are defined,
    // these flags are ignored. They will always be false after canonicalize.
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs,
    OptionalAttr<TPU_ContractPrecisionEnum>:$precision,
    // NOTE: User-level optional, once canonicalized, always present.
    OptionalAttr<TPU_DotDimensionNumbersAttr>:$dimension_numbers
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def TPU_ConcatenateOp : TPU_Op<"concatenate", [Pure]> {
  let arguments = (ins
    Variadic<AnyVectorOfNonZeroRank>:$sources,
    I32Attr:$dimension
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $sources `in` $dimension attr-dict `:` type($sources) `->` type($output)
  }];
  let hasVerifier = 1;
}

def TPU_BitcastOp : TPU_Op<"bitcast", [Pure]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$input);
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_BitcastVregOp : TPU_Op<"bitcast_vreg", [Pure]> {
  let arguments = (ins TPU_Vreg:$input);
  let results = (outs TPU_Vreg:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
}

def TPU_WeirdOp : TPU_Op<"weird", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyType:$input);  // F32 vector or scalar
  let results = (outs AnyType:$output);  // I1 vector or scalar
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_ReciprocalOp : TPU_Op<"reciprocal", [Pure, SameOperandsAndResultType, ElementwiseMappable]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$input,
    DefaultValuedAttr<BoolAttr, "false">:$approx
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_RollVectorsOp : TPU_Op<"roll_vectors", [Pure]> {
  let arguments = (ins Variadic<AnyVectorOfNonZeroRank>:$input);
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TPU_UnrollVectorsOp : TPU_Op<"unroll_vectors", [Pure]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$input);
  let results = (outs Variadic<AnyVectorOfNonZeroRank>:$output);
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TPU_CreateMaskOp : TPU_Op<"create_mask", [Pure, SameVariadicOperandSize]> {
  // high is exclusive
  let arguments = (ins Variadic<Index>:$low, Variadic<Index>:$high);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    `[` $low `]``[` $high `]` attr-dict `:` type($output)
  }];
}

def TPU_CreateSubelementMaskOp : TPU_Op<"create_subelement_mask", [Pure]> {
  let summary = "Create a mask masking contiguous rows of subelements.";
  let description = [{
    The "half-sublanes", "quarter-sublanes", etc. (unit is determined by
    the type of `output`) of the mask are masked in the range specified by
    `from` and `to`.

    - If `from <= to`, the range `[from, to)` is set and the rest is unset.
    - If `to <= from`, the range `[to, from)` is unset and the rest is set.

    All lanes are set identically.

    Example:

    ```mlir
    %msk = tpu.create_subelement_mask 3, 9 : vector<8x128x2xi1>
    ```

    This creates a mask `%msk` where, for all `lane`s, `%msk[*][lane][*]` is:

    ```
    [[0, 0], [0, 1], [1, 1], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]
    ```

    It is currently only supported:
    - In TPU v4, for `num_subelems` of 1 and 2.
    - In TPU v5, for `num_subelems` of 1, 2, and 4.
  }];
  let arguments = (ins
    I32Attr:$from,  // inclusive
    I32Attr:$to  // exclusive
  );
  let results = (outs AnyType:$output);  // Verify this is a vmsk with num_subelems
  let assemblyFormat = [{
    $from `,` $to attr-dict `:` type($output)
  }];
}

def TPU_AssumeMultipleOp : TPU_Op<"assume_multiple", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins
    AnyTypeOf<[Index, AnyInteger]>:$value,
    I32Attr:$multiple
  );
  let results = (outs AnyTypeOf<[Index, AnyInteger]>:$result);
  let hasVerifier = 1;
}

def TPU_MemRefSliceOp : TPU_Op<"memref_slice", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$mem_ref,
    Variadic<I32>:$base_idx,
    Variadic<I32>:$dynamic_sizes
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $mem_ref `[` $base_idx `]` (`<` $dynamic_sizes^ `>`)?
    attr-dict `:` type($mem_ref) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_MemRefSqueezeOp : TPU_Op<"memref_squeeze", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_MemRefReshapeOp : TPU_Op<"memref_reshape", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_MemRefBitcastOp : TPU_Op<"memref_bitcast", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_ReinterpretCastOp : TPU_Op<"reinterpret_cast", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_AssumeLayoutOp : TPU_Op<"assume_layout", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def TPU_EraseLayoutOp : TPU_Op<"erase_memref_layout", [Pure]> {
  let arguments = (ins AnyMemRef:$operand);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def TPU_DeviceIdOp : TPU_Op<"device_id", [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def TPU_SemaphoreReadOp : TPU_Op<"sem_read"> {
  let arguments = (ins MemRefOf<[TPU_SemaphoreType, TPU_DMASemaphoreType]>:$semaphore);
  let results = (outs I32:$result);
  let assemblyFormat = [{ $semaphore attr-dict `:` type($semaphore) `->` type($result)}];
}

def TPU_SemaphoreWaitOp : TPU_Op<"sem_wait"> {
  let arguments = (ins
    MemRefOf<[TPU_SemaphoreType]>:$semaphore,
    I32:$amount
  );
  let results = (outs);
  let assemblyFormat = [{ $semaphore `,` $amount attr-dict `:` type($semaphore)}];
  let hasVerifier = 1;
}

def TPU_AllocaSemaphoreOp : TPU_Op<"sem_alloc"> {
  let arguments = (ins);
  let results = (outs MemRefOf<[TPU_SomeSemaphoreType]>:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def TPU_GetBarrierSemaphoreOp : TPU_Op<"sem_barrier"> {
  let arguments = (ins);
  let results = (outs MemRefOf<[TPU_SemaphoreType]>:$semaphore);
  let assemblyFormat = [{ attr-dict `:` type($semaphore) }];
  let hasVerifier = 1;
}

def TPU_SemaphoreSignalOp : TPU_Op<"sem_signal", [AttrSizedOperandSegments]> {
  let arguments = (ins
    MemRefOf<[TPU_SemaphoreType]>:$semaphore,
    I32:$amount,
    Optional<I32>:$device_id, // For remote DMAs
    Optional<I32>:$core_id, // For megacore
    OptionalAttr<TPU_CoreTypeEnum>:$core_type
  );
let assemblyFormat = [{
    $semaphore `,` $amount (`device_id` $device_id^)? (`core_id` $core_id^)? (`core_type` $core_type^)? attr-dict `:` type($semaphore)
  }];
  let hasVerifier = 1;
  let builders = [
    // A backward-compatible builder that sets `core_type` to nullptr.
    OpBuilder<(ins "Value":$semaphore, "Value":$amount,
               "Value":$device_id, "Value":$core_id)>,
  ];
}

def TPU_EnqueueDMAOp : TPU_Op<"enqueue_dma", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$source,
    Optional<MemRefOf<[TPU_DMASemaphoreType]>>:$source_semaphore, // For remote DMAs
    AnyMemRef:$target,
    MemRefOf<[TPU_DMASemaphoreType]>:$target_semaphore,
    Optional<I32>:$device_id, // For remote DMAs
    Optional<I32>:$core_id, // For megacore
    // Smaller number means higher priority. 0 is the highest and the default.
    DefaultValuedAttr<I32Attr, "0">:$priority
  );
  let hasVerifier = 1;
}

def TPU_WaitDMA2Op : TPU_Op<"wait_dma2"> {
  let arguments = (ins
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    AnyMemRef:$src,
    AnyMemRef:$dst
  );
  let hasVerifier = 1;
}

// TODO(mvoz): Remove once a month has passed. b/395630795
def TPU_WaitDMAOp : TPU_Op<"wait_dma"> {
  let arguments = (ins
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    AnyMemRef:$ref
  );
  let hasVerifier = 1;
}

def TPU_RegionOp : TPU_Op<"region", [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"tpu::YieldOp">]> {
  let arguments = (ins);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
  let hasVerifier = 1;
}

def TPU_TraceOp : TPU_Op<"trace", [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"tpu::YieldOp">]> {
  let arguments = (ins StrAttr:$message, I32Attr:$level);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$region);
}

def TPU_TraceStartOp : TPU_Op<"trace_start", []> {
  let arguments = (ins StrAttr:$message, I32Attr:$level);
  let results = (outs);
}

def TPU_TraceStopOp : TPU_Op<"trace_stop", []> {
  let arguments = (ins);
  let results = (outs);
}

def TPU_YieldOp : TPU_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def TPU_DelayOp : TPU_Op<"delay"> {
  let arguments = (ins I32:$nanos);
  let results = (outs);
}

// Expands the granularity of mask to subelements.
def TPU_MaskCastOp : TPU_Op<"mask_cast", [Pure]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$input);
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TPU_GetIterationBoundOp : TPU_Op<"iteration_bound"> {
  let arguments = (ins I32Attr:$dim);
  let results = (outs I32:$result);
  let assemblyFormat = [{ $dim attr-dict `:` type($result) }];
}

def TPU_GetInternalScratchOp : TPU_Op<"internal_scratch">  {
  let arguments = (ins);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def TPU_PRNGSeed32Op : TPU_Op<"prng_set_seed_32"> {
  let arguments = (ins Variadic<I32>:$seeds);
  let results = (outs);
}

def TPU_PRNGRandomBitsOp : TPU_Op<"prng_random_bits"> {
  let arguments = (ins);
  let results = (outs AnyVectorOfNonZeroRank:$output);
}

def TPU_SublaneShuffleOp : TPU_Op<"sublane_shuffle", [SameOperandsAndResultType]> {
  // This op takes 2 physical vregs and a pattern, applies the pattern,
  // and returns the result as 1 vreg.
  //
  // The pattern is a list of integers, where the integer value is the
  // index of the sublane in the *combined input* [lhs, rhs], and the
  // position of the integer in the list is the index of the sublane
  // in the *output* vreg.
  //
  // The pattern size must match the operand/result sublane count.
  //
  // Example:
  //   %0 = tpu.single_output_sublane_shuffle %a, %b,
  //        [0, 1, 2, 3, 4, 5, 6, 7] // Result is %a
  //   %1 = tpu.single_output_sublane_shuffle %a, %b,
  //        [8, 9, 10, 11, 12, 13, 14, 15] // Result is %b
  //   %2 = tpu.single_output_sublane_shuffle %a, %b,
  //        [7, 6, 5, 4, 11, 10, 9, 8] // Result uses high half of a
  //                                  // and low half of b, reversed.
  let arguments = (ins
    TPU_Vreg:$lhs,
    TPU_Vreg:$rhs,
    DenseI32ArrayAttr:$pattern
  );
  let results = (outs TPU_Vreg:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $pattern attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

def TPU_TransposeOp : TPU_Op<"transpose", [Pure]> {
  let summary = "tpu transpose operation";
  let arguments = (ins AnyVectorOfAnyRank:$vector,
                       DenseI64ArrayAttr:$permutation);
  let results = (outs AnyVectorOfAnyRank:$result);

  let builders = [
    OpBuilder<(ins "Value":$vector, "ArrayRef<int64_t>":$permutation)>
  ];
  let assemblyFormat = [{
    $vector `,` $permutation attr-dict `:` type($vector) `->` type($result)
  }];
  let extraClassDeclaration = [{
    VectorType getSourceVectorType() {
      return ::llvm::cast<VectorType>(getVector().getType());
    }
    VectorType getResultVectorType() {
      return ::llvm::cast<VectorType>(getResult().getType());
    }
  }];
  let hasVerifier = 1;
}

def TPU_LogOp : TPU_Op<"log"> {
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    StrAttr:$tag,
    DefaultValuedAttr<BoolAttr, "false">:$formatted
  );
  let results = (outs);
  let assemblyFormat = [{ $tag attr-dict (`:`  `[` $inputs^ `]` `:` type($inputs))? }];
  let hasVerifier = 1;
}

def TPU_LogBufferOp : TPU_Op<"log_buffer"> {
  let arguments = (ins
    AnyMemRef:$input,
    DenseI64ArrayAttr:$shape,
    StrAttr:$tag
  );
  let results = (outs);
  let assemblyFormat = [{ $tag attr-dict `:` $input `:` type($input) }];
  let hasVerifier = 1;
}

def DebugAssertInsertionPass : Pass<"debug-assert-insertion", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createDebugAssertInsertionPass()";
}

def LogicalToPhysicalDeviceIdPass : Pass<"logical-to-physical-device-id", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createLogicalToPhysicalDeviceIdPass(-1)";
  let options = [Option<"total_devices", "total-devices", "int", "", "">];
}

def InferMemRefLayoutPass : Pass<"tpu-infer-memref-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let constructor = "::mlir::tpu::createInferMemRefLayoutPass()";
  let options = [
    // If hardware_generation is not set, the default value of -1 will crash on
    // runOnOperation.
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
    Option<"tpu_tiling_flags", "tpu-tiling-flags", "::mlir::tpu::TpuTilingFlags", /*default=*/"::mlir::tpu::TpuTilingFlags{}", "">,
  ];
}

def CanonicalizeMosaicPass : Pass<"tpu-canonicalize-mosaic", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createCanonicalizeMosaicPass()";
  let options = [
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"compatibility_mode", "compatibility-mode", "bool", /*default=*/"1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
  ];
}

def InferVectorLayoutPass : Pass<"tpu-infer-vector-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createInferVectorLayoutPass()";
  let options = [
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
  ];
}

def RelayoutInsertionPass : Pass<"tpu-relayout-insertion", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createRelayoutInsertionPass()";
  let options = [
    // If hardware_generation is not set, the default value of -1 will crash on
    // runOnOperation.
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
  ];
}

def ApplyVectorLayoutPass : Pass<"tpu-apply-vector-layout", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createApplyVectorLayoutPass()";
  let options = [
    // If hardware_generation is not set, the default value of -1 will crash on
    // runOnOperation.
    Option<"hardware_generation", "hardware-generation", "int", /*default=*/"-1", "">,
    Option<"lane_count", "lane-count", "int", /*default=*/"128", "">,
    Option<"sublane_count", "sublane-count", "int", /*default=*/"8", "">,
    Option<"mxu_contracting_size", "mxu-contracting-size", "int", /*default=*/"128", "">,
    Option<"mxu_noncontracting_size", "mxu-noncontracting-size", "int", /*default=*/"128", "">,
    Option<"max_sublanes_in_scratch", "max-sublanes-in-scratch", "int", /*default=*/"0", "">,
    Option<"vmem_banks", "vmem-banks", "int", /*default=*/"-1", "">,
    Option<"max_shuffle_sublane_offset", "max-shuffle-sublane-offset", "int", /*default=*/"-1", "Max sublane offset per shuffled load/store">,
  ];
}

def LinalgVectorizationPass : Pass<"linalg-vectorization", "::mlir::func::FuncOp"> {
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::tpu::TPUDialect",
  ];
  let constructor = "::mlir::tpu::createLinalgVectorizationPass(false)";
  let options = [
    Option<"supports_bf16_alu_instructions", "supports-bf16-alu-instructions", "bool", "", "">,
    Option<"supports_bf16_matmul", "supports-bf16-matmul", "bool", "", "">,
  ];
}

#endif  // TPU_ATTRS
