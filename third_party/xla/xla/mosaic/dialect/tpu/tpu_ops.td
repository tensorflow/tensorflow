/* Copyright 2026 The OpenXLA Authors.

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

#ifndef TPU_OPS
#define TPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Pass/PassBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "xla/mosaic/dialect/tpu/tpu.td"
include "xla/mosaic/dialect/tpu/tpu_enums.td"
include "xla/mosaic/dialect/tpu/tpu_types.td"

def TPU_Vreg : Type<IsVectorOfNonZeroRankTypePred, "native-sized vreg", "::mlir::VectorType">;

def TPU_Vmask : VectorOfRankAndType<[2, 3], [I1]>;

def TPU_SemaphoreType : TPU_Type<"Semaphore", "semaphore", [MemRefElementTypeInterface]>;
def TPU_DMASemaphoreType : TPU_Type<"DMASemaphore", "dma_semaphore", [MemRefElementTypeInterface]>;
def TPU_SomeSemaphoreType : AnyTypeOf<[TPU_SemaphoreType, TPU_DMASemaphoreType]>;

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

def TPU_VectorLayoutAttr : TPU_Attr<"VectorLayout", "vpad"> {
  let description = [{TODO}];

  let parameters = (ins "Layout":$layout);
  let hasCustomAssemblyFormat = 1;
}

def TPU_TiledLayoutAttr
  : TPU_Attr<"TiledLayout", "tiled",
             [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface, ["verifyLayout"]>]> {
  let description = [{
    This attribute represents tiled layouts in memrefs.

    Multiple levels of tiling are supported with the following restriction:
    - Additional levels of tiling may not add any padding.
    - Additional levels of tiling may not tile previously untiled dimensions,
      that is, they cannot tile across first-level tiles.

    Tile strides encode the stride when moving along a given dimension. They
    must have the same rank as the shape and must be decreasing with increasing
    dimension number. For tiled dimensions, the stride applies only when moving
    across first-level tiles. The strides are in units of the size of the first
    tile, or 1 if there are no tiles.
  }];
  let parameters = (ins
    ArrayRefParameter<"::xla::Tile", "">:$tiles,
    ArrayRefParameter<"int64_t", "">:$tile_strides
  );
  let extraClassDeclaration = [{
    static ::llvm::SmallVector<int64_t> getDefaultTileStrides(::llvm::ArrayRef<::xla::Tile> tiles, ::llvm::ArrayRef<int64_t> shape);
    static TiledLayoutAttr getContiguous(::mlir::MLIRContext* context, ::llvm::ArrayRef<::xla::Tile> tiles, ::llvm::ArrayRef<int64_t> shape);

    // Returns the number of trailing dimensions where tiles can be statically
    // determined to be contiguous (there are no gaps or overlap between tiles).
    // NOTE: Tiles being contiguous does not imply the _data_ is contiguous
    //   since the tiles may contain padding. This means this function is
    //   different from MemRefType::getNumContiguousTrailingDims.
    int64_t getNumTrailingDimsWithContiguousTiles(
        ::llvm::ArrayRef<int64_t> shape) const;
    bool tilesAreKnownContiguous(const ::llvm::ArrayRef<int64_t> shape) const {
      return getNumTrailingDimsWithContiguousTiles(shape) == getRank();
    }

    int64_t getRank() const {
      return getTileStrides().size();
    }
    int64_t getUntiledRank() const;

    ::llvm::SmallVector<int64_t> getExpandedShape(::llvm::ArrayRef<int64_t> shape) const;
    ::llvm::SmallVector<int64_t> getExpandedStrides() const;
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def TPU_MemorySpaceAttr : AttrDef<TPU_Dialect, "MemorySpace"> {
  let mnemonic = "memory_space";
  let cppNamespace = "::mlir::tpu";
  let parameters = (ins
    "::mlir::tpu::MemorySpace":$value,
    OptionalParameter<"::std::optional<::mlir::tpu::CoreType>">:$core_type
  );
  let assemblyFormat = "`<` $value (`,` $core_type^)? `>`";
  let builders = [
    AttrBuilder<(ins "::mlir::tpu::MemorySpace":$value), [{
        return get($_ctxt, value, std::nullopt);
    }]>
  ];
}

class TPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits> {
}

def DefaultMemWrite : MemoryEffects<[MemWrite<DefaultResource>]>;
def DefaultMemRead : MemoryEffects<[MemRead<DefaultResource>]>;

def TPU_AllReduceOp : TPU_Op<"all_reduce", [Pure]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$input, I64Attr:$dim, TPU_ReductionKindAttr:$kind);
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  let hasVerifier = 1;
}

def TPU_ReduceIndexOp : TPU_Op<"reduce_index", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$input,
    I32Attr:$axis,
    TPU_ReductionKindAttr:$kind
  );
  let results = (outs VectorOfNonZeroRankOf<[I32]>:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

// tpu.scan performs a scan across a vector.
//
// If a mask is provided, all output elements before the first unmasked input
// element is undefined. Subsequent masked elements will hold the result
// of the last unmasked element.
//
// For example, a "kSum" reduction over a input vector [1, 2, 3, 4]
// with mask [0, 1, 0, 1] will produce the output vector [X, 2, 2, 6].
// where X is some undefined value.
//
//  output : Result vector. Must have the same shape as source.
//  input  : Vector to scan.
//  kind   : Reduction operator. Must be one of "kSum", "kMax", or "kMin".
//           Must be "kSum" if input is an I1 vector.
//  mask   : Elementwise vector mask. The scan operation starts from the
//           lowest-indexed non-masked vector element (all previous elements
//           have undefined values). Not taken for I1 input vectors.
def TPU_ScanOp : TPU_Op<"scan"> {
  let arguments = (ins
    VectorOfNonZeroRankOf<[I1, I16, I32, BF16, F32]>:$input,
    TPU_ReductionKindAttr:$kind,
    Optional<VectorOfNonZeroRankOf<[I1]>>:$mask
  );
  let results = (outs VectorOfNonZeroRankOf<[I16, I32, BF16, F32]>:$output);
  let assemblyFormat = [{
    $kind `,` $input (`masked` $mask^)? attr-dict `:` type($input) `,` type($mask) `->` type($output)
  }];
  let hasVerifier = 1;
}

def TPU_SortOp : TPU_Op<"sort", [Pure]> {
  let summary = "Sorts key/value pairs based on keys.";
  let description = [{
    tpu.sort performs a stable sort of key/value pairs in ascending or
    descending order based on keys. Masked-out keys and values are placed at the
    end of the output vectors. An output mask indicates which outputs
    correspond to the valid inputs.
  }];
  let arguments = (ins
    VectorOfNonZeroRankOf<[I32,F32]>:$keys,
    VectorOfNonZeroRankOf<[I32,F32]>:$values,
    Optional<VectorOfNonZeroRankOf<[I1]>>:$mask,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs
    VectorOfNonZeroRankOf<[I1]>:$output_mask,
    VectorOfNonZeroRankOf<[I32,F32]>:$sorted_keys,
    VectorOfNonZeroRankOf<[I32,F32]>:$sorted_values
  );
  let assemblyFormat = [{
    $keys `,` $values (`masked` $mask^)? attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TPU_StoreOp : TPU_Op<"store", [DefaultMemWrite, AttrSizedOperandSegments]> {
  let arguments = (ins
    TPU_Vreg:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    Optional<AnyType>:$mask,
    DefaultValuedAttr<I32Attr, "1">:$sublane_stride,  // In sublane-sized units
    DefaultValuedAttr<BoolAttr, "false">:$add
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? `sublanes` $sublane_mask  (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($valueToStore) `,` type($mask)
  }];
}

def TPU_LoadOp : TPU_Op<"load", [DefaultMemRead]> {
  let arguments = (ins
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseBoolArrayAttr:$sublane_mask,
    DefaultValuedAttr<I32Attr, "1">:$sublane_stride  // In sublane-sized units
  );
  let results = (outs TPU_Vreg:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` `sublanes` $sublane_mask (`sublane_stride` $sublane_stride^)? attr-dict `:` type($base) `,` type($result)
  }];
  let description = [{
    Similar to `vector::LoadOp` but with `sublane_mask` and `sublane_stride`.
    When `indices` are negative, it means loading from negative offset
    of `base` address.
  }];
}

// TODO(jevinjiang): migrate tpu.strided_store to general vector store op.
def TPU_VectorStoreOp :TPU_Op<"vector_store", [DefaultMemWrite, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$valueToStore,
    AnyMemRef:$base,
    Variadic<Index>:$indices,
    DenseI32ArrayAttr:$strides,
    Optional<AnyVectorOfNonZeroRank>:$mask,   // Elementwise mask.
    DefaultValuedAttr<BoolAttr, "false">:$add
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? attr-dict `:` type($base) `,` type($valueToStore) `,` type($mask)
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$valueToStore, "::mlir::Value":$base, "::mlir::ValueRange":$indices, CArg<"::mlir::Value", "nullptr">:$mask, CArg<"bool", "false">:$add)>
  ];
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
  let builders = [
    OpBuilder<(ins "::mlir::Type":$result_type, "::mlir::Value":$base, "::mlir::ValueRange":$indices, CArg<"::mlir::Value", "nullptr">:$mask)>
  ];
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

// TODO: b/435258666 - Merge with tpu.vector_load_idx.
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

// TODO: b/435258666 - Merge with tpu.vector_store_idx.
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

// tpu.vector_load_idx loads values from arbitrary locations in memory.
//
// Each element in the output vector is loaded from an index in the base memref
// specified by the corresponding elements in the 'indices' vectors. The shape
// of each index vector must match the shape of the output vector. The number
// of index vectors must equal the rank of the base memref.
//
// For example, for a vector of length n with rank 2, the indices will look like:
//   indices = [[idx0, idx1, ...], [idxn, idxn+1, ...]]
// where [idx0, idxn] is the offset of the first vector element.
//
//  base        : Memref specifying the base address.
//  indices     : Vectors of indices for each dimension of the base memref.
//  mask        : Optional elementwise vector mask.
def TPU_VectorLoadIdxOp :TPU_Op<"vector_load_idx", [DefaultMemRead, AttrSizedOperandSegments]> {
  let arguments = (ins
    MemRefOf<[I32, F32]>:$base,
    Variadic<VectorOfNonZeroRankOf<[I32]>>:$indices,
    Optional<VectorOfNonZeroRankOf<[I1]>>:$mask
  );
  let results = (outs VectorOfNonZeroRankOf<[I32, F32]>:$value);
  let assemblyFormat = [{
    $base `[` $indices `]` (`masked` $mask^)? attr-dict `:` type($base) `[` type($indices) `]` `,` type($value) `,` type($mask)
  }];
  let hasVerifier = 1;
}

// tpu.vector_store_idx stores values to arbitrary locations in memory.
//
// Each element in the input vector is stored to an index in the base memref
// specified by the corresponding elements in the 'indices' vectors. The shape
// of each index vector must match the shape of the input vector. The number
// of index vectors must equal the rank of the base memref.
//
// For example, for a vector of length n with rank 2, the indices will look like:
//   indices = [[idx0, idx1, ...], [idxn, idxn+1, ...]]
// where [idx0, idxn] is the offset of the first vector element.
//
// When multiple vector elements have the same index to store to, the data from
// the highest lane will be the one stored. If add is true, then the data will
// be added from the lowest lane to the highest lane.
//
//  valueToStore: Vector to be stored.
//  base        : Memref specifying the base address.
//  indices     : Vectors of indices for each dimension of the base memref.
//  mask        : Optional elementwise vector mask.
//  add         : If true, add source values to target values. Otherwise, overwrite.
def TPU_VectorStoreIdxOp :TPU_Op<"vector_store_idx", [DefaultMemWrite, AttrSizedOperandSegments]> {
  let arguments = (ins
    VectorOfNonZeroRankOf<[I32, F32]>:$valueToStore,
    MemRefOf<[I32, F32]>:$base,
    Variadic<VectorOfNonZeroRankOf<[I32]>>:$indices,
    Optional<VectorOfNonZeroRankOf<[I1]>>:$mask,
    DefaultValuedAttr<BoolAttr, "false">:$add
  );
  let results = (outs);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $valueToStore (`masked` $mask^)? attr-dict `:` type($base) `[` type($indices) `]` `,` type($valueToStore) `,` type($mask)
  }];
  let hasVerifier = 1;
}

def TPU_DynamicRotateOp : TPU_Op<"dynamic_rotate", [Pure, AllTypesMatch<["value", "result"]>]> {
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

def TPU_ScanCountOp : TPU_Op<"scan_count", [Pure, InferTypeOpAdaptor, SameOperandsAndResultShape]> {
let summary = [{
    ScanCountOp calculates the running duplicate occurrence count of the elements
    in the input vector. Elements eligible for counting are specified by the
    input mask vector. The output mask vector indicates one unique occurrence
    per duplicate that was counted.
  }];

  let description = [{
    ScanCountOp calculates the running duplicate occurrence count of the elements
    in the input vector, %values. The output vector, %counts, contains the running
    duplicate occurrence count for the corresponding element in
    the input vector, where the count is performed in ascending order of element
    indices. For example, if the elements of %values at indices 0, 5, and 7 had
    duplicate values, then the elements of %counts at indices 0, 5, and 7 would
    be 1, 2, and 3, respectively.

    A mask vector, %in_mask, specifies which of the elements in the input vector
    are eligible for counting. An element in %values that has its mask set to 0
    will always have a count of 1 in %counts, regardless of the position in the
    vector, or whether there were duplicates or not.
  }];

  let arguments = (ins
    VectorOfNonZeroRankOf<[I1]>:$in_mask,
    AnyVectorOfNonZeroRank:$values
  );
  let results = (outs
    VectorOfNonZeroRankOf<[I1]>:$out_mask,
    VectorOfNonZeroRankOf<[I32]>:$counts
  );

  let assemblyFormat = [{
    `mask` `(` $in_mask `:` type($in_mask) `)`
    `value` `(` $values `:` type($values) `)`
    attr-dict `:` type(results)
  }];

}

def TPU_IotaOp : TPU_Op<"iota", [Pure]> {
  let description = [{
    Creates a vector that with values that start at 0 and increase along a
    dimension resulting from collapsing the given `dimensions` together in
    row-major order.

    Example:
    ```
    tpu.iota {dimensions = array<i32: 2, 0>} : vector<4x3x2xi16>
    ```
    This produces a vector with the following values:
    ```
    [[[0, 4], [0, 4], [0, 4]]
     [[1, 5], [1, 5], [1, 5]]
     [[2, 6], [2, 6], [2, 6]]
     [[3, 7], [3, 7], [3, 7]]]
    ```
  }];
  let arguments = (ins DenseI32ArrayAttr:$dimensions);
  let results = (outs VectorOfNonZeroRankOf<[AnyInteger, Index]>:$output);
  let assemblyFormat = [{ attr-dict `:` type($output) }];
  let hasVerifier = 1;
}

def TPU_ReshapeOp : TPU_Op<"reshape", [Pure]> {
  let arguments = (ins AnyVectorOfNonZeroRank:$source);
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{ $source attr-dict `:` type($source) `->` type($result) }];
  let hasVerifier = 1;
  let hasFolder = 1;
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
    For each sublane `i`, broadcasts the value in lane `lane + i` along the
    entire sublane. For packed type, imagine the data is compressed unpacked
    along sublane dimension, and the sublane count is multiplied by the packing
    factor.
    For example, for i16 with sublane count 8, `i` above is in [0, 8 * 2).
    If `lane + i` is not in [0, lane_count), then the value in sublane `i` is
    not defined (can be anything).
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

// `unsigned_integers` controls the signedness interpretation of the source
// integer type. It applies both to integer-to-float conversions and to
// integer-to-integer upcasts (selecting sign extension vs zero extension
// when `integer_extended` is true).
//
// When unpacking integers to integers, setting `integer_extended` to false will
// leave bits higher than source bitwidth as undefined.
//
// Take int4 to int16 interleaved unpacking and `index = 1` as an example:
//
// Source:
//
// Bits    28  24  20  16  12   8   4   0
//       --------abcd------------efgh----
//
// where "a" and "e" are the sign bits of the values to be unpacked, and "-" are
// bits to be ignored.
//
// Unpacked, integer_extended = true, unsigned_integers = false (sign extend):
//
// Bits    28  24  20  16  12   8   4   0
//       aaaaaaaaaaaaabcdeeeeeeeeeeeeefgh
//
// Unpacked, integer_extended = true, unsigned_integers = true (zero extend):
//
// Bits    28  24  20  16  12   8   4   0
//       0000000000000bcd00000000000000gh
//
// Unpacked, integer_extended = false:
//
// Bits    28  24  20  16  12   8   4   0
//       ------------abcd------------efgh
def TPU_UnpackSubelementsOp : TPU_Op<"unpack_subelements", [Pure]> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    I32Attr:$index,
    TPU_PackFormatEnum:$pack_format,
    DefaultValuedAttr<BoolAttr, "true">:$integer_extended,
    DefaultValuedAttr<BoolAttr, "false">:$unsigned_integers
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $source `,` $index attr-dict `:` type($source) `->` type($output) }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// unsigned_integers applies to the result type (for the source type the sign
// doesn't matter: integer-to-integer conversions truncate the high bits,
// floats are always signed).
// Float to integer packing rounds to nearest even.
// WARNING: pack(pack(a, b), pack(c, d)) == pack(a, b, c, d) only holds for
// compressed packing!
// Below, we use [ ... ] to denote the bounds of the vreg and use regular parens
// ( ... ) to denote packing of multiple subelements into a single 32-bit word.
//
//                          Interleaved packing
//
// Interleaved packing downcasts each value and interleaves the subelement
// dimension of the sources (below, the subelement dimension is made explicit
// for unpacked types).
//
// result[..., result_subelem]
//   = source[result_subelem % num_sources][..., result_subelem / num_sources]
//
// Take 16-bit vregs A, B, C and D:
///
//   [ (A000 A001) (A010 A011) ... ]
//   [ (A100 A101) (A110 A111) ... ]
//   ...
//
// An interleaved pack(a, b) from 16-bit values produces:
//
//   [ (A000 B000 A001 B001) (A010 B010 A011 B011) ...]
//   ...
//
// While an interleaved pack(a, b, c, d) produces the following subelements in
// each vreg word:
//
//   [ (A000 B000 C000 D000 A001 B001 C001 D001) ... ]
//   ...
//
//                          Compressed packing
//
// Compressed packing downcasts each value and concatenates the combined sublane
// and subelement dimensions (the "row" dimension) of the sources (below, the
// subelement dimension is made explicit for unpacked types).
//
// result[result_sublane, ..., result_subelem]
//   = sources[source_idx][source_sublane, ..., source_subelem]
//
//  where:
//
// - source_subelem = source_row % source_packing
// - source_sublane = source_row / source_packing
// - source_row = result_row % source_rows
// - source_idx = result_row / source_rows
// - result_row = result_sublane * result_packing + result_subelem
// - source_rows = sublane_count * source_packing
//
// A compressed pack(a, b) from 16-bit values produces:
//
//  [ (A000 A001 A100 A101) (A010 A011 A110 A111) ... ]
//  [ (A200 A201 A300 A301) (A210 A211 A310 A311) ... ]
//  ... # 2 more sublanes
//  [ (B000 B001 B100 B101) (B010 B011 B110 B111) ... ]
//  [ (B200 B201 B300 B301) (B210 B211 B310 B311) ... ]
//  ...
def TPU_PackSubelementsOp : TPU_Op<"pack_subelements", [Pure, SameTypeOperands]> {
  let arguments = (ins
    Variadic<TPU_Vreg>:$sources,
    DenseI32ArrayAttr:$positions,
    TPU_PackFormatEnum:$pack_format,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unsigned_integers
  );
  let results = (outs TPU_Vreg:$output);
  let assemblyFormat = [{ $sources attr-dict `:` type($sources) `->` type($output) }];
  let builders = [
    OpBuilder<(ins "::mlir::VectorType":$output_type, "::mlir::ArrayRef<::mlir::Value>":$padded_sources, "::mlir::tpu::PackFormat":$pack_format)>,
    OpBuilder<(ins "::mlir::VectorType":$output_type, "::mlir::ArrayRef<::mlir::Value>":$padded_sources, "::mlir::tpu::PackFormat":$pack_format, "bool":$unsigned_integers)>,
  ];
  let hasVerifier = 1;
}

def TPU_PackElementwiseOp : TPU_Op<"pack_elementwise", [Pure, SameTypeOperands, ElementwiseMappable]> {
  let description = [{
    Packs multiple `sources` elementwise into a single vector of a narrower `target_type`.

    The number of `sources` must equal the packing factor, which is the ratio of
    the element bitwidth of the `sources` to the element bitwidth of the
    `target_type`. Elements from the `sources` are interleaved and packed into
    the `output`, ordered from lowest to highest bits, corresponding to their
    order in the `sources`.  The `output` is then bitcasted to the signless
    integer type of the same bitwidth as the `sources`.

    Note that for integer packing, the bits in `sources` that exceed the
    bitwidth of the `target_type` are just truncated.
    For example, given the `sources` are int8 xxxx'1001 and yyyy'0011,
    `target_type` is int4, the output will be 0011'1001.
  }];
  let arguments = (ins
    Variadic<VectorOfNonZeroRankOf<[F32, AnyInteger]>>:$sources,
    TypeAttr:$target_type
  );
  let results = (outs VectorOfNonZeroRankOf<[AnyInteger]>:$output);
  let assemblyFormat = [{ $sources attr-dict `:` type($sources) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_UnpackElementwiseOp : TPU_Op<"unpack_elementwise", [Pure, ElementwiseMappable]> {
  let description = [{
    Unpacks a single vector from `source`, which contains multiple `source_type`
    vectors packed elementwise.

    The `index` selects which packed value to extract from each word of `source`.
    An `index` of 0 corresponds to the lowest bits. The extracted values are
    cast to the output element type.

    NOTE: The op follows the *interleaved format* during unpacking.
    See `TPU_UnpackSubelementsOp` for more details.

    For example, if `source_type` is i4, result type is i8, and `source` is
    packed i8 with x'y'z'w'm'n'i'j' in a word, where each character represents 4
    bits:

    When `index=0`, the result is a packed i8 with s_y'y's_w'w's_n'n's_j'j';
    When `index=1`, the result is a packed i8 with s_x'x's_z'z's_m'm's_i'i'.
    With `s_x` indicating the MSB of `x`, and so on.
  }];
  let arguments = (ins
    VectorOfNonZeroRankOf<[AnyInteger]>:$source,
    TypeAttr:$source_type,
    I32Attr:$index
  );
  let results = (outs VectorOfNonZeroRankOf<[F32, AnyInteger]>:$output);
  let assemblyFormat = [{ $source `,` $index attr-dict `:` type($source) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_RelayoutOp : TPU_Op<"relayout", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyVectorOfAnyRank:$input);
  let results = (outs AnyVectorOfAnyRank:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_PackMaskOp : TPU_Op<"pack_vmsk", [Pure, SameTypeOperands]> {
  let description = [{
    Packs multiple vmasks into a single vmask with more elements in each word.
  }];

  let arguments = (ins
    Variadic<TPU_Vmask>:$sources,
    DenseI32ArrayAttr:$positions
  );
  let results = (outs TPU_Vmask:$output);
  let assemblyFormat = [{ $sources attr-dict `:` type($sources) `->` type($output) }];
  let builders = [
    OpBuilder<(ins "::mlir::VectorType":$output_type, "::mlir::ArrayRef<::mlir::Value>":$padded_sources)>,
  ];
  let hasVerifier = 1;
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

def TPU_DynamicGatherOp : TPU_Op<"dynamic_gather", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>, AllShapesMatch<["indices", "output"]>, AllElementTypesMatch<["source", "output"]>]> {
  let description = [{
    Gathers elements from `source` using `indices`.

    The specified `dimensions` of `source` are collapsed together and indexed by
    `indices`.

    Given a shape `N0 x N1 x ...`,  the `output[i0, i1, ...]` is given by
    `collapsed_source[j0, j1, ..., indices[i0, i1, ...] mod M]` where
    - `collapsed_source` is the result of collapsing `dimensions` of `source`
      into a new trailing dimension of size `M`.
    - `jk` is the subsequence of `in` for `n` not in `dimensions`.

    When a single dimension is specified, this is similar to
    `np.take_along_axis`.
  }];
  let arguments = (ins
    AnyVectorOfNonZeroRank:$source,
    VectorOfNonZeroRankOf<[AnyInteger]>:$indices,
    DenseI32ArrayAttr:$dimensions
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{
    $source `[` $indices `]` `in` $dimensions attr-dict
    `:` type($source) `,` type($indices) `->` type($output)
  }];
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def AnyFloatLike : AnyTypeOf<[AnyFloat, VectorOfAnyRankOf<[AnyFloat]>]>;
def AnySignlessIntegerLike : AnyTypeOf<[AnySignlessInteger, VectorOfAnyRankOf<[AnySignlessInteger]>]>;

// Internal operation. All arith.fptosi operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_FPToSIOp : TPU_Op<"fptosi", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyFloatLike:$in, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnySignlessIntegerLike:$output);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($output) }];
  let hasCanonicalizeMethod = 1;
}

// Internal operation. All arith.fptoui operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_FPToUIOp : TPU_Op<"fptoui", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyFloatLike:$in, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnySignlessIntegerLike:$output);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($output) }];
}

// Internal operation. All arith.sitofp operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_SIToFPOp : TPU_Op<"sitofp", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnySignlessIntegerLike:$in, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnyFloatLike:$output);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($output) }];
}

// Internal operation. All arith.uitofp operations that change the bitwidth
// must be canonicalized to this operation.
def TPU_UIToFPOp : TPU_Op<"uitofp", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnySignlessIntegerLike:$in, TPU_RoundingModeEnum:$rounding_mode);
  let results = (outs AnyFloatLike:$output);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($output) }];
}

// Internal operation.
def TPU_ExtFOp : TPU_Op<"extf", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyFloatLike:$in);
  let results = (outs AnyFloatLike:$out);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($out) }];
  let hasFolder = 1;
}

// Internal operation.
def TPU_TruncFOp : TPU_Op<"truncf", [Pure, ElementwiseMappable]> {
  let arguments = (
    ins AnyFloatLike:$in,
    TPU_RoundingModeEnum:$rounding_mode
  );
  let results = (outs AnyFloatLike:$out);
  let assemblyFormat = [{ $in attr-dict `:` type($in) `->` type($out) }];
  let hasFolder = 1;
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
    OptionalAttr<TPU_DotDimensionNumbersAttr>:$dimension_numbers,
    // Hint to compilers whether to fuse the transposed LHS in the matmul.
    BoolAttr:$transpose_lhs_hint
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def TPU_MatmulPushRhsOp : TPU_Op<"matmul_push_rhs"> {
  let arguments = (ins
    AnyVectorOfNonZeroRank:$rhs,
    I32Attr:$mxu_index,
    DefaultValuedAttr<I32Attr, "0">:$staging_register,
    DefaultValuedAttr<BoolAttr, "false">:$transpose
  );
  let results = (outs);
  let assemblyFormat = [{
    $rhs attr-dict `:` type($rhs)
  }];
}

def TPU_MatmulAccLhsOp : TPU_Op<"matmul_acc_lhs"> {
  let arguments = (ins
    I32Attr:$acc,
    AnyVectorOfNonZeroRank:$lhs,
    I32Attr:$mxu_index,
    OptionalAttr<I32Attr>:$load_staged_rhs
  );
  let results = (outs);
  let assemblyFormat = [{
    $acc `,` $lhs attr-dict `:` type($lhs)
  }];
}

def TPU_MatmulPopOp : TPU_Op<"matmul_pop"> {
  let arguments = (ins
    I32Attr:$acc,
    I32Attr:$mxu_index
  );
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $acc attr-dict `:` type($result)
  }];
}

def TPU_ConcatenateOp : TPU_Op<"concatenate", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
  let hasFolder = 1;
}

def TPU_WeirdOp : TPU_Op<"weird", [Pure, ElementwiseMappable]> {
  let arguments = (ins AnyType:$input);  // F32 vector or scalar
  let results = (outs AnyType:$output);  // I1 vector or scalar
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_ReciprocalOp : TPU_Op<"reciprocal", [Pure, SameOperandsAndResultType, ElementwiseMappable]> {
  let summary = "Computes the reciprocal of the input.";
  let description = [{
    Computes the reciprocal of the input.
    `approx` controls whether to use an approximate reciprocal.
    `full_range` controls whether the input and result are handled correctly for
    all possible values. Otherwise, edge cases may be handled non-IEEE compliant
    ways. When `approx` is False, `full_range` has no effect as every
    computation is exact.
  }];
  let arguments = (ins
    AnyVectorOfNonZeroRank:$input,
    DefaultValuedAttr<BoolAttr, "false">:$approx,
    DefaultValuedAttr<BoolAttr, "true">:$full_range
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $input attr-dict `:` type($input) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_StochasticConvertOp : TPU_Op<"stochastic_convert", [Pure, SameOperandsAndResultShape]> {
  let arguments = (ins
    VectorOfNonZeroRankOf<[F32]>:$input,
    VectorOfNonZeroRankOf<[I32]>:$random
  );
  let results = (outs AnyVectorOfNonZeroRank:$output);
  let assemblyFormat = [{ $input `,` $random attr-dict `:` type($input) `,` type($random) `->` type($output) }];
}

def TPU_StochasticConvertElementwiseOp : TPU_Op<"stochastic_convert_elementwise", [Pure, ElementwiseMappable]> {
  // Stochastically converts the input to the target dtype based on the mode.
  // When the target dtype is less than 32 bits, the result occupies the lowest {bitwidth} bits in the I32 output.
  let arguments = (ins
    VectorOfNonZeroRankOf<[F32]>:$input,
    VectorOfNonZeroRankOf<[I32]>:$random,
    TypeAttr:$dst_type
  );
  let results = (outs VectorOfNonZeroRankOf<[I32]>:$output);
  let assemblyFormat = [{ $input `,` $random attr-dict `:` type($input) `,` type($random) `->` type($output) }];
  let hasVerifier = 1;
}

def TPU_RollVectorsOp : TPU_Op<"roll_vectors", [Pure]> {
  let arguments = (ins Variadic<AnyVectorOfAnyRank>:$input);
  let results = (outs AnyVectorOfAnyRank:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def TPU_UnrollVectorsOp : TPU_Op<"unroll_vectors", [Pure]> {
  let arguments = (ins AnyVectorOfAnyRank:$input);
  let results = (outs Variadic<AnyVectorOfAnyRank>:$output);
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
  let summary = "Assumes that a value is a multiple of a given integer.";
  let description = [{
    This operation is a hint to the compiler that the input `value` is guaranteed
    to be a multiple of `multiple`. This can be used to satisfy divisibility checks
    in some compiler passes.

    The result is the same as the input `value`.

    Example:

    ```mlir
    %val = tpu.assume_multiple %arg0, 16 : index
    ```
  }];
  let arguments = (ins
    AnyTypeOf<[Index, AnyInteger]>:$value,
    I32Attr:$multiple
  );
  let results = (outs AnyTypeOf<[Index, AnyInteger]>:$result);
  let assemblyFormat = [{$value `,` $multiple attr-dict `:` type($result)}];
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
  let extraClassDeclaration = [{
    mlir::InFlightDiagnostic verifyOffsetAndSizeTileAlignment(
        std::array<int64_t, 2> tc_target_shape,
        MemRefType source_type_override = nullptr);
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def TPU_MemRefSqueezeOp : TPU_Op<"memref_squeeze", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let extraClassDeclaration = [{
    mlir::InFlightDiagnostic verifyTiling();
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TPU_MemRefReshapeOp : TPU_Op<"memref_reshape", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let extraClassDeclaration = [{
    mlir::InFlightDiagnostic verifyTiling();
  }];
  let hasVerifier = 1;
}

def TPU_MemRefBitcastOp : TPU_Op<"memref_bitcast", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let extraClassDeclaration = [{
    mlir::InFlightDiagnostic verifyTiling();
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def TPU_ReinterpretCastOp : TPU_Op<"reinterpret_cast", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$input,
    // Dynamic offset shifts the memref pointer by the specified number of
    // elements of the output type.
    // NOTE: The size of the shift must be a multiple of the memory space word
    // size for lowering to succeed.
    Optional<I32>:$dynamic_offset,
    Variadic<I32>:$dynamic_sizes
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $input ($dynamic_offset^)? (`<` $dynamic_sizes^ `>`)? attr-dict `:` type($input) `->` type($result)
  }];
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$result_type, "::mlir::Value":$input, CArg<"::mlir::Value", "nullptr">:$dynamic_offset, CArg<"::mlir::ValueRange", "{}">:$dynamic_sizes)>
  ];
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
  let description = [{
    Erases the layout attribute and, optionally, the memory space from a memref.
    The resulting memref is otherwise identical to the input.

    This is intended to exist only temporarily during tiling and memory space
    propagation. Accesses to the result memref are expected to behave
    identically to the operand memref, i.e. operand[indices] == result[indices]
    should always hold. This clearly may not be possible if the layouts are
    different, so the existence of this op should be limited to tiling and
    memory space propagation.
  }];
  let arguments = (ins AnyMemRef:$operand);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Returns the ID of the current device.
//
// On the input to the compiler the return value is a logical ID in the XLA
// device assignment. It changes to a physical ID after the
// logical-to-physical-device-id pass.
def TPU_DeviceIdOp : TPU_Op<"device_id", [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

// Returns the logical ID of the current core.
// Querying Core ID is not supported in kernels that use dimension semantics.
def TPU_CoreIdOp : TPU_Op<"core_id", [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
  let hasVerifier = 1;
}

// Returns the logical ID of the current subcore.
// Querying Subcore ID is not supported in kernels that use dimension semantics.
def TPU_SubcoreIdOp : TPU_Op<"subcore_id", [Pure]> {
  let arguments = (ins);
  let results = (outs I32:$result);
  let assemblyFormat = [{ attr-dict `:` type($result) }];
  let hasVerifier = 1;
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
    Optional<I32>:$core_id,
    Optional<I32>:$subcore_id
  );
let assemblyFormat = [{
    $semaphore `,` $amount (`device_id` $device_id^)? (`core_id` $core_id^)? (`subcore_id` $subcore_id^)? attr-dict `:` type($semaphore)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    mlir::tpu::CoreType getTargetCoreType();
  }];
  let builders = [
    // A backward-compatible builder that sets `subcore_id` to nullptr.
    OpBuilder<(ins "Value":$semaphore, "Value":$amount, "Value":$device_id, "Value":$core_id)>,
  ];
}

def TPU_BarrierOp : TPU_Op<"barrier"> {
  let summary = [{Barrier synchronization across SC vector subcores.}];
  let description = [{
    Performs barrier synchronization across all SC vector subcores at the
    specified barrier id.
  }];
  let arguments = (ins Index:$barrier_id);
  let results = (outs);
  let assemblyFormat = [{ `barrier_id` `(` $barrier_id `)` attr-dict }];
}

// tpu.enqueue_dma enqueues a DMA operation.
//
// source            : Memref to copy from.
// source_semaphore  : Semaphore to signal after the DMA completes. If a regular
//                     semaphore is provided, it will be incremented by 1 when
//                     the DMA is complete on platforms that support this mode
//                     and should be waited on with semaphore_wait(sem, 1).
//                     Otherwise WaitDMA should be used.
// target            : Memref to copy to.
// target_semaphore  : Semaphore to wait on before the DMA completes. If a regular
//                     semaphore is provided, it will be incremented by 1 when
//                     the DMA is complete on platforms that support this mode
//                     and should be waited on with semaphore_wait(sem, 1).
//                     Otherwise WaitDMA should be used.
// device_id         : The id of the device to copy to for remote DMAs.
// core_id           : The id of the core to copy to for remote and cross-core
//                     DMAs.
// subcore_id        : The id of the subcore to copy to for remote and cross-core
//                     DMAs, if applicable.
// priority          : The priority of the DMA.
// strict_ordering   : True if the DMA requires strict ordering. If false, the
//                     ordering is either strict or relaxed depending on the
//                     source and destination.
def TPU_EnqueueDMAOp : TPU_Op<"enqueue_dma", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$source,
    Optional<MemRefOf<[TPU_SomeSemaphoreType]>>:$source_semaphore,
    AnyMemRef:$target,
    Optional<MemRefOf<[TPU_SomeSemaphoreType]>>:$target_semaphore,
    Optional<I32>:$device_id, // For remote DMAs
    Optional<I32>:$core_id,
    Optional<I32>:$subcore_id,
    // Smaller number means higher priority. 0 is the highest and the default.
    DefaultValuedAttr<I32Attr, "0">:$priority,
    DefaultValuedAttr<BoolAttr, "false">:$strict_ordering
  );
  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)`
    `target` `(` $target `:` type($target) `)`
    (`source_semaphore` `(` $source_semaphore^ `:` type($source_semaphore) `)`)?
    (`target_semaphore` `(` $target_semaphore^ `:` type($target_semaphore) `)`)?
    (`device_id` `(` $device_id^ `)`)?
    (`core_id` `(` $core_id^ `)`)?
    (`subcore_id` `(` $subcore_id^ `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    mlir::tpu::CoreType getTargetCoreType();
  }];
  let builders = [
    // A backward-compatible builder that sets `subcore_id` to nullptr.
    OpBuilder<(ins "Value":$source, "Value":$source_semaphore, "Value":$target, "Value":$target_semaphore, "Value":$device_id, "Value":$core_id, CArg<"uint32_t", "0">:$priority, CArg<"bool", "false">:$strict_ordering)>,
  ];
}

// A base class for all ops that need to differentiate between gather and
// scatter.
class IndirectDMAOp {
  code extraBaseClassDeclaration = [{
    // Return true if this op performs a gather. Returns false if it performs a
    // scatter.
    FailureOr<bool> isGather();
  }];
}

// tpu.enqueue_indirect_dma copies data between HBM and VMEM, or between
// VMEM_SHARED and VMEM using indirect HBM offsets.
//
// If the source is in HBM or VMEM_SHARED and the target is in VMEM, performs a
// gather from the source (operand) at the offsets to the target (gather
// result).
// If the source is in VMEM and the target is in HBM or VMEM_SHARED, performs a
// scatter of the source (updates) to the target (operand) at the offsets.
//
// The memory region being indexed (the `source` for gather, `target` for
// scatter) must have a shape `[z, s0, ..., sm]` or `[1, z, s0, ..., sm]`,
// where `z` is the indexed dimension.
//
// The `offsets` argument must be of shape `[o]` or `[1, o]`.
// Each of the `o` values indexes the `z` dimension within the range `[0, z-1]`.
//
// Each offset selects a slice of shape `[s0, ..., sm]` (which can be elemental).
//
// source          : Memref to copy from.
// target          : Memref to copy to.
// offsets         : Gather or scatter offsets.
// semaphore       : Semaphore to wait on; receive semaphore for scatter, send semaphore for gather.
// add             : If true, add source values to target values. Otherwise, overwrite.
// offset_filter   : If set, don't write values at offsets whose value is equal to
//                   the filter value.
def TPU_EnqueueIndirectDMAOp : TPU_Op<"enqueue_indirect_dma">, IndirectDMAOp {
  let arguments = (ins
    AnyMemRef:$source,
    AnyMemRef:$target,
    AnyTypeOf<[MemRefOf<[I32]>, VectorOfRankAndType<[1], [I32]>]>:$offsets,
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    Optional<I32>:$offset_filter,
    DefaultValuedAttr<BoolAttr, "false">:$add
  );
  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)`
    `target` `(` $target `:` type($target) `)`
    `offsets` `(` $offsets `:` type($offsets) `)`
    (`offset_filter` `(` $offset_filter^ `)`)?
    `semaphore` `(` $semaphore `:` type($semaphore) `)`
    attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = extraBaseClassDeclaration;
}

// tpu.wait_dma2 waits for a DMA to complete.
//
// The number of bytes to wait for is determined based on the size of the
// destination memref.
def TPU_WaitDMA2Op : TPU_Op<"wait_dma2", [AttrSizedOperandSegments]> {
  let arguments = (ins
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    AnyMemRef:$src,
    AnyMemRef:$dst,
    Optional<I32>:$device_id, // For remote DMAs
    Optional<I32>:$core_id,
    DefaultValuedAttr<BoolAttr, "false">:$strict_ordering
  );
  let assemblyFormat = [{
    `semaphore` `(` $semaphore `:` type($semaphore) `)`
    `src` `(` $src `:` type($src) `)`
    `dst` `(` $dst `:` type($dst) `)`
    (`device_id` `(` $device_id^ `)`)?
    (`core_id` `(` $core_id^ `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
  // A backward-compatible builder that sets `device_id` and `core_id` to nullptr.
  let builders = [
    OpBuilder<(ins "Value":$semaphore, "Value":$src, "Value":$dst)>
  ];
}

// tpu.wait_dma waits for a DMA to complete.
def TPU_WaitDMAOp : TPU_Op<"wait_dma", [AttrSizedOperandSegments]> {
  let arguments = (ins
    AnyMemRef:$source,
    Optional<MemRefOf<[TPU_SomeSemaphoreType]>>:$source_semaphore,
    AnyMemRef:$target,
    Optional<MemRefOf<[TPU_SomeSemaphoreType]>>:$target_semaphore,
    Optional<I32>:$device_id, // For remote DMAs
    Optional<I32>:$core_id,
    Optional<I32>:$subcore_id,
    DefaultValuedAttr<I32Attr, "0">:$priority,
    DefaultValuedAttr<BoolAttr, "false">:$strict_ordering,
    BoolAttr:$wait_target
  );
  let assemblyFormat = [{
    `source` `(` $source `:` type($source) `)`
    `target` `(` $target `:` type($target) `)`
    (`source_semaphore` `(` $source_semaphore^ `:` type($source_semaphore) `)`)?
    (`target_semaphore` `(` $target_semaphore^ `:` type($target_semaphore) `)`)?
    (`device_id` `(` $device_id^ `)`)?
    (`core_id` `(` $core_id^ `)`)?
    (`subcore_id` `(` $subcore_id^ `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

// Like tpu.wait_dma2, but for indirect DMAs.
//
// The number of bytes to wait for is determined based on the size of the
// destination memref in a gather, and the size of the source memref in a
// scatter. The op differentiates between gather and scatter based on the memory
// spaces of the source and destination memrefs.
def TPU_WaitIndirectDMAOp : TPU_Op<"wait_indirect_dma">, IndirectDMAOp {
  let arguments = (ins
    MemRefOf<[TPU_DMASemaphoreType]>:$semaphore,
    AnyMemRef:$src,
    AnyMemRef:$dst
  );
  let assemblyFormat = [{
    `semaphore` `(` $semaphore `:` type($semaphore) `)`
    `src` `(` $src `:` type($src) `)`
    `dst` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = extraBaseClassDeclaration;
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

def TPU_TraceValueOp : TPU_Op<"trace_value", []> {
  let summary = "Emit a scalar value as a trace event.";
  let description = [{
    Emits a dynamic scalar value as a trace event.
    When lowered to LLO, creates its own trace scope (start/value/stop).
    Supported types: i32, f32.

    Example:
    ```mlir
    %count = arith.constant 42 : i32
    tpu.trace_value %count, "my_value" : i32
    ```
  }];
  let arguments = (ins
    AnyTypeOf<[I32, F32]>:$value,
    StrAttr:$label
  );
  let results = (outs);
  let assemblyFormat = [{ $value `,` $label attr-dict `:` type($value) }];
}

def TPU_YieldOp : TPU_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def TPU_DelayOp : TPU_Op<"delay"> {
  let arguments = (ins I32:$nanos);
  let results = (outs);
  let assemblyFormat = [{ $nanos attr-dict }];
}

// Expands the granularity of mask to subelements.
def TPU_MaskCastOp : TPU_Op<"mask_cast", [Pure]> {
  let description = [{
    Cast a mask register into a different packing.

    If casting to a type with smaller packing, then values being packed together
    must be identical. For example, for 8x128x4xi1 -> 8x128x2xi1,
    input[i, j, 0] == input[i, j, 1] and input[i, j, 2] == input[i, j, 3] must
    hold for all i, j. Otherwise, the result is undefined.
  }];
  let arguments = (ins VectorOfNonZeroRankOf<[I1]>:$input);
  let results = (outs VectorOfNonZeroRankOf<[I1]>:$result);
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

def TPU_FetchAndAddSyncOp : TPU_Op<"fetch_and_add_sync", [DefaultMemRead, DefaultMemWrite]> {
  let summary = [{
    FetchAndAddSyncOp atomically increments an SMEM element on another core and
    returns the old value.
  }];

  let description = [{
    FetchAndAddSyncOp atomically increments the element at `base[indices]` on
    core `core_id` by `$value` and returns the old value. The number of
    `$indices` must match the rank of the `$base` memref. The `$base` memref
    must be in SMEM. The update is synchronous.

    The `$core_id` specify the core on which the increment is performed.

    Example:

    ```mlir
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %tile_id = arith.constant 3 : i32
    %old = tpu.fetch_and_add_sync %smem[%c0], %c1, <sc_vector_subcore>, %tile_id
        : memref<1xi32, #tpu.memory_space<smem>>, i32 -> i32
    ```
  }];

  let arguments = (ins
    MemRefOf<[I32]>:$base,
    Variadic<I32>:$indices,
    I32:$value,
    I32:$core_id
  );
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` `,` $value `,` $core_id
    attr-dict `:` type($base) `,` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

#endif  // TPU_OPS
