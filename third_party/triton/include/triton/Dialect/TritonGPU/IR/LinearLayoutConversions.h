// Conversions from TritonGPU layouts (e.g. BlockedEncodingAttr) to
// LinearLayout.

#ifndef TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {
enum class ScaleDotElemType : uint32_t;
} // namespace mlir::triton

namespace mlir::triton::gpu {
class SwizzledSharedEncodingAttr;
class NVMMASharedEncodingAttr;
class AMDRotatingSharedEncodingAttr;
class AMDMfmaEncodingAttr;

// - BlockedEncodingAttrs have the following input dimensions.
//
//   "register": elements in one thread
//   "lane": threads in a warp
//   "warp": warps in a block/CTA
//   "block": blocks in a cluster
//
// - An n-dimensional SwizzledSharedEncodingAttr has the following input
// dimensions.
//
//   "offset": the n'th element in the allocation, within a particular thread
//      block (i.e. within a CTA).  The offset is measured in elements, not
//      bytes.
//   "block": blocks in a cluster
//
// All layouts have the following output dimensions.
//
//  "dimi" for i in 0..n-1: the location in the n'th logical dimension of the
//  output tensor.  These also are not reordered according to the layout's
//  `order`.
//
// You can flatten the input or output dimensions into a single dimension using
// LinearLayout::flattenIns/Outs().
//
// elemBitWidth is the bit width of one element in the layout.  This is required
// to compute the linear layout for MMAv3 (i.e. Hopper) shared layouts (i.e.
// shared layouts with nvmma_shared layout) but is otherwise unused.
//
// Returns std::nullopt if the given layout can't be converted to an LL.
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);

// Convert the shared encoding of a tensor with `nvmma_shared` layout to a
// LinearLayout that maps from a linear shared memory offset to tensor index.
//
// If `disableSwizzle` is set, then the resulting layout does not include
// swizzling.
LinearLayout sharedToLinearLayoutLeadingOffset(ArrayRef<int64_t> shape,
                                               NVMMASharedEncodingAttr shared,
                                               bool disableSwizzle = false);

// Given a linear layout where the input dimensions contain a "block" dimension,
// this method sets the "block" dimension to 0 and removes the corresponding
// output dimensions.
//
// Note that this behavior differs from calling
// `LinearLayout::sublayout(inDimNames, outDimNames)` when "block" is not in
// `inDimNames`. The latter does not modify the output sizes.
LinearLayout getLayoutWithinBlock(const LinearLayout &layout);

// In this function, we construct a linear layout representing the
// <shared memory offset, iteration, block> -> <tensor element index> mapping
// for entire `src` and `dst` tensors.  We determine the shape of the
// intermediate shared memory buffer needed for a register-to-register
// conversion using the maximum size accessed in each dimension from `src`'s
// layout and `dst`'s layout.  See the getRepShapeForCvt function in
// Allocation.cpp for details. Note that the buffer might be smaller than the
// tensor being converted, so we need multiple "iterations" to move a subregion
// of the `src` tensor to the corresponding subregion of the `dst` tensor.  The
// pesudo code of layout conversion is as follows:
//
// for iter in 0..numIterations:
//   sync threads
//   for vecIdx in [0..numRegisters/storeVec]:
//     registers <- get registers used in iter
//     offsets <- get offsets using the intermediate linear layout
//     store registers[vecIdx * storeVec, (vecIdx + 1) * storeVec)] to shared
//     memory
//   sync threads
//   for vecIdx in [0..numRegisters/loadVec]:
//     registers <- get registers used in iter
//     offsets <- get offsets using the intermediate linear layout
//     load registers[vecIdx * loadVec, (vecIdx + 1) * loadVec)] from shared
//     memory
LinearLayout chooseShemLayoutForRegToRegConversion(
    MLIRContext *ctx, ArrayRef<unsigned> tensorShape,
    ArrayRef<unsigned> repShape, ArrayRef<unsigned> order);

// This function constructs a linear layout that maps
// <register, lane, warp> to <shared memory offset, iteration>.
// The primary goal is to efficiently store 2D tiles of a tensor into shared
// memory using the `stmatrix` instruction, with each thread responsible for
// storing `N` elements.  If `stmatrix` cannot be used for the given tensor
// encoding, this function returns `std::nullopt`.
//
// Unlike standard vectorized stores, such as `st.shared.v4 [%offset],
// %vec_reg`, where `%vec_reg` contains four consecutive data elements, the
// `stmatrix` instruction allows `N` registers to point to non-contiguous
// locations within a tensor tile.
//
// For instance, the `stmatrix [%offset], %mat_reg` instruction on NVIDIA GPUs
// enables `%mat_reg` to store `N` elements that do not need to be consecutive.
// However, it is crucial that the address (`%offset`) of each row in a tensor
// tile should be aligned to `N` * `elemBitWidth`.  The `%offset` of each thread
// is calculated based on the provided tensor encoding.
//
// Currently, we support only the NVIDIA MMAv3 encoding and the `stmatrix.x4`
// instruction.  Each `stmatrix.x4` instruction stores eight 16-bit elements per
// thread, resulting in a total of 8 * 32 = 256 elements per warp, or 16 * 16
// elements per warp when distributed across four 8x8 tiles.  Each thread's
// `%offset` points to an address aligned with 8 * 16 bits, denoting a row in
// the 8x8 tile.  The values in `%mat_reg` are non-consecutive elements,
// composed of 4 pairs of consecutive elements.  These matrix addresses are
// distributed as follows:
//
//              col[0-7]     col[8-15]
//   row[0-7]  lane[0-7]    lane[16-23]
//   row[8-15] lane[8-15]   lane[24-31]
//
// The matrix elements of thread 0 are distributed in the following pattern:
//
//           col0       col8
//   row0  reg[0-1]   reg[4-5]
//   row8  reg[2-3]   reg[6-7]
//
// When `swizzleByteSize` is non-zero, the layout is constructed
// differently due to leading dimension offset and swizzling.
// There are two key concepts to understand:
//
//   1. Chunks: The leading dimension (i.e., the column dimension) is divided
//   into chunks, where each chunk's size is determined by `swizzleByteSize`.
//   2. Swizzling within tiles: Each tile applies a swizzling pattern to its
//   rows to optimize memory access.
//
// - Concept 1: Chunks
//
// In the swizzled layout, the leading dimension is strided by
// `swizzleByteSize`. This introduces the concept of a "chunk", where each chunk
// spans a certain number of columns.
//
// For a tile size of `stmatrix.x4` (16x16 elements), with each element being 16
// bits (2 bytes), each tile occupies 16 rows and 32 bytes per row (since 16
// elements * 2 bytes per element = 32 bytes per row).
//
// Given a `swizzleByteSize` of 128 bytes, the number of tiles per chunk can be
// calculated as:
//
//   Number of tiles per chunk = swizzleByteSize / (bytes per row) = 128 bytes /
//   32 bytes = 4 tiles
//
// Therefore, each chunk contains 4 tiles horizontally, spanning 64 columns
// (since each tile is 16 columns):
//
//             col0-15    col16-31   col32-47   col48-63
//   row0-15    tile0      tile1      tile2      tile3
//
// For a tensor of size 128x128 elements (#rows x #columns), and each element
// being 16 bits, the tensor can be divided into multiple chunks both
// horizontally and vertically.  Chunks are stored in memory in a "column-major"
// order based on chunks, meaning chunk1's address follows chunk0's.
//
// Assuming we have 8 warps, and we assign each warp to process a chunk of 16
// rows (rows per tile) and 128 columns (the width of two chunks). This results
// in each warp handling one horizontal slice of the tensor.
//
// The overall layout can be visualized as:
//
//                        |<- 128 * 128 bytes ->|<- 128 * 128 bytes ->|
//                              columns 0-63         columns 64-127
//   warp0 | rows 0-15            chunk0               chunk8
//   warp1 | rows 16-31           chunk1               chunk9
//   warp2 | rows 32-47           chunk2               chunk10
//   warp3 | rows 48-63           chunk3               chunk11
//   warp4 | rows 64-79           chunk4               chunk12
//   warp5 | rows 80-95           chunk5               chunk13
//   warp6 | rows 96-111          chunk6               chunk14
//   warp7 | rows 112-127         chunk7               chunk15
//
// - Concept 2: Swizzling within tiles
//
// Within each 16x16 tile, rows are swizzled to optimize memory access patterns.
// This swizzling is similar to what's defined in `TritonGPUAttrDefs.td`. at the
// level of each 16x16 tile rather than the entire tensor.
//
// Key parameters for swizzling:
//
//   - `perPhase`: The number of rows over which to apply a XOR operation at
//   each phase.
//   - `maxPhase`: The total number of phases.
//   - `vectorWidth`: The number of elements per vector, which is 8 in this case
//   because `stmatrix` stores 8 contiguous elements per thread.
//
// The offset of each element within a tile is calculated using the formula:
//
//   offset = row * swizzleByteSize + (vectorWidth * ((row / perPhase) %
//   maxPhase)) * elementSize
//
// where `elementSize` is the size of each element in bytes (2 bytes for 16-bit
// elements).
//
// For example, consider the element at index `(row=1, col=0)` in chunk0:
//
// Without swizzling:
//
//   offset = row * swizzleByteSize + col * elementSize
//          = 1 * 128 bytes + 0 * 2 bytes
//          = 128 bytes
//
// With swizzling (assuming `perPhase=1`, `maxPhase=8`, `vectorWidth=8`):
//
//   offset = row * swizzleByteSize + (vectorWidth * ((row / perPhase) %
//   maxPhase)) * elementSize
//          = 1 * 128 bytes + (8 * ((1 / 1) % 8)) * 2 bytes
//          = 128 bytes + (8 * (1 % 8)) * 2 bytes
//          = 128 bytes + 8 * 2 bytes
//          = 128 bytes + 16 bytes
//          = 144 bytes
//
// This swizzling ensures that elements are stored in a way that optimizes for
// memory bandwidth and reduces bank conflicts.
//
// - Verification through Linear Layout
//
// We can verify the offsets with the following outputs of the corresponding
// linear layout, where each element is 16 bits (2 bytes):
//
//   - register=1 -> offset=1
//     register=2 -> offset=2
//     register=4 -> offset=4
//     register=8 -> offset=16
//     register=16 -> offset=32
//     register=32 -> offset=8192
//   - lane=1 -> offset=72
//     lane=2 -> offset=144
//     lane=4 -> offset=288
//     lane=8 -> offset=512
//     lane=16 -> offset=8
//   - warp=1 -> offset=1024
//     warp=2 -> offset=2048
//     warp=4 -> offset=4096
//
// For index `(row=1, col=0)`, which corresponds to `reg=0` and `lane=1` in
// `warp=0`, the offset is calculated as 72 * 2 bytes = 144 bytes.  The result
// matches our earlier calculation.
//
// TODO(Keren): We should replace tensorTy with a LinearLayout and the element
// bit width of the tensor in the future to support more flexible tensor
// encodings
LinearLayout chooseStMatrixLayout(MLIRContext *ctx, RankedTensorType tensorTy,
                                  int swizzleByteSize);

// The primary goal of this function is to efficiently store 2D tiles of a
// tensor into shared memory using the `ldmatrix` instruction.
LinearLayout chooseLdMatrixLayout(Attribute enc, ArrayRef<int64_t> shape,
                                  bool needTrans, int32_t elemBitWidth);

// The primary goal of this function is to efficiently load 2D tiles of a
// tensor from shared memory using the `ds_read_tr` instruction for AMD GPUs.
LinearLayout chooseDsReadB64TrLayout(Attribute enc, ArrayRef<int64_t> shape,
                                     int32_t elemBitWidth);

LinearLayout getScaleTMEMStoreLinearLayout(RankedTensorType scaleType,
                                           int numWarps);

// Create LinearLayout for scale in scaled mfma.
LinearLayout chooseScaledMfmaScaleLayout(
    MLIRContext *ctx, int dotOperandIdx,
    const std::vector<std::vector<int32_t>> &dotOperandWarpBasis,
    ArrayRef<int64_t> dotOperandShape, unsigned mfmaMDim);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
