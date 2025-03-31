#ifndef TRITON_TOOLS_LAYOUTUTILS_H
#define TRITON_TOOLS_LAYOUTUTILS_H

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {
// Is the sublayout defined from dimNames to dimNames the identity?
// In particular, is the input and  output size in these dimensions
// the same, and are the bases the identity?
bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames);

// Is the sublayout defined from dimNames to dimNames a subpermutation matrix?
// I.e. the layout matrix is formed by selecting unique columns from the
// identity matrix and adding zero columns. A zero column in the layout means
// that changing a bit in the inputs does not change the bits of the outputs
// (broadcasting).
bool squareSublayoutIsPermutation(const LinearLayout &ll,
                                  ArrayRef<StringAttr> dimNames);

// For each output dimension d, ensure that the layout's output size (i.e., its
// codomain) does not exceed shape[d]. Do this without changing the size of the
// layout's inputs (i.e., leave its domain unchanged).
//
// This function is invariant to the order of the layout's input and output
// dimensions.
//
// We achieve this by setting the largest value in each output dimension d to 0
// because bases that map to a location larger than shape[d]
// effectively duplicate along that dimension.  For example, consider a layout
// with an output dimension size of 32, and we call ensureLayoutNotLargerThan to
// shrink the output dimension size to 8:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 16
//
// In the first step, we shrink the output dimension size to 16 by setting
// L(lane=2) to 0:
//
//   L(register=1) = 8
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0
//
// This means that lane=2 has the same data as lane=0.
//
// Now the output dimension of this layout has a size of 16, which is still
// larger than 8.  We find the current largest value in the output dimension,
// which is L(register=1) = 8, and we set L(register=1) to 0:
//
//   L(register=1) = 0
//   L(register=2) = 4
//   L(register=4) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0
//
// Now the output dimension of this layout has a size of 8, which is the desired
// size.  Note that this method works only because the bases are powers of two,
// which is the case for DistributedLayouts If broadcastRegisters is false, we
// remove any register that's larger than the desired shape. In the example
// above we would have
//   L(register=1) = 4
//   L(register=2) = 1
//   L(lane=1) = 2
//   L(lane=2) = 0
LinearLayout
ensureLayoutNotLargerThan(const LinearLayout &layout,
                          const llvm::SmallDenseMap<StringAttr, int64_t> &shape,
                          bool broadcastRegisters = true);

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along its most-minor dimension ("register" for register layouts, "offset" for
// shared layouts).
//
// This function is invariant to the order of the layout's input dimensions, but
// it cares about the order of the output dims, which should be minor-to-major.
LinearLayout ensureLayoutNotSmallerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape);

// Return a vector of the standard out dimension names for tensor layouts. These
// are "dim0", "dim1", etc.
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank);

// Return a vector of the standard out dimension name/value pairs, i.e.
// ("dim0", dstShape[0]), ("dim1", dstShape[1]), etc.
SmallVector<std::pair<StringAttr, int32_t>>
standardOutDimPairs(MLIRContext *ctx, ArrayRef<int64_t> dstShape);

// Return an identity mapping from `inDimName` to the standard out dimensions,
// with the dimensions sized according to the shape. The bases are sorted
// according to `order`, with the most minor dimension first.
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order);

// Compute the supremum of two lists.
// Error out if the supremum does not exist (e.g. [a, b] and [b, a]).
// If the supremum is not unique, we return the first list first
// (e.g. [a, b], [a, c] -> [a, b, c]).
SmallVector<StringAttr> supremum(const SmallVector<StringAttr> &x,
                                 const SmallVector<StringAttr> &y);
} // namespace mlir::triton

#endif // TRITON_TOOLS_LAYOUTUTILS_H
