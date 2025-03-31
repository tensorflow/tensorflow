#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton {

static bool checkSquareSublayout(const LinearLayout &ll,
                                 ArrayRef<StringAttr> dimNames,
                                 function_ref<bool(int, int32_t)> checkBasis) {
  // The empty layout is the identity
  if (dimNames.size() == 0) {
    return true;
  }
  // Check that the input-output sizes are the same
  LinearLayout sl = ll.sublayout(dimNames, dimNames);
  for (StringAttr dim : dimNames) {
    if (ll.getInDimSize(dim) != ll.getOutDimSize(dim)) {
      return false;
    }
  }
  // Once the inputs and output dimensions are the same, we can just check
  // that the basis for the single remaining dimension is the identity.
  sl = sl.flattenIns().flattenOuts();
  const auto &inDimBases = sl.getBases().begin()->second;
  for (auto [b, basis] : llvm::enumerate(inDimBases)) {
    if (!checkBasis(b, basis[0])) {
      return false;
    }
  }
  return true;
}

bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames) {
  return checkSquareSublayout(
      ll, dimNames, [](int b, int32_t basis) { return basis == (1 << b); });
}

bool squareSublayoutIsPermutation(const LinearLayout &ll,
                                  ArrayRef<StringAttr> dimNames) {
  int32_t mask = 0;
  return checkSquareSublayout(ll, dimNames, [&](int b, int32_t basis) {
    if (!llvm::isPowerOf2_32(basis))
      return false;
    if (mask & basis)
      return false; // check if this bit is already set
    mask |= basis;
    return true;
  });
}

LinearLayout
ensureLayoutNotLargerThan(const LinearLayout &layout,
                          const llvm::SmallDenseMap<StringAttr, int64_t> &shape,
                          bool broadcastRegisters) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }
  MLIRContext *ctx = shape.begin()->first.getContext();

  auto bases = layout.getBases();

  auto kRegister = StringAttr::get(ctx, "register");
  std::set<int32_t> broadcastedDims;

  for (auto outDim : llvm::enumerate(layout.getOutDimNames())) {
    auto outDimName = outDim.value();
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    if (actualSize <= desiredSize) {
      continue;
    }
    assert(actualSize % desiredSize == 0);
    // <inDimName, basisIdx, outValue>
    std::vector<std::tuple<StringAttr, int, int>> sortedBases;
    for (auto [inDimName, basis] : bases) {
      for (size_t basisIdx = 0; basisIdx < basis.size(); basisIdx++) {
        auto outValue = basis[basisIdx][outDim.index()];
        if (outValue == 0) {
          continue;
        }
        assert(llvm::isPowerOf2_32(outValue));
        sortedBases.emplace_back(inDimName, basisIdx, outValue);
      }
    }
    // From the largest basis to the smallest.
    llvm::sort(sortedBases,
               [](auto a, auto b) { return std::get<2>(a) > std::get<2>(b); });
    for (auto [inDimName, basisIdx, outValue] : sortedBases) {
      if (actualSize <= desiredSize) {
        break;
      }
      if (!broadcastRegisters && inDimName == kRegister) {
        broadcastedDims.insert(basisIdx);
      } else {
        bases[inDimName][basisIdx][outDim.index()] = 0;
      }
      actualSize >>= 1;
    }
  }
  if (!broadcastRegisters) {
    // Remove broadcasted registers
    std::vector<std::vector<int32_t>> newBasesRegister;
    for (auto [idx, basis] : llvm::enumerate(bases[kRegister])) {
      // Remove if it's broadcasted
      if (broadcastedDims.find(idx) == broadcastedDims.end()) {
        newBasesRegister.push_back(std::move(basis));
      }
    }
    bases[kRegister] = std::move(newBasesRegister);
  }

  return LinearLayout(std::move(bases),
                      llvm::to_vector(layout.getOutDimNames()));
}

// For each out-dim d, ensure the layout's out-size (i.e. its codomain) is no
// smaller than shape[d].  Do this by increasing the size of the layout's inputs
// along its most-minor dimension ("register" for register layouts, "offset" for
// shared layouts).
//
// This function is invariant to the order of the layout's input dimensions, but
// it cares about the order of the output dims, which should be minor-to-major.
LinearLayout ensureLayoutNotSmallerThan(
    const LinearLayout &layout,
    const llvm::SmallDenseMap<StringAttr, int64_t> &shape) {
  assert(shape.size() == layout.getNumOutDims());
  if (shape.empty()) {
    return layout;
  }

  StringAttr kDim = *layout.getInDimNames().begin();
  assert(kDim == "register" || kDim == "offset");

  LinearLayout ret = layout;
  for (StringAttr outDimName : layout.getOutDimNames()) {
    int32_t actualSize = layout.getOutDimSize(outDimName);
    int32_t desiredSize = shape.lookup(outDimName);
    assert(actualSize > desiredSize || desiredSize % actualSize == 0);
    ret *= LinearLayout::identity1D(desiredSize / actualSize, kDim, outDimName);
    assert(ret.getOutDimSize(outDimName) >= desiredSize);
  }
  return ret;
}

// Returns ["dim0", "dim1", ..., "dim<rank-1>"].
SmallVector<StringAttr> standardOutDimNames(MLIRContext *ctx, int rank) {
  SmallVector<StringAttr> ret;
  for (int i = 0; i < rank; i++) {
    ret.push_back(StringAttr::get(ctx, "dim" + llvm::Twine(i)));
  }
  return ret;
}

// Returns [("dim0", dstShape[0]), ("dim1", dstShape[1]), ...,
// ("dim<rank-1>", dstShape[rank-1])].
SmallVector<std::pair<StringAttr, int32_t>>
standardOutDimPairs(MLIRContext *ctx, ArrayRef<int64_t> dstShape) {
  auto newRank = dstShape.size();
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto [dim, size] :
       llvm::zip(standardOutDimNames(ctx, newRank), dstShape)) {
    newOutDims.emplace_back(dim, size);
  }
  return newOutDims;
}

// Returns a 1D -> ND layout into [dim0, dim1, ...] that's equivalent to
// creating a 1D -> 1D mapping of size product(shape) and then reshaping to
// permute(shape, order).
LinearLayout identityStandardND(StringAttr inDimName, ArrayRef<unsigned> shape,
                                ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  MLIRContext *ctx = inDimName.getContext();
  auto rank = shape.size();

  // The order in triton is written wrt. [dim0, dim1, ...].
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < shape.size(); i++) {
    // Start with the most-minor dimension, which is order[0].
    int dim = order[i];
    ret *= LinearLayout::identity1D(shape[dim], inDimName, outDimNames[dim]);
  }
  return ret;
}

// Compute the supremum of two lists.
// If the supremum is not unique, we return the first list first
// Error out if the supremum does not exist
// e.g. sup([a, b], [a, c]) = [a, b, c], sup([a, b], [b, c]) = [a, b, c]
//      sup([a, b], [b, a]) = error! Supremum does not exist.
SmallVector<StringAttr> supremum(const SmallVector<StringAttr> &x,
                                 const SmallVector<StringAttr> &y) {
  llvm::SetVector<StringAttr> result;
  DenseMap<StringAttr, int> posX, posY;
  for (auto [idx, elem] : llvm::enumerate(x))
    posX[elem] = idx;
  for (auto [idx, elem] : llvm::enumerate(y))
    posY[elem] = idx;
  int i = 0, j = 0;
  const int INF = std::numeric_limits<int>::max();
  while (i < x.size() || j < y.size()) {
    while (i < x.size() && result.contains(x[i]))
      ++i;
    while (j < y.size() && result.contains(y[j]))
      ++j;
    if (i >= x.size() && j >= y.size())
      break;
    if (i < x.size() && j < y.size() && x[i] == y[j]) {
      if (posY[x[i]] < j)
        llvm_unreachable("Supremum does not exist");
      result.insert(x[i]);
      ++i, ++j;
      continue;
    }
    int candX = INF, candY = INF;
    if (i < x.size()) {
      if (posY.count(x[i]) && posY[x[i]] >= j)
        candX = posY[x[i]];
    }
    if (j < y.size()) {
      if (posX.count(y[j]) && posX[y[j]] >= i)
        candY = posX[y[j]];
    }
    if (i < x.size() && candX == INF) {
      result.insert(x[i]);
      ++i;
      continue;
    }
    if (j < y.size() && candY == INF) {
      result.insert(y[j]);
      ++j;
      continue;
    }
    if (candX <= candY) {
      if (posY[x[i]] < j)
        llvm_unreachable("Supremum does not exist");
      result.insert(x[i]);
      ++i;
    } else {
      if (posX[y[j]] < i)
        llvm_unreachable("Supremum does not exist");
      result.insert(y[j]);
      ++j;
    }
  }
  return to_vector(result);
}

} // namespace mlir::triton
