#include "triton/Tools/LinearLayout.h"

#include <cstdint>
#include <set>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/f2reduce/f2reduce.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "linear_layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}

static int __builtin_ctzll(unsigned long long x) {
  unsigned long r;
  _BitScanForward64(&r, x);
  return static_cast<int>(r);
}

#endif

namespace mlir::triton {

namespace {
using BasesT = LinearLayout::BasesT;
using llvm::SmallDenseSet;
using llvm::Twine;

BasesT makeBasesMap(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases) {
  BasesT ret;
  for (const auto &[inDim, inDimBases] : bases) {
    ret[inDim] = inDimBases;
  }
  return ret;
}

// Dump the matrix to stderr in a human-readable format for debugging.
void dumpMatrix(uint64_t *m, int numRows, int numCols) {
  assert(numCols <= 64);
  for (int r = 0; r < numRows; r++) {
    llvm::errs() << "0b";
    for (int c = 0; c < numCols; c++) {
      llvm::errs() << ((m[r] & (1 << c)) != 0 ? "1" : "0");
    }
    llvm::errs() << "\n";
  }
}

// Build a matrix of size sum(outDimSizeLog2) x sum(inDimSizeLog2) representing
// the bases of the given layout.  This can then be used by f2reduce.
//
// This function is called from the constructor of LinearLayout, so be careful
// not to use any functions that create LLs in here.
std::unique_ptr<uint64_t[]> getMatrix(const LinearLayout &layout) {
  int numRows = layout.getTotalOutDimSizeLog2();
  int numCols = layout.getTotalInDimSizeLog2();

  // Don't handle giant LLs.  This makes some things easier; for example, each
  // row can be a single uint64_t.
  assert(numCols <= 64 && "LinearLayout too large");
  assert(numRows <= 64 && "LinearLayout too large");

  // Suppose we have a layout specified by the following values.
  //
  //   L(0,1) = (0b01, 0b1)
  //   L(0,2) = (0b10, 0b0)
  //   L(1,0) = (0b10, 0b0)
  //   L(2,0) = (0b11, 0b0)
  //
  // We will create one column per entry above.  The max bit width of the
  // codomain is (2,1), so our matrix will have 2+1=3 rows.  The final matrix
  // will be
  //
  //  | L(0,1)[0] L(0,2)[0] L(1,0)[0] L(2,0)[0] |   | 0b1001 |
  //  |    ↓         ↓         ↓         ↓      |   | 0b0111 |
  //  | L(0,1)[1] L(0,2)[1] L(1,0)[1] L(2,0)[1] | = | 0b1000 |
  //  |    ↓         ↓         ↓         ↓      |
  //
  // Note `new uint64_t[n]()` is zero-initialized, but `new uint64_t[n]` is not.
  std::unique_ptr<uint64_t[]> m(new uint64_t[numRows]());
  int r = 0;
  for (StringAttr outDim : layout.getOutDimNames()) {
    int c = 0;
    for (StringAttr inDim : layout.getInDimNames()) {
      for (int i = 0; i < layout.getInDimSizeLog2(inDim); i++) {
        uint64_t basis = layout.getBasis(inDim, i, outDim);
        for (int j = 0; j < layout.getOutDimSizeLog2(outDim); j++) {
          m[r + j] |= ((basis >> j) & 1) << c;
        }
        c++;
      }
    }
    r += layout.getOutDimSizeLog2(outDim);
  }

  return m;
}

// Compute the rank of the matrix formed by taking the bases for the given
// outDim as columns.  In other words, finds the number of linearly-independent
// bases for this output dimension.
int getMatrixRank(std::unique_ptr<uint64_t[]> m, int numRows, int numCols) {
  // stride is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  assert(numCols <= 64);
  f2reduce::inplace_rref_strided(m.get(), numRows, numCols, /*stride=*/1);

  // The rank of the reduced matrix is simply the number of nonzero rows.
  int rank = 0;
  for (int i = 0; i < numRows; i++) {
    if (m[i] != 0)
      rank++;
  }
  return rank;
}

template <typename T, typename U>
void assertDimsEqualIgnoringOrder(T &&a, U &&b) {
  SmallDenseSet<StringAttr> as(a.begin(), a.end());
  SmallDenseSet<StringAttr> bs(b.begin(), b.end());
  if (as != bs) {
    llvm::report_fatal_error("Dimensions must match, ignoring order, but they "
                             "don't.  Got dims: [" +
                             Twine(triton::join(a, ", ")) + "] and [" +
                             triton::join(b, ", ") + "]");
  }
}

template <typename T, typename U>
void assertDimsSubsetIgnoringOrder(T &&small, U &&big) {
  SmallDenseSet<StringAttr> smallSet(small.begin(), small.end());
  SmallDenseSet<StringAttr> bigSet(big.begin(), big.end());
  if (!llvm::set_is_subset(smallSet, bigSet)) {
    llvm::report_fatal_error("Dimensions must be a subset, ignoring order, but "
                             "they aren't.  Got dims: [" +
                             Twine(triton::join(small, ", ")) + "] and [" +
                             triton::join(big, ", ") + "]");
  }
}
} // anonymous namespace

/*static*/ std::optional<LinearLayout>
LinearLayout::tryCreate(BasesT bases,
                        ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                        bool requireSurjective) {
  LinearLayout ll(std::move(bases), std::move(outDims), NoCheckInvariants{});
  std::optional<std::string> error = ll.checkInvariants(requireSurjective);
  if (error) {
    return std::nullopt;
  }
  return ll;
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           NoCheckInvariants)
    : bases(std::move(bases)) {
  for (auto [outDim, size] : outDims) {
    this->outDims[outDim] = size;
  }
}

LinearLayout::LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames)
    : bases(std::move(bases)) {
  // Infer out-dim sizes.
  for (StringAttr outDim : outDimNames) {
    outDims[outDim] = 1;
  }
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        int32_t &size = outDims[outDimNames[i]];
        size = std::max<int32_t>(size, llvm::NextPowerOf2(basis[i]));
      }
    }
  }

  std::optional<std::string> error =
      checkInvariants(/*requireSurjective=*/true);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

LinearLayout::LinearLayout(BasesT bases,
                           ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                           bool requireSurjective)
    : LinearLayout(std::move(bases), std::move(outDims), NoCheckInvariants{}) {
  std::optional<std::string> error = checkInvariants(requireSurjective);
  if (error.has_value()) {
    llvm::report_fatal_error(StringRef(*error));
  }
}

std::optional<std::string>
LinearLayout::checkInvariants(bool requireSurjective) {
  LDBG("checkInvariants: " << toString());
  // Check that basis values are non-negative.
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t b) { return b < 0; })) {
        return "Invalid bases passed to LinearLayout.  Expected all basis "
               "values to be non-negative, but found a negative value for "
               "in dimension '" +
               inDim.str() + "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Check that the bases all have length equal to outDimNames.size().
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      if (basis.size() != outDims.size()) {
        return "Invalid bases passed to LinearLayout.  Expect all bases to "
               "have the same size, equal to outDimNames.size() (" +
               std::to_string(outDims.size()) +
               ").  But this failed for in dimension '" + inDim.str() +
               "'.  Full list of bases:" + toString() + "\n";
      }
    }
  }

  // Check that the out-dim sizes are powers of 2.
  for (const auto &[outDim, size] : outDims) {
    if (!llvm::isPowerOf2_32(size)) {
      return "Invalid out-dim size " + std::to_string(size) + " for out-dim '" +
             outDim.str() + "'.  Out-dim sizes must be powers of 2.\n";
    }
  }

  // Check that the bases are smaller than the out-dim sizes.
  SmallVector<StringAttr> outDimNames = llvm::to_vector(getOutDimNames());
  for (const auto &[inDim, inDimBases] : this->bases) {
    for (const auto &basis : inDimBases) {
      for (int i = 0; i < basis.size(); i++) {
        if (basis[i] >= outDims[outDimNames[i]]) {
          return "Invalid basis " + std::to_string(basis[i]) + " for in-dim '" +
                 inDim.str() + "' and out-dim '" + outDimNames[i].str() +
                 "'.  Basis must be less than the out-dim size.\n";
        }
      }
    }
  }

  // Determine whether the this layout is surjective, i.e. that every `out`
  // coordinate can be reached by some `in` coordinate.
  //
  // It's prohibitively slow to calculate this naively, but thankfully, this
  // is equivalent to checking that the number of linearly-independent bases
  // is equal to sum(getOutDimSizeLog2).  This can be computed by finding
  // the rank of the matrix whose columns are those bases.  We can compute
  // the rank of our matrix using Gaussian elimination, which runs in O(n^3)
  // for an n x n matrix.  Our matrix size is sum(inDimSizeLog2) x
  // sum(outDimSizeLog2), so this should be plenty fast.
  this->surjective =
      getMatrixRank(getMatrix(*this), /*numRows=*/getTotalOutDimSizeLog2(),
                    /*numCols=*/getTotalInDimSizeLog2()) ==
      getTotalOutDimSizeLog2();

  if (requireSurjective && !surjective) {
    return "Layout is expected to be surjective, i.e. every `out` coordinate "
           "can be reached by some `in` coordinate, but was not:" +
           toString();
  }

  return std::nullopt;
}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<StringAttr> outDimNames)
    : LinearLayout(makeBasesMap(bases), outDimNames) {}

LinearLayout::LinearLayout(
    ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
    ArrayRef<std::pair<StringAttr, int32_t>> outDims, bool requireSurjective)
    : LinearLayout(makeBasesMap(bases), outDims, requireSurjective) {}

/*static*/ LinearLayout LinearLayout::identity1D(int32_t size,
                                                 StringAttr inDimName,
                                                 StringAttr outDimName) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  std::vector<std::vector<int32_t>> powersOf2;
  for (int32_t i = 1; i < size; i *= 2) {
    powersOf2.emplace_back().push_back(i);
  }
  return LinearLayout({{inDimName, std::move(powersOf2)}}, {outDimName});
}

/*static*/ LinearLayout LinearLayout::zeros1D(int32_t size,
                                              StringAttr inDimName,
                                              StringAttr outDimName,
                                              int32_t outDimSize) {
  if (size == 0)
    return LinearLayout::empty();

  assert(llvm::isPowerOf2_32(size));
  std::vector<std::vector<int32_t>> zeros;
  for (int i = 0; i < llvm::Log2_32(size); i++) {
    zeros.emplace_back().push_back(0);
  }
  return LinearLayout({{inDimName, zeros}}, {{outDimName, outDimSize}},
                      /*requiresSurjective=*/outDimSize == 1);
}

int32_t LinearLayout::getOutDimIndex(StringAttr outDim) const {
  int i = 0;
  for (auto [name, _] : outDims) {
    if (name == outDim) {
      return i;
    }
    i++;
  }
  llvm::report_fatal_error("outDim " + Twine(outDim) + " is not in layout" +
                           toString());
}

int32_t LinearLayout::getInDimSizeLog2(StringAttr inDim) const {
  auto it = bases.find(inDim);
  assert(it != bases.end());
  return it->second.size();
}

int32_t LinearLayout::getTotalInDimSizeLog2() const {
  return std::accumulate(getInDimNames().begin(), getInDimNames().end(), 0,
                         [&](int32_t acc, StringAttr inDim) {
                           return acc + getInDimSizeLog2(inDim);
                         });
}

int32_t LinearLayout::getOutDimSizeLog2(StringAttr outDim) const {
  auto it = outDims.find(outDim);
  assert(it != outDims.end());
  return llvm::Log2_32(it->second);
}

int32_t LinearLayout::getTotalOutDimSizeLog2() const {
  return std::accumulate(getOutDimNames().begin(), getOutDimNames().end(), 0,
                         [&](int32_t acc, StringAttr outDim) {
                           return acc + getOutDimSizeLog2(outDim);
                         });
}

int32_t LinearLayout::getNumConsecutiveInOut() const {
  if (bases.empty() || getNumOutDims() == 0)
    return 1;

  // Count how many of the initial bases for the first in-dim are
  // (2^i, 0, ..., 0).
  const auto &firstInDimBases = bases.begin()->second;
  int consec = 0;
  for (; consec < firstInDimBases.size(); consec++) {
    const auto &basis = firstInDimBases[consec];
    if (basis[0] != (1 << consec) ||
        !std::all_of(basis.begin() + 1, basis.end(),
                     [](int32_t x) { return x == 0; })) {
      break;
    }
  }

  // `or` together all other bases' first out-dim.
  int32_t otherBits = 0;
  for (const auto &[inDim, inDimBases] : bases) {
    for (int i = 0; i < inDimBases.size(); i++) {
      if (inDim != bases.begin()->first || i >= consec) {
        otherBits |= inDimBases[i][0];
      }
    }
  }
  int32_t trailingZeros = otherBits != 0 ? __builtin_ctz(otherBits) : 31;

  return 1 << std::min(consec, trailingZeros);
}

LinearLayout LinearLayout::transposeIns(ArrayRef<StringAttr> newInDims) const {
  assertDimsEqualIgnoringOrder(newInDims, getInDimNames());

  BasesT newBases;
  for (const auto &inDim : newInDims) {
    newBases[inDim] = bases.find(inDim)->second;
  }
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
                      surjective);
}

LinearLayout
LinearLayout::transposeOuts(ArrayRef<StringAttr> newOutDims) const {
  assertDimsEqualIgnoringOrder(newOutDims, getOutDimNames());

  std::vector<int32_t> permutation;
  for (const auto &outDim : newOutDims) {
    permutation.push_back(getOutDimIndex(outDim));
  }

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      std::vector<int32_t> newBasis;
      for (int32_t i : permutation) {
        newBasis.push_back(basis[i]);
      }
      newInDimBases.push_back(std::move(newBasis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> newOutDimSizes;
  for (auto outDim : newOutDims) {
    newOutDimSizes.push_back({outDim, getOutDimSize(outDim)});
  }
  return LinearLayout(std::move(newBases), newOutDimSizes, surjective);
}

LinearLayout LinearLayout::reshapeIns(
    ArrayRef<std::pair<StringAttr, int32_t>> newInDims) const {
  assert(llvm::all_of(newInDims, [&](auto &inDim) {
    return llvm::isPowerOf2_32(inDim.second);
  }));
  assert(getTotalInDimSize() == std::accumulate(newInDims.begin(),
                                                newInDims.end(), 1,
                                                [&](int32_t acc, auto &inDim) {
                                                  return acc * inDim.second;
                                                }));

  // First flatten into a single in-dimension.  Then split it up according
  // to `newInDims`.
  SmallVector<std::vector<int32_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    for (const auto &basis : inDimBases) {
      flatBases.push_back(basis);
    }
  }

  BasesT newBases;
  int i = 0;
  for (const auto &[inDim, inDimSize] : newInDims) {
    auto &newInDimBases = newBases[inDim];
    for (int j = 0; j < llvm::Log2_32(inDimSize); j++) {
      newInDimBases.push_back(flatBases[i++]);
    }
  }
  return LinearLayout(std::move(newBases), llvm::to_vector(outDims),
                      surjective);
}

LinearLayout LinearLayout::reshapeOuts(
    ArrayRef<std::pair<StringAttr, int32_t>> newOutDims) const {
  assert(llvm::all_of(newOutDims, [&](auto &outDim) {
    return llvm::isPowerOf2_32(outDim.second);
  }));
  assert(getTotalOutDimSize() ==
         std::accumulate(
             newOutDims.begin(), newOutDims.end(), 1,
             [&](int32_t acc, auto &outDim) { return acc * outDim.second; }));

  SmallVector<int32_t> shifts;
  shifts.push_back(0);
  for (StringAttr outDim : getOutDimNames()) {
    shifts.push_back(shifts.back() + getOutDimSizeLog2(outDim));
  }

  // Flatten into a single out-dimension.  Then split it up according to
  // `newOutDims`.
  llvm::MapVector<StringAttr, std::vector<int32_t>> flatBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &flatInBases = flatBases[inDim];
    for (const auto &basis : inDimBases) {
      int b = 0;
      for (int i = 0; i < basis.size(); i++) {
        b += basis[i] << shifts[i];
      }
      flatInBases.push_back(b);
    }
  }

  BasesT newBases;
  for (const auto &[inDim, flatInBases] : flatBases) {
    std::vector<std::vector<int32_t>> &newInDimBases = newBases[inDim];
    for (int32_t b : flatInBases) {
      std::vector<int32_t> multiDimBasis;
      for (int32_t newSize : llvm::make_second_range(newOutDims)) {
        multiDimBasis.push_back(b % newSize);
        b /= newSize;
      }
      newInDimBases.push_back(std::move(multiDimBasis));
    }
  }

  return LinearLayout(std::move(newBases), newOutDims, surjective);
}

LinearLayout LinearLayout::concatIns(const LinearLayout &other) const {
  assert(llvm::to_vector(getOutDimNames()) ==
             llvm::to_vector(other.getOutDimNames()) &&
         "layouts must have the same output dimensions");
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) == other.getOutDimSize(outDim) &&
           "layouts must have the same output dimension sizes");
  }

  LinearLayout::BasesT resultBases = getBases();
  for (auto &bases : other.getBases())
    resultBases.insert(bases);
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto &[outDim, outDimSize] : outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  return LinearLayout(std::move(resultBases), newOutDims,
                      /*requiresSurjective=*/false);
}

LinearLayout LinearLayout::concatOuts(const LinearLayout &other) const {
  assert(llvm::to_vector(getInDimNames()) ==
             llvm::to_vector(other.getInDimNames()) &&
         "layouts must have the same input dimensions");
  for (StringAttr inDim : getInDimNames()) {
    assert(getInDimSize(inDim) == other.getInDimSize(inDim) &&
           "layouts must have the same input dimension sizes");
  }

  LinearLayout::BasesT result;
  for (auto [lhsBases, rhsBases] : llvm::zip(getBases(), other.getBases())) {
    auto &resultBases = result[lhsBases.first];
    assert(lhsBases.first == rhsBases.first);
    for (auto [lhsBasis, rhsBasis] :
         llvm::zip(lhsBases.second, rhsBases.second)) {
      std::vector<int32_t> resultBasis;
      llvm::append_range(resultBasis, lhsBasis);
      llvm::append_range(resultBasis, rhsBasis);
      resultBases.push_back(std::move(resultBasis));
    }
  }
  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto &[outDim, outDimSize] : outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  for (auto &[outDim, outDimSize] : other.outDims)
    newOutDims.emplace_back(outDim, outDimSize);
  return LinearLayout(std::move(result), newOutDims,
                      /*requiresSurjective=*/false);
}

LinearLayout operator*(LinearLayout inner, LinearLayout outer) {
  // Check that dims common to outer and inner have the same relative order.
  auto inDims = supremum(llvm::to_vector(inner.getInDimNames()),
                         llvm::to_vector(outer.getInDimNames()));
  auto outDims = supremum(llvm::to_vector(inner.getOutDimNames()),
                          llvm::to_vector(outer.getOutDimNames()));

  // Get the sizeLog2 of all input and output dimensions we're going to
  // consider, in order.  `inner` is more minor, so its dimensions come
  // first.
  llvm::MapVector<StringAttr, int32_t> inDimSizesLog2;
  llvm::MapVector<StringAttr, int32_t> outDimSizesLog2;
  for (const auto &dim : inDims)
    inDimSizesLog2.insert({dim, 0});
  for (const auto &dim : outDims)
    outDimSizesLog2.insert({dim, 0});
  for (const auto &layout : {inner, outer}) {
    for (StringAttr inDim : layout.getInDimNames()) {
      inDimSizesLog2[inDim] += layout.getInDimSizeLog2(inDim);
    }
    for (StringAttr outDim : layout.getOutDimNames()) {
      outDimSizesLog2[outDim] += layout.getOutDimSizeLog2(outDim);
    }
  }

  BasesT allBases;
  for (auto [inDimName, inDimSizeLog2] : inDimSizesLog2) {
    std::vector<std::vector<int32_t>> &inDimBases = allBases[inDimName];

    // Fill with zeros.
    inDimBases = std::vector<std::vector<int32_t>>(
        inDimSizeLog2, std::vector<int32_t>(outDimSizesLog2.size(), 0));

    for (auto [outDimIdx, outDimNameAndSize] :
         llvm::enumerate(outDimSizesLog2)) {
      auto [outDimName, outDimSize] = outDimNameAndSize;
      if (inner.hasInDim(inDimName) && inner.hasOutDim(outDimName)) {
        for (int i = 0; i < inner.getInDimSizeLog2(inDimName); i++) {
          inDimBases[i][outDimIdx] = inner.getBasis(inDimName, i, outDimName);
        }
      }
      if (outer.hasInDim(inDimName) && outer.hasOutDim(outDimName)) {
        int offset =
            inner.hasInDim(inDimName) ? inner.getInDimSizeLog2(inDimName) : 0;
        int shift = inner.hasOutDim(outDimName)
                        ? inner.getOutDimSizeLog2(outDimName)
                        : 0;
        for (int i = 0; i < outer.getInDimSizeLog2(inDimName); i++) {
          inDimBases[offset + i][outDimIdx] =
              outer.getBasis(inDimName, i, outDimName) << shift;
        }
      }
    }
  }

  llvm::SmallVector<std::pair<StringAttr, int32_t>> outDimSizes;
  for (auto [outDim, sizeLog2] : outDimSizesLog2) {
    outDimSizes.push_back({outDim, 1 << sizeLog2});
  }
  return LinearLayout(std::move(allBases), outDimSizes,
                      inner.isSurjective() && outer.isSurjective());
}

bool LinearLayout::isTrivialOver(ArrayRef<StringAttr> dimNames) const {
  for (StringAttr dim : dimNames) {
    if (!llvm::is_contained(getInDimNames(), dim) &&
        !llvm::is_contained(getOutDimNames(), dim)) {
      return false;
    }
  }

  auto getRemainingDimNames = [&](auto allDimNames) {
    SmallVector<StringAttr> remainingDimNames;
    for (StringAttr dim : allDimNames) {
      if (!llvm::is_contained(dimNames, dim)) {
        remainingDimNames.push_back(dim);
      }
    }
    return remainingDimNames;
  };
  SmallVector<StringAttr> remainingInDimNames =
      getRemainingDimNames(getInDimNames());
  SmallVector<StringAttr> remainingOutDimNames =
      getRemainingDimNames(getOutDimNames());

  // Think of this as a block-matrix multiplying a vector:
  // [[A, B],  *  [v_1,
  //  [C, D]]      v_2]
  // where v_2 is the dimNames and v_1 is the remainingInDimNames
  // We can quotient out dimNames iff they don't affect the remainingInDimNames
  // in the result. In other words, we want to check that B is zero, and C is
  // zero, and D is the identity
  return squareSublayoutIsIdentity(*this, dimNames) &&
         sublayoutIsZero(remainingInDimNames, dimNames) &&
         sublayoutIsZero(dimNames, remainingOutDimNames);
}

std::optional<LinearLayout>
LinearLayout::quotient(ArrayRef<StringAttr> dimNames) const {
  if (!isTrivialOver(dimNames)) {
    return std::nullopt;
  }

  // This should probably be even less general, where we ask inDimNames ==
  // outDimNames
  auto getRemainingDimNames = [&](auto allDimNames) {
    SmallVector<StringAttr> remainingDimNames;
    for (StringAttr dim : allDimNames) {
      if (!llvm::is_contained(dimNames, dim)) {
        remainingDimNames.push_back(dim);
      }
    }
    return remainingDimNames;
  };

  SmallVector<StringAttr> inDimNames = getRemainingDimNames(getInDimNames());
  SmallVector<StringAttr> outDimNames = getRemainingDimNames(getOutDimNames());

  return sublayout(inDimNames, outDimNames);
}

LinearLayout LinearLayout::sublayout(ArrayRef<StringAttr> inDimNames,
                                     ArrayRef<StringAttr> outDimNames) const {
  assertDimsSubsetIgnoringOrder(inDimNames, getInDimNames());
  assertDimsSubsetIgnoringOrder(outDimNames, getOutDimNames());
  SmallDenseSet<StringAttr> inDimSet(inDimNames.begin(), inDimNames.end());
  SmallDenseSet<StringAttr> outDimSet(outDimNames.begin(), outDimNames.end());

  SmallVector<int> outDimIndicesToKeep;
  for (auto [i, outDim] : llvm::enumerate(getOutDimNames())) {
    if (outDimSet.contains(outDim)) {
      outDimIndicesToKeep.push_back(i);
    }
  }
  BasesT newBases;
  for (auto [inDim, inDimBases] : bases) {
    if (!inDimSet.contains(inDim)) {
      continue;
    }
    auto &newInDimBases = newBases[inDim];
    for (auto &basis : inDimBases) {
      auto &newBasis = newInDimBases.emplace_back();
      for (int i : outDimIndicesToKeep) {
        newBasis.push_back(basis[i]);
      }
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
  for (auto [outDim, outDimSize] : outDims) {
    if (outDimSet.contains(outDim)) {
      newOutDims.push_back({outDim, outDimSize});
    }
  }
  return LinearLayout(std::move(newBases), std::move(newOutDims),
                      /*requireSurjective=*/false);
}

bool LinearLayout::sublayoutIsZero(ArrayRef<StringAttr> inDimNames,
                                   ArrayRef<StringAttr> outDimNames) const {
  LinearLayout ss = sublayout(inDimNames, outDimNames);
  for (auto [inDim, inDimBases] : ss.bases) {
    for (auto basis : inDimBases) {
      if (!llvm::all_of(basis, [](int32_t b) { return b == 0; })) {
        return false;
      }
    }
  }
  return true;
}

SmallVector<std::pair<StringAttr, int32_t>>
LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  assertDimsEqualIgnoringOrder(llvm::make_first_range(ins), getInDimNames());

  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (StringAttr outDim : getOutDimNames()) {
    int32_t outVal = 0;
    for (auto &[inDim, val] : ins) {
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        if (val & (1 << i))
          outVal ^= getBasis(inDim, i, outDim);
      }
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}

LinearLayout LinearLayout::compose(const LinearLayout &outer) const {
  assertDimsEqualIgnoringOrder(getOutDimNames(), outer.getInDimNames());
  for (StringAttr outDim : getOutDimNames()) {
    assert(getOutDimSize(outDim) <= outer.getInDimSize(outDim));
  }

  BasesT newBases;
  for (const auto &[inDim, inDimBases] : bases) {
    auto &newInDimBases = newBases[inDim];
    for (const auto &basis : inDimBases) {
      SmallVector<std::pair<StringAttr, int32_t>> bases;
      for (auto [outDim, b] : llvm::zip(getOutDimNames(), basis)) {
        bases.push_back({outDim, b});
      }
      auto newBases = outer.apply(bases);
      auto newBasesRange = llvm::make_second_range(newBases);
      newInDimBases.push_back(
          std::vector<int32_t>(newBasesRange.begin(), newBasesRange.end()));
    }
  }

  bool compositionIsSurjective =
      isSurjective() && outer.isSurjective() &&
      llvm::all_of(getOutDimNames(), [&](StringAttr outDim) {
        return getOutDimSize(outDim) == outer.getInDimSize(outDim);
      });
  return LinearLayout(std::move(newBases), llvm::to_vector(outer.outDims),
                      compositionIsSurjective);
}

namespace {
std::unique_ptr<uint64_t[]> concatMatrices(const LinearLayout &A,
                                           const LinearLayout &B) {
  // In plain words, "convert_layout does not change the shape of a tensor"
  assert(A.getTotalOutDimSizeLog2() == B.getTotalOutDimSizeLog2() &&
         "Matrices must have the same number of output dimensions");
  int numRows = A.getTotalOutDimSizeLog2();
  int numColsA = A.getTotalInDimSizeLog2();

  // rref expects the lower bits to be the lower indices of the matrix
  auto concat = getMatrix(A);
  auto BMat = getMatrix(B);
  for (int r = 0; r < numRows; r++) {
    concat[r] |= BMat[r] << numColsA;
  }
  return concat;
}

LinearLayout lstsq(const LinearLayout &A, const LinearLayout &B) {
  // Solve the least square system AX = B for A = outer, B = *this
  // and return the least square solution X of minimal norm
  // A and B may not be surjective, but we assume that Im(B) \subset Im(A)
  // Sketch of the algorithm:
  // https://github.com/triton-lang/triton/pull/5309#discussion_r1869084111
  int numRows = A.getTotalOutDimSizeLog2();
  int numColsA = A.getTotalInDimSizeLog2();
  int numColsB = B.getTotalInDimSizeLog2();
  int numCols = numColsA + numColsB;
  std::unique_ptr<uint64_t[]> combinedMat = concatMatrices(A, B);
  f2reduce::inplace_rref_strided(combinedMat.get(), numRows, numCols,
                                 /*stride=*/1);

  // Compute the pivot columns
  // Since A and B have the same image, each row will either have a pivot
  // or will be all zeros
  SmallVector<int32_t> pivotCols;
  for (int r = 0; r < numRows; r++) {
    auto row = combinedMat[r];
    if (row == 0) {
      continue;
    }
    int c = __builtin_ctzll(row);
    assert(c < numColsA && "Precondition broken. Im(B) not contained in Im(A)");
    assert(pivotCols.empty() ||
           pivotCols.back() < c && "Pivot columns are not in increasing order");
    pivotCols.push_back(c);
  }

  // Extract A^{-1}B and complete the matrix using zeros
  std::unique_ptr<uint64_t[]> retMat(new uint64_t[numColsA]());
  int j = 0;
  for (int r = 0; r < numColsA; r++) {
    auto isPivot = j < pivotCols.size() && pivotCols[j] == r;
    retMat[r] = isPivot ? combinedMat[j++] >> numColsA : 0;
  }

  // We need names for the in/out dim of the flattened layout we're going to
  // read off from `m`.  These could be anything, doesn't matter.
  StringAttr inDim1D = *A.getInDimNames().begin();
  StringAttr outDim1D = *A.getOutDimNames().begin();

  // Read off the new bases.  These are for a flattened 1D -> 1D
  LinearLayout::BasesT retBases;
  auto &bs = retBases[inDim1D];
  for (int c = 0; c < numColsB; c++) {
    int32_t basis = 0;
    for (int r = 0; r < numColsA; r++) {
      basis |= (retMat[r] >> c & 1) << r;
    }
    bs.push_back({basis});
  }

  LinearLayout retFlattened(std::move(retBases),
                            {{outDim1D, A.getTotalInDimSize()}},
                            /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> retInDims;
  SmallVector<std::pair<StringAttr, int32_t>> retOutDims;
  for (StringAttr dim : B.getInDimNames()) {
    retInDims.push_back({dim, B.getInDimSize(dim)});
  }
  for (StringAttr dim : A.getInDimNames()) {
    retOutDims.push_back({dim, A.getInDimSize(dim)});
  }
  return retFlattened.reshapeIns(retInDims).reshapeOuts(retOutDims);
}

} // namespace

LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const {
  // TODO(Lezcano) Make friend and perhaps rename to `convertFrom` or `lstsq`
  // For this, we need to implement our LLVM lowerings by inverting the "outer"
  // layout, and then iterating over the elements from the "this" layout and
  // fetching the corresponding element from the "outer" layout. This exercises
  // the broadcasting that we incentivise via choosing the minimum norm solution
  // in lstsq.

  // The order of dims does not matter. We choose to transpose outer
  auto outDims = llvm::to_vector(getOutDimNames());
  assertDimsEqualIgnoringOrder(outDims, outer.getOutDimNames());
  const auto &B = *this;
  const auto A = outer.transposeOuts(outDims);
  for (auto dim : outDims) {
    assert(A.getOutDimSize(dim) == B.getOutDimSize(dim) &&
           "Convert layout does not change the shape of a tensor");
  }

  // We'll write A^{-1} to mean the inverse or the pseudo-inverse of A
  // We are computing A^{-1}B so A must be surjective so that
  // it has a left inverse.
  assert(A.isSurjective());

  // Broadcasting heuristic
  // Imagine we have two layouts with `warps = [[0, 0],  [0, 0]]`
  // (broadcasting) on both layouts. We could map any warp to any warp in the
  // conversion. Now, we want to map them as the identity map, to mark that
  // nothing needs to be done there (`lstsq` would map all the warps to the
  // zero warp, minimum norm solution). The heuristic here is as follows:
  // - If a dimension is the same for both layouts, we want to map it as the
  // identity
  //   Equivalently, we don't add it to the conversion
  // - Otherwise, we just call lstsq (i.e. map all the equivalent elements
  //   to the same input element) to take advantage of broadcasting in shared
  //   memory and avoid saving repeated elements in shared memory
  SmallVector<StringAttr> identityDims;
  for (auto dim : A.getInDimNames()) {
    if (B.hasInDim(dim) &&
        A.sublayout(dim, outDims) == B.sublayout(dim, outDims)) {
      identityDims.push_back(dim);
    }
  }
  SmallVector<StringAttr> ANonIdentityInDims;
  SmallVector<StringAttr> BNonIdentityInDims;
  for (auto dim : A.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      ANonIdentityInDims.push_back(dim);
    }
  }
  for (auto dim : B.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim)) {
      BNonIdentityInDims.push_back(dim);
    }
  }

  auto AReduced = A.sublayout(ANonIdentityInDims, outDims);
  auto BReduced = B.sublayout(BNonIdentityInDims, outDims);

  // If one is empty, the other must be empty as well
  assert((AReduced == LinearLayout::empty()) ==
         (BReduced == LinearLayout::empty()));
  bool isEmpty = AReduced == LinearLayout::empty();

  auto ret = isEmpty ? LinearLayout::empty() : lstsq(AReduced, BReduced);

  // TODO(Lezcano): We should return the reduced layout instead of re-adding the
  // identity maps. With this, we'll be able to kill `minimalCvtLayout`

  // Add the identity maps for the dimensions that are the same for both layouts
  for (auto dim : identityDims) {
    ret *= LinearLayout::identity1D(A.getInDimSize(dim), dim, dim);
  }

  // Reorder the dimensions in the result to match the order expected by the
  // current and outer layouts.
  return ret.transposeIns(llvm::to_vector(B.getInDimNames()))
      .transposeOuts(llvm::to_vector(A.getInDimNames()));
}

LinearLayout LinearLayout::invert() const {
  assert(isInvertible() &&
         "A linear layout must be surjective and square to be invertible");
  return pseudoinvert();
}

LinearLayout LinearLayout::pseudoinvert() const {
  // A^-1(x) = A^-1(I(x)), thus A.invert() = I.invertAndCompose(A)
  assert(isSurjective() &&
         "A linear layout must be surjective to compute its pseudoinverse");
  LinearLayout identity = LinearLayout::empty();
  for (auto outDim : getOutDimNames()) {
    identity *= LinearLayout::identity1D(getOutDimSize(outDim), outDim, outDim);
  }
  return identity.invertAndCompose(*this);
}

llvm::MapVector<StringAttr, int32_t>
LinearLayout::getFreeVariableMasks() const {
  std::unique_ptr<uint64_t[]> mat = getMatrix(*this);
  int numRows = getTotalOutDimSizeLog2();
  int numCols = getTotalInDimSizeLog2();

  // stride is specified in number of 64-bit words per row, and we pack our
  // matrix so that there's only one uint64_t per row.
  assert(numCols <= 64);
  f2reduce::inplace_rref_strided(mat.get(), numRows, numCols, /*stride=*/1);

  // For each row in the RREF matrix, identify the column with the first "1".
  // These columns correspond to the basic (i.e. non-free) variables.
  std::set<int32_t> basicVars;
  for (int r = 0; r < numRows; r++) {
    if (mat[r] == 0) {
      continue;
    }
    basicVars.insert(__builtin_ctzll(mat[r]));
  }

  llvm::MapVector<StringAttr, int32_t> ret;
  int c = 0;
  for (StringAttr dim : getInDimNames()) {
    int32_t mask = 0;
    for (int i = 0; i < getInDimSizeLog2(dim); i++, c++) {
      if (basicVars.count(c) == 0) {
        mask |= (1 << i);
      }
    }
    ret[dim] = mask;
  }
  return ret;
}

LinearLayout LinearLayout::removeZeroBasesAlongDim(StringAttr stripDim) const {
  LinearLayout::BasesT result;
  for (auto &[inDim, inDimBases] : getBases()) {
    auto &newInDimBases = result[inDim];
    if (inDim != stripDim) {
      newInDimBases = inDimBases;
      continue;
    }
    for (auto &basis : inDimBases) {
      if (llvm::any_of(basis, [](int32_t val) { return val != 0; })) {
        newInDimBases.push_back(basis);
      }
    }
  }
  return LinearLayout(std::move(result), llvm::to_vector(getOutDimNames()));
}

size_t hash_value(const LinearLayout &layout) {
  size_t seed = 0;

  // Hash the bases
  for (const auto &base : layout.getBases()) {
    // Hash the input dimension name
    seed = llvm::hash_combine(seed, base.first);

    // Hash the vectors in bases
    for (const auto &vec : base.second) {
      for (int32_t val : vec) {
        seed = llvm::hash_combine(seed, val);
      }
    }
  }

  // Hash the output dimensions and their sizes
  for (const auto &outDim : layout.getOutDimNames()) {
    seed = llvm::hash_combine(seed, outDim, layout.getOutDimSize(outDim));
  }
  // Don't hash the surjective flag as it's a cached property
  return seed;
}

bool operator==(LinearLayout lhs, LinearLayout rhs) {
  if (!lhs.equalIgnoringOutDimSizes(rhs))
    return false;

  for (const auto &[lhsOutDimAndSize, rhsOutDimAndSize] :
       llvm::zip(lhs.outDims, rhs.outDims)) {
    if (lhsOutDimAndSize.second != rhsOutDimAndSize.second)
      return false;
  }
  return true;
}

bool LinearLayout::equalIgnoringOutDimSizes(const LinearLayout &other) const {
  // llvm::MapVector doesn't have an operator== :(.
  if (llvm::to_vector(this->getOutDimNames()) !=
      llvm::to_vector(other.getOutDimNames()))
    return false;
  if (this->bases.size() != other.bases.size())
    return false;
  for (auto it1 = this->bases.begin(), it2 = other.bases.begin();
       it1 != this->bases.end(); ++it1, ++it2) {
    if (*it1 != *it2)
      return false;
  }
  return true;
}

std::string LinearLayout::toString() const {
  // Start with a newline because we print out a bulleted list; it doesn't
  // make sense for the first line of this list to be on the same line as
  // any previous text.
  std::string ret = "\n";
  std::string outDimsStr =
      "[" +
      join(outDims, ", ",
           [](auto dimAndSize) {
             auto [outDim, size] = dimAndSize;
             return outDim.str() + " (size " + std::to_string(size) + ")";
           }) +
      "]";

  if (bases.empty()) {
    if (outDims.empty()) {
      return "\n(empty layout)";
    } else {
      return "\n(empty layout with out-dims " + outDimsStr + ")";
    }
  }

  // TODO: Add spaces for alignment.
  for (const auto &[inDim, inDimBases] : bases) {
    if (inDimBases.empty()) {
      ret += " - " + inDim.str() + " is a size 1 dimension\n";
      continue;
    }

    ret += " - " +
           join(llvm::seq(inDimBases.size()), "\n   ",
                [&, &inDim = inDim, &inDimBases = inDimBases](int i) {
                  return inDim.str() + "=" + std::to_string(1 << i) + " -> (" +
                         join(inDimBases[i], ", ") + ")";
                }) +
           "\n";
  }
  ret += "where out dims are: " + outDimsStr;
  return ret;
}

} // namespace mlir::triton
