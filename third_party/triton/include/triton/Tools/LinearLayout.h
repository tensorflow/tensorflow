#ifndef TRITON_TOOLS_LINEARLAYOUT_H
#define TRITON_TOOLS_LINEARLAYOUT_H

#include <cstdint>
#include <numeric>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton {

// # High-level overview of linear layouts
//
// The idea for linear layouts is due to Adam P. Goucher.
//
// In Triton, a linear layout (LL) is a function that maps from a "hardware
// location" to a "logical tensor index".
//
// For example, suppose we have a 2D tensor T stored in GPU registers.  T's
// layout (i.e., L) is the function that, given a "hardware location" tuple of
// (thread-id, warp-id), returns an index (x,y) into T.  In other words, if
// L(t,w) = (x,y) is our linear layout func, then a register in thread t in warp
// w contains the value T[x,y].
//
// The key fact about LLs is, the mapping from (t,w) to (x,y) is not arbitrary.
// We only need to specify the value of L(t,w) at certain special points
// (namely, the values L(t,0) and L(0,w) where t and w are powers of 2), and
// from those we can compute all the other values of L.
//
// Here's an example LL where we have 4 warps and 4 threads per warp, and the
// tensor T has shape 4x4.  We define the function L by choosing the values of
// L(0,1), L(0,2), L(1,0), and L(2,0).  Our choices are shown below.
//
//               t/w    0     1     2    3
//               0      ? (0,1) (0,2)    ?
//    L(t,w) =   1  (1,1)     ?     ?    ?
//               2  (2,2)     ?     ?    ?
//               3      ?     ?     ?    ?
//
// You only need to specify these four values to define the whole linear layout.
// These special values are called the "basis vectors" or "bases" of the layout.
// We complete the table by xor'ing together the bases, according to the
// following rule.  (I write "⊕" for xor.)
//
//    L(t1 ⊕ t2, w1 ⊕ w2) = L(t1, w1) ⊕ L(t2, w2)  (linearity rule).
//
// The linearity rule plus our four choices allows us to fill in the whole
// table.  Here's how we might compute some of the values.
//
//    L(0,0) = L(1 ⊕ 1, 0 ⊕ 0) = L(1,0) ⊕ L(1,0) = (1,1) ⊕ (1,1) = (0,0)
//    L(0,3) = L(0 ⊕ 0, 2 ⊕ 1) = L(0,2) ⊕ L(0,1) = (0,2) ⊕ (0,1) = (0,3)
//    L(3,0) = L(2 ⊕ 1, 0 ⊕ 0) = L(2,0) ⊕ L(1,0) = (2,2) ⊕ (1,1) = (3,3)
//    L(3,3) = L(3 ⊕ 0, 0 ⊕ 3) = L(3,0) ⊕ L(0,3) = (3,3) ⊕ (0,3) = (3,0).
//
// (Notice it's a consequence of the linearity rule that L(0,0) = (0,0), no
// matter what values we chose for the table.)
//
// The whole table looks like this.
//
//              t/w   0     1     2     3
//              0  (0,0) (0,1) (0,2) (0,3)
//    L(t,w) =  1  (1,1) (1,0) (1,3) (1,2)
//              2  (2,2) (2,3) (2,0) (2,1)
//              3  (3,3) (3,2) (3,1) (3,0).
//
// Careful readers will recognize this as a classic "swizzled" layout where
// (t, w) -> (t, w ⊕ t).  To go from this formula to an LL, you only need to
// compute the results at input points (0,1), (0,2), (1,0), and (2,0).

// Indeed the whole point of LLs is that they allow us to specify transposed and
// swizzled layouts as a "general case".  Instead of a layout class for
// registers in a thread, and another layout for registers in a thread but in
// MMAv2 order, and so on, all of these can be represented by different LLs.
// This gets rid of special cases and lets us write more general code.
//
// In this example, L was a 2D -> 2D function, but LLs are general MD -> ND
// functions.  In practice, a GPU register layout usually has input dims (reg,
// thread-id, warp-id, block-id), where reg represents the fact that one thread
// may store values for the tensor in multiple registers.
//
// To summarize, a linear layout is a function from tuples of integers to tuples
// of integers.  We specify some key values of the function, and then we can
// compute all the other values using the linearity rule.
//
// Here are the key things you can do with linear layout objects.
//
//  1. Given an LL, construct a new LL by modifying it or combining it with
//     another LL.
//
//  2. "Apply" an LL, i.e. use it to map an input index to an output index.
//     A function for this that uses LLVM-dialect MLIR as its input and output
//     lives in TritonGPUToLLVM.h.
//
//  3. Convert an existing Triton layout (e.g. BlockedLayoutAttr) to an LL.
//     These functions live in TritonGPU/LinearLayoutConversions.h.  During
//     TTGIR -> LLVM codegen, we convert Triton layouts to linear layouts and
//     then apply them.  In the future, we intend to remove the Triton layouts
//     entirely.
//
// # Examples of linear layouts
//
// 1. The 1D identity layout.  This maps L(x) = x.
//
//    Recall that our bases are the values of L(x) where x is a power of two.
//    So for e.g. an 8-element layout, we have L(1) = 1, L(2) = 2, L(4) = 4, and
//    therefore our bases are [1, 2, 4].
//
// 2. The 1D zeros layout.  This maps L(x) = 0.
//
//    For an 8-element layout, we have L(1) = L(2) = L(4) = 0, so our bases are
//    [0, 0, 0].
//
// 3. A 2D -> 2D identity layout.  Our basis vectors are the values of L(x,0)
//    and L(0,y) where x and y are powers of two.  The bases are
//
//    - L(0,1) = (0,1)
//    - L(0,2) = (0,2)
//    - L(1,0) = (1,0)
//    - L(2,0) = (2,0).
//
// 4. A 2D -> 2D transpose layout.  For a 4x4 layout, we have:
//
//    - L(0,1) = (1,0)
//    - L(0,2) = (2,0)
//    - L(1,0) = (0,1)
//    - L(2,0) = (0,2).
//
// 5. A 1D -> 1D "transpose" layout.  Consider the 16-element layout that maps
//
//    x    = 0 1 2 3 4 5 6 7 8 9 A B C D E F
//    L(x) = 0 4 8 C 1 5 9 D 2 6 A E 3 7 B F.
//
//    The bases are [L(1), L(2), L(4), L(8)] = [4, 8, 1, 2].  You can also think
//    of this as a rearrangement of the 1D identity layout [1, 2, 4, 8].
//
// 6. A 2D -> 1D broadcasted layout.  L(x,y) = x.  For a 4x4 -> 4 layout, our
//    bases are
//
//    - L(0,1) = 0
//    - L(0,2) = 0
//    - L(1,0) = 1
//    - L(2,0) = 2.
//
// # Implementation notes
//
// ## Dimension order
//
// An LL's input and output dimensions have an order.  This order only affects
// the reshapeIns/Outs and similar operations, where the layout is logically
// flattened according to the dimension order and then chopped up again.
//
// ## Surjectivity and injectivity
//
// Most LLs are surjective, i.e. all output values are covered by some input
// value.  But occasionally you might create a non-surjective layout, usually
// via invertAndCompose.  We aggressively assert that LLs are surjective unless
// you explicitly create one that's not.
//
// LLs are not, in general, injective.  There might exist multiple input values
// that map to the same output value.  This represents the idea that the same
// logical tensor elements can be stored in multiple places in the hardware.
//
// ## Why map hardware loc -> tensor index and not the other way around?
//
// In Triton, a linear layout usually tells us which logical tensor value is
// stored at a particular place in the hardware.  For example, an LL might map
// the tuple (thread-id, warp-id, block-id) to a 2D index into a tensor, (x,y),
// meaning that the register at (t,w,b) has value tensor[x,y].  Or it might map
// from a shared memory (offset, block) to a tensor index.
//
// It might seem more natural to go the other way around, from tensor index to
// place in the hardware.  But a particular tensor[x,y] value might be stored in
// more than one place in the hardware, so if we went in this direction, the
// layout would no longer be a proper function.  This would complicate
// everything else.
//
// # Optional mathematical background: Linear functions over GF(2)
//
// (You shouldn't need to understand this math to use linear layouts, but it
// helps with the implementation.)
//
// One way to define a linear function is to say it's any function F that can be
// written as
//
//    L(a) = a1 * B1 + a2 * B2 + ... + aM * BM,
//
// where
//
//   - a is a vector [a1...aM], and ai is a scalar in some field 𝔽 (for
//     example, ai might be a real number), and
//   - each Bj is a vector [b1j, b1j, ..., bNj] of N scalars in 𝔽.
//
// We can also write this as a matrix-vector product Ba, where
//
//    - a is the column vector [a1, ..., aM] and
//
//    - B is the matrix formed by concatenating the column vectors B1, ..., BM:
//
//           | ↑    ↑         ↑ |
//       B = | B1,  B2, ...,  BM|
//           | ↓    ↓         ↓ |
//
//           |b11, b12, ..., b1M|
//           |b21, b22, ..., b2M|
//         = | ↓    ↓         ↓ |
//           |bN1, bN2, ..., bNM|.
//
// Usually when we do linear algebra, the field 𝔽 from which `ai` and `bij` are
// drawn is the real or complex numbers.  But in linear layouts, we let	𝔽 be a
// different field: GF(2).
//
// GF(2) is the two-element field of bits.  To define a field, I need to give
// you the set of elements and also addition and multiplication operations.  For
// GF(2) the elements are simply {0,1}.  We define addition as xor, and
// multiplication as binary `and`.
//
// Here's an example of a 4x4 matrix-vector multiply where the elements are in
// GF(2).  I'm using ⊕ to represent GF(2)'s addition operation (i.e xor) and ×
// to represent multiplication (i.e. binary `and`).
//
//    | 1 0 0 0 | | 0 |     | 1 |         | 0 |         | 0 |         | 0 |
//    | 0 1 1 0 | | 1 |  =  | 0 | × 0  ⊕  | 1 | × 1  ⊕  | 1 | × 1  ⊕  | 0 | × 0
//    | 0 0 1 1 | | 1 |     | 0 |         | 0 |         | 1 |         | 1 |
//    | 0 0 1 1 | | 0 |     | 0 |         | 0 |         | 1 |         | 1 |
//
//                                        | 0 |         | 0 |
//                       =                | 1 |    ⊕    | 1 |
//                                        | 0 |         | 1 |
//                                        | 0 |         | 1 |
//
//                          | 0 |
//                       =  | 0 |.
//                          | 1 |
//                          | 1 |
//
// This works, but it's cumbersome.  It's more compact to think of the vector
// `a` as an M-bit integer, and each column Bi of the matrix B as an N-bit
// integer.  Here's the same matrix-vector product written this way.
//
//   = | 1 2 14 12 | × 6
//   = | 1 2 14 12 | × 0b0110
//   = (1 × 0) ⊕ (2 × 1) ⊕ (14 × 1) ⊕ (12 × 0)
//   = 2 ⊕ 14
//   = 12.
//
// And we confirm that our answer of 12 is equal to the binary value 0b1100 we
// got before.
//
// Notice that the function F(a) is fully specified by the matrix B, and that
// the four columns of B tell us the values of F at power-of-two values for `a`,
// namely F(1), F(2), F(4), and F(8).  In other words, we specify four results
// of F(x) (we call these the function's "basis vectors" or its "bases") and we
// can then compute any other value by xor'ing together subsets of the bases.
//
// In the case of a 1D -> 1D layout, the implementation of an LL is
// straightforward from the mathematical description.  If the LL is
// higher-dimensional, we can "stack" the bit vectors to create 1D vectors.
// For example, if we have a 2D LL and we're given input tuple (0b0011, 0b1100),
// we can treat this like a 1D input 0b0011'1100 and then do the regular 1D LL
// computation.  Similarly we can "unstack" the output from 1D to ND.
//
// The linearity rule presented earlier is perhaps misleading at this point.  In
// the 1D view of things, we really only need
//
//    L(x ⊕ y) = L(x) ⊕ L(y)  (1D linearity rule),
//
// which is part of the definition of L being a linear function.  The new 1D
// linearity rule plus stacking/unstacking is equivalent to the earlier
// N-dimensional linearity rule.
//
// That's all we need in order to define linear layouts mathematically!
//
// # Comparison to Nvidia CuTe
//
// (Note, I'm not an expert on CuTe; this is my best understanding.)
//
// CuTe is a programmatic layout system that's part of Nvidia CUTLASS; see
// https://github.com/NVIDIA/cutlass/blob/629f465/media/docs/cute/00_quickstart.md
//
// LLs and CuTe solve similar problems.  Before CuTe, CUTLASS v2 had many
// handcrafted layouts, "RowMajor", "VoltaTensorOpMultiplicandCongruous", etc,
// see https://www.youtube.com/watch?v=QLdUML5MCfE&t=574s.  Each of these was a
// special case.  CUTLASS v3 introduced CuTe layouts, which are programmable and
// subsume all of these special cases.  The CUTLASS folks say this simplified
// CUTLASS, in the same way that we hope LLs will simplify Triton.
//
// Like CuTe layouts, LLs are also programmable and composable.  But there are
// also some differences.
//
//  - Dimensions in LLs are named; CuTe dimensions are numbered.
//  - CuTe layouts can be nested; LLs cannot be.  (Nesting doesn't give CuTe
//    layouts additional power; any nested layout can be flattened.)
//  - CuTe layouts support non-power-of-two shapes; LLs do not.  In particular
//    this means that LLs cannot represent padded layouts.
//  - In CuTe, swizzling is a separate step applied after specifying a layout.
//    In LLs, swizzling is part of the layout itself.
//  - The structure of LLs allows us to programmatically search for layouts that
//    satisfy certain requirements, for example a shared layout that doesn't
//    have bank conflicts when read into a particular register layout.  CuTe
//    expects a human to choose the layout using their brain.
//  - CuTe emits code that is in the critical path of your CPU and GPU programs,
//    therefore it needs to be fast.  It uses C++ template magic to specialize
//    on known-sized dimensions, and so on.  LLs themselves do not need to be
//    fast; only the emitted `apply` code is on the critical path.
//  - CuTe requires a CUDA compiler such as nvcc; LLs do not.
//
class LinearLayout {
private:
  // bases[inDim][i] = L(0, ..., inDim=2^i, ..., 0).  All other values of L are
  // computed by xor'ing bases together, using the linearity rule.  In addition:
  //
  // - Each inDim has the same set of outDims, in the same order.
  // - The order of dims is minor-to-major, although this only affects reshape.
  llvm::MapVector<StringAttr /*inDim*/,
                  std::vector<std::vector<int32_t> /*size=getNumOutDims()*/>
                  /*size=getInDimSizeLog2(inDim)*/>
      bases;

  llvm::MapVector<StringAttr, int32_t /*size*/> outDims;
  bool surjective;

public:
  using BasesT = decltype(bases);

  // The 0-dimensional layout that maps everything to 0.  This is useful as a
  // starting point when doing something like
  //
  //   LinearLayout ret = LinearLayout::empty();
  //   for (...) ret *= ...;
  //   return ret;
  static LinearLayout empty() { return LinearLayout(BasesT{}, {}); }

  // Creates a 1D -> 1D layout that's the identity function, i.e. L(x) = x
  // for x in [0, size).
  static LinearLayout identity1D(int32_t size, StringAttr inDim,
                                 StringAttr outDim);

  // Creates a 1D -> 1D layout that maps every input value to 0, i.e. L(x) = 0
  // for x in [0, size). By default this creates a surjective layout where
  // `outDim` has size 1 (the only element is 0). If `outDimSize` is specified
  // to be greater than 1, then this creates a non-surjective layout with a
  // specific size for `outDim`.
  static LinearLayout zeros1D(int32_t size, StringAttr inDim, StringAttr outDim,
                              int32_t outDimSize = 1);

  // Creates a LinearLayout from a list of bases.  These are interpreted
  // according to the rules written for the member variable `bases`.
  //
  // Calculates the out-dim sizes according to the bases.  Consider the
  // following example.
  //
  //   L(in1=1) = (out1=1, out2=0)
  //   L(in1=2) = (out1=5, out2=1)
  //   L(in1=4) = (out1=2, out2=2)
  //
  // To calculate the out-dim sizes, we first find the largest values for out1
  // and out2, namely 5 and 2, then round these up to the next power of 2,
  // namely 8 and 4.  These are the out-dim sizes.
  //
  // Assert-fails if the layout is not surjective given these out-dim sizes.
  // That is, every possible out-dim in range [0, size) must be produced by
  // xor'ing some combination of bases.
  explicit LinearLayout(BasesT bases, ArrayRef<StringAttr> outDimNames);

  // Creates a LinearLayout given a list of bases and the explicit out-dimension
  // sizes.  Allows the layout to be non-surjective.
  //
  // To see why we need to explicitly pass out-dim sizes when creating a
  // non-surjective layout, consider the following example.
  //
  //   L(in1=1) = 1
  //   L(in1=2) = 4
  //
  // If we naively infer the out-dim sizes from these bases, we'd infer a size
  // of nextPow2(4) = 8.  But given that the layout is non-surjective, who is to
  // say that the codomain is not (say) [0,32)?  We can't tell, thus we need to
  // be explicit about the sizes.
  explicit LinearLayout(BasesT bases,
                        ArrayRef<std::pair<StringAttr, int32_t>> outDims,
                        bool requireSurjective);

  // Construct a LinearLayout from an explicit list of bases.  (This constructor
  // is needed because llvm::MapVector does not have a constructor that accepts
  // an initializer_list.)
  //
  // For example, given these bases
  //
  //   L(in1=1, in2=0) = (out1=0, out2=1)
  //   L(in1=2, in2=0) = (out1=0, out2=2)
  //   L(in1=0, in2=1) = (out1=0, out2=4)
  //   L(in1=0, in2=2) = (out1=0, out2=8)
  //   L(in1=0, in2=4) = (out1=1, out2=1)
  //
  // we can use this constructor to build an equivalent LL:
  //
  // LinearLayout({
  //     {"in1", {/*L(in1=1)=*/{0,1}, /*L(in1=2)=*/{0,2}}},
  //     {"in2", {/*L(in2=1)=*/{0,4}, /*L(in2=2)=*/{0,8}, /*L(in2=4)=*/{1,1}}},
  //   },
  //   {"out1", "out2"})
  //
  // The overload that infers out-dim sizes assert-fails if the layout is not
  // surjective.
  explicit LinearLayout(
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
      ArrayRef<StringAttr> outDimNames);
  explicit LinearLayout(
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases,
      ArrayRef<std::pair<StringAttr, int32_t>> outDims, bool requireSurjective);

  bool isSurjective() const { return surjective; }

  bool isInvertible() const {
    return surjective && getTotalInDimSize() == getTotalOutDimSize();
  }

  const BasesT &getBases() const { return bases; }

  // Get the pos'th basis vector for the inDim -> outDim mapping.
  // getBasis(inDim, pos) = L(0, ..., inDim = 2^pos, ..., 0).
  ArrayRef<int32_t> getBasis(StringAttr inDim, int32_t pos) const {
    auto it = bases.find(inDim);
    assert(it != bases.end());
    assert(pos < it->second.size());
    return it->second[pos];
  }

  int32_t getBasis(StringAttr inDim, int32_t pos, StringAttr outDim) const {
    return getBasis(inDim, pos)[getOutDimIndex(outDim)];
  }

  // These are in minor-to-major order, although if you don't flatten the dims
  // (e.g. by reshaping) then the order doesn't really affect anything.
  auto getInDimNames() const { return llvm::make_first_range(bases); }
  auto getOutDimNames() const { return llvm::make_first_range(outDims); }
  auto getOutDimSizes() const { return llvm::make_second_range(outDims); }

  // Gets the position that this outDim occupies in getOutDimNames().  Asserts
  // if the dim is not present.
  int32_t getOutDimIndex(StringAttr outDim) const;

  bool hasInDim(StringAttr inDim) const { return bases.contains(inDim); }
  bool hasOutDim(StringAttr outDim) const { return outDims.contains(outDim); }

  int32_t getNumInDims() const { return bases.size(); }
  int32_t getNumOutDims() const { return outDims.size(); }

  // Asserts if the dimension is not present.
  int32_t getInDimSizeLog2(StringAttr inDim) const;
  int32_t getInDimSize(StringAttr inDim) const {
    return 1 << getInDimSizeLog2(inDim);
  }

  int32_t getTotalInDimSizeLog2() const;
  int32_t getTotalInDimSize() const { return 1 << getTotalInDimSizeLog2(); }

  // getOutDimSize(dim) == s means that there exists an input value that will
  // produce each output value in [0,s) (if the layout is surjective).
  //
  // For example, if our bases are
  //
  //   L(in0=1) = 1
  //   L(in0=2) = 4
  //   L(in1=1) = 2
  //   L(in1=2) = 8
  //
  // then the largest value we can produce is L(3,3) = 1 ⊕ 4 ⊕ 2 ⊕ 8 = 15 (and
  // indeed we can produce all values in [0,16) by xor'ing subsets of the bases
  // 1,2,4,8), so getOutDimSize(out_dim0) == 16.
  //
  // Asserts if the dimension is not present.
  int32_t getOutDimSizeLog2(StringAttr outDim) const;
  int32_t getOutDimSize(StringAttr outDim) const {
    return 1 << getOutDimSizeLog2(outDim);
  }

  int32_t getTotalOutDimSizeLog2() const;
  int32_t getTotalOutDimSize() const { return 1 << getTotalOutDimSizeLog2(); }

  // Finds the number of consecutive input elements in the first input dimension
  // that map to consecutive output elements in the first output dimension.
  //
  // Mathematically, finds the maximum value V such that for any a, b, c, and
  // for all v in [0,V),
  //
  //   L(a*V + v, b, c, ...) = L(a*V, b, c, ...) + (v, 0, ..., 0)
  //
  // Note that's +, not ⊕, in the RHS.  (Equivalently, we could use binary-or
  // instead of +.  In other words, we require that L(a*V, b, c, ...) have no
  // bits that overlap with v.)
  //
  // For example, if L maps (register, lane) to (dim1, dim0), then this tells
  // you how many consecutive registers map to consecutive elements of dim1.
  //
  // This only works across the first (i.e. the most-minor) dimension of in/out.
  // If you want it to work across more dimensions, flatten the layout.
  //
  // TODO(jlebar): Replace with divideLeft.
  int32_t getNumConsecutiveInOut() const;

  // Reorders the in/out dimensions of the layout.  This is mostly cosmetic
  // (affecting e.g. the order of getIn/OutDimNames), but it also affects the
  // behavior of reshape.
  [[nodiscard]] LinearLayout
  transposeIns(ArrayRef<StringAttr> newInDimOrder) const;
  [[nodiscard]] LinearLayout
  transposeOuts(ArrayRef<StringAttr> newOutDimOrder) const;

  [[nodiscard]] LinearLayout reshapeIns(
      ArrayRef<std::pair<StringAttr /*inDimName*/, int32_t /*size*/>> newInDims)
      const;

  // Reshapes to a single input dim (named whatever our first in-dim is named).
  [[nodiscard]] LinearLayout flattenIns() const {
    if (getNumInDims() == 0) {
      return reshapeIns({});
    }
    return reshapeIns({{*getInDimNames().begin(), getTotalInDimSize()}});
  }

  [[nodiscard]] LinearLayout
  reshapeOuts(ArrayRef<std::pair<StringAttr /*outDimName*/, int32_t /*size*/>>
                  newOutDims) const;

  // Reshapes to a single out dim (named whatever our first out-dim is named).
  [[nodiscard]] LinearLayout flattenOuts() const {
    if (getNumOutDims() == 0) {
      return reshapeOuts({});
    }
    return reshapeOuts({{*getOutDimNames().begin(), getTotalOutDimSize()}});
  }

  // Concatenates two layouts by their input dimensions. The layouts must have
  // the same output dimensions and sizes and different input dimensions. The
  // input dimensions of this layout are placed before those of 'other'. This
  // can be thought of as the opposite of `sublayout`, which slices a layout
  // from a larger one.
  [[nodiscard]] LinearLayout concatIns(const LinearLayout &other) const;
  // Concatenates two layouts by their output dimensions. The layouts must have
  // the same input dimensions and sizes and different output dimensions. The
  // output dimensions of this layout are placed before those of 'other'. This
  // can be thought of as the opposite of `sublayout`, which slices a layout
  // from a larger one.
  [[nodiscard]] LinearLayout concatOuts(const LinearLayout &other) const;

  // Computes the direct sum of two layouts.
  // https://en.wikipedia.org/wiki/Direct_sum#Direct_sum_of_matrices
  //
  // Roughly speaking, the first layout acts on the first part of the input
  // dimensions, and the second layout acts on the second part.
  // In other words, it's the generalisation of concatenation of the inputs
  // to linear maps.
  //
  // Examples:
  //
  //  - empty() is the multiplicative identity:
  //
  //      L * empty() == empty() * L == L.
  //
  //  - Multiplying two identity1D layouts with disjoint in/out dimensions gives
  //    a 2D identity layout:
  //
  //      identity1D(4, "i1", "o1") * identity1D(8, "i2", "o2") =>
  //      L(i1,i2) = (i1,i2),
  //
  //    with in-dims ("i1", "i2") and out-dims ("o1", "o2"), in that order.
  //
  //  - If out-dims overlap, they are combined, as in the following examples.
  //
  //    - identity1D(4, "i", "o") * identity1D(2, "i", "o") ==
  //      identity1D(8, "i", "o")
  //      The output matrix is [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  //
  //    - identity1D(4, "i", "o") * zeros1D(2, "i", "o") => L(x) = x % 4
  //      for x in [0,8).
  //      The output matrix is [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
  //
  //    - zeros1D(2, "i", "o") * identity1D(4, "i", "o") => L(x) = x / 2
  //      for x in [0,8).
  //      The output matrix is [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

  //    - identity1D(4, "i", "o1") * identity1D(8, "i", "o2") =>
  //      L(x) = (x % 4, x / 4) for x in [0,32).
  //      The output dims are ("o1", "o2") in that order.
  //
  // If the input (or output) dims of the layouts are not the same, we take
  // the supremum of the two ordered lists with the inclusion, respecting the
  // order. If multiple suprema exist, we bias towards the first list.
  // e.g. sup([a, b], [a, c]) = [a, b, c], sup([a, b], [b, c]) = [a, b, c]
  //      sup([a, b], [b, a]) = error! Supremum does not exist.
  //
  // Notice that this operation is not commutative, but it is associative.
  //
  // Requires: Any in/out dimensions which are in both outer and inner appear in
  // the same relative order.
  //
  // Postcondition: If both inner and outer are surjective, the result is
  // surjective.
  friend LinearLayout operator*(LinearLayout inner, LinearLayout outer);
  LinearLayout &operator*=(LinearLayout outer) {
    *this = *this * outer;
    return *this;
  }

  // Returns true if this layout acts trivially (as the identity) on the given
  // dimensions. This means that it's the identity on those dimensions, and it
  // does not map other dimensions onto those or these onto other dimensions.
  bool isTrivialOver(ArrayRef<StringAttr> dimNames) const;

  // For an endomorphism on dimNames (linear map that maps dimNames to dimNames)
  // checks whether it is the identity map on these dimensions (i.e
  // LinearLayouts::isTrivialOver) and if so, returns the sublayout of the
  // remaining dimensions.
  // nb. The isTrivialOver condition is more restrictive than the usual
  //     "leaves the subspace invariant" condition in maths.
  //     We can always relax it if we know how to take advantage of a conversion
  //     layout being block-diagonal in the future.
  std::optional<LinearLayout> quotient(ArrayRef<StringAttr> dimNames) const;

  // Gets a layout with only these in/out dimensions.
  //
  // In other words, gets a layout where the in-dims not mentioned in inDimNames
  // are set to 0, and the out-dims not mentioned in outDimNames are omitted.
  //
  // The output-dim sizes are unchanged.  The order of the in/out dims in the
  // returned layout matches the order of the original layout, not the order of
  // the arguments.
  LinearLayout sublayout(ArrayRef<StringAttr> inDimNames,
                         ArrayRef<StringAttr> outDimNames) const;

  // Is the sublayout restricted to inDimNames + outDimNames all zeros?
  bool sublayoutIsZero(ArrayRef<StringAttr> inDimNames,
                       ArrayRef<StringAttr> outDimNames) const;

  // Computes and returns L(x, y, z).
  //
  // If you want to apply the layout to mlir Values instead of integers, that
  // function lives in TritonGPUToLLVM/Utility.h.
  SmallVector<std::pair<StringAttr, int32_t>>
  apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const;

  // Creates a new layout which is equivalent to running this layout, then
  // running `outer`.  That is,
  //
  //  - let this layout be L(x), and
  //  - let `outer` be O(x).
  //  - Then compose(outer) returns the layout (O∘L)(x), aka O(L(x)).
  //
  // Requires:
  //   - The output dimensions of this layout equal the input dimensions of
  //     outer (order doesn't matter).
  //   - For each output dim d of this layout, this->getOutDimSize(d) <=
  //     outer.getInDimSize(d).
  //
  // Postcondition: The result is surjective iff `this` and `outer` are
  // surjective and this->getOutDimSize(d) == outer.getInDimSize(d) for each of
  // this->getOutDimNames().
  //
  [[nodiscard]] LinearLayout compose(const LinearLayout &outer) const;

  // Inverts or pseudo-inverts `outer` and composes it with `this`.
  //
  // Formally, if C = A.invertAndCompose(B), then for all x, C(x) = y implies
  // A(x) = B(y), or in other words A(x) = B(C(x)).  If B is invertible, then
  // C(x) = B^-1(A(x)), which is how this function gets its name.
  //
  // For example, suppose you have the following two LLs.
  //
  //   - R is an LL representing registers, mapping (lane, warp) to a 2D index.
  //   - S is an LL representing shared memory, mapping offset to a 2D index.
  //
  // Suppose you want to store tensor values from registers into shared memory.
  // That is, given a (lane, warp), you want to know the corresponding shared
  // memory offset to store into.
  //
  // This is equivalent to converting a (lane, warp) into a 2D index (i.e.
  // applying R), then converting a 2D index into a shmem offset (i.e. applying
  // the inverse of S).  R.invertAndCompose(S) computes this transformation.
  //
  // Notice the following requirements in order for this to work.
  //
  //   - R and S must have the same output dimension names (different order is
  //     allowed).
  //   - S must be surjective, i.e. there must be some offset for each output
  //     dimension of S.  This way when we compose S^-1 with R, every possible
  //     2D index that we might get from R has some shmem offset.
  //   - The codomain of S must be at least as large as the codomain of R.
  //     Otherwise, R could map some tensor index that is not stored in S.
  //
  // One requirement we *don't* have is that S is injective; we allow two shmem
  // offsets to hold the same 2D index.  If S is not injective,
  // the algorithm chooses the smallest offset for a given (lane, warp).
  [[nodiscard]] LinearLayout invertAndCompose(const LinearLayout &outer) const;

  // Get the layout that is the inverse of this layout.
  [[nodiscard]] LinearLayout invert() const;
  // Compute and return a psueodinverse of this layout. This is a layout such
  // that `B = A.psuedoinvert()` implies that `A(B(x)) = I`. If `A` is
  // invertible, then this returns `A^-1`.
  [[nodiscard]] LinearLayout pseudoinvert() const;

  // For each in-dim, returns a bitmask of the "free variables" in the layout
  // function.
  //
  // These are the bits in the input that can be changed without changing the
  // output.  If all of the free variables are 0, then the layout is injective
  // (i.e. every input bit affects the output).
  llvm::MapVector<StringAttr, int32_t> getFreeVariableMasks() const;

  // Take the current linear layout and remove all zero bases for the provided
  // dimension and return the resulting layout. This is useful for deriving a
  // layout that returns just the unique output values when varying a given
  // input dimension that has broadcasting.
  [[nodiscard]] LinearLayout removeZeroBasesAlongDim(StringAttr stripDim) const;

  std::string toString() const;

  friend bool operator==(LinearLayout lhs, LinearLayout rhs);
  friend bool operator!=(LinearLayout lhs, LinearLayout rhs) {
    return !(lhs == rhs);
  }
  bool equalIgnoringOutDimSizes(const LinearLayout &other) const;
  friend size_t hash_value(const LinearLayout &layout);

private:
  // Factory function that gracefully fails rather than asserts if the layout is
  // not well-formed.
  static std::optional<LinearLayout>
  tryCreate(BasesT bases, ArrayRef<std::pair<StringAttr, int32_t>> outDims,
            bool requireSurjective);

  // Constructor that does not check invariants.  Used by tryCreate.
  struct NoCheckInvariants {};
  LinearLayout(BasesT bases, ArrayRef<std::pair<StringAttr, int32_t>> outDims,
               NoCheckInvariants);

  [[nodiscard]] std::optional<std::string>
  checkInvariants(bool requireSurjective);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const LinearLayout &layout) {
  os << layout.toString();
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const LinearLayout &layout) {
  os << layout.toString();
  return os;
}

} // namespace mlir::triton

#endif // TRITON_TOOLS_LINEARLAYOUT_H
