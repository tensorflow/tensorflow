#ifndef TRITONGPU_ATTRDEFS
#define TRITONGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

//===----------------------------------------------------------------------===//
// TritonGPU Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonGPU_AttrTrait : AttrInterface<"TritonGPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let methods = [
  ];
}

class TritonGPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonGPU_AttrTrait], traits), baseCppClass> {

  let description = [{
TritonGPU tensors differ from usual tensors in that they contain a _layout_ attribute which determines
how the data should be partitioned across CUDA threads. Formally speaking, we define a layout as a function
\mathcal{L} that maps a multi-dimensional tensor index $i \in \mathbb{Z}^d$ to a set of integers T corresponding
to the indices of the CUDA threads allowed to access some data at index $i$.

For example, let us consider the layout function:
\mathcal{L}(0, 0) = {0, 4}
\mathcal{L}(0, 1) = {1, 5}
\mathcal{L}(1, 0) = {2, 6}
\mathcal{L}(1, 1) = {3, 7}

Then, attaching $\mathcal{L} to a tensor $T$ would mean that:
- T[0,0] is owned by both cuda thread 0 and 4
- T[0,1] is owned by both cuda thread 1 and 5
- T[1,0] is owned by both cuda thread 2 and 6
- T[1,1] is owned by both cuda thread 3 and 7

Right now, Triton implements two main classes of layouts: shared, and distributed.
  }];
  let attrName = "triton.gpu." # attrMnemonic;

  code extraBaseClassDeclaration = [{
  }];
}

//===----------------------------------------------------------------------===//
// CTA Layout
//===----------------------------------------------------------------------===//

def CTALayoutAttr : TritonGPU_Attr<"CTALayout", "cta_layout"> {
  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$CTAsPerCGA,
    ArrayRefParameter<"unsigned">:$CTASplitNum,
    ArrayRefParameter<"unsigned">:$CTAOrder
  );

  let description = [{
Describes how blocks are distributed among the cooperate thread arrays (aka
CTAs, aka thread blocks) in a cooperate thread group (aka CTG, aka thread group
cluster).  CGAs were introduced in Hopper (sm90).

The tensor is divided up into CTASplitNum pieces, which are distributed among
the CTAsPerCGA thread blocks.  Each CTA processes a subtensor of shape
`tensor_shape / CTASplitNum`.

Example 0: The tensor shape is [64, 128] and, there are two CTAs, each
processing half the tensor [64, 64]. Then CTAsPerCGA = [1, 2] and
CTASplitNum = [1, 2].

Example 1: The tensor shape is [64, 128] and, there are two CTAs, both
processing the complete tensor [64, 128]. This happens when multicast is
enabled. In this case, CTAsPerCTA = [1, 2] but CTASplitNum = [1, 1].

Example 2: Consider a matmul AxB=C, where A=[M,K], B=[K,N], C=[M,N].  The
CTAsPerCGA for A, B, C are the same, [SplitM, SplitN], but the CTASplitNum are
different. CTASplitNum_A = [SplitM, 1], which means multicast on dim1,
CTASplitNum_B = [1, SplitN], which means multicast on dim0, CTASplitNum_C =
[SplitM, SplitN]  which means no multicast.

Currently programs with multiple CTAs per CGA are an experimental feature in
Triton, not enabled by default.

You can leave off the CTALayout properties in the textual IR and Triton will
fill in the "default" CTALayout of CTAsPerCGA = CTASplitNum = [1...1].  In
addition, if there's only one CTA per CGA, then Triton canonicalizes CTAOrder to
[n-1,...,0] (it doesn't matter in this case).
  }];

  // CTALayout::get canonicalizes CTAOrder to [n,n-1,...,0] if CTAsPerCGA is
  // [1...1].  The CTAOrder doesn't matter in this case.
  //
  // This is a little weird because if you write textual IR with a one order and
  // then print it back out, you might get a different order.  But it seems this
  // is the best way to canonicalize an attribute in MLIR.
  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$CTAsPerCGA,
                     "ArrayRef<unsigned>":$CTASplitNum,
                     "ArrayRef<unsigned>":$CTAOrder), [{
        if (llvm::all_of(CTAsPerCGA, [](unsigned x) { return x == 1; })) {
          SmallVector<unsigned> order;
          for (int i = CTAsPerCGA.size() - 1; i >= 0; --i)
            order.push_back(i);
          return $_get(context, CTAsPerCGA, CTASplitNum, order);
        }
        return $_get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }]>,
  ];

  let extraClassDeclaration = [{
    static CTALayoutAttr getDefault(MLIRContext *context, int rank) {
      SmallVector<unsigned> CTAsPerCGA(rank, 1);
      SmallVector<unsigned> CTASplitNum(rank, 1);
      SmallVector<unsigned> CTAOrder;
      for (int i = rank - 1; i >= 0; --i)
        CTAOrder.push_back(i);
      return get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
    }
    unsigned getRank() const {
      return getCTAOrder().size();
    }
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}


def LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let description = [{
    Common trait for all TTGIR layouts.
  }];
  let methods = [
    InterfaceMethod<"Get the shape of the CTAs per CGA.",
                    "SmallVector<unsigned>",
                    "getCTAsPerCGA">,
    InterfaceMethod<"Get the order of the CTAs per CGA. The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getCTAOrder">,
    InterfaceMethod<"Each CTA processes 1/CTASplitNum of the tensor.",
                    "SmallVector<unsigned>",
                    "getCTASplitNum">,
  ];
}

//===----------------------------------------------------------------------===//
// Shared Layout Encoding
//===----------------------------------------------------------------------===//

def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
    Common trait describing shared memory.
  }];
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t",
                    "getAlignment">,
  ];
}

def SwizzledSharedEncodingAttr :
  TritonGPU_Attr<"SwizzledSharedEncoding", "swizzled_shared_encoding", [SharedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "swizzled_shared";

  let description = [{
An encoding for tensors whose elements may be simultaneously accessed by
different cuda threads in the programs, via shared memory. In other words,
for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.

In order to avoid shared memory bank conflicts, elements may be swizzled.
Here are some examples.  In all cases, the input tensor is [0, 1, ..., n-1].

1. Basic swizzling

  #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // xor with 0
  [ 5,  4,  7,  6],  // xor with 1
  [10, 11,  8,  9],  // xor with 2
  [15, 14, 13, 12]   // xor with 3

Here elements of row r are xor'ed with r (or more properly, in[r][c] ->
out[r][c^r]).

2. Multiple rows per phase

  #shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14]

Elements of row r are xor'ed with r/2.  In other words, perPhase=2
means that pairs of 2 rows get the same swizzling.

3. Max-phase applied

  $shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 5,  4,  7,  6],  // phase 1 (xor with 1)
  [ 8,  9, 10, 11],  // phase 0
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // ...
  [21, 20, 23, 22],
  [24, 25, 26, 27],
  [29, 28, 31, 30]

Elements of row r are xor'ed with (r/2) % 2.  In other words, maxPhase=m has the
effect of limiting the maximum value of the xor to m-1.

4. Max-phase and per-phase

  #shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
  [ 0,  1,  2,  3],  // phase 0 (xor with 0)
  [ 4,  5,  6,  7],  // phase 0
  [ 9,  8, 11, 10],  // phase 1 (xor with 1)
  [13, 12, 15, 14],  // phase 1
  [16, 17, 18, 19],  // phase 0
  [20, 21, 22, 23],  // phase 0
  [25, 24, 27, 26],  // phase 1
  [29, 28, 31, 30]]  // phase 1

Here the xor value (the "phase", I guess?) changes every perPhase rows, up to a
maximum value of maxPhase-1.  In other words, elements of row r are xor'ed with
(r/2) % 2.

5. Adding vec

  #shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>
  [ 0,  1,  2,  3,  4,  5,  6,  7],
  [10, 11,  8,  9, 14, 15, 12, 13],
  [20, 21, 22, 23, 16, 17, 18, 19],
  [30, 31, 28, 29, 26, 27, 24, 25]

When vec=2, elements are swizzled in pairs of 2.  In other words, the element at
(r,c) has value

  ((c / 2) ^ r) * 2 + (c % 2).
  }];

  // swizzle info: vec, perPhase, maxPhase
  // order: the fastest-changing axis first
  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CTALayoutAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit), [{
        bool needTrans = false; // default value
        return get(context, dotOpEnc, shape, order, CTALayout, typeWidthInBit, needTrans);
    }]>,

    // TODO(jlebar): This should not be an overload of
    // SwizzledSharedEncodingAttr::get().  It's misleading, because it does a bunch of
    // nontrivial work based on the given dotOpEnc.
    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "unsigned":$typeWidthInBit,
                     "bool":$needTrans), [{

        // ---- begin MFMA ----
        if (auto mfmaEnc = mlir::dyn_cast<AMDMfmaEncodingAttr>(dotOpEnc.getParent())) {
          return mfmaEnc.composeSharedLayoutForOperand(
              CTALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
              typeWidthInBit, needTrans);
        }

        // ---- begin WMMA ----
        if (mlir::isa<AMDWmmaEncodingAttr>(dotOpEnc.getParent())) {
          if (dotOpEnc.getOpIdx() == 0) {
            const int numBanks = 32;
            const int bankBitWidth = 32;

            // number of inner dimension rows per one pattern repeat
            int innerDimLength = shape[order[0]];
            int elemsPerOneBanksRow = (numBanks * bankBitWidth) / typeWidthInBit;

            int perPhase = std::max(1, elemsPerOneBanksRow / innerDimLength);
            int vecSize = ((typeWidthInBit == 16) ? 64 : 32 ) / typeWidthInBit;
            int maxPhase = 16 / perPhase;

            return get(context, vecSize, perPhase, maxPhase, order, CTALayout);
          } else {
            // Do not swizzle in case k dimension is not innermost.
            // In this case accesses will go in different banks even without swizzling.
            return get(context, 1, 1, 1, order, CTALayout);
          }
        }


        auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

        if(!mmaEnc)
          return get(context, 1, 1, 1, order, CTALayout);

        int opIdx = dotOpEnc.getOpIdx();
        auto shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

        // number of rows per phase

        // index of the inner dimension in `order`
        unsigned inner = (opIdx == 0) ? 0 : 1;

        // ---- begin Ampere & Hopper ----
        if (mmaEnc.isAmpere() || mmaEnc.isHopper()) {
          int perPhase = 128 / (shapePerCTA[order[0]] * 4 / dotOpEnc.getKWidth());
          perPhase = std::max<int>(perPhase, 1);
          std::vector<size_t> matShape = {8, 8, 4 * dotOpEnc.getKWidth()};
          int vecWidth = 32 / typeWidthInBit;
          if (vecWidth != dotOpEnc.getKWidth() && order[0] == inner) {
              perPhase = std::max<int>(perPhase, 2 * vecWidth);
          }
          int rank = order.size();
          // --- handle A operand ---
          if (opIdx == 0) { // compute swizzling for A operand
              int m = (needTrans) ? matShape[2] : matShape[0];
              int k = (needTrans) ? matShape[0] : matShape[2];
              int vec = (order[0] == rank-1) ? k : m;
              int mmaStride = (order[0] == rank-1) ? m : k;
              int maxPhase = std::max(mmaStride / perPhase, 1);
              return get(context, vec, perPhase, maxPhase, order, CTALayout);
          }

          // --- handle B operand ---
          if (opIdx == 1) {
              // we compute vec and maxPhase m, n and k size of the mma
              // instruction. when matmul operands is transposed, we should
              // consider that to get m, n and k.
              int n = needTrans ? matShape[2] : matShape[1];
              int k = needTrans ? matShape[1] : matShape[2];
              int vec = (order[0] == rank-1) ? n : k;
              int mmaStride = (order[0] == rank-1) ? k : n;
              int maxPhase = std::max(mmaStride / perPhase, 1);
              return get(context, vec, perPhase, maxPhase, order, CTALayout);
          }

          llvm_unreachable("invalid operand index");
        }

        // ---- not implemented ----
        llvm_unreachable("unsupported swizzling for provided MMA version");
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth);
    }]>,

    AttrBuilder<(ins "DotOperandEncodingAttr":$dotOpEnc,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool":$needTrans), [{
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      return get(context, dotOpEnc, shape, order, CTALayout, bitwidth, needTrans);
    }]>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int32_t getAlignment() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

def NVMMASharedEncodingAttr :
  TritonGPU_Attr<"NVMMASharedEncoding", "nvmma_shared_encoding", [SharedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "nvmma_shared";

  let description = [{
    Represent blocked shared memory matching MMAv3/MMAv5 shared memory input.
    This is meant to represent 2d tiled blocked layout.
    The full layout representation is described here:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
  }];


  // fp4Padded: Indicates that this encoding represents a mixed-precision fp4 operand in MMAv5 scaled dot, which needs
  // to be in the special padded layout as described in https://docs.nvidia.com/cuda/parallel-thread-execution/#packing-format-used-for-matrix-a-and-b-by-kind-mxf8f6f4-in-shared-memory
  let parameters = (
    ins
    "unsigned":$swizzlingByteWidth,
    "bool":$transposed,
    "unsigned":$elementBitWidth,
    "bool":$fp4Padded,
    "CTALayoutAttr":$CTALayout
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$order,
                     "CTALayoutAttr":$CTALayout,
                     "Type":$eltTy,
                     "bool": $fp4Padded), [{
        auto shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);
        int32_t swizzlingByteWidth = 0;
        unsigned eleBitWidth = eltTy.getIntOrFloatBitWidth();
        int packingFactor = fp4Padded ? 2 : 1;

        // get proper shared memory swizzling mode from the contiguous dimension
        // size of the origin blocked layout.
        auto contigDimSizeInByte = shapePerCTA[order[0]] * packingFactor * eleBitWidth / 8;
        if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
          swizzlingByteWidth = 128;
        } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
          swizzlingByteWidth = 64;
        } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
          swizzlingByteWidth = 32;
        } else {
          llvm_unreachable("unsupported NVMMA layout (MMAv3 or TMA)");
        }
        bool transposed = order[0] == 0;
        return $_get(context, swizzlingByteWidth, transposed, eleBitWidth, fp4Padded, CTALayout);
    }]>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int32_t getAlignment() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getOrder() const {
      return getTransposed() ? SmallVector<unsigned>({0, 1}) : SmallVector<unsigned>({1, 0});
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def AMDRotatingSharedEncodingAttr :
  TritonGPU_Attr<"AMDRotatingSharedEncoding", "amd_rotating_shared_encoding",
                 [SharedEncodingTrait, LayoutEncodingTrait]> {
  let mnemonic = "amd_rotating_shared";

  let description = [{
This shared encoding is similar to SwizzledSharedEncodingAttr, but instead of
repeating swizzling pattern every `maxPhase*perPhase` rows of the memory object,
called a block, this layout changes swizzling pattern `maxPhase` times, then
repeats the pattern. The name "rotating" comes from the fact that first tensor
element of each block is swizzled with different phase, which is equal to
current block number: 0, 1, 2.. maxPhase-1, 0, 1, 2 ...

This layout is used to reduce bank conflicts in cases where shared memory writes
and reads are performed on layouts with different order. It's meant for hardware
without native shared memory tranpose support.

Swizzling pattern affects only 2 fastest dimensions of a tensor.
In the following text these two dimensions are called row and column:
- row is a fastest dimension
- column is a second fastest dimension

Elements in a row dimension are stored in memory contiguously.

If a matrix of size [128x64] is stored in this shared layout with order [1, 0],
dim 1 (64) will be stored contiguously and called row, dim 0 (128) is will be
called column. If order of shared layout is [0, 1], dim 0 (128) is stored
contiguously becomes a row, dim 1 (64) becomes a column.

Swizzling pattern is following:

Let's consider an element with logical coordinates = (inRowId, inColId).
For simplicity, we do not vectorize memory in examples,
i.e. vec == 1 and layout swizzles inidividual elements.
For vec != 1 example, take a look at SwizzledSharedEncodingAttr documentation.

Swizzled coordinates within memory object are (outRowId, outColId):

  outRowId = inRowId
  phase   = (inRowId / perPhase) % maxPhase
  blockNo = (inRowId / (perPhase * maxPhase)) % maxPhase
  combinedPhase = phase ^ blockNo
  outColId   = inColId ^ combinedPhase

Actual offset in memory could be computed with following function:

memmory_offset = (outColId + outRowId * num_of_element_in_row) * sizeof(element)


Swizzling examples (matrix is filled with numbers 0, 1, 2, .. columns*rows-1):

  #shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
  row      elements
    0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
    1  [ 5,  4,  7,  6],  // phase = 1 blockNo = 0 (xor with 1)
    2  [ 9,  8, 11, 10],  // phase = 0 blockNo = 1 (xor with 1)
    3  [12, 13, 14, 15]   // phase = 1 blockNo = 1 (xor with 0)
    4  [16, 17, 18, 19],  // phase = 0 blockNo = 0 (xor with 0)
    5  [21, 20, 23, 22],  // phase = 1 blockNo = 0 (xor with 1)
    6  [25, 24, 27, 26],  // phase = 0 blockNo = 1 (xor with 1)
    7  [28, 29, 30, 31]   // phase = 1 blockNo = 1 (xor with 0)

  #shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
  row      elements
    0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
    1  [ 4,  5,  6,  7],  // phase = 0 blockNo = 0 (xor with 0)
    2  [ 9,  8, 11, 10],  // phase = 1 blockNo = 0 (xor with 1)
    3  [13, 12, 15, 14]   // phase = 1 blockNo = 0 (xor with 1)
    4  [17, 16, 19, 18],  // phase = 0 blockNo = 1 (xor with 1)
    5  [21, 20, 23, 22],  // phase = 0 blockNo = 1 (xor with 1)
    6  [24, 25, 26, 27],  // phase = 1 blockNo = 1 (xor with 0)
    7  [28, 29, 30, 31]   // phase = 1 blockNo = 1 (xor with 0)

  #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  row      elements
    0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
    1  [ 5,  4,  7,  6],  // phase = 1 blockNo = 0 (xor with 1)
    2  [10, 11,  8,  9],  // phase = 2 blockNo = 0 (xor with 2)
    3  [15, 14, 13, 12]   // phase = 3 blockNo = 0 (xor with 3)
    4  [17, 16, 19, 18],  // phase = 0 blockNo = 1 (xor with 1)
    5  [20, 21, 22, 23],  // phase = 1 blockNo = 1 (xor with 0)
    6  [27, 26, 25, 24],  // phase = 2 blockNo = 1 (xor with 3)
    7  [30, 31, 28, 29]   // phase = 3 blockNo = 1 (xor with 2)
  }];

  let parameters = (
    ins
    "unsigned":$vec,
    "unsigned":$perPhase,
    "unsigned":$maxPhase,
    ArrayRefParameter<"unsigned">:$order,
    "CTALayoutAttr":$CTALayout
  );

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    int32_t getAlignment() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;
  }];
  let hasCustomAssemblyFormat = 1;
}


//===----------------------------------------------------------------------===//
// Distributed Layout Encoding
//===----------------------------------------------------------------------===//
def DistributedEncodingTrait : AttrInterface<"DistributedEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";

  let description = [{
The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.

For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
For example, for a shape/order pair defines a distribution layout
shape = [4, 4]
order = [0, 1] // The fastest-changing axis first
->
layout = [0  4  8  12]
         [1  5  9  13]
         [2  6  10 14]
         [3  7  11 15]

For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.

If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
We call each individual tile "rep".
  }];

  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrder">,
    InterfaceMethod<"Return total element size per thread.",
                    "unsigned",
                    "getTotalElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*methodBody=*/[{}],
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_attr, shape).getTotalElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Return element size per thread in each dimension.",
                    "SmallVector<unsigned>",
                    "getElemsPerThread",
                     (ins "ArrayRef<int64_t>":$shape),
                     /*methodBody=*/[{}],
                     /*defaultImplementation=*/[{
                         return toLinearEncoding($_attr, shape).getElemsPerThread(shape);
                     }]>,
    InterfaceMethod<"Convert to LinearLayout.",
                    "LinearLayout",
                    "toLinearLayout",
                    (ins "ArrayRef<int64_t>":$shape)>,
  ];
}

class DistributedEncoding<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonGPU_Dialect>
  : TritonGPU_Attr<name, attrMnemonic, !listconcat([DistributedEncodingTrait, LayoutEncodingTrait], traits), dialect> {

  let description = [{
Distributed encodings have a layout function L that is entirely characterized
by a d-dimensional tensor T. Note that L doesn't need to have the same shape
(or even the same rank) as the tensor it is encoding.

The layout function \mathcal{L} of this layout is then defined, for an
index `i` \in Z^d, as follows:

\mathcal{L}(T)[i_d] = L[(i_d + k_d*T.shape[d]) % L.shape[d]] \forall k_d such as i_d + k_d*T.shape[d] < L.shape[d]

Intuitively, when the tensor dim size T.shape[d] is larger than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "wrapped around" manner, with
each thread owning multiple values.

OTOH, when the tensor dim size T.shape[d] is smaller than the layout
dim size L.shape[d], on that particular dim, we distribute values from the
tensor to threads mapped in the layout in a "broadcasted" manner, with
each value owned by multiple threads.

For example, for a tensor/layout pair
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

Then the data of T would be distributed as follow between the 16 CUDA threads:
L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
  }];

  code extraDistributedDeclaration  = extraBaseClassDeclaration # [{
    unsigned getRank() const { return getCTAOrder().size(); }
    // Implemented in subclasses
    SmallVector<unsigned> getRepOrder() const;
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTAOrder() const;
    SmallVector<unsigned> getCTASplitNum() const;

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}

//===----------------------------------------------------------------------===//
// Linear Layout Encoding
//===----------------------------------------------------------------------===//

def LinearLayoutParam : AttrOrTypeParameter<"LinearLayout",
                                            "linear layout"> {
  let cppAccessorType = "const LinearLayout &";
}

def LinearEncodingAttr : DistributedEncoding<"LinearEncoding", "linear_encoding"> {
  let mnemonic = "linear";

  let description = [{
    See the docs in LinearLayout.h for the definition of linear layouts.
  }];

  let parameters = (ins LinearLayoutParam:$linearLayout);

  let extraClassDeclaration = extraDistributedDeclaration # [{
    // Generic distributed encoding methods
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

    SmallVector<unsigned int> getContig(const char *, SmallVector<unsigned int>) const;
    SmallVector<unsigned> getContigPerThread() const;
    SmallVector<unsigned> getContigPerWarp() const;
    SmallVector<unsigned> getOrder() const;
    SmallVector<unsigned> getWarpOrder() const;
    SmallVector<unsigned> getThreadOrder() const;


    // Generalizes get{Warp,Thread,CTA}Order to linear layouts.
    // Returns the order of the dimensions `dimName` of the layout.
    // If more than dimension is of size one, it uses defaultOrder to determine
    // the order of the dimensions of size one.
    SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                      ArrayRef<unsigned> defaultOrder) const;

    // Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
    // Returns the bases of the dimensions `dimName` of the layout.
    // If skipBroadcast is false, we count a base zero
    SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                      bool skipBroadcast = true) const;
    SmallVector<unsigned> getThreadsPerWarp() const;
    SmallVector<unsigned> getWarpsPerCTA() const;

    // [FIXME LL] Supports legacy behaviour. We should remove these functions
    SmallVector<unsigned> getShapePerCTATile() const;
    SmallVector<unsigned> getSizePerThread() const;
  }];

  let genVerifyDecl = 1;
  // Example of assembly format:
  // <{register = [[0, 1], [8, 0], [0, 8], [64, 0]],
  //   lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
  //   warp = [[16, 0], [32, 0]],
  //   block = []}>
  let hasCustomAssemblyFormat = 1;
}


//===----------------------------------------------------------------------===//
// Blocked Layout Encoding
//===----------------------------------------------------------------------===//

def BlockedEncodingAttr : DistributedEncoding<"BlockedEncoding", "blocked_encoding"> {
  let mnemonic = "blocked";

  let description = [{
An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
used to promote memory coalescing in LoadInst and StoreInst.
It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
specify the amount of elements owned by each CUDA thread, warp and CTA respectively.

Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}>

Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:

[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                 ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {1, 1}
  CTASplitNum = {1, 1}
}>

Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
4 CTAs (taking 2x2 for example) as follows:

CTA [0,0]                                              CTA [0,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

CTA [1,0]                                              CTA [1,1]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                    ...
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

#ttg.blocked_layout<{
  sizePerThread = {2, 2}
  threadsPerWarp = {8, 4}
  warpsPerCTA = {1, 2}
  CTAsPerCGA = {2, 2}
  CTASplitNum = {2, 2}
}>
}];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    ArrayRefParameter<"unsigned">:$order, // the fastest-changing axis first

    // CTALayout is optional in the textual IR.  If omitted, we infer it to be a
    // single CTA (so CTAsPerCGA = [1,...,1], CTASplitNum = [1,...,1],
    // CTAOrder=[n,n-1,...,0]).
    "CTALayoutAttr":$CTALayout
  );
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "CTALayoutAttr":$CTALayout), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> threadsPerWarp(rank);
      SmallVector<unsigned, 4> warpsPerCTA(rank);
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(CTALayout.getCTASplitNum(), shape);

      unsigned remainingLanes = numThreadsPerWarp;
      unsigned remainingThreads = numWarps * numThreadsPerWarp;
      unsigned remainingWarps = numWarps;
      unsigned prevLanes = 1;
      unsigned prevWarps = 1;

      // starting from the contiguous dimension
      for (unsigned d = 0; d < rank - 1; ++d) {
        unsigned i = order[d];
        unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, std::max<unsigned>(1, shapePerCTA[i] / sizePerThread[i]));
        threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
        warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
        remainingWarps /= warpsPerCTA[i];
        remainingLanes /= threadsPerWarp[i];
        remainingThreads /= threadsPerCTA;
        prevLanes *= threadsPerWarp[i];
        prevWarps *= warpsPerCTA[i];
      }

      // Expand the last dimension to fill the remaining lanes and warps
      threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
      warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

      return $_get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout);
    }]>,

    AttrBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "ArrayRef<unsigned>":$sizePerThread,
                     "ArrayRef<unsigned>":$order,
                     "unsigned":$numWarps,
                     "unsigned":$numThreadsPerWarp,
                     "unsigned":$numCTAs), [{
      unsigned rank = sizePerThread.size();
      SmallVector<unsigned, 4> CTAsPerCGA(rank);
      SmallVector<unsigned, 4> CTASplitNum(rank);
      ArrayRef<unsigned> CTAOrder = order;

      unsigned remainingCTAs = numCTAs;

      // starting from the most strided dimension
      for (int d = rank - 1; d >= 0; --d) {
        unsigned i = order[d];
        CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, std::max<unsigned>(1, shape[i] / sizePerThread[i]));
        CTASplitNum[i] = CTAsPerCGA[i];
        remainingCTAs /= CTAsPerCGA[i];
      }

      CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

      CTALayoutAttr CTALayout = CTALayoutAttr::get(context, CTAsPerCGA, CTASplitNum, CTAOrder);
      return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CTALayout);
    }]>
  ];

  let extraClassDeclaration = extraDistributedDeclaration;

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MMA Layout Encoding
//===----------------------------------------------------------------------===//

def MmaEncodingTrait : AttrInterface<"MmaEncodingTrait"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first",
                    "SmallVector<unsigned>",
                    "getRepOrderForOperand",
                    (ins "int":$opIdx)>,
  ];
}

def AMDMfmaEncodingAttr : DistributedEncoding<"AMDMfmaEncoding", "amd_mfma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_mfma";

  let description = [{
An encoding for tensors that have been produced by MFMA matrix core instructions,
available on AMD Instinct GPUs of CDNA architectures.

It is characterized by the following parameters:
- `versionMajor` and `versionMinor` indicates the GPU architecture:
  - 1.0: gfx908, i.e. CDNA1
  - 2.0: gfx90a: i.e. CDNA2
  - 3.0: gfx942: CDNA3
  - 4.0: gfx950: CDNA4
- `warpsPerCTA` indicates the warp layout in the block.
- `MDim` and `NDim` indicate the dimension of the output of the mfma instruction.
- `isTransposed` indicates the result tensor is transposed so that it can be converted to dotOperand layout
without going to shared memory. This is used in the case of chained dot (E.g. Flash-Attention kernel).

Example 1:
Suppose we have a tensor with a shape of [32, 64], warpsPerCTA set to [1, 2] and MDim=NDim=32.
The data will be distributed between threads as follows:

                warp 0                                 warp 1
-----------------/\--------------      -----------------/\--------------
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
[ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]

Example 2:
Suppose we have a tensor with a shape of [16, 32], warpsPerCTA set to [1, 2] and MDim=NDim=16.
The data will be distributed between threads as follows:

                warp 0                                 warp 1
-----------------/\-------------      ------------------/\---------------
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
[ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]

Example 3:
Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and nonKDim set to 4.
The data will be distributed between threads as follows(note that each element is duplicated in 16 threads):
Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and MDim=NDim=4.
The data will be distributed between threads as follows(note that each element is duplicated in 16 threads):

M  N ->                    warp 0                                                       warp 2
| --------------------------/\--------------------------   ------------------------------/\------------------------------
V [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
  [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
                           warp 1                                                       warp 3
  --------------------------/\--------------------------   ------------------------------/\------------------------------
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
  [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
}];

  let parameters = (
    ins
    "unsigned": $versionMajor,
    "unsigned": $versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "unsigned":$MDim,
    "unsigned":$NDim,
    "bool":$isTransposed,
    "CTALayoutAttr":$CTALayout
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<int64_t> getInstrShapeForOperand(int kWidth, int opIdx) const;
    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape, int kWidth, int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;

    // Returns a swizzled shared layout matching this MFMA layout for the
    // dot operand at the given |operandIdx| with |operandShape|.
    SwizzledSharedEncodingAttr composeSharedLayoutForOperand(
        CTALayoutAttr ctaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
        ArrayRef<unsigned> sharedOrder, unsigned vectorSize,
        unsigned elemBitWidth, bool needTrans) const;
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def AMDWmmaEncodingAttr : DistributedEncoding<"AMDWmmaEncoding", "amd_wmma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "amd_wmma";

  let description = [{
An encoding for tensors that have been produced by WMMA matrix core instructions,
available on AMD Radeon GPUs of RDNA architectures.
- A `version` parameter specifies instruction version to lower in. The data
  distribution within one warp is also depends on it. Following architectures are
  supported:
  - 1: gfx11
  - 2: gfx12
- A `warpsPerCTA` parameter characterizes data distribution between warps.
  An important limitation of WMMA for layout is a shape for tiles processed
  by a single warp. It is [16, 16].
  This encoding assumes specific access to matrix elements by threads.

Example:
Suppose we have a tensor with shape [32, 64], `warpsPerCTA` set to [2, 2].
Matrix elements represent which lane owns the element. Currently only wave32 mode
is supported.

// ----------------------------------- version = 1 ----------------------------------- //

Row |                  warp 0                                    warp 1
    |/-------------------^-------------------\ /-------------------^-------------------\
0   |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
1   |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
2   |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
3   |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
    | ...                  ...                  ...                  ...
14  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
15  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]

    |                  warp 2                                    warp 3
16  |/-------------------^-------------------\ /-------------------^-------------------\
17  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
18  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
19  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
20  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
    | ...                  ...                  ...                  ...
30  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
31  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]

// ------------------------ version = 2, isTransposed = false ------------------------ //

Row |       warp 0                warp 1
    |/--------^---------\ /---------^--------\
0   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
1   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
..  | ...                    ...
6   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
7   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
8   |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
9   |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
..  | ...                  ...
14  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
15  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
    |
    |       warp 2                warp 3
    |/--------^---------\ /---------^--------\
16  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
17  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
..  | ...                    ...
22  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
23  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
24  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
25  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
..  | ...                  ...
30  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
31  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]

// ------------------------ version = 2, isTransposed = true ------------------------ //

    |               warp 0                     warp 1
    |/----------------^----------------\ /-------^-------\
Col>| 0  1  2  3  4  5  6  7  8  ... 15  16 17 18  ... 32
Row |
0   |[0  0  0  0  0  0  0  0  16 ... 16] [0  0  0  ... 16]
1   |[1  1  1  1  1  1  1  1  17 ... 17] [1  1  1  ... 17]
..  | ...                  ...
14  |[14 14 14 14 14 14 14 14 30 ... 30] [14 14 14 ... 30]
15  |[15 15 15 15 15 15 15 15 31 ... 31] [15 15 15 ... 31]
    |
    |               warp 2                     warp 3
    |/----------------^----------------\ /-------^-------\
16  |[0  0  0  0  0  0  0  0  16 ... 16] [0  0  0  ... 16]
17  |[1  1  1  1  1  1  1  1  17 ... 17] [1  1  1  ... 17]
..  | ...                  ...
30  |[14 14 14 14 14 14 14 14 30 ... 30] [14 14 14 ... 30]
31  |[15 15 15 15 15 15 15 15 31 ... 31] [15 15 15 ... 31]
  }];

  let parameters = (
    ins
    "unsigned": $version,
    "bool":$isTransposed,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CTALayoutAttr":$CTALayout
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<int64_t> getElemsPerInstrForOperands() const;
    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape,
                                          Type elemType, int kWidth, int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    unsigned getKWidthForOperands() const;
    static SmallVector<unsigned> getMNKDimPerInstr();
  }];
}

def NvidiaMmaEncodingAttr : DistributedEncoding<"NvidiaMmaEncoding", "nvidia_mma_encoding", [MmaEncodingTrait]> {
  let mnemonic = "nvidia_mma";

  let description = [{
An encoding for tensors that have been produced by tensor cores.

It is characterized by two parameters:
- A 'versionMajor' which specifies the generation the tensor cores
  whose output is being partitioned:
  - 1 for first-gen tensor cores (Volta), and
  - 2 for second-gen tensor cores (Turing/Ampere).
- A 'versionMinor' which indicates the specific layout of a tensor core
  generation, e.g. for Volta, there might be multiple kinds of layouts
  annotated by 0,1,2 and so on.
- A `blockTileSize` to indicate how data should be partitioned between warps.

// -------------------------------- version = 1 --------------------------- //

For first-gen tensor cores, the implicit warpTileSize is [16, 16].
Note: the layout is different from the recommended in PTX ISA
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.884 section, FP32 accumulator).

For example, when versionMinor=1, the matrix L corresponding to
blockTileSize=[32,16] is:

                               warp 0
--------------------------------/\-------------------------------
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
[ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
[ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
[ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
[ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
[ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]

                          warp 1 = warp0 + 32
--------------------------------/\-------------------------------
[ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
[ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
[ ............................................................... ]


// -------------------------------- version = 2 --------------------------- //

For second-gen tensor cores, the implicit warpTileSize is [16, 8].
Information about this layout can be found in the official PTX documentation
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
(mma.16816 section, FP32 accumulator).

For example, the matrix L corresponding to blockTileSize=[32,16] is:
                warp 0                          warp 2
-----------------/\-------------  ----------------/\-------------
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
[ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
[ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
[ ..............................  ..............................
[ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63

              warp 1                           warp 3
----------------/\-------------   ----------------/\-------------
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
[ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
[ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
[ ..............................  ...............................
[ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127

}];

  let parameters = (
    ins
    "unsigned":$versionMajor,
    "unsigned":$versionMinor,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CTALayoutAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape
  );


  let extraClassDeclaration = extraDistributedDeclaration # [{
    bool isVolta() const;
    bool isTuring() const;
    bool isAmpere() const;
    bool isHopper() const;

    SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                          int bitwidth, int kWidth,
                                          int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def SliceEncodingAttr : DistributedEncoding<"SliceEncoding", "slice_encoding"> {
  let mnemonic = "slice";

  let description = [{
    Given a `parent` layout and a `dim`, squeezes the given `dim` in the `parent`
    layout and distributes values in a tensor T according to the new layout.

    For example, given

    T = [x  x  x  x  x  x  x  x]
    L_parent = [0  1  2  3 ]
               [4  5  6  7 ]
               [8  9  10 11]
               [12 13 14 15] (with 16 CUDA threads)

    With dim = 0, squeezing out dim 0, we have
    L = [{0,4,8,12},  {1,5,9,13}, {2,6,10,14},  {3,7,11,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L(T) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]

    With dim = 1, squeezing out dim 1, we have
    L = [ {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15} ]

    Then the data of T would be distributed as follow between the 16 CUDA threads:
    L = [ {0,1,2,3}, {4,5,6,7}, ..., {12,13,14,15}, {0,1,2,3}, ..., {12,13,14,15} ]

    This is useful for constructing the inverse layout of an expand_dims operation
    during some optimization passes.
  }];

  let parameters = (
    ins
    "unsigned":$dim,
    "DistributedEncodingTrait":$parent
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    template<class T>
    SmallVector<T> paddedShape(ArrayRef<T> shape) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def DotOperandEncodingAttr : DistributedEncoding<"DotOperandEncoding", "dot_operand_encoding"> {
  let mnemonic = "dot_op";

  let description = [{
In the TritonGPU dialect, given `d = tt.dot a, b, c` tt.dot's operands a and b
must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e.
pre-Hopper).  For MMA v3, the operands are *almost always* in a regular shared
encoding, but sometimes the LHS is also a dot-operand encoding.

a's opIdx is 0, b's opIdx is 1.

The parent field is the layout of d.

kWidth defines number of consecutive elements stored by one thread along k dimension.
Some layouts do not use this parameter, either because they have a fixed number of
elements along the K dim, or they use all elements of the tensor along the K dim.

# WGMMA Notes
We require kWidth to be provided for Hopper because the dtype at loading might be
different from the dtype at WGMMA, due to casting. The kWidth is determined by the
dtype at WGMMA.

The encoded tensor consists of operand A for possibly multiple wgmma instructions.
For each wgmma, each warp in a warp group feeds a single "warp matrix"
Each warp matrix consists of 2x2 "quads".
Each thread holds several elements in each quad. Right before a wgmma,
the sum of bitwidth of
the elements in each quad should add up to 32.

These values are stored unrolled in `elements`.
The ordering of dimensions is as follows by convention:
batch (only 1 batch for Hopper currently)
matM (m-index of the "warp matrix")
matK (k-index of the "warp matrix")
quadK (k-index of the "quad" in the core matrix)
quadM (m-index of the "quad" in the core matrix)
vecIdx (index of the element in the quad; this is always along the k-dim)
  }];

  let parameters = (
    ins
    "unsigned":$opIdx,
    "Attribute":$parent,
    DefaultValuedParameter<"unsigned", "0">:$kWidth
  );

  let builders = [
    AttrBuilder<(ins "unsigned":$opIdx,
                     "Attribute":$parent,
                     "Type":$eltTy), [{
      NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
      if (!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
        return $_get(context, opIdx, parent, 0);
      // For MMAV2 and V3
      unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
      unsigned kWidth = 32 / bitwidth;
      return $_get(context, opIdx, parent, kWidth);
    }]>
  ];

  let assemblyFormat = "`<` `{` struct(params) `}` `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = extraDistributedDeclaration;
}

def TTG_SharedMemorySpace : AttrDef<TritonGPU_Dialect, "SharedMemorySpace"> {
  let mnemonic = "shared_memory";
  let description = [{
    Attribute to indicate that the memory descriptor points to shared memory.
  }];
}
#endif
