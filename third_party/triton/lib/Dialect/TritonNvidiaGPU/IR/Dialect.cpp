/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

static constexpr int numTmemRows = 128;

TMemAllocation getTmemAllocSizes(MemDescType memDescType) {
  const int rowSizeInBytes = 4;
  auto shapePerCTA = triton::gpu::getShapePerCTA(memDescType);
  if (isa<TensorMemoryScalesEncodingAttr>(memDescType.getEncoding())) {
    // For scales the data are packed and replicated 4 times.
    assert(memDescType.getElementType().getIntOrFloatBitWidth() == 8);
    assert(memDescType.getShape().size() == 2 &&
           "TODO handle multibuffering of scales.");
    int k = shapePerCTA[1];
    int m = shapePerCTA[0];
    int numColumn = ceil<int>(m, 32) * ceil<int>(k, 4);
    return TMemAllocation(numColumn, numTmemRows);
  }
  assert(isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
             memDescType.getEncoding()) &&
         "Expecting a tensor memory encoding attribute");
  triton::nvidia_gpu::TensorMemoryEncodingAttr attr =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          memDescType.getEncoding());
  bool isUnpacked = attr.getUnpacked();
  int64_t elementSizeInBytes =
      isUnpacked ? rowSizeInBytes
                 : memDescType.getElementType().getIntOrFloatBitWidth() / 8;
  int sizeInBytes = product(shapePerCTA) * elementSizeInBytes;
  int numRows = numTmemRows;
  // BlockM of 64 is and interleaved format, where for single message only the
  // first 16 rows are used. For multiple blocks, the rows are interleaved, i.e.
  //  0                   N/2                     N
  //  ---------------------------------------------
  // 0  0,0 0,1... 0,N/2-1   0,N/2 0,N/2+1 ... 0, N-1  \
  //...                                                  Block 0
  // 15 15,0 15,1..15,N/2-1  15,N/2 15,N/2+1...15, N-1 /
  // 16 0,0 0,1... 0,N/2-1   0,N/2 0,N/2+1 ... 0, N-1  \
  //...                                                  Block 1
  // 31 15,0 15,1..15,N/2-1  15,N/2 15,N/2+1...15, N-1 /
  // Note that allocations that consists of single block of 64 rows are
  // "sparse" and only half of the rows are used.
  // Note that even for 3D shapes for which 2D slices are big enough to fit
  // entire tensor block, we will use "sparse" allocation.
  int blockM = attr.getBlockM();
  int blockN = attr.getBlockN();
  int lastDim = shapePerCTA.size() - 1;
  int isSingleBlock =
      (shapePerCTA[lastDim - 1] <= blockM) && (shapePerCTA[lastDim] <= blockN);
  if (blockM == 64 && isSingleBlock)
    numRows = 64;
  int numColumn = ceil<int>(sizeInBytes, (numRows * rowSizeInBytes));
  return TMemAllocation(numColumn, numRows);
}

Attribute getTmemCompatibleLayout(unsigned M, unsigned N,
                                  ArrayRef<int64_t> shape, unsigned numWarps,
                                  triton::gpu::CTALayoutAttr ctaLayout) {
  assert(numWarps == 4 || numWarps == 8);
  assert(shape.size() == 2);
  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  SmallVector<unsigned> blocksPerTile = {(unsigned)shape[0] / M,
                                         (unsigned)shape[1] / N};
  int numBlocks = blocksPerTile[0] * blocksPerTile[1];
  if (M == 64) {
    unsigned numWarpGroups = numWarps / 4;
    if (numBlocks == 1) {
      // Split along the N dimension
      sizePerThread = {1, N / (numWarpGroups * 2)};
      threadsPerWarp = {16, 2};
      warpsPerCTA = {4, numWarpGroups};
    } else {
      sizePerThread = {1, N / 2};
      threadsPerWarp = {16, 2};
      warpsPerCTA = {0, 0};
      // Distribute at most as many warp groups as there is blocks
      // along M dimension.
      warpsPerCTA[0] = 4 * std::min(blocksPerTile[0], numWarpGroups);
      // Distribute rest of the warp groups along N dimension.
      warpsPerCTA[1] = ceil<int>(numWarpGroups, warpsPerCTA[0] / 4);
    }
  } else {
    unsigned numWarpGroups = numWarps / 4;
    if (shape[0] > 128) {
      // Split along M dimension
      sizePerThread = {1, N};
      threadsPerWarp = {32, 1};
      warpsPerCTA = {4 * numWarpGroups, 1};
    } else {
      // Split along N dimension
      sizePerThread = {1, N / numWarpGroups};
      threadsPerWarp = {32, 1};
      warpsPerCTA = {4, numWarpGroups};
    }
  }
  order = {0, 1};
  return triton::gpu::BlockedEncodingAttr::get(ctaLayout.getContext(),
                                               sizePerThread, threadsPerWarp,
                                               warpsPerCTA, order, ctaLayout);
}

// Verify if the distributed layout can be mapped onto tensor memory.
bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       MemDescType memType) {
  int numWarps = lookupNumWarps(op);
  assert(numWarps % 4 == 0);
  int numWarpGroups = numWarps / 4;

  int blockM = 0;
  int blockN = 0;
  bool scalesEncoding = false;
  if (auto attr = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          memType.getEncoding())) {
    blockM = attr.getBlockM();
    blockN = attr.getBlockN();
  } else {
    assert(isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(
               memType.getEncoding()) &&
           "Expecting a tensor memory encoding attribute");
    return tensorType.getEncoding() ==
           triton::gpu::LinearEncodingAttr::get(
               tensorType.getContext(),
               getScaleTMEMStoreLinearLayout(tensorType, numWarps));
  }
  auto shapePerCTA = mlir::triton::gpu::getShapePerCTA(tensorType);
  int numElements = product(shapePerCTA);
  int numBlocks = ceil<int>(numElements, blockM * blockN);
  bool useStridedMessage = blockM == 64;

  int numWarpGroupsPerBlock = ceil<int>(numWarpGroups, numBlocks);

  auto tensorEncoding =
      cast<triton::gpu::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sizePerThread = tensorEncoding.getSizePerThread();
  auto threadsPerWarp = tensorEncoding.getThreadsPerWarp();
  auto warpsPerCTA = tensorEncoding.getWarpsPerCTA();
  auto order = tensorEncoding.getOrder();

  if (order.size() != 2)
    return false;

  if (order[0] != 0 || order[1] != 1)
    return false;

  if (useStridedMessage) {
    // For blockM=64 we need to use 16x32bx2 message, meaning the distributed
    // layout needs to be organized into 16x2 threads per warp and one row
    // access per thread.
    if (threadsPerWarp[0] != 16 || threadsPerWarp[1] != 2 ||
        sizePerThread[0] != 1)
      return false;

    if (numBlocks == 1) {
      // with blockM=64 and just single block we cannot split along the M
      // dimension. Check that if we split, we split along N.
      if (numWarpGroupsPerBlock > 1) {
        if (warpsPerCTA[1] == 1)
          return false;
      }
    }
  } else {
    // For blockM=128, we need to use a 32x32b message, which requires 32
    // threads to be sequentially ordered across the M dimension, ensuring
    // that each thread accesses a single and unique TMEM datapath.
    if (threadsPerWarp[0] != 32 || sizePerThread[0] != 1)
      return false;
  }
  return true;
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//
namespace {
class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto sharedAttr = mlir::dyn_cast<TensorMemoryEncodingAttr>(attr)) {
      os << "tmem";
      return AliasResult::FinalAlias;
    }
    if (mlir::isa<TensorMemoryScalesEncodingAttr>(attr)) {
      os << "tmem_scales";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};
} // namespace

//===----------------------------------------------------------------------===//

void TritonNvidiaGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonGPUOpAsmInterface>();
}

// verify TritonNvidiaGPU ops
LogicalResult
TritonNvidiaGPUDialect::verifyOperationAttribute(Operation *op,
                                                 NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
