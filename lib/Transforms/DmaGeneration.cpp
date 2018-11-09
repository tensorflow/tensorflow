//===- DmaGeneration.cpp - DMA generation pass ------------------------ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a pass to automatically promote accessed memref regions
// to buffers in a faster memory space that is explicitly managed, with the
// necessary data movement operations expressed as DMAs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/StmtVisitor.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "dma-generate"

using namespace mlir;

namespace {

// A region of memory in a lower memory space.
struct Region {
  // Memref corresponding to the region.
  MLValue *memref;
  // Read or write.
  bool isWrite;
  // Region of memory accessed.
  // TODO(bondhugula): Replace this to exploit HyperRectangularSet.
  std::unique_ptr<FlatAffineConstraints> cst;
};

/// Generates DMAs for memref's living in 'lowMemorySpace' into newly created
/// buffers in 'highMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
struct DmaGeneration : public FunctionPass, StmtWalker<DmaGeneration> {
  explicit DmaGeneration(unsigned lowMemorySpace = 0,
                         unsigned highMemorySpace = 1,
                         int minDmaTransferSize = 1024)
      : FunctionPass(&DmaGeneration::passID), lowMemorySpace(lowMemorySpace),
        highMemorySpace(highMemorySpace),
        minDmaTransferSize(minDmaTransferSize) {}

  PassResult runOnMLFunction(MLFunction *f) override;
  // Not applicable to CFG functions.
  PassResult runOnCFGFunction(CFGFunction *f) override { return success(); }
  bool runOnForStmt(ForStmt *forStmt);

  void visitOperationStmt(OperationStmt *opStmt);
  void generateDma(const Region &region, Location loc, MLFuncBuilder *b);

  // List of memory regions to promote.
  std::vector<Region> regions;

  static char passID;
  const unsigned lowMemorySpace;
  const unsigned highMemorySpace;
  const int minDmaTransferSize;
};

} // end anonymous namespace

char DmaGeneration::passID = 0;

/// Generates DMAs for memref's living in 'lowMemorySpace' into newly created
/// buffers in 'highMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO(bondhugula): extend this to store op's.
FunctionPass *mlir::createDmaGenerationPass(unsigned lowMemorySpace,
                                            unsigned highMemorySpace,
                                            int minDmaTransferSize) {
  return new DmaGeneration(lowMemorySpace, highMemorySpace, minDmaTransferSize);
}

// Gather regions to promote to buffers in higher memory space.
// TODO(bondhugula): handle store op's; only load's handled for now.
void DmaGeneration::visitOperationStmt(OperationStmt *opStmt) {
  if (auto loadOp = opStmt->dyn_cast<LoadOp>()) {
    if (loadOp->getMemRefType().getMemorySpace() != lowMemorySpace)
      return;

    // TODO(bondhugula): eventually, we need to be performing a union across all
    // regions for a given memref instead of creating one region per memory op.
    // This way we would be allocating O(num of memref's) sets instead of
    // O(num of load/store op's).
    auto memoryRegion = std::make_unique<FlatAffineConstraints>();
    if (!getMemoryRegion(opStmt, memoryRegion.get())) {
      LLVM_DEBUG(llvm::dbgs() << "Error obtaining memory region");
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "Memory region");
    LLVM_DEBUG(memoryRegion->dump());

    regions.push_back(
        {cast<MLValue>(loadOp->getMemRef()), false, std::move(memoryRegion)});
  }
}

// Create a buffer in the higher (faster) memory space for the specified region;
// generate a DMA from the lower memory space to this one, and replace all loads
// to load from the buffer.
// TODO: handle write regions by generating outgoing DMAs; only read regions are
// handled for now.
void DmaGeneration::generateDma(const Region &region, Location loc,
                                MLFuncBuilder *b) {
  // Only memref read regions handled for now.
  if (region.isWrite)
    return;

  auto *memref = region.memref;
  auto memRefType = memref->getType().cast<MemRefType>();

  SmallVector<SSAValue *, 4> srcIndices, destIndices;

  SSAValue *zeroIndex = b->create<ConstantIndexOp>(loc, 0);

  unsigned rank = memRefType.getRank();
  SmallVector<int, 4> shape;
  shape.reserve(rank);

  // Index start offsets for faster memory buffer relative to the original.
  SmallVector<int, 4> offsets;
  offsets.reserve(rank);

  unsigned numElements = 1;
  for (unsigned d = 0; d < rank; d++) {
    auto lb = region.cst->getConstantLowerBound(d);
    auto ub = region.cst->getConstantUpperBound(d);

    if (!lb.hasValue() || !ub.hasValue()) {
      LLVM_DEBUG(llvm::dbgs() << "Non-constant loop bounds");
      return;
    }

    offsets.push_back(lb.getValue());
    int dimSize = ub.getValue() - lb.getValue() + 1;
    if (dimSize <= 0)
      return;
    shape.push_back(dimSize);
    numElements *= dimSize;
    srcIndices.push_back(b->create<ConstantIndexOp>(loc, lb.getValue()));
    destIndices.push_back(zeroIndex);
  }

  // Create the faster memref buffer.
  auto fastMemRefType =
      b->getMemRefType(shape, memRefType.getElementType(), {}, highMemorySpace);

  auto fastMemRef = b->create<AllocOp>(loc, fastMemRefType)->getResult();
  // Create a tag (single element 1-d memref) for the DMA.
  auto tagMemRefType = b->getMemRefType({1}, b->getIntegerType(32));
  auto tagMemRef = b->create<AllocOp>(loc, tagMemRefType);
  auto numElementsSSA = b->create<ConstantIndexOp>(loc, numElements);

  // TODO(bondhugula): check for transfer sizes not being a multiple of
  // minDmaTransferSize and handle them appropriately.

  // TODO(bondhugula): Need to use strided DMA for multi-dimensional (>= 2-d)
  // case.
  b->create<DmaStartOp>(loc, memref, srcIndices, fastMemRef, destIndices,
                        numElementsSSA, tagMemRef, zeroIndex);
  b->create<DmaWaitOp>(loc, tagMemRef, zeroIndex, numElementsSSA);

  // Replace all uses of the old memref with the promoted one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    auto d0 = b->getAffineDimExpr(i);
    remapExprs.push_back(d0 - offsets[i]);
  }
  auto indexRemap = b->getAffineMap(rank, 0, remapExprs, {});
  replaceAllMemRefUsesWith(memref, cast<MLValue>(fastMemRef), {}, indexRemap);
}

bool DmaGeneration::runOnForStmt(ForStmt *forStmt) {
  walk(forStmt);

  MLFuncBuilder b(forStmt);
  for (const auto &region : regions) {
    generateDma(region, forStmt->getLoc(), &b);
  }

  // This function never leaves the IR in an invalid state.
  return false;
}

PassResult DmaGeneration::runOnMLFunction(MLFunction *f) {
  bool ret = false;

  for (auto &stmt : *f) {
    // Run on all 'for' statements for now.
    if (auto *forStmt = dyn_cast<ForStmt>(&stmt)) {
      ret = ret | runOnForStmt(forStmt);
    }
  }
  return ret ? failure() : success();
}

static PassRegistration<DmaGeneration>
    pass("dma-generate", "Generate DMAs for memory operations");
