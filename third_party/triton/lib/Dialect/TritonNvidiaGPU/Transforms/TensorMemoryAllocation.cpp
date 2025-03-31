#include "mlir/Analysis/Liveness.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

namespace {

// Granularity of row allocations.
static constexpr int allocGranularity = 64;
struct TMemChunk {
  int startRow;
  int startCol;
  int numCols;
  int numRows;
};

// Use a simple bitmap to track memory usage. This is a slow but it allows us to
// handle 2D memory without extra algorithmic complexity. The number of
// allocations is expected to be small so the compile time is unlikely to be a
// problem.
struct MemoryBitMap {
  MemoryBitMap() : elements(512 * kNumRows, false) {}
  void free(const TMemChunk &chunk) {
    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, false);
      }
    }
  }
  void alloc(const TMemChunk &chunk) {
    // Ensure the underlying data fits the allocation.
    while ((chunk.startCol + chunk.numCols) * kNumRows >= elements.size())
      elements.resize(2 * elements.size(), false);

    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, true);
      }
    }
  }

  TMemChunk findFirstFit(TMemAllocation allocSize,
                         std::optional<int> rowIdConstraint,
                         int columnAlignment) const {
    int numRows = allocSize.numRows / allocGranularity;
    assert(kNumRows - numRows >= 0);
    assert(allocSize.numRows % allocGranularity == 0);
    int startCol = 0;
    while (1) {
      // Skip to the next aligned address.
      if (startCol % columnAlignment != 0) {
        startCol = (startCol / columnAlignment + 1) * columnAlignment;
      }
      // Iterate over possible starting rows
      for (int startRow = 0; startRow <= kNumRows - numRows; ++startRow) {
        if (rowIdConstraint && *rowIdConstraint != startRow)
          continue;
        bool fits = true;

        // Check if the block starting at (startRow, startCol) is free
        for (int i = 0; i < allocSize.numCols && fits; ++i) {
          for (int j = 0; j < numRows; ++j) {
            if (isUsed(startRow + j, startCol + i)) {
              fits = false;
              break;
            }
          }
        }

        // If a suitable block is found, return it
        if (fits) {
          TMemChunk chunk;
          chunk.startRow = startRow;
          chunk.startCol = startCol;
          chunk.numRows = numRows;
          chunk.numCols = allocSize.numCols;
          return chunk;
        }
      }
      startCol++;
    }
    return TMemChunk();
  }

private:
  bool isUsed(int row, int col) const {
    if (row + col * kNumRows >= elements.size())
      return false;
    return elements[row + col * kNumRows];
  }
  void setUsed(int row, int col, bool used) {
    assert(row + col * kNumRows < elements.size());
    elements[row + col * kNumRows] = used;
  }

  static constexpr int kNumRows = 2;
  std::vector<bool> elements;
};

static Interval<int> getLiveIntervals(Value value, Liveness &liveness,
                                      DenseMap<Operation *, int> &operationId) {
  auto liveOperations = liveness.resolveLiveness(value);
  // Merge the alloc liverange with the liverange of any subview of the
  // allocation.
  SmallVector<Operation *> users(value.getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    if (!isa<triton::gpu::MemDescSubviewOp>(user))
      continue;
    auto usersLivness = liveness.resolveLiveness(user->getResult(0));
    liveOperations.insert(liveOperations.end(), usersLivness.begin(),
                          usersLivness.end());
    users.append(user->getResult(0).getUsers().begin(),
                 user->getResult(0).getUsers().end());
  }
  auto minId = std::numeric_limits<int>::max();
  auto maxId = std::numeric_limits<int>::min();
  std::for_each(liveOperations.begin(), liveOperations.end(),
                [&](Operation *liveOp) {
                  if (operationId[liveOp] < minId) {
                    minId = operationId[liveOp];
                  }
                  if ((operationId[liveOp] + 1) > maxId) {
                    maxId = operationId[liveOp] + 1;
                  }
                });
  return Interval(minId, maxId);
}

static void updateMap(MemoryBitMap &memoryMap, Interval<int> liveInterval,
                      std::map<int, TMemChunk> &intervalLiverangeEnd) {
  int start = liveInterval.start();
  // Add any dead liverange to the list of free intervals.
  for (auto it = intervalLiverangeEnd.begin();
       it != intervalLiverangeEnd.end();) {
    if (it->first > start)
      break;
    memoryMap.free(it->second);
    it = intervalLiverangeEnd.erase(it);
  }
}

static TMemChunk allocFirstFit(MemoryBitMap &memoryMap,
                               TMemAllocation allocSize,
                               std::optional<int> rowIdConstraint,
                               ArrayRef<TMemChunk> coexistingChunks,
                               int columnAlignment) {
  // `coexistingChunks` are all the allocations that might need to be live at
  // the same time as the current allocation plus what is known to be currently
  // live. Union those allocations with a copy of the current memory map and use
  // that to find the actual offsets.
  MemoryBitMap mapForAlloc = memoryMap;
  for (const TMemChunk &chunk : coexistingChunks)
    mapForAlloc.alloc(chunk);
  TMemChunk chunk =
      mapForAlloc.findFirstFit(allocSize, rowIdConstraint, columnAlignment);

  // Mark this chunk as allocated in the actual memory map.
  memoryMap.alloc(chunk);
  return chunk;
}

static Operation *getAlloc(Value value) {
  Operation *op = value.getDefiningOp();
  while (isa<triton::gpu::MemDescSubviewOp>(op)) {
    op = op->getResult(0).getDefiningOp();
  }
  assert(isa<triton::nvidia_gpu::TMEMAllocOp>(op) && "Expected a TMEMAllocOp");
  return op;
}

class RowIdConstraints {
  llvm::EquivalenceClasses<Operation *> dependentAllocs;
  llvm::SmallDenseMap<Operation *, int> rowIndex;

public:
  void joinOps(Operation *op1, Operation *op2) {
    dependentAllocs.unionSets(op1, op2);
  }

  std::optional<int> getRowIdConstraint(Operation *op) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return std::nullopt;
    auto rowIt = rowIndex.find(*it);
    if (rowIt == rowIndex.end())
      return std::nullopt;
    return rowIt->second;
  }

  void addConstraints(Operation *op, int rowId) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return;
    rowIndex[*it] = rowId;
  }
};

static int
allocateTMem(Operation *parentOp,
             DenseMap<triton::nvidia_gpu::TMEMAllocOp, int> &offsets) {
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  DenseMap<Operation *, int> operationId;
  RowIdConstraints rowIdConstraints;
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    operationId[op] = operationId.size();
    if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
      allocs.push_back(alloc);
    }
    if (auto mmaOp = dyn_cast<triton::nvidia_gpu::TCGen5MMAOp>(op)) {
      if (isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
              mmaOp.getA().getType().getEncoding())) {
        TMemAllocation allocSize = getTmemAllocSizes(mmaOp.getA().getType());
        if (allocSize.numRows == 64) {
          // HW restriction, the A alloc and accumulator needs to be in the same
          // rows.
          rowIdConstraints.joinOps(getAlloc(mmaOp.getA()),
                                   getAlloc(mmaOp.getD()));
        } else {
          // TODO: we need to handle cases where the format is blockM and we
          // have multiple blocks.
          assert((cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
                      mmaOp.getA().getType().getEncoding())
                          .getBlockM() != 64 &&
                  cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
                      mmaOp.getD().getType().getEncoding())
                          .getBlockM() != 64) &&
                 "interleaved layout with TMEM operand is not supported yet.");
        }
      }
    }
  });
  int totalMemorySize = 0;
  MemoryBitMap memoryMap;
  Liveness liveness(parentOp);
  std::map<int, TMemChunk> intervalLiverangeEnd;
  DenseMap<TMEMAllocOp, TMemChunk> allocChunks;
  // Implement a linear scan first fit algorithm. We expect that fragmentation
  // won't be a problem, if it is this should be revisited.
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    TMEMAllocOp alloc = *it;

    // Find all allocations in code that may execute at the same time. Only look
    // at processed allocations.
    SmallVector<TMemChunk> coexistingChunks;
    if (auto ws = alloc->getParentOfType<WarpSpecializeOp>()) {
      for (auto prevIt = allocs.begin(); prevIt != it; ++prevIt) {
        TMEMAllocOp prevAlloc = *prevIt;
        auto prevWs = prevAlloc->getParentOfType<WarpSpecializeOp>();
        if (prevWs && prevWs == ws &&
            alloc->getParentRegion() != prevAlloc->getParentRegion())
          coexistingChunks.push_back(allocChunks.at(prevAlloc));
      }
    }

    Interval<int> liveInterval = getLiveIntervals(alloc, liveness, operationId);
    auto memDescType = alloc.getType();
    TMemAllocation allocSize = getTmemAllocSizes(memDescType);
    updateMap(memoryMap, liveInterval, intervalLiverangeEnd);

    std::optional<int> rowIdConstraint =
        rowIdConstraints.getRowIdConstraint(alloc);
    // TODO: clarify the alignment requirements for different allocations. For
    // now enforce an alignment of 4 columns.
    const int columnAlignment = 4;
    TMemChunk chunkAllocated =
        allocFirstFit(memoryMap, allocSize, rowIdConstraint, coexistingChunks,
                      columnAlignment);
    allocChunks.insert({alloc, chunkAllocated});
    // currently naively constraint allocs based on the first one we find.
    rowIdConstraints.addConstraints(alloc, chunkAllocated.startRow);
    intervalLiverangeEnd[liveInterval.end()] = chunkAllocated;
    int colOffset = chunkAllocated.startCol;
    int rowOffset = chunkAllocated.startRow * 16;

    alloc->setAttr(
        "tensor_memory_col_offset",
        IntegerAttr::get(IntegerType::get(parentOp->getContext(), 32),
                         colOffset));
    alloc->setAttr(
        "tensor_memory_row_offset",
        IntegerAttr::get(IntegerType::get(parentOp->getContext(), 32),
                         rowOffset));
    totalMemorySize = std::max(totalMemorySize, colOffset + allocSize.numCols);
  }
  return totalMemorySize;
}

class TritionTensorMemoryAllocationPass
    : public TritionTensorMemoryAllocationPassBase<
          TritionTensorMemoryAllocationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    DenseMap<triton::nvidia_gpu::TMEMAllocOp, int> offsets;
    // TODO: handle cases with multiple function with TMEMAllocOp.
    int totalMemorySize = allocateTMem(mod, offsets);

    std::array<int, 6> possibleAllocations = {0, 32, 64, 128, 256, 512};
    if (totalMemorySize <= 512) {
      for (int size : possibleAllocations) {
        if (totalMemorySize <= size) {
          totalMemorySize = size;
          break;
        }
      }
    }
    // if totalMemorySize > 512 we exceeded the maximum amount of tensor memory,
    // let the compilation finish so that we can raise an exception in python
    // for auto-tuner.
    if (totalMemorySize > 0) {
      assert(mod->getAttr("ttg.shared") != nullptr &&
             cast<IntegerAttr>(mod->getAttr("ttg.shared")).getInt() != 0 &&
             "Shared memory is required for allocation of Tensor Core memory.");
    }

    mod->setAttr("ttg.tensor_memory_size",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        totalMemorySize));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTensorMemoryAllocationPass() {
  return std::make_unique<TritionTensorMemoryAllocationPass>();
}
