#ifndef TRITON_ANALYSIS_ALLOCATION_H
#define TRITON_ANALYSIS_ALLOCATION_H

#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <atomic>
#include <limits>

namespace mlir {

namespace triton {
class AllocationAnalysis;

/// Callback to allow backends to specify target-specific scratch sizes for
/// some operations.
using AllocationAnalysisScratchSizeFn = std::function<unsigned(Operation *)>;

unsigned defaultAllocationAnalysisScratchSizeFn(Operation *op);

// To convert a tensor from one layout to another, we need to allocate a
// temporary buffer (i.e., scratch buffer) in shared memory. The conversion may
// require multiple iterations, with each iteration involving multiple
// vectorized loads/stores. The scratch buffer has a shape (`repShape`) that
// represents the maximum size accessed in each dimension during each iteration.
// It is padded (`paddedRepShape`) to avoid bank conflicts and is accessed in a
// specific `order`.
struct ScratchConfig {
  SmallVector<unsigned> repShape;
  SmallVector<unsigned> paddedRepShape;
  SmallVector<unsigned> order;
  unsigned inVec;
  unsigned outVec;

  ScratchConfig(SmallVector<unsigned> repShape,
                SmallVector<unsigned> paddedRepShape, unsigned inVec = 1,
                unsigned outVec = 1)
      : repShape(repShape), paddedRepShape(paddedRepShape), inVec(inVec),
        outVec(outVec) {}

  void print(llvm::raw_ostream &os) const {
    os << "repShape: [";
    llvm::interleaveComma(repShape, os);
    os << "]";
    os << ", paddedRepShape: [";
    llvm::interleaveComma(paddedRepShape, os);
    os << "]";
    os << ", order: [";
    llvm::interleaveComma(order, os);
    os << "]";
    os << ", inVec: " << inVec << ", outVec: " << outVec << "\n";
  }
};

// For a layout conversion between `srcTy` and `dstTy`, return the vector length
// that can be used for the stores to and loads from shared memory,
// respectively.
std::pair</*inVec*/ unsigned, /*outVec*/ unsigned>
getScratchCvtInOutVecLengths(RankedTensorType srcTy, RankedTensorType dstTy);

ScratchConfig getScratchConfigForCvt(RankedTensorType srcTy,
                                     RankedTensorType dstTy);

} // namespace triton

/// Modified from llvm-15.0: llvm/ADT/AddressRanges.h
/// A class that represents an interval, specified using a start and an end
/// values: [Start, End).
template <typename T> class Interval {
public:
  Interval() {}
  Interval(T S, T E) : Start(S), End(E) { assert(Start <= End); }
  T start() const { return Start; }
  T end() const { return End; }
  T size() const { return End - Start; }
  bool contains(T Addr) const { return Start <= Addr && Addr < End; }
  bool intersects(const Interval &R) const {
    return Start < R.End && R.Start < End;
  }
  bool operator==(const Interval &R) const {
    return Start == R.Start && End == R.End;
  }
  bool operator!=(const Interval &R) const { return !(*this == R); }
  bool operator<(const Interval &R) const {
    return std::make_pair(Start, End) < std::make_pair(R.Start, R.End);
  }

private:
  T Start = std::numeric_limits<T>::min();
  T End = std::numeric_limits<T>::max();
};

template <class T> Interval(T, T) -> Interval<T>;

class Allocation {
public:
  /// A unique identifier for shared memory buffers
  using BufferId = size_t;
  using BufferIdSetT = DenseSet<BufferId>;
  using FuncAllocMapT = CallGraph<Allocation>::FuncDataMapT;

  static constexpr BufferId InvalidBufferId =
      std::numeric_limits<BufferId>::max();

  Allocation() = default;
  /// Creates a new Allocation analysis that computes the shared memory
  /// information for all associated shared memory values.
  explicit Allocation(Operation *operation) : operation(operation) {}

  /// Runs allocation analysis on the given top-level operation.
  void run(FuncAllocMapT &funcAllocMap,
           triton::AllocationAnalysisScratchSizeFn scratchSizeGetter);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Returns the offset of the given buffer in the shared memory.
  size_t getOffset(BufferId bufferId) const {
    return bufferSet.at(bufferId).offset;
  }

  /// Returns the size of the given buffer in the shared memory.
  size_t getAllocatedSize(BufferId bufferId) const {
    return bufferSet.at(bufferId).size;
  }

  /// Returns the allocated interval of the given buffer.
  Interval<size_t> getAllocatedInterval(BufferId bufferId) const {
    auto &buffer = bufferSet.at(bufferId);
    return Interval<size_t>(buffer.offset, buffer.offset + buffer.size);
  }

  /// Returns the buffer id of the given value.
  /// This interface only returns the allocated buffer id.
  /// If you want to get all the buffer ids that are associated with the given
  /// value, including alias buffers, use getBufferIds.
  BufferId getBufferId(Value value) const {
    if (valueBuffer.count(value)) {
      return valueBuffer.lookup(value)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns all the buffer ids of the given value, including alias buffers.
  BufferIdSetT getBufferIds(Value value) const {
    BufferIdSetT bufferIds;
    auto allocBufferId = getBufferId(value);
    if (allocBufferId != InvalidBufferId)
      bufferIds.insert(allocBufferId);
    for (auto *buffer : aliasBuffer.lookup(value)) {
      if (buffer->id != InvalidBufferId)
        bufferIds.insert(buffer->id);
    }
    return bufferIds;
  }

  /// Returns the scratch buffer id of the given value.
  BufferId getBufferId(Operation *operation) const {
    if (opScratch.count(operation)) {
      return opScratch.lookup(operation)->id;
    } else if (opVirtual.count(operation)) {
      return opVirtual.lookup(operation)->id;
    } else {
      return InvalidBufferId;
    }
  }

  /// Returns if the given buffer is a virtual buffer.
  bool isVirtualBuffer(BufferId bufferId) const {
    return bufferSet.at(bufferId).kind == BufferT::BufferKind::Virtual;
  }

  /// Returns the size of total shared memory allocated
  size_t getSharedMemorySize() const { return sharedMemorySize; }

  /// Returns mapping from operation to list of live LDS buffers
  std::map<Operation *, SmallVector<BufferId>> getLiveBuffers();

private:
  /// A class that represents a shared memory buffer
  struct BufferT {
    /// Explicit: ttg.local_alloc
    /// Scratch: ttg.convert_layout
    /// Virtual: triton.call
    enum class BufferKind { Explicit, Scratch, Virtual };

    BufferKind kind;
    BufferId id;
    Operation *owner;
    size_t size;
    size_t alignment;
    size_t offset;

    bool operator==(const BufferT &other) const { return id == other.id; }
    bool operator<(const BufferT &other) const { return id < other.id; }

    BufferT(BufferKind kind, BufferId id, Operation *owner, size_t size,
            size_t alignment = 4, size_t offset = 0)
        : kind(kind), id(id), owner(owner), size(size), alignment(alignment),
          offset(offset) {}

    size_t setOffsetAligned(size_t newOffset) {
      return offset = llvm::alignTo(newOffset, alignment);
    }
  };

  /// Op -> Scratch Buffer
  using OpScratchMapT = llvm::MapVector<Operation *, BufferT *>;
  /// Value -> Explicit Buffer
  using ValueBufferMapT = llvm::MapVector<Value, BufferT *>;
  /// Value -> Alias Buffer
  using AliasBufferMapT = llvm::MapVector<Value, llvm::SetVector<BufferT *>>;
  /// BufferId -> Buffer
  using BufferSetT = std::map<BufferId, BufferT>;

private:
  template <BufferT::BufferKind Kind, typename KeyType, typename... Args>
  void addBuffer(KeyType &key, Args &&...args) {
    BufferId nextId = bufferIdCounter++;
    auto [it, inserted] = bufferSet.insert_or_assign(
        nextId, BufferT(Kind, nextId, key, std::forward<Args>(args)...));
    BufferT *buffer = &it->second;
    if constexpr (Kind == BufferT::BufferKind::Explicit) {
      valueBuffer[key] = buffer;
    } else if constexpr (Kind == BufferT::BufferKind::Virtual) {
      opVirtual[key] = buffer;
    } else {
      opScratch[key] = buffer;
    }
  }

  void addAlias(Value value, Value alloc) {
    aliasBuffer[value].insert(valueBuffer[alloc]);
  }

private:
  Operation *operation = nullptr;
  OpScratchMapT opScratch;
  OpScratchMapT opVirtual;
  ValueBufferMapT valueBuffer;
  AliasBufferMapT aliasBuffer;
  BufferSetT bufferSet;
  size_t sharedMemorySize = 0;

  size_t bufferIdCounter = 0;

  friend class triton::AllocationAnalysis;
};

/// Static analysis that computes the allocation of shared memory buffers
/// of the entire call graph.
/// The allocation is performed in a post-order walk of the call graph.
/// Each call op is treated like convert_layout that allocates a scratch buffer.
/// At each call, we compute the start offset of the scratch buffer and pass it
/// as an argument to the callee.
class ModuleAllocation : public CallGraph<Allocation> {
public:
  using FuncOffsetMapT = DenseMap<FunctionOpInterface, Value>;

  ModuleAllocation(ModuleOp moduleOp,
                   triton::AllocationAnalysisScratchSizeFn scratchSizeGetter =
                       triton::defaultAllocationAnalysisScratchSizeFn)
      : CallGraph<Allocation>(moduleOp) {
    walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
        // Pre-order edge walk callback
        [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
        // Post-order node walk callback
        [&](FunctionOpInterface funcOp) {
          auto [iter, inserted] = funcMap.try_emplace(funcOp, funcOp);
          if (inserted)
            iter->second.run(funcMap, scratchSizeGetter);
        });
  }

  size_t getSharedMemorySize() {
    size_t size = 0;
    for (auto funcOp : getRoots()) {
      auto *alloc = getFuncData(funcOp);
      size = std::max(size, alloc->getSharedMemorySize());
    }
    return size;
  }

  size_t getSharedMemorySize(FunctionOpInterface funcOp) {
    return getFuncData(funcOp)->getSharedMemorySize();
  }

  void setFunctionSharedMemoryValue(FunctionOpInterface funcOp, Value value) {
    sharedMemoryValue[funcOp] = value;
  }

  Value getFunctionSharedMemoryBase(FunctionOpInterface funcOp) {
    return sharedMemoryValue[funcOp];
  }

private:
  FuncOffsetMapT sharedMemoryValue;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_ALLOCATION_H
