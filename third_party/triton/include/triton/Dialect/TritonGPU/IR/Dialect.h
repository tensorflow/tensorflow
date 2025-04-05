#ifndef TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include <unordered_map>

// LinearLayoutCache Utils
using CacheKey = std::tuple<std::vector<int64_t>, mlir::Attribute>;

namespace llvm {
template <typename T> size_t hash_value(const std::vector<T> &vec) {
  return hash_combine_range(vec.begin(), vec.end());
}
} // namespace llvm

namespace std {
template <> struct hash<CacheKey> {
  size_t operator()(const CacheKey &key) const noexcept {
    using llvm::hash_value;
    size_t seed = 0;
    std::apply(
        [&seed](const auto &...elems) {
          ((seed = llvm::hash_combine(seed, hash_value(elems))), ...);
        },
        key);
    return seed;
  }
};
} // namespace std

namespace mlir::triton::gpu {

constexpr static char AttrNumWarpsName[] = "ttg.num-warps";
constexpr static char AttrNumCTAsName[] = "ttg.num-ctas";
constexpr static char AttrTargetName[] = "ttg.target";
constexpr static char AttrNumThreadsPerWarp[] = "ttg.threads-per-warp";

// Find the contextual number of warps on which this operation is executed.
int lookupNumWarps(Operation *op);
// Try to find the contextual number of warps on which this operation is
// executed. Returns nullopt if a warp size cannot be find. This is used for
// verifiers.
std::optional<int> maybeLookupNumWarps(Operation *op);

// FIXME: Make this API and that of maybeLookupNumWarps consistent!
// Utility to find the number of threads per warp
int lookupThreadsPerWarp(OpBuilder &rewriter);

template <typename Key, typename Value> class Cache {
public:
  std::optional<Value> get(const Key &key) {
    std::shared_lock lock(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void set(Key key, Value result) {
    std::scoped_lock lock(mutex);
    cache.emplace(std::move(key), std::move(result));
  }

private:
  std::unordered_map<Key, Value> cache;
  llvm::sys::SmartRWMutex<true> mutex;
};

using LinearLayoutCache = Cache<CacheKey, LinearLayout>;
using LinearEncodingCache = Cache<CacheKey, LinearEncodingAttr>;
} // namespace mlir::triton::gpu

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Dialect.h.inc"
#include "triton/Dialect/TritonGPU/IR/Ops.h.inc"

namespace mlir::triton::gpu {
struct SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
  StringRef getName() final { return "<SharedMemory>"; }
};

// Convert a distributed layout to a linear encoding
LinearEncodingAttr toLinearEncoding(RankedTensorType type);
LinearEncodingAttr toLinearEncoding(Attribute layout, ArrayRef<int64_t> shape);

unsigned getTotalElemsPerThread(Type type);

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape);

SmallVector<unsigned> getElemsPerThread(Type type);

// Returns the number of warps per CTA that have access to non-replicated
// elements of the tensor. E.g. for a blocked layout with sizePerThread = [1,
// 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4] and tensor shape = [2, 2],
// returns [1, 1], since the first warp has access to the full tensor, whereas
// the other warps have access to replicated elements.
SmallVector<unsigned> getWarpsPerCTA(Attribute layout,
                                     ArrayRef<int64_t> tensorShape);
inline SmallVector<unsigned> getWarpsPerCTA(RankedTensorType type) {
  return getWarpsPerCTA(type.getEncoding(), type.getShape());
}

// Returns the number of contiguous elements of the logical tensor that each
// thread has access to, on each dimension of the tensor. For a blocked layout
// with sizePerThread = [1, 4] and tensor shape = [128, 1], the elements
// for thread 0 would be [A_{0, 0}, A_{0, 0}, A_{0, 0}, A_{0, 0}], returns [1,
// 1]. Whereas for a tensor shape [128, 128], the elements for thread 0 would be
// [A_{0, 0}, A_{0, 1}, A_{0, 2}, A_{0, 3}], returns [1, 4].
SmallVector<unsigned> getContigPerThread(RankedTensorType tensorType);

// Returns the number of threads per warp that have access to non-replicated
// elements of the tensor. E.g. for a blocked layout with sizePerThread = [1,
// 1], threadsPerWarp = [2, 16] and tensor shape = [2, 2], threads 0, 1, 16, 17
// have access to the full tensor, whereas the other threads have access to
// replicated elements, so this function returns [2, 2].
SmallVector<unsigned> getThreadsPerWarp(Attribute layout,
                                        ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getThreadsPerWarp(RankedTensorType type) {
  return getThreadsPerWarp(type.getEncoding(), type.getShape());
}

// Returns the dimensions of the tensor from minor (fast-varying) to
// major (slow-varying). For distributed layouts, this represents
// the order of the elements within a thread.
// For shared Layout, the order refers to which dimension of the original tensor
// is contiguous in shared memory.
SmallVector<unsigned> getOrder(DistributedEncodingTrait layout,
                               ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getOrder(RankedTensorType type) {
  return getOrder(cast<DistributedEncodingTrait>(type.getEncoding()),
                  type.getShape());
}

SmallVector<unsigned> getOrder(SharedEncodingTrait layout,
                               ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getOrder(MemDescType type) {
  return getOrder(cast<SharedEncodingTrait>(type.getEncoding()),
                  type.getShape());
}
inline SmallVector<unsigned> getOrder(TensorOrMemDesc type) {
  if (auto memDesc = dyn_cast<MemDescType>(type)) {
    return getOrder(memDesc);
  } else {
    auto tensorTy = cast<RankedTensorType>(type);
    return getOrder(tensorTy);
  }
}

// To be removed once we implement arbitrary swizzled layouts
// It chooses heuristically an order for the memory layout in which to save
// a distributed layout taking into account the order of the elements
// and the threads.
SmallVector<unsigned> getOrderForMemory(DistributedEncodingTrait layout,
                                        ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getOrderForMemory(RankedTensorType type) {
  return getOrderForMemory(cast<DistributedEncodingTrait>(type.getEncoding()),
                           type.getShape());
}
inline SmallVector<unsigned> getOrderForMemory(TensorOrMemDesc type) {
  if (auto memDesc = dyn_cast<MemDescType>(type)) {
    return getOrder(memDesc);
  } else {
    auto tensorTy = cast<RankedTensorType>(type);
    return getOrderForMemory(tensorTy);
  }
}

// Returns the dimensions along which warpId's are distributed.
// warpsPerCTA only tells the warp layout in the CTA, e.g. warpsPerCTA = [2, 4]
// tells there are 2 warps along dim0 and 4 warps along dim1.
// warpOrder tells the specific order when distributing warp IDs.
// E.g. warpOrder = [0, 1] means the warp IDs are distributed as follows
// [warp0  warp2  warp4 warp6]
// [warp1  warp3  warp5 warp7]
SmallVector<unsigned> getWarpOrder(DistributedEncodingTrait layout,
                                   ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getWarpOrder(RankedTensorType type) {
  return getWarpOrder(cast<DistributedEncodingTrait>(type.getEncoding()),
                      type.getShape());
}

// Returns the dimensions along which threadId's are distributed.
// Similar to warpOrder, threadOrder is necessary to tell the specific thread
// distribution in the warp.
SmallVector<unsigned> getThreadOrder(DistributedEncodingTrait layout,
                                     ArrayRef<int64_t> shape);
inline SmallVector<unsigned> getThreadOrder(RankedTensorType type) {
  return getThreadOrder(cast<DistributedEncodingTrait>(type.getEncoding()),
                        type.getShape());
}

CTALayoutAttr getCTALayout(Attribute layout);

SmallVector<unsigned> getCTAsPerCGA(Attribute layout);

SmallVector<unsigned> getCTASplitNum(Attribute layout);

SmallVector<unsigned> getCTAOrder(Attribute layout);

/* The difference between ShapePerCTATile and ShapePerCTA:
 * (1) ShapePerCTATile is defined by SizePerThread * ThreadsPerWarp *
 *     WarpsPerCTA in each dimension and is independent from the tensor shape.
 * (2) ShapePerCTA is defined by shape / CTASplitNum in each dimension.
 * (3) In the implementation of emitIndices, ShapePerCTATile will
 *     be replicated or wrapped to fit ShapePerCTA.
 */
// [FIXME LL] Kill this function
SmallVector<unsigned> getShapePerCTATile(RankedTensorType layout);

// Returns the "logical" shape per CTA
SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape);
SmallVector<int64_t> getShapePerCTA(Type type);

// Returns the shape per CTA, which is "physically" allocated
// Such shapes may be bigger than the logical one due to, for example, padding
// in shared memory.
SmallVector<int64_t> getAllocationShapePerCTA(Attribute layout,
                                              ArrayRef<int64_t> shape);
SmallVector<int64_t> getAllocationShapePerCTA(Type type);

unsigned getNumCTAs(Attribute layout);

// Return the order that represents that the batch is in row-major or
// column-major order for a batch of matrices of shape [*, m, n] with
// len(shape) == rank.
SmallVector<unsigned> getMatrixOrder(unsigned rank, bool rowMajor);

// Return the order that represents that the dot operand is in kContig
// (contiguous in the inner dimension) or it's contiguous on the outer
// dimension.
SmallVector<unsigned> getOrderForDotOperand(unsigned opIdx, unsigned rank,
                                            bool kContig);

bool isExpensiveCat(CatOp cat, Attribute targetEncoding);

// Return true if a view between the two types cannot be implemented as a no-op.
bool isExpensiveView(Type srcType, Type dstType);

// Return a blocked encoding where the shape is distributed contiguously amongst
// the threads, warps, CTAs with 1 element per threads.
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs);

// Dump information about which threads/registers contain each of the tensor
// elements.
void dumpLayout(RankedTensorType tensorType);

// Dump the layout from HW point of view and prints what tensor element is held
// by each thread and register.
void dumpHWLayout(RankedTensorType tensorType);

// Return a string representation of the layout of the tensor.
std::string getLayoutStr(RankedTensorType tensorType, bool useHWPointOfView);

template <typename T>
llvm::SmallVector<T> expandMatrixShapeWithBatch(llvm::ArrayRef<T> s);

llvm::SmallVector<unsigned>
expandMatrixOrderWithBatch(llvm::ArrayRef<unsigned> o);
} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_DIALECT_H_
