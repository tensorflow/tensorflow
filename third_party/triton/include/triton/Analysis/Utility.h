#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir {

inline bool isZeroConst(Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isZero();
  return false;
}

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp op)
      : op(op.getOperation()), axis(op.getAxis()) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  ArrayRef<int64_t> getSrcShape() { return srcShape; }

  Attribute getSrcLayout() { return srcEncoding; }

  triton::ReduceOp getOperation() { return op; }

  unsigned getThreadOffsetOnReductionAxis();

  bool isWarpSynchronous();

  unsigned getInterWarpSizeWithUniqueData();

  unsigned getIntraWarpSizeWithUniqueData();

  // The shape of the shared memory space needed for the reduction.
  SmallVector<unsigned> getScratchRepShape();

  SmallVector<unsigned> getOrderWithAxisAtBeginning();

  unsigned getScratchSizeInBytes();

  bool isReduceWithinCTA();

private:
  triton::ReduceOp op;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;
};

class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op) : scanOp(op) {
    auto firstTy = cast<RankedTensorType>(op.getOperands()[0].getType());
    srcShape = firstTy.getShape();
    legacyEncoding = firstTy.getEncoding();
    srcEncoding = triton::gpu::toLinearEncoding(legacyEncoding, srcShape);
    srcElementTypes = op.getElementTypes();
    // The codegen does not support different element/thread/warp order so
    // we choose one a priori. We choose that of the blocked encoding.
    // When we generalise this code to other layouts we'll probably need to
    // get rid of all this logic and the *Stride auxiliary methods
    // and replace them by transposes and reshapes on the LinearLayout
    if (auto blockedEncoding =
            dyn_cast<triton::gpu::BlockedEncodingAttr>(legacyEncoding)) {
      order = llvm::to_vector(blockedEncoding.getOrder());
    } else {
      order = srcEncoding.getOrder();
    }

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != legacyEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }
  // Return true if the lowering of the scan op is supported.
  bool isSupported();
  // Return the number of elements per thread along axis dim.
  unsigned getAxisNumElementsPerThread();
  // Return the number of elements per thread along non-axis dims.
  unsigned getNonAxisNumElementsPerThread();
  // Return the number of threads per warp along non-axis dims.
  unsigned getNonAxisNumThreadsPerWarp();
  // Return the flat numbers of threads computing independent scan results.
  unsigned getNonAxisNumThreadsPerCTA();
  // Return the number of warps per CTA along axis dim with unique data.
  unsigned getAxisNumWarpsWithUniqueData();
  // Return the number of threads per warp along axis dim with unique data.
  unsigned getAxisNumThreadsPerWarpWithUniqueData();
  // Return the number of blocks along axis dim.
  unsigned getAxisNumBlocks();
  // Return the number of blocks along non axis dim.
  unsigned getNonAxisNumBlocks();
  // Return the size of the scratch space needed for scan lowering.
  unsigned getScratchSizeInBytes();
  // Return the number of elements of the scratch space needed for scan
  // lowering.
  unsigned getScratchSizeInElems();

  // Stride between contiguous element along axis dim.
  unsigned getAxisElementStride();
  // Stride between contiguous threads along axis dim.
  unsigned getAxisThreadStride();
  // Stride between contiguous blocks along axis dim.
  unsigned getAxisBlockStride();

  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  bool getReverse() { return scanOp.getReverse(); }
  triton::gpu::LinearEncodingAttr getEncoding() { return srcEncoding; }
  llvm::ArrayRef<int64_t> getShape() { return srcShape; }
  unsigned getNumOperands() { return scanOp.getNumOperands(); }
  SmallVector<Type> getElementTypes() { return srcElementTypes; }
  SmallVector<unsigned> getOrder() { return order; }
  Region &getCombineOp();

private:
  triton::ScanOp scanOp;
  triton::gpu::LinearEncodingAttr srcEncoding;
  Attribute legacyEncoding;
  llvm::ArrayRef<int64_t> srcShape;
  SmallVector<Type> srcElementTypes;
  SmallVector<unsigned> order;
};

// Helper class for lowering `tt.gather` operations. This class shares lowering
// logic between shared memory allocation and LLVM codegen.
class GatherLoweringHelper {
public:
  GatherLoweringHelper(triton::GatherOp gatherOp);

  // Get the shared memory scratch size required by this op.
  unsigned getScratchSizeInBytes();
  // Determine if the gather can be performed completely within a warp.
  bool isWarpLocal();

private:
  triton::GatherOp gatherOp;
};

// This struct represents a decomposed layout conversion within a warp into
// three transformations: P1 and P2 represent lane-dependent register shuffles
// and W represents a warp shuffle. P2^-1 is returned because it represents the
// (reg, lane) -> (reg) mapping from the perspective of the destination element.
//
// Nearly all layout conversions that only require data movement within a warp
// can be implemented this way.
struct DecomposedWarpConversion {
  triton::LinearLayout P1, W, P2inv;
  triton::LinearLayout reducedP1, reducedP2inv;
};

// Given the source and destination tensor types where a layout conversion only
// involves data movement within warps, attempt to find a decomposition for a
// warp layout conversion.
std::optional<DecomposedWarpConversion>
getWarpLayoutConvertDecomposition(RankedTensorType srcTy,
                                  RankedTensorType dstTy);

// Decomposes a reshape into simpler pieces.
//
// As an example, suppose we have a reshape from [4,4,4] to [2,2,8,2].
// You might explain what this does as follows.
//
//  - Split the first input dimension into [2,2].
//  - Take the remaining two input dimensions, merge them into a single [16]
//    dim, and then split that into [8,2].
//
// In general, a reshape can be described a sequence of smushing one or more
// input dimensions together and then breaking them apart into one or more
// output dimensions.  So we could represent the example above as follows.
//
//   [
//     ([0], [0, 1]),  # input dim [0] -> output dims [0, 1]
//     ([1, 2], [2, 3]),  # input dims [1, 2] -> output dims [2, 3]
//   ]
//
// Notice that the input dims (first tuple elems) appear in sequential order if
// you read left-to-right-top-to-bottom, and so do the output dims.
//
// This function returns the above decomposition.
SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape);

// Returns the number of elements in the scratch space needed.
// If shape is empty, it means no shared memory is needed.
unsigned getNumScratchElements(ArrayRef<unsigned> shape);

bool supportWMMA(triton::DotOp op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

// Conversion from `srcTy` to `dstTy` involving the minimum amount of data
// transfer provided that both types can be converted to LL (if it can't it'll
// return nullopt). The output will be such that layout.getInDimNames() ==
// layout.getOutDimNames() and the conversion will not include kBlock (resp.
// kWarp or kLane) if it can be avoided
triton::LinearLayout minimalCvtLayout(RankedTensorType srcTy,
                                      RankedTensorType dstTy);

// Conversion from `srcTy` to `dstTy` only involves reordering of registers.
// There is no need for data exchange across threads, warps, or blocks.
bool cvtReordersRegisters(RankedTensorType srcTy, RankedTensorType dstTy);

// Conversion from `srcTy` to `dstTy` involves data exchange across threads
// within a warp.  No data exchange across warps or blocks is needed.
bool cvtNeedsWarpShuffle(RankedTensorType srcTy, RankedTensorType dstTy);

// Conversion from `srcTy` to `dstTy` involves data exchange across threads,
// warps, and possibly blocks.
bool cvtNeedsSharedMemory(RankedTensorType srcTy, RankedTensorType dstTy);

bool atomicNeedsSharedMemory(Value result);

// Return true if the src and dst layout match.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);

// Check if MFMA layout can be converted to the dot operand
// layout using warp shuffle.
bool matchMFMAAndDotOperandShuffleCase(RankedTensorType srcTy,
                                       RankedTensorType dstTy);

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Operation in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort);

/// This uses the toplogicalSort above
SetVector<Operation *>
multiRootGetSlice(Operation *op, TransitiveFilter backwardFilter = nullptr,
                  TransitiveFilter forwardFilter = nullptr);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

/// This class represents a call graph for a given ModuleOp and holds
/// data of type T associated with each FunctionOpInterface.
template <typename T> class CallGraph {
public:
  using FuncDataMapT = DenseMap<FunctionOpInterface, T>;

  /// Constructor that builds the call graph for the given moduleOp.
  explicit CallGraph(ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  /// Walks the call graph and applies the provided update functions
  /// to the edges and nodes.
  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void walk(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    DenseSet<FunctionOpInterface> visited;
    for (auto root : roots) {
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
                                               updateNodeFn);
    }
  }

  /// Retrieves the data associated with a function
  T *getFuncData(FunctionOpInterface funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  /// Getters
  ModuleOp getModuleOp() const { return moduleOp; }
  SmallVector<FunctionOpInterface> getRoots() const { return roots; }
  size_t getNumFunctions() const { return funcMap.size(); }

  /// Returns true if the given function is a root.
  bool isRoot(FunctionOpInterface funcOp) const {
    return llvm::is_contained(roots, funcOp);
  }

  /// Maps the data and the graph nodes associated with a funcOp to a
  /// targetFuncOp.
  template <typename FROM, typename TO>
  void mapFuncOp(FROM funcOp, TO targetFuncOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.second == funcOp) {
          edge.second = targetFuncOp;
        }
      }
    }
    graph[targetFuncOp] = graph[funcOp];
    // Replace in roots
    for (auto it = roots.begin(); it != roots.end(); ++it) {
      if (*it == funcOp) {
        *it = targetFuncOp;
        break;
      }
    }
    // Replace in funcMap
    funcMap[targetFuncOp] = funcMap[funcOp];
  }

  /// Maps the graph edges associated with a callOp to a targetCallOp.
  template <typename FROM, typename TO>
  void mapCallOp(FROM callOp, TO targetCallOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.first == callOp) {
          edge.first = targetCallOp;
        }
      }
    }
  }

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseSet<FunctionOpInterface> visited;
    // Build graph
    moduleOp.walk([&](Operation *op) {
      auto caller = op->getParentOfType<FunctionOpInterface>();
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto *callee = callOp.resolveCallableInTable(&symbolTable);
        auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callee);
        if (funcOp) {
          graph[caller].emplace_back(
              std::pair<CallOpInterface, FunctionOpInterface>(callOp, funcOp));
          visited.insert(funcOp);
        }
      }
    });
    // Find roots
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      if (!visited.count(funcOp)) {
        roots.push_back(funcOp);
      }
    });
  }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doWalk(FunctionOpInterface funcOp,
              DenseSet<FunctionOpInterface> &visited, UpdateEdgeFn updateEdgeFn,
              UpdateNodeFn updateNodeFn) {
    if (visited.count(funcOp)) {
      llvm::report_fatal_error("Cycle detected in call graph");
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PreOrder) {
      updateNodeFn(funcOp);
    }
    for (auto [callOp, callee] : graph[funcOp]) {
      if constexpr (UpdateEdgeOrder == WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(callee, visited, updateEdgeFn,
                                               updateNodeFn);
      if constexpr (UpdateEdgeOrder == WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PostOrder) {
      updateNodeFn(funcOp);
    }
    visited.erase(funcOp);
  }

protected:
  ModuleOp moduleOp;
  DenseMap<FunctionOpInterface,
           SmallVector<std::pair<CallOpInterface, FunctionOpInterface>>>
      graph;
  FuncDataMapT funcMap;
  SmallVector<FunctionOpInterface> roots;
};
// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

triton::MakeTensorPtrOp getMakeTensorPtrOp(Value v);

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
