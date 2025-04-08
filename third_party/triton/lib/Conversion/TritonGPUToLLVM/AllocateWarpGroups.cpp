#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUALLOCATEWARPGROUPS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
struct AllocateWarpGroups
    : public mlir::triton::gpu::impl::TritonGPUAllocateWarpGroupsBase<
          AllocateWarpGroups> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Compute the total number of warps required at any given time.
    int baseNumWarps = lookupNumWarps(mod);
    int maxExtraWarps = 0;
    mod.walk([&](WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();
      int req = op.getTotalPartitionWarps();
      maxExtraWarps = std::max(maxExtraWarps, req);

      // Allocate the start IDs such that the largest warpgroups have lower
      // starting warp IDs.
      // FIXME: Handle aligning warp group IDs to 4 for TMEM.
      SmallVector<std::pair<unsigned, int32_t>> idxAndSize;
      for (auto [i, size] : llvm::enumerate(arr))
        idxAndSize.emplace_back(i, size);
      llvm::sort(idxAndSize,
                 [&](auto lhs, auto rhs) { return lhs.second > rhs.second; });

      SmallVector<int32_t> startIds(arr.size());
      int startId = baseNumWarps;
      for (auto [i, size] : idxAndSize) {
        startIds[i] = startId;
        startId += size;
      }
      op.setWarpGroupStartIds(startIds);
    });

    Builder b(&getContext());
    mod->setAttr("ttg.total-num-warps",
                 b.getI32IntegerAttr(baseNumWarps + maxExtraWarps));
  }
};
} // namespace
