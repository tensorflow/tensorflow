#ifndef TRITONCOMMONGPU_CONVERSION_PASSES
#define TRITONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"allocate-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

def TritonGPUGlobalScratchAllocationPass : Pass<"tritongpu-global-scratch-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign global scratch memory allocation";

  let description = [{
    Decide on global scratch space memory allocation and assign attributes to each allocation.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def TritonGPUAllocateWarpGroups : Pass<"tritongpu-allocate-warp-groups", "mlir::ModuleOp"> {
  let summary = "Allocate warp groups";

  let description = [{
    The `tritongpu-allocate-warp-groups` pass performs warpgroup allocation for
    a GPU program. When a GPU program contains warp specialization, additional
    warps are launched in addition to the "default" warp group. The "default"
    warpgroup executes top-level code in a `tt.func` and its size is specified
    by the user via the `num_warps` argument.

    This pass analyzes `ttg.warp_specialize` ops in the program and determines
    the total number of needed warps, then attaches the range of warp IDs to
    each warpgroup function.
  }];
}

#endif
