#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonAMDGPUStreamPipeline : Pass<"tritonamdgpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonAMDGPUStreamPipelinePass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"numStages", "num_stages",
           "int32_t", /*default*/"2",
           "Number of Pipeline stages">,
    Option<"globalPrefetch", "global_prefetch",
           "int32_t", /*default*/"0",
           "Set global prefetch stage count">,
    Option<"localPrefetch", "local_prefetch",
           "int32_t", /*default*/"0",
           "Set local prefetch stage count">,
    Option<"useAsyncCopy", "use_async_copy",
           "bool", /*default*/"false",
           "Use AsyncCopyGlobalToLocal to directly load to shared memory">,
  ];
}

def TritonAMDGPUAccelerateMatmul : Pass<"tritonamdgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., AMD matrix cores)
  }];

  let constructor = "mlir::createTritonAMDGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"kPack", "kPack",
           "int32_t", /*default*/"1",
           "KWidth / kBase">
  ];
}

def TritonAMDGPUOptimizeEpilogue : Pass<"tritonamdgpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPUOptimizeEpiloguePass()";

  let dependentDialects = [];

}

def TritonAMDGPUHoistLayoutConversions : Pass<"tritonamdgpu-hoist-layout-conversions", "mlir::triton::FuncOp"> {
  let summary = "Hoist layout conversions out of the loop";

  let description = [{
  This pass tries to hoist a convert_layout op out of the loop if 1) its dst is a tensor
  of dotOperand layout, and 2) its src is defined out of the loop.
  The rational is as follows:
  1. When the defining op of the src is out of the loop, it means the src is loop-invariant.
     Then we can potentially hoist this convert_layout op, since it's also loop-invariant.
  2. The drawback of this LICM is higher register pressure. However, on AMD GPUs, we have
     a larger register file but smaller shared memory. It's beneficial to keep loop-invariant
     variables in registers rather than loading them from shared memory in the loop.
  }];

  let constructor = "mlir::createTritonAMDGPUHoistLayoutConversionsPass()";

}

def TritonAMDGPUCanonicalizePointers : Pass<"tritonamdgpu-canonicalize-pointers", "mlir::triton::FuncOp"> {
  let summary = "Canonicalize pointers: rewrite pointers passed to load/store operation as a `<basePtr, offset>` pair.";

  let description = [{
  This pass pushes all the constant pointer arithmetic on a scalar basePtr, while all the vector
  pointer arithmetic to a vector offset. I.e., if we consider the following IR:
  ```
    %v_ptr = tt.splat %s_ptr
    %c_offset = tt.splat %s_offset
    %v_offset0 = tt.make_range
    %v_offset1 = tt.make_range
    %v_ptr0 = tt.addptr %v_ptr, %c_offset
    %v_ptr1 = tt.addptr %v_ptr0, %v_offset0
    %v_ptr2 = tt.addptr %v_ptr0, %v_offset1
    %data = tt.load(%v_ptr2)
  ```
  We transform this into:
  ```
    %s_ptr0 = tt.addptr %s_ptr, %s_offset
    %v_offset = %zero
    %v_offset = arith.addi %v_offset, %v_offset0
    %v_offset = arith.addi %v_offset, %v_offset1
    %c_ptr = tt.splat %s_ptr0
    %v_ptr = tt.addptr %c_ptr, %v_offset
    %data = tt.load(%v_ptr)
  ```
  In the above IR:
  -  `v_` means "variable vector across the program"
  -  `c_` means "constant vector across the program"
  -  `s_` means "scalar"
  So we transform the IR such that the constant updates become scalar updates, and the variable updates happen on the offset. Note that
  when we have to load the data, we splat the scalar pointer, add the "variable" offset and then issue the load.
  }];

  let constructor = "mlir::createTritonAMDGPUCanonicalizePointersPass()";

  let dependentDialects = [];

}

def TritonAMDGPUReorderInstructions: Pass<"tritonamdgpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonAMDGPUReorderInstructionsPass()";

  let dependentDialects = [];
}

def TritonAMDGPUConvertToBufferOps : Pass<"tritonamdgpu-convert-buffer-ops", "mlir::ModuleOp"> {
  let summary = "Convert memory operations to buffer operations";

  let description = "This pass converts memory and atomic operations (e.g., tt.load/tt.store/tt.atomic_rmw) to  amdgpu buffer operations, if possible";

  let constructor = "mlir::createTritonAMDGPUConvertToBufferOpsPass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
  ];
}

def TritonAMDGPUBlockPingpong: Pass<"tritonamdgpu-block-pingpong", "mlir::ModuleOp"> {
  let summary = "Interleaving instructions from two warps on the same SIMD to better utilize matrix core";

  let description = [{
    This pass reorder instructions to interleave instructions from two warps on the same SIMD unit.
    We call this a ping-pong scheduling pattern, where two warps run concurrently in the synchronized fashion
    This block ping-pong pattern could be beneficial under few conditions including
    occupancy and number of warps.
  }];

  let constructor = "mlir::createTritonAMDGPUBlockPingpongPass()";

  let dependentDialects = ["mlir::ROCDL::ROCDLDialect, mlir::triton::amdgpu::TritonAMDGPUDialect"];

  let options = [
    Option<"numStages", "num-stages",
        "int32_t", /*default*/"2",
        "Number of Pipeline stages">,
    ];
}

def TritonAMDGPUInThreadTranspose: Pass<"tritonamdgpu-in-thread-transpose", "mlir::triton::FuncOp"> {
  let summary = "Extend global load sizePerThread to 2D shape and perform transpose within registers per thread before writing to shared memory";

  let description = [{
    Pass looks for inefficient load->local_store->local_load chains.
    In particular, this pass optimizes dot operand loading from shared memory
    in cases when operand is stored in global memory in non-K-continous way.

    ```
      #blocked = #ttg.blocked<{sizePerThread = [1, 8], ..., order = [1, 0]}>
      #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
      #mma = #ttg.amd_mfma<{...}>

      // pass consider global loads are coalesced at this point
      %loaded_data = tt.load ... : tensor<#blocked>
      %local_data = ttg.local_alloc %loaded_data : (tensor<#blocked>) -> !ttg.memdesc<#shared>
      // following local_load is not vectorized because of different mma dot register order and memory order of shared layout
      %dot_operand = ttg.local_load %local_data : !ttg.memdesc<#shared> -> tensor<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    ```

    transforms it into code with vectorized local_loads and local_store with specialized shared layout to minimize bank conflicts:

    ```
      #blocked = #ttg.blocked<{sizePerThread = [1, 8], ..., order = [1, 0]}>
      #transposable_layout = #ttg.blocked<{sizePerThread = [4, 8], ..., order = [1, 0]}>
      // layout identical to #transposable_layout, but with transposed register values
      // transposition makes it possible to do vectorized shared memory stores
      #linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4] ... }>
      // shared layout with order compatible with mma layout, so shared loads are vectorized
      #shared = #ttg.amd_rotating_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>

      %loaded_data = tt.load ... : tensor<#transposable_layout>
      %tmp1 = ttg.convert_layout %loaded_data : tensor<#transposable_layout> -> tensor<#blocked>
      %tmp2 = ttg.convert_layout %tmp1 : tensor<#blocked> -> tensor<#transposable_layout>
      %transposed = amdgpu.in_thread_transpose %tmp2 : tensor<#transposable_layout> -> tensor<#linear>
      %local_data = ttg.local_alloc %transposed : tensor<#linear> -> !ttg.memdesc<#shared>
      %dot_operand = ttg.local_load %local_data : !ttg.memdesc<#shared> -> tensor<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    ```

    After transformation tt.load stays coalesced, because optimization do not change anything across fastest dimension.
    local_alloc is vectorized and uses swizzled memory, number of bank conflics reduced
    local_load is vectorized, because shared memory order matches destination layout register order.

    This pass introduces two ttg.convert_layouts to properly cover cases when between ttg.load and ttg.local_alloc/ttg.local_store
    exist more operations like scf or ttg.memdesc_subview. These convert_layouts ops are optimized out by later passes.
  }];

  let constructor = "mlir::createTritonAMDGPUInThreadTransposePass()";

  let dependentDialects = ["mlir::triton::amdgpu::TritonAMDGPUDialect", "mlir::triton::gpu::TritonGPUDialect"];
}

def TritonAMDGPUCoalesceAsyncCopy: Pass<"tritonamdgpu-coalesce-async-copy", "mlir::ModuleOp"> {
  let summary = "Improve coalescing for async global to local copies";

  let description = [{
    GFX9:
      For AsyncCopyGlobalToLocal ops where the blocked encoding's sizePerThread is larger than the contiguity of the
      source or the supported load vector size we clip it to the largest supported size. This ensures we get coalesced writes to
      shared memory as required by the hardware. Does only work for non swizzled shared memory layouts
  }];

  let constructor = "mlir::createTritonAMDGPUCoalesceAsyncCopyPass()";

  let dependentDialects = [];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
  ];
}


#endif
