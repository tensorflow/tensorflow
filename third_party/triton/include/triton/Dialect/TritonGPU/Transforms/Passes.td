#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Applies software pipelining to loops in the module based on number of stages.
    This may convert some load into asynchronous loads, and multi-buffer the data.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"dumpIntermediateSteps", "dump-intermediate-steps",
           "bool", /*default*/"false",
           "Dump intermediate steps">
  ];
}

def TritonGPUTC05MMAPipeline : Pass<"tritongpu-tc05mma-pipeline", "mlir::ModuleOp"> {
  let summary = "Test pass calling TC05MMA pipeline";

  let description = [{
    This pass is used to test the TC05MMA pipelining under LIT. Internally it calls
    `getTC05MMASchedule` to get the schedule and then applies the pipelining.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"disableExpander", "disable-expander", "bool", /*default*/"false", "Run only loop pre-process">
  ];
}

def TritonGPUTestPipelineAssignLatencies : Pass<"tritongpu-test-pipeline-assign-latencies", "mlir::ModuleOp"> {
  let summary = "test assigning latencies to interesting ops ahead of pipelining";

  let description = [{
    This is a test pass that tests `assignLatencies` method of `TritonGPUPipeline`.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def TritonGPUTestPipelineScheduleLoop : Pass<"tritongpu-test-pipeline-schedule-loop", "mlir::ModuleOp"> {
  let summary = "test scheduling a loop for software pipelining";

  let description = [{
    This is a test pass that tests `scheduleLoop` method of `TritonGPUPipeline`.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonGPUTestPipelineLowerLoop : Pass<"tritongpu-test-pipeline-lower-loop", "mlir::ModuleOp"> {
  let summary = "test lowering a loop for software pipelining";

  let description = [{
    This is a test pass that tests `lowerLoop` method of `TritonGPUPipeline`.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonGPUFuseNestedLoops : Pass<"tritongpu-fuse-nested-loops", "mlir::ModuleOp"> {
  let summary = "fuse nested loops for pipelining";

  let description = [{
    The `tritongpu-fuse-nested-loops` pass will analyze loop nests in the module
    that need to be pipelined and fuse them into a single loop. This composes
    with the pipeliner to pipeline loop nests.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::arith::ArithDialect",
    "mlir::ub::UBDialect",
  ];
}

def TritonGPUAutomaticWarpSpecialization : Pass<"tritongpu-automatic-warp-specialization", "mlir::ModuleOp"> {
  let summary = "automatic warp specialization of loops";

  let description = [{
    The `tritongpu-automatic-warp-specialization` pass applies automatic
    warp specialization to eligible loops in the module. The pass will analyze
    the loops in the kernel and attempt to create a partition schedule, which
    if successful lowers the loop by duplicating it into `ttg.warp_specialize`
    partition regions.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def TritonGPURewritePartitionDependencies : Pass<"tritongpu-rewrite-partition-dependencies", "mlir::ModuleOp"> {
  let summary = "test pass for rewriting partition dependencies";

  let description = [{
    The `tritongpu-rewrite-partition-dependencies` pass analyzes the partitions
    assigned to a loop and their SSA dependencies. It rewrites the dependencies
    to be passed through shared memory, applying multi-buffering according to
    the assigned stages of the partitions.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonGPUPartitionLoops : Pass<"tritongpu-partition-loops", "mlir::ModuleOp"> {
  let summary = "split scheduled loops into `ttg.warp_specialize`";

  let description = [{
    The `tritongpu-partition-loops` pass will analyze the loops in the module
    that have been scheduled for warp specialization and split them into
    `ttg.warp_specialize` partition regions. This requires no SSA dependencies
    between any of the partitions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def TritonGPULoadMMASpecialization : Pass<"tritongpu-load-mma-specialization", "mlir::ModuleOp"> {
  let summary = "load MMA specialization";

  let description = [{
    The `tritongpu-load-mma-specialization` pass looks for matmul loops in the
    module and attempts to create a partition schedule, separating async loads
    and async MMAs into separate partitions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def TritonGPUF32DotTC : Pass<"tritongpu-F32DotTC", "mlir::ModuleOp"> {
  let summary = "3xTF32 trick";

  let description = [{
    Decompose fp32 `DotOp` instructions into 4 pointwise ops and 3 fp16 `DotOp`s
    to allow using TensorCores. See https://github.com/NVIDIA/cutlass/discussions/385
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

def TritonGPUPrefetch : Pass<"tritongpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    This pass attempts to prefetch from shared memory the operands (A and B)
    of a `tt.dot`, when this operation is located in a loop.
    Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
    that may have their operands constructed at the end of the previous
    iteration.
    Transformations are performed in five different places:
      1. The pass emits a prologue to the loop where the data for the first
         loop iteration are prefetched.
      2. The loop arguments are extended with the new prefetched values.
      3. The dotOp parameters is updated with the new args.
      4. The prefetch operations for the next iteration are added to the loop.
      5. The yieldOp is updated by adding the prefetched values for the next
         iteration.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonGPUAccelerateMatmul : Pass<"tritongpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUOptimizeDotOperands : Pass<"tritongpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "fuse transpositions";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
    hardware-accelerated transpositions.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"hoistLayoutConversion", "hoist-layout-conversion",
           "bool", /*default*/"true",
           "whether to move conver to dot operand earlier pass elementwise ops">
  ];
}

def TritonGPUCoalesce: Pass<"tritongpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    The pass analyses loads/stores with type `tensor<tt.ptr<>>` or
    `tt.ptr<tensor<>>` and replaces the layouts of these operations with
    coalesced layouts, i.e. cache friendly access patterns.
    Layout conversions are inserted before and after the load/store op
    to maintain consistency with the rest of the program.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}


def TritonGPURemoveLayoutConversions : Pass<"tritongpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
    The purpose of this pass is to rewrite the `ConvertLayoutOps` to reduce
    the number of operations and to prefer favorable layouts like
    `BlockedEncodingAttr` layout for "expensive" loads and stores
    (good for coalescing) and `NvidiaMmaEncodingAttr` otherwise
    (good for tensor ops).
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonGPUOptimizeThreadLocality : Pass<"tritongpu-optimize-thread-locality", "mlir::ModuleOp"> {
  let summary = "Reduce the cost of synchronization between threads in an SM";

  let description = [{
    The aim of this pass is to reduce cross-thread communication for certain
    operations, like reductions, reshapes, and gathers.

    For reduction operations, this pass attempts to adjust the reduction size
    (or layout) to avoid splitting the reduction operation between multiple
    threads. Currently, this pass only optimizes reduction yielded by loop to be
    thread-local until after the loop completes.

    For gathers, this pass will attempt to pick an optimized layout for gather
    operations in the module. This is determined based on the shapes of the
    gather operands as well as their existing layouts. The pass applies
    heuristics to determine when it is appropriate to assign specific layouts
    and trigger their respective codegen paths. For now, the pass only attempts
    to apply layouts that result in warp-synchronous gathers.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReorderInstructions: Pass<"tritongpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReduceDataDuplication: Pass<"tritongpu-reduce-data-duplication", "mlir::ModuleOp"> {
  let summary = "Reduce data duplication in register by decomposing convert[distributed -> dotOperand] "
                "into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and reuse #shared tensors";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCombineTensorSelectAndIf: Pass<"tritongpu-combine-tensor-select-and-if", "mlir::ModuleOp"> {
  let summary = "Combine tensor select and if";

  let description = "For select instruction that uses the same condition as the if instruction in the same block "
                    "this pass combines the select into the if instruction, making the select operands returned by the "
                    "then/else yields.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUOptimizeAccumulatorInit: Pass<"tritongpu-optimize-accumulator-init", "mlir::ModuleOp"> {
  let summary = "Replace accumulator zero-initialization with the flag indicating first use of the accumulator";

  let description = "For the dot operations that support accumulator-use flag this pass replaces the zero-initialization "
                    "of the accumulator with the flag indicating the first use of the accumulator.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCoalesceAsyncCopy: Pass<"tritongpu-coalesce-async-copy", "mlir::ModuleOp"> {
  let summary = "Improve coalescing for async global to local copies";

  let description = "For AsyncCopyGlobalToLocal ops where the shared encoding's vec is less than "
                    "the blocked encoding's sizePerThread, this pass improves coalescing by clipping the "
                    "sizePerThread value";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
