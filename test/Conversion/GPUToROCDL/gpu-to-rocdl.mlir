// RUN: mlir-opt %s -lower-gpu-ops-to-rocdl-ops | FileCheck %s

module attributes {gpu.kernel_module} {
  // CHECK-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      attributes { gpu.kernel } {
    // CHECK: rocdl.workitem.id.x : !llvm.i32
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workitem.id.y : !llvm.i32
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workitem.id.z : !llvm.i32
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.dim.x : !llvm.i32
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.y : !llvm.i32
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.z : !llvm.i32
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.id.x : !llvm.i32
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.id.y : !llvm.i32
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.id.z : !llvm.i32
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.grid.dim.x : !llvm.i32
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.grid.dim.y : !llvm.i32
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.grid.dim.z : !llvm.i32
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

    std.return
  }
}
