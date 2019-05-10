// RUN: mlir-opt -gpu-kernel-outlining -split-input-file %s | FileCheck %s

func @launch() {
  %0 = "op"() : () -> (f32)
  %1 = "op"() : () -> (memref<?xf32, 1>)
  %gDimX = constant 8 : index
  %gDimY = constant 12 : index
  %gDimZ = constant 16 : index
  %bDimX = constant 20 : index
  %bDimY = constant 24 : index
  %bDimZ = constant 28 : index

  // CHECK: "gpu.launch_func"(%c8, %c12, %c16, %c20, %c24, %c28, %0, %1) {kernel: @launch_kernel : (f32, memref<?xf32, 1>) -> ()} : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
  // CHECK-NOT: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY,
                                       %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY,
                                        %block_z = %bDimZ)
             args(%arg0 = %0, %arg1 = %1) : f32, memref<?xf32, 1> {
    "use"(%arg0): (f32) -> ()
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %42 = load %arg1[%tx] : memref<?xf32, 1>
    return
  }
  return
}

// CHECK: func @launch_kernel(%arg0: f32, %arg1: memref<?xf32, 1>)
// CHECK-NEXT: attributes {gpu.kernel}
// CHECK-NEXT: %0 = "gpu.block_id"() {dimension: "x"} : () -> index
// CHECK-NEXT: %1 = "gpu.block_id"() {dimension: "y"} : () -> index
// CHECK-NEXT: %2 = "gpu.block_id"() {dimension: "z"} : () -> index
// CHECK-NEXT: %3 = "gpu.thread_id"() {dimension: "x"} : () -> index
// CHECK-NEXT: %4 = "gpu.thread_id"() {dimension: "y"} : () -> index
// CHECK-NEXT: %5 = "gpu.thread_id"() {dimension: "z"} : () -> index
// CHECK-NEXT: %6 = "gpu.grid_dim"() {dimension: "x"} : () -> index
// CHECK-NEXT: %7 = "gpu.grid_dim"() {dimension: "y"} : () -> index
// CHECK-NEXT: %8 = "gpu.grid_dim"() {dimension: "z"} : () -> index
// CHECK-NEXT: %9 = "gpu.block_dim"() {dimension: "x"} : () -> index
// CHECK-NEXT: %10 = "gpu.block_dim"() {dimension: "y"} : () -> index
// CHECK-NEXT: %11 = "gpu.block_dim"() {dimension: "z"} : () -> index
// CHECK-NEXT: "use"(%arg0) : (f32) -> ()
// CHECK-NEXT: "some_op"(%0, %9) : (index, index) -> ()
// CHECK-NEXT: %12 = load %arg1[%3] : memref<?xf32, 1>

// -----

func @multiple_launches() {
  %cst = constant 8 : index
  // CHECK: "gpu.launch_func"(%c8, %c8, %c8, %c8, %c8, %c8) {kernel: @multiple_launches_kernel : () -> ()} : (index, index, index, index, index, index) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    return
  }
  // CHECK: "gpu.launch_func"(%c8, %c8, %c8, %c8, %c8, %c8) {kernel: @multiple_launches_kernel_0 : () -> ()} : (index, index, index, index, index, index) -> ()
  gpu.launch blocks(%bx2, %by2, %bz2) in (%grid_x2 = %cst, %grid_y2 = %cst,
                                          %grid_z2 = %cst)
             threads(%tx2, %ty2, %tz2) in (%block_x2 = %cst, %block_y2 = %cst,
                                           %block_z2 = %cst) {
    return
  }
  return
}

// CHECK: func @multiple_launches_kernel()
// CHECK: func @multiple_launches_kernel_0()
