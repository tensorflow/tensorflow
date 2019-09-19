// RUN: mlir-opt -gpu-kernel-outlining -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @launch()
func @launch() {
  // CHECK: %[[ARG0:.*]] = "op"() : () -> f32
  %0 = "op"() : () -> (f32)
  // CHECK: %[[ARG1:.*]] = "op"() : () -> memref<?xf32, 1>
  %1 = "op"() : () -> (memref<?xf32, 1>)
  // CHECK: %[[GDIMX:.*]] = constant 8
  %gDimX = constant 8 : index
  // CHECK: %[[GDIMY:.*]] = constant 12
  %gDimY = constant 12 : index
  // CHECK: %[[GDIMZ:.*]] = constant 16
  %gDimZ = constant 16 : index
  // CHECK: %[[BDIMX:.*]] = constant 20
  %bDimX = constant 20 : index
  // CHECK: %[[BDIMY:.*]] = constant 24
  %bDimY = constant 24 : index
  // CHECK: %[[BDIMZ:.*]] = constant 28
  %bDimZ = constant 28 : index

  // CHECK: "gpu.launch_func"(%[[GDIMX]], %[[GDIMY]], %[[GDIMZ]], %[[BDIMX]], %[[BDIMY]], %[[BDIMZ]], %[[ARG0]], %[[ARG1]]) {kernel = @launch_kernel} : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
  // CHECK-NOT: gpu.launch blocks
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDimX, %grid_y = %gDimY,
                                       %grid_z = %gDimZ)
             threads(%tx, %ty, %tz) in (%block_x = %bDimX, %block_y = %bDimY,
                                        %block_z = %bDimZ)
             args(%arg0 = %0, %arg1 = %1) : f32, memref<?xf32, 1> {
    "use"(%arg0): (f32) -> ()
    "some_op"(%bx, %block_x) : (index, index) -> ()
    %42 = load %arg1[%tx] : memref<?xf32, 1>
    gpu.return
  }
  return
}

// CHECK-LABEL: func @launch_kernel
// CHECK-SAME: (f32, memref<?xf32, 1>)
// CHECK-NEXT: attributes {gpu.kernel}

// CHECK-LABEL: func @launch_kernel
// CHECK-SAME: (%[[KERNEL_ARG0:.*]]: f32, %[[KERNEL_ARG1:.*]]: memref<?xf32, 1>)
// CHECK-NEXT: attributes {gpu.kernel}
// CHECK-NEXT: %[[BID:.*]] = "gpu.block_id"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.block_id"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.block_id"() {dimension = "z"} : () -> index
// CHECK-NEXT: %[[TID:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.thread_id"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.thread_id"() {dimension = "z"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.grid_dim"() {dimension = "z"} : () -> index
// CHECK-NEXT: %[[BDIM:.*]] = "gpu.block_dim"() {dimension = "x"} : () -> index
// CHECK-NEXT: = "gpu.block_dim"() {dimension = "y"} : () -> index
// CHECK-NEXT: = "gpu.block_dim"() {dimension = "z"} : () -> index
// CHECK-NEXT: "use"(%[[KERNEL_ARG0]]) : (f32) -> ()
// CHECK-NEXT: "some_op"(%[[BID]], %[[BDIM]]) : (index, index) -> ()
// CHECK-NEXT: = load %[[KERNEL_ARG1]][%[[TID]]] : memref<?xf32, 1>

// -----

func @multiple_launches() {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  // CHECK: "gpu.launch_func"(%[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]]) {kernel = @multiple_launches_kernel} : (index, index, index, index, index, index) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    gpu.return
  }
  // CHECK: "gpu.launch_func"(%[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]]) {kernel = @multiple_launches_kernel_0} : (index, index, index, index, index, index) -> ()
  gpu.launch blocks(%bx2, %by2, %bz2) in (%grid_x2 = %cst, %grid_y2 = %cst,
                                          %grid_z2 = %cst)
             threads(%tx2, %ty2, %tz2) in (%block_x2 = %cst, %block_y2 = %cst,
                                           %block_z2 = %cst) {
    gpu.return
  }
  return
}

// CHECK: func @multiple_launches_kernel()
// CHECK: func @multiple_launches_kernel_0()

// -----

func @extra_constants(%arg0 : memref<?xf32>) {
  // CHECK: %[[CST:.*]] = constant 8 : index
  %cst = constant 8 : index
  %cst2 = constant 2 : index
  %cst3 = constant 3 : index
  // CHECK: "gpu.launch_func"(%[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]], %[[CST]], %{{.*}}) {kernel = @extra_constants_kernel} : (index, index, index, index, index, index, memref<?xf32>) -> ()
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst)
             args(%kernel_arg0 = %cst2, %kernel_arg1 = %arg0, %kernel_arg2 = %cst3) : index, memref<?xf32>, index {
    "use"(%kernel_arg0, %kernel_arg1, %kernel_arg2) : (index, memref<?xf32>, index) -> ()
    gpu.return
  }
  return
}

// CHECK-LABEL: func @extra_constants_kernel(%{{.*}}: memref<?xf32>)
// CHECK: constant
// CHECK: constant

// -----

func @function_call(%arg0 : memref<?xf32>) {
  %cst = constant 8 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst,
                                       %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst,
                                        %block_z = %cst) {
    // TODO(b/141098412): Support function calls.
    // expected-error @+1 {{'device_function' does not reference a valid function}}
    call @device_function() : () -> ()
    gpu.return
  }
  return
}

func @device_function() {
  gpu.return
}
