// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL:func @no_args(%arg0: index)
func @no_args(%sz : index) {
  // CHECK: gpu.launch blocks(%i0, %i1, %i2) in (%i6 = %arg0, %i7 = %arg0, %i8 = %arg0) threads(%i3, %i4, %i5) in (%i9 = %arg0, %i10 = %arg0, %i11 = %arg0)
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
             threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
    return
  }
  return
}

// CHECK-LABEL:func @args(%arg0: index, %arg1: index, %arg2: f32, %arg3: memref<?xf32, 1>) {
func @args(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
  // CHECK: gpu.launch blocks(%i0, %i1, %i2) in (%i6 = %arg0, %i7 = %arg0, %i8 = %arg0) threads(%i3, %i4, %i5) in (%i9 = %arg1, %i10 = %arg1, %i11 = %arg1) args(%i12 = %arg2, %i13 = %arg3) : f32, memref<?xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
             threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd)
	     args(%kernel_arg0 = %float, %kernel_arg1 = %data) : f32, memref<?xf32, 1> {
    return
  }
  return
}

// It is possible to use values passed into the region as arguments.
// CHECK-LABEL: func @passing_values
func @passing_values(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
  // CHECK: gpu.launch blocks(%i0, %i1, %i2) in (%i6 = %arg0, %i7 = %arg0, %i8 = %arg0) threads(%i3, %i4, %i5) in (%i9 = %arg1, %i10 = %arg1, %i11 = %arg1) args(%i12 = %arg2, %i13 = %arg3) : f32, memref<?xf32, 1>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
             threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd)
	     args(%kernel_arg0 = %float, %kernel_arg1 = %data) : f32, memref<?xf32, 1> {
    // CHECK: "use"(%i12)
    "use"(%kernel_arg0): (f32) -> ()
    return
  }
  return
}

// It is possible to use values defined in nested regions as long as they don't
// cross kernel launch region boundaries.
// CHECK-LABEL: func @nested_isolation
func @nested_isolation(%sz : index) {
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
             threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
    "region"() ({
      // CHECK: %0 = "produce"()
      %val = "produce"() : () -> (index)
      "region"() ({
        // CHECK: "use"(%0)
        "use"(%val) : (index) -> ()
      }) : () -> ()
    }) : () -> ()
  }
  return
}

func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>)
    attributes { gpu.kernel } {
  %tIdX = "gpu.thread_id"() {dimension: "x"} : () -> (index)
  %tIdY = "gpu.thread_id"() {dimension: "y"} : () -> (index)
  %tIdZ = "gpu.thread_id"() {dimension: "z"} : () -> (index)

  %bDimX = "gpu.block_dim"() {dimension: "x"} : () -> (index)
  %bDimY = "gpu.block_dim"() {dimension: "y"} : () -> (index)
  %bDimZ = "gpu.block_dim"() {dimension: "z"} : () -> (index)

  %bIdX = "gpu.block_id"() {dimension: "x"} : () -> (index)
  %bIdY = "gpu.block_id"() {dimension: "y"} : () -> (index)
  %bIdZ = "gpu.block_id"() {dimension: "z"} : () -> (index)

  %gDimX = "gpu.grid_dim"() {dimension: "x"} : () -> (index)
  %gDimY = "gpu.grid_dim"() {dimension: "y"} : () -> (index)
  %gDimZ = "gpu.grid_dim"() {dimension: "z"} : () -> (index)

  "some_op"(%bIdX, %tIdX) : (index, index) -> ()
  %42 = load %arg1[%bIdX] : memref<?xf32, 1>
  return
}

func @foo() {
  %0 = "op"() : () -> (f32)
  %1 = "op"() : () -> (memref<?xf32, 1>)
  // CHECK: %c8 = constant 8
  %cst = constant 8 : index

  // CHECK: "gpu.launch_func"(%c8, %c8, %c8, %c8, %c8, %c8, %0, %1) {kernel: @kernel_1 : (f32, memref<?xf32, 1>) -> ()} : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
  "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1)
      {kernel: @kernel_1 : (f32, memref<?xf32, 1>) -> ()}
      : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> ()
  return
}
