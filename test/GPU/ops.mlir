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
