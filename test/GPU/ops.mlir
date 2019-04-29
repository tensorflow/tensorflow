// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL:func @no_args(%arg0: index)
func @no_args(%sz : index) {
// CHECK:  "gpu.launch"(%arg0, %arg0, %arg0, %arg0, %arg0, %arg0) : (index, index, index, index, index, index) -> () {
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz)
      : (index, index, index, index, index, index) -> () {
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index):
    return
  }
  return
}

// CHECK-LABEL:func @args(%arg0: index, %arg1: index, %arg2: f32, %arg3: memref<?xf32, 1>) {
func @args(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
// CHECK:  "gpu.launch"(%arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg2, %arg3) : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> () {
  "gpu.launch"(%blk, %blk, %blk, %thrd, %thrd, %thrd, %float, %data)
      : (index, index, index, index, index, index, f32, memref<?xf32,1>) -> () {
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index,
       %data0: f32, %data1: memref<?xf32,1>):
    return
  }
  return
}

// It is possible to use values passed into the region as arguments.
// CHECK-LABEL: func @passing_values
func @passing_values(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
// CHECK:  "gpu.launch"(%arg0, %arg0, %arg0, %arg1, %arg1, %arg1, %arg2, %arg3) : (index, index, index, index, index, index, f32, memref<?xf32, 1>) -> () {
  "gpu.launch"(%blk, %blk, %blk, %thrd, %thrd, %thrd, %float, %data)
      : (index, index, index, index, index, index, f32, memref<?xf32,1>) -> () {
// CHECK: ^bb1(%i0: index, %i1: index, %i2: index, %i3: index, %i4: index, %i5: index, %i6: index, %i7: index, %i8: index, %i9: index, %i10: index, %i11: index, %i12: f32, %i13: memref<?xf32, 1>)
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index,
       %data0: f32, %data1: memref<?xf32,1>):
// CHECK: "use"(%i12)
    "use"(%data0): (f32) -> ()
    return
  }
  return
}

// It is possible to use values defined in nested regions as long as they don't
// cross kernel launch region boundaries.
// CHECK-LABEL: func @nested_isolation
func @nested_isolation(%sz : index) {
  "gpu.launch"(%sz, %sz, %sz, %sz, %sz, %sz)
      : (index, index, index, index, index, index) -> () {
  ^bb1(%bx: index, %by: index, %bz: index,
       %tx: index, %ty: index, %tz: index,
       %szbx: index, %szby: index, %szbz: index,
       %sztx: index, %szty: index, %sztz: index):
    "region"() : () -> () {
// CHECK: %0 = "produce"()
      %val = "produce"() : () -> (index)
      "region"() : () -> () {
// CHECK: "use"(%0)
        "use"(%val) : (index) -> ()
      }
    }
  }
  return
}
