// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @null() -> !deallocation.ownership {
  %null = deallocation.null
  return %null : !deallocation.ownership
}

// CHECK-LABEL: @null
// CHECK-NEXT: Results
// CHECK-NEXT: null

func.func @get_buffer() -> (index, index, index) {
  %null = deallocation.null
  %a = memref.alloc() : memref<f32>
  %a_owned = deallocation.own %a : memref<f32>
  %null_buffer = deallocation.get_buffer %null : !deallocation.ownership
  %a_buffer = deallocation.get_buffer %a : memref<f32>
  %a_owned_buffer = deallocation.get_buffer %a_owned : !deallocation.ownership
  return %null_buffer, %a_buffer, %a_owned_buffer : index, index, index
}

// CHECK-LABEL: @get_buffer
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 0
// CHECK-NEXT: i64: [[ADDR:[0-9]+$]]
// CHECK-NEXT: i64: [[ADDR]]

func.func @retain() -> (memref<1xf32>, memref<f32>, !deallocation.ownership) {
  %a = memref.alloc() : memref<1xf32>
  %a_owned = deallocation.own %a : memref<1xf32>
  %b = memref.alloc() : memref<f32>
  %b_owned = deallocation.own %b : memref<f32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1.0 : f32
  memref.store %c1, %a[%c0] : memref<1xf32>

  %d = deallocation.null

  %e = memref.cast %a : memref<1xf32> to memref<?xf32>
  %f = deallocation.retain(%e) of (%b_owned, %a_owned, %d) :
    (memref<?xf32>, !deallocation.ownership, !deallocation.ownership, !deallocation.ownership)
    -> (!deallocation.ownership)
  return %a, %b, %f : memref<1xf32>, memref<f32>, !deallocation.ownership
}

// CHECK-LABEL: @retain
// CHECK-NEXT: Results
// CHECK-NEXT: <1xf32>: [1.000000e+00]
// CHECK-NEXT: <<deallocated>>
// CHECK-NEXT: <1xf32>: [1.000000e+00]
