// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @null() -> memref<*xf32> {
  %null = deallocation.null : memref<*xf32>
  return %null : memref<*xf32>
}

// CHECK-LABEL: @null
// CHECK-NEXT: Results
// CHECK-NEXT: null

func.func @get_buffer() -> (index, index) {
  %null = deallocation.null : memref<*xf32>
  %a = memref.alloc() : memref<f32>
  %null_buffer = deallocation.get_buffer %null : memref<*xf32>
  %a_buffer = deallocation.get_buffer %a : memref<f32>
  return %null_buffer, %a_buffer : index, index
}

// CHECK-LABEL: @get_buffer
// CHECK-NEXT: Results
// CHECK-NEXT: i64: 0
// CHECK-NEXT: i64: {{[0-9]+$}}

func.func @retain() -> (memref<i32>, memref<i32>, memref<*xi32>) {
  %a = memref.alloc() : memref<i32>
  %b = memref.alloc() : memref<i32>

  %c1 = arith.constant 1 : i32
  memref.store %c1, %a[] : memref<i32>

  %d = deallocation.null : memref<*xi32>

  %e = memref.cast %a : memref<i32> to memref<*xi32>
  %f = deallocation.retain(%e) of (%b, %a, %d) :
    (memref<*xi32>, memref<i32>, memref<i32>, memref<*xi32>) -> (memref<*xi32>)
  return %a, %b, %f : memref<i32>, memref<i32>, memref<*xi32>
}

// CHECK-LABEL: @retain
// CHECK-NEXT: Results
// CHECK-NEXT: <i32>: 1
// CHECK-NEXT: <<deallocated>>
// CHECK-NEXT: <i32>: 1
