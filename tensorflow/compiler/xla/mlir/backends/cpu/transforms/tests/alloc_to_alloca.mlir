// RUN: xla-cpu-opt %s -split-input-file -xla-alloc-to-alloca \
// RUN:   -allow-unregistered-dialect \
// RUN:   | FileCheck %s

func.func @convert_alloc() {
  %alloc = memref.alloc() { yolo = 42 } : memref<2x2x2xf32>
  "test.dummy"(%alloc) : (memref<2x2x2xf32>) -> ()
  return
}

// CHECK-LABEL: func @convert_alloc
// CHECK:         memref.alloca() {yolo = 42 : i64} : memref<2x2x2xf32>

// -----

func.func @do_not_convert_large_alloc() {
  %alloc = memref.alloc() : memref<8x8x8xf32>
  "test.dummy"(%alloc) : (memref<8x8x8xf32>) -> ()
  return
}

// CHECK-LABEL: func @do_not_convert_large_alloc
// CHECK-NOT:     memref.alloca
// CHECK:         memref.alloc

// -----

func.func @do_not_convert_alloc_in_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %alloc = memref.alloc() : memref<1xf32>
    "test.dummy"(%alloc) : (memref<1xf32>) -> ()
  }
  return
}

// CHECK-LABEL: func @do_not_convert_alloc_in_loop
// CHECK-NOT:     memref.alloca
// CHECK:         memref.alloc

// -----

func.func @do_not_convert_escaping_alloc() -> memref<1xf32> {
  %alloc = memref.alloc() { yolo = 42 } : memref<f32>
  %something:2 = "test.dummy"(%alloc) : (memref<f32>) -> (memref<1xf32>, f32)
  return %something#0 : memref<1xf32>
}

// CHECK-LABEL: func @do_not_convert_escaping_alloc
// CHECK-NOT:     memref.alloca
// CHECK:         memref.alloc