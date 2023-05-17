// RUN: (! mlir-interpreter-runner %s -run-all 2>&1) | FileCheck %s

func.func @out_of_bounds_load() -> i32 {
  %c3 = arith.constant 3 : index
  %cst = arith.constant dense<[1, 2]> : memref<2xi32>
  %ret = memref.load %cst[%c3] : memref<2xi32>
  return %ret : i32
}

// CHECK-LABEL: @out_of_bounds_load
// CHECK-NEXT: array index out of bounds

func.func @out_of_bounds_store() {
  %c3 = arith.constant 3 : index
  %v = arith.constant 32 : i32
  %cst = arith.constant dense<[1, 2]> : memref<2xi32>
  memref.store %v, %cst[%c3] : memref<2xi32>
  return
}

// CHECK-LABEL: @out_of_bounds_store
// CHECK-NEXT: array index out of bounds

func.func @out_of_bounds_subview() -> memref<?xi32, strided<[?], offset: ?>> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %v = arith.constant 32 : i32
  %cst = arith.constant dense<[1, 2]> : memref<2xi32>
  %subview = memref.subview %cst[%c1][%c2][%c1]
    : memref<2xi32> to memref<?xi32, strided<[?], offset: ?>>
  return %subview : memref<?xi32, strided<[?], offset: ?>>
}

// CHECK-LABEL: @out_of_bounds_subview
// CHECK-NEXT: subview out of bounds

func.func @collapse_shape_no_common_stride()
    -> (memref<1x2x3xi32>, memref<1x6xi32>) {
  %a = arith.constant dense<[[[[0, 1, 2], [3, 4, 5]],
                              [[6, 7, 8], [9, 10,11]]]]>
    : memref<1x2x2x3xi32>
  %b = memref.subview %a[0, 0, 0, 0][1, 2, 1, 3][1, 1, 1, 1]
    : memref<1x2x2x3xi32> to memref<1x2x3xi32>
  %c = memref.collapse_shape %b [[0], [1, 2]]
    : memref<1x2x3xi32> into memref<1x6xi32>
  return %b, %c : memref<1x2x3xi32>, memref<1x6xi32>
}

// CHECK-LABEL: @collapse_shape_no_common_stride
// CHECK-NEXT: cannot collapse dimensions without a common stride

func.func @double_free() {
  %a = memref.alloc() : memref<i32>
  memref.dealloc %a : memref<i32>
  memref.dealloc %a : memref<i32>
  return
}

// CHECK-LABEL: @double_free
// CHECK-NEXT: Interpreter failure: double-free
// CHECK-NEXT: Note: allocated by %alloc = memref.alloc() : memref<i32>
// CHECK-NEXT: Note: previously freed by memref.dealloc %alloc : memref<i32>
// CHECK-NEXT{2}: Encountered failure while executing memref.dealloc %alloc : memref<i32>

func.func @use_after_free() {
  %a = memref.alloc() : memref<i32>
  memref.dealloc %a : memref<i32>
  %b = arith.constant 1 : i32
  memref.store %b, %a[] : memref<i32>
  return
}

// CHECK-LABEL: @use_after_free
// CHECK-NEXT: Interpreter failure: use-after-free
// CHECK-NEXT: Note: allocated by %alloc = memref.alloc() : memref<i32>
// CHECK-NEXT: Note: previously freed by memref.dealloc %alloc : memref<i32>
// CHECK-NEXT{2}: Encountered failure while executing memref.store %c1_i32, %alloc[] : memref<i32>
