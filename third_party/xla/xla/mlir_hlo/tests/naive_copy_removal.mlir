// RUN: mlir-hlo-opt %s --split-input-file --naive-copy-removal | FileCheck %s

func.func @target_is_alloc(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  memref.copy %arg0, %alloc_4: memref<8x8xf32> to memref<8x8xf32>
  return %arg0 : memref<8x8xf32>
}

// CHECK-LABEL: func @target_is_alloc(
// CHECK-SAME:      %[[INPUT:.*]]: memref<8x8xf32>)

// CHECK-NOT:     memref.copy
// CHECK:         return %[[INPUT]]

// -----

func.func @target_is_alloc_with_other_stores(%arg0: memref<8x8xf32>)
                                             -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  memref.copy %arg0, %alloc_4: memref<8x8xf32> to memref<8x8xf32>
  linalg.fill ins(%cst_0 : f32) outs(%alloc_4 : memref<8x8xf32>)
  memref.store %cst_0, %alloc_4[%c4, %c4] : memref<8x8xf32>
  return %arg0 : memref<8x8xf32>
}

// CHECK-LABEL: func @target_is_alloc_with_other_stores(
// CHECK-SAME:      %[[INPUT:.*]]: memref<8x8xf32>)

// CHECK:         memref.alloc
// CHECK-NOT:     memref.copy
// CHECK:         linalg.fill
// CHECK:         memref.store
// CHECK:         return %[[INPUT]]

// -----

func.func @target_is_subview(%arg0: memref<8x8xf32>) -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %subview_5 = memref.subview %alloc_4[0, 0] [%c4, %c4] [1, 1] :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %arg0, %subview_5 :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  return %arg0 : memref<8x8xf32>
}

// CHECK-LABEL: func @target_is_subview(
// CHECK-SAME:      %[[INPUT:.*]]: memref<8x8xf32>)

// CHECK-NOT:     memref.copy
// CHECK:         return %[[INPUT]]

// -----

func.func @target_is_subview_of_subview(%arg0: memref<8x8xf32>)
                                        -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %subview_5 = memref.subview %alloc_4[0, 0] [%c4, %c4] [1, 1] :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  %subview_6 = memref.subview %subview_5[0, 0] [%c4, %c4] [1, 1] :
        memref<?x?xf32, strided<[8, 1]>> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %arg0, %subview_6 :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  return %arg0 : memref<8x8xf32>
}

// CHECK-LABEL: func @target_is_subview_of_subview(
// CHECK-SAME:      %[[INPUT:.*]]: memref<8x8xf32>)

// CHECK-NOT:     memref.copy
// CHECK:         return %[[INPUT]]

// -----

func.func @do_not_simplify_subview(%arg0: memref<8x8xf32>) -> vector<8x8xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %subview_5 = memref.subview %alloc_4[0, 0] [%c4, %c4] [1, 1] :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %arg0, %subview_5 :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  %27 = vector.transfer_read %subview_5[%c0, %c0], %cst_0 :
        memref<?x?xf32, strided<[8, 1]>>, vector<8x8xf32>
  return %27 : vector<8x8xf32>
}

// CHECK-LABEL: func @do_not_simplify_subview(

// CHECK:         memref.alloc
// CHECK:         memref.subview
// CHECK:         memref.copy

// -----

func.func @do_not_simplify_alloc(%arg0: memref<8x8xf32>) -> vector<8x8xf32> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  memref.copy %arg0, %alloc_4 : memref<8x8xf32> to memref<8x8xf32>
  %27 = vector.transfer_read %alloc_4[%c0, %c0], %cst_0 :
        memref<8x8xf32>, vector<8x8xf32>
  return %27 : vector<8x8xf32>
}

// CHECK-LABEL: func @do_not_simplify_alloc(

// CHECK:         memref.alloc
// CHECK:         memref.copy

// -----

func.func @do_not_simplify_subview_with_other_use(%arg0: memref<8x8xf32>)
                                                  -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  %subview_5 = memref.subview %alloc_4[0, 0] [%c4, %c4] [1, 1] :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  %subview_6 = memref.subview %alloc_4[0, 0] [%c4, %c4] [1, 1] :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %arg0, %subview_6 :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  memref.copy %arg0, %subview_5 :
        memref<8x8xf32> to memref<?x?xf32, strided<[8, 1]>>
  return %arg0 : memref<8x8xf32>
}


// CHECK-LABEL: func @do_not_simplify_subview_with_other_use(

// CHECK:         memref.alloc
// CHECK:         memref.subview
// CHECK:         memref.subview
// CHECK:         memref.copy
// CHECK:         memref.copy

// -----

func.func @target_is_alloc_with_loads_stores(%arg0: memref<8x8xf32>)
                                             -> memref<8x8xf32> {
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  memref.copy %arg0, %alloc_4: memref<8x8xf32> to memref<8x8xf32>
  "lmhlo.custom_call"(%alloc_4, %alloc_4) ({
  }) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effect = false,
    operandSegmentSizes = array<i32: 1, 1>
  } : (memref<8x8xf32>, memref<8x8xf32>) -> ()

  return %arg0 : memref<8x8xf32>
}

// CHECK-LABEL: func @target_is_alloc_with_loads_stores(
// CHECK-SAME:      %[[INPUT:.*]]: memref<8x8xf32>)

// CHECK:         memref.alloc
// CHECK:         memref.copy
// CHECK:         "lmhlo.custom_call"
// CHECK:         return %[[INPUT]]
