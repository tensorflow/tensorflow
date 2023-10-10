// RUN: mlir-hlo-opt -split-input-file -shape-simplification %s | FileCheck %s

// Incompatible shapes. No folding.
// CHECK-LABEL: func @f
func.func @f() -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [2] : !shape.shape
  %1 = shape.const_shape [7] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  func.return %2 : !shape.shape
}

// -----

// Broadcast of partially dynamic shapes yields a static shape.
// CHECK-LABEL: func @f
func.func @f(%arg0 : tensor<42x?x42x?xf32>, %arg1 : tensor<42x?x?xf32>) -> !shape.shape {
  // CHECK: %[[CST:.*]] = shape.const_shape [42, 42, 42, 256] : !shape.shape
  // CHECK: return %[[CST]]
  %0 = shape.const_shape [256] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<42x?x42x?xf32> -> !shape.shape
  %2 = shape.shape_of %arg1 : tensor<42x?x?xf32> -> !shape.shape
  %3 = shape.broadcast %0, %1, %2 : !shape.shape, !shape.shape, !shape.shape -> !shape.shape
  func.return %3 : !shape.shape
}

// -----

// Remove operands that don't contribute to the result.
// CHECK-LABEL: func @f
func.func @f(%arg0 : tensor<?x?x42x42xf32>, %arg1 : tensor<42x42xf32>) -> tensor<?xindex> {
  // CHECK: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?x?x42x42xf32> -> tensor<?xindex>
  // CHECK: return %[[SHAPE0]]
  %0 = shape.const_shape [42, 1] : tensor<2xindex>
  %1 = shape.shape_of %arg0 : tensor<?x?x42x42xf32> -> tensor<?xindex>
  %2 = shape.shape_of %arg1 : tensor<42x42xf32> -> tensor<2xindex>
  %3 = shape.broadcast %0, %1, %2 : tensor<2xindex>, tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
  func.return %3 : tensor<?xindex>
}

// -----

// The constant shape needs to stay alive or the result will be smaller.
// CHECK-LABEL: func @f
func.func @f(%arg0 : tensor<?xf32>) -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [1, 1] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  func.return %2 : !shape.shape
}

// -----

// [256] is the only contributor of that constant, keep it.
// CHECK-LABEL: func @f
func.func @f(%arg0 : tensor<?x?xf32>) -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [256] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<?x?xf32> -> !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  func.return %2 : !shape.shape
}

// -----

// CHECK-LABEL: func @static_non1_succeeds
// CHECK-NEXT:   %[[C2:.*]] = arith.constant 2
// CHECK-NEXT:   return %[[C2]]
func.func @static_non1_succeeds(%arg0 : tensor<?x?xf64>, %arg1 : tensor<?x1xf64>,
                           %arg2: tensor<?x2xf64>) -> index {
 %c1 = arith.constant 1 : index
 %1 = shape.shape_of %arg0 : tensor<?x?xf64> -> tensor<2xindex>
 %2 = shape.shape_of %arg1 : tensor<?x1xf64> -> tensor<2xindex>
 %3 = shape.shape_of %arg2 : tensor<?x2xf64> -> tensor<2xindex>
 %4 = shape.broadcast %1, %2, %3 : tensor<2xindex>, tensor<2xindex>,
                                   tensor<2xindex> -> tensor<2xindex>
 %result = tensor.extract %4[%c1] : tensor<2xindex>
 func.return %result : index
}

// -----

// CHECK-LABEL: func @all_static_1s_succeeds
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1
// CHECK-NEXT:   return %[[C1]]
func.func @all_static_1s_succeeds(%arg0 : tensor<?x1xf64>, %arg1 : tensor<?x1xf64>)
                            -> index {
 %c1 = arith.constant 1 : index
 %1 = shape.shape_of %arg0 : tensor<?x1xf64> -> tensor<2xindex>
 %2 = shape.shape_of %arg1 : tensor<?x1xf64> -> tensor<2xindex>
 %3 = shape.broadcast %1, %2 : tensor<2xindex>, tensor<2xindex>
                               -> tensor<2xindex>
 %result = tensor.extract %3[%c1] : tensor<2xindex>
 func.return %result : index
}

// -----

// CHECK-LABEL: func @single_non_static_1_succeeds
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?x?xf64>
// CHECK:        %[[C1:.*]] = arith.constant 1
// CHECK:        %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK:        return %[[DIM]]
func.func @single_non_static_1_succeeds(%arg0 : tensor<?x?xf64>,
                                   %arg1 : tensor<?x1xf64>) -> index {
 %c0 = arith.constant 1 : index
 %1 = shape.shape_of %arg0 : tensor<?x?xf64> -> tensor<2xindex>
 %2 = shape.shape_of %arg1 : tensor<?x1xf64> -> tensor<2xindex>
 %3 = shape.broadcast %1, %2 : tensor<2xindex>, tensor<2xindex>
                               -> tensor<2xindex>
 %result = tensor.extract %3[%c0] : tensor<2xindex>
 func.return %result : index
}

// -----

// CHECK-LABEL: func @multiple_non_static_1_fails
// CHECK-NEXT:   constant 0
// CHECK-NEXT:   shape.shape_of
// CHECK-NEXT:   shape.shape_of
// CHECK-NEXT:   shape.broadcast
// CHECK-NEXT:   %[[RESULT:.*]] = tensor.extract
// CHECK-NEXT:   return %[[RESULT]]
func.func @multiple_non_static_1_fails(%arg0 : tensor<?x?xf64>,
                                  %arg1 : tensor<?x1xf64>) -> index {
 %c0 = arith.constant 0 : index
 %1 = shape.shape_of %arg0 : tensor<?x?xf64> -> tensor<2xindex>
 %2 = shape.shape_of %arg1 : tensor<?x1xf64> -> tensor<2xindex>
 %3 = shape.broadcast %1, %2 : tensor<2xindex>, tensor<2xindex>
                               -> tensor<2xindex>
 %result = tensor.extract %3[%c0] : tensor<2xindex>
 func.return %result : index
}

// -----

// CHECK-LABEL: func @extract_no_crash
// CHECK-NEXT:   tensor.extract
func.func @extract_no_crash(%arg0 : tensor<index>) -> index {
 %result = tensor.extract %arg0[] : tensor<index>
 func.return %result : index
}
