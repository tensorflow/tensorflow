// RUN: kernel-gen-opt -split-input-file -kernelgen-shape-simplification %s | FileCheck %s

// Incompatible shapes. No folding.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [2] : !shape.shape
  %1 = shape.const_shape [7] : !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

// -----

// Broadcast of partially dynamic shapes yields a static shape.
// CHECK-LABEL: func @f
func @f(%arg0 : tensor<42x?x42x?xf32>, %arg1 : tensor<42x?x?xf32>) -> !shape.shape {
  // CHECK: %[[CST:.*]] = shape.const_shape [42, 42, 42, 256] : !shape.shape
  // CHECK: return %[[CST]]
  %0 = shape.const_shape [256] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<42x?x42x?xf32> -> !shape.shape
  %2 = shape.shape_of %arg1 : tensor<42x?x?xf32> -> !shape.shape
  %3 = shape.broadcast %0, %1, %2 : !shape.shape, !shape.shape, !shape.shape -> !shape.shape
  return %3 : !shape.shape
}

// -----

// Remove operands that don't contribute to the result.
// CHECK-LABEL: func @f
func @f(%arg0 : tensor<?x?x42x42xf32>, %arg1 : tensor<42x42xf32>) -> tensor<?xindex> {
  // CHECK: %[[SHAPE0:.*]] = shape.shape_of %arg0 : tensor<?x?x42x42xf32> -> tensor<?xindex>
  // CHECK: return %[[SHAPE0]]
  %0 = shape.const_shape [42, 1] : tensor<?xindex>
  %1 = shape.shape_of %arg0 : tensor<?x?x42x42xf32> -> tensor<?xindex>
  %2 = shape.shape_of %arg1 : tensor<42x42xf32> -> tensor<?xindex>
  %3 = shape.broadcast %0, %1, %2 : tensor<?xindex>, tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return %3 : tensor<?xindex>
}

// -----

// The constant shape needs to stay alive or the result will be smaller.
// CHECK-LABEL: func @f
func @f(%arg0 : tensor<?xf32>) -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [1, 1] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<?xf32> -> !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}

// -----

// [256] is the only contributor of that constant, keep it.
// CHECK-LABEL: func @f
func @f(%arg0 : tensor<?x?xf32>) -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [256] : !shape.shape
  %1 = shape.shape_of %arg0 : tensor<?x?xf32> -> !shape.shape
  %2 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}
