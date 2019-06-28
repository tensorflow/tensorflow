// RUN: tf-opt -tfl-lower-static-tensor-list %s | FileCheck %s

func @tensorlistGetItem(tensor<3x10xf32>, tensor<1xi32>, tensor<i32>) -> (tensor<10xf32>, tensor<3x10xf32>) {
^bb0(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>):
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<*x!tf.variant>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<*x!tf.variant>, tensor<i32>, tensor<1xi32>) -> tensor<10xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<*x!tf.variant>, tensor<1xi32>) -> tensor<3x10xf32>
  return %1, %2 : tensor<10xf32>, tensor<3x10xf32>

// CHECK-LABEL: tensorlistGetItem
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<3x10xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK: return %0, %arg0 : tensor<10xf32>, tensor<3x10xf32>
}

func @tensorlistGetItemWithUnknownRank(tensor<*xf32>, tensor<1xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>):
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<*xf32>, tensor<1xi32>) -> tensor<*x!tf.variant>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<*x!tf.variant>, tensor<i32>, tensor<1xi32>) -> tensor<*xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<*x!tf.variant>, tensor<1xi32>) -> tensor<*xf32>
  return %1, %2 : tensor<*xf32>, tensor<*xf32>

// CHECK-LABEL: tensorlistGetItemWithUnknownRank
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %0, %arg0 : tensor<*xf32>, tensor<*xf32>
}

func @tensorlistSetItem(tensor<3x10xf32>, tensor<1xi32>, tensor<i32>, tensor<10xf32>) -> tensor<3x10xf32> {
^bb0(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>):
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<*x!tf.variant>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<*x!tf.variant>, tensor<i32>, tensor<10xf32>) -> tensor<*x!tf.variant>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<*x!tf.variant>, tensor<1xi32>) -> tensor<3x10xf32>
  return %2 : tensor<3x10xf32>

// CHECK-LABEL: tensorlistSetItem
// CHECK:  %cst = constant dense<1> : tensor<1xi32>
// CHECK:  %cst_0 = constant dense<0> : tensor<i32>
// CHECK:  %cst_1 = constant dense<-1> : tensor<i32>
// CHECK:  %0 = "tf.Rank"(%arg0) : (tensor<3x10xf32>) -> tensor<i32>
// CHECK:  %1 = "tf.Rank"(%arg3) : (tensor<10xf32>) -> tensor<i32>
// CHECK:  %2 = "tf.ExpandDims"(%0, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %3 = "tf.Fill"(%2, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %4 = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:  %5 = "tf.ExpandDims"(%1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %6 = "tf.Fill"(%5, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %7 = "tf.Concat"(%cst_0, %4, %6) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %8 = "tf.ExpandDims"(%arg2, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %9 = "tf.Fill"(%5, %cst_1) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %10 = "tf.Concat"(%cst_0, %8, %9) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %11 = "tf.Fill"(%2, %cst_1) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %12 = "tf.Slice"(%arg0, %3, %10) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %13 = "tf.Slice"(%arg0, %7, %11) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %14 = "tf.ExpandDims"(%arg3, %cst_0) : (tensor<10xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  %15 = "tf.Concat"(%cst_0, %12, %14, %13) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<3x10xf32>
// CHECK:  return %15 : tensor<3x10xf32>
}

func @tensorlistSetItemWithScalarElements(tensor<5xf32>, tensor<0xi32>, tensor<i32>, tensor<f32>) -> tensor<5xf32> {
^bb0(%arg0: tensor<5xf32>, %arg1: tensor<0xi32>, %arg2: tensor<i32>, %arg3: tensor<f32>):
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<5xf32>, tensor<0xi32>) -> tensor<*x!tf.variant>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<*x!tf.variant>, tensor<i32>, tensor<f32>) -> tensor<*x!tf.variant>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<*x!tf.variant>, tensor<0xi32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>

// CHECK-LABEL: tensorlistSetItemWithScalarElements
// CHECK:  %cst = constant dense<1> : tensor<1xi32>
// CHECK:  %cst_0 = constant dense<0> : tensor<i32>
// CHECK:  %cst_1 = constant dense<-1> : tensor<i32>
// CHECK:  %0 = "tf.Rank"(%arg0) : (tensor<5xf32>) -> tensor<i32>
// CHECK:  %1 = "tf.Rank"(%arg3) : (tensor<f32>) -> tensor<i32>
// CHECK:  %2 = "tf.ExpandDims"(%0, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %3 = "tf.Fill"(%2, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %4 = "tf.Add"(%arg2, %cst) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:  %5 = "tf.ExpandDims"(%1, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %6 = "tf.Fill"(%5, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %7 = "tf.Concat"(%cst_0, %4, %6) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %8 = "tf.ExpandDims"(%arg2, %cst_0) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %9 = "tf.Fill"(%5, %cst_1) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %10 = "tf.Concat"(%cst_0, %8, %9) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %11 = "tf.Fill"(%2, %cst_1) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %12 = "tf.Slice"(%arg0, %3, %10) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %13 = "tf.Slice"(%arg0, %7, %11) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %14 = "tf.ExpandDims"(%arg3, %cst_0) : (tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  %15 = "tf.Concat"(%cst_0, %12, %14, %13) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<5xf32>
// CHECK:  return %15 : tensor<5xf32>
}

func @tensorlistReserve(tensor<3xi32>, tensor<i32>, tensor<i32>) -> tensor<3xf32> {
^bb0(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
  %0 = "tf.TensorListReserve"(%arg0, %arg1) {element_dtype = f32} : (tensor<3xi32>, tensor<i32>) -> tensor<*x!tf.variant>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<*x!tf.variant>, tensor<i32>, tensor<3xi32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>

// CHECK-LABEL: tensorlistReserve
// CHECK:  %cst = constant dense<0> : tensor<i32>
// CHECK:  %cst_0 = constant dense<0.000000e+00> : tensor<f32>
// CHECK:  %0 = "tf.ExpandDims"(%arg1, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %1 = "tf.Concat"(%cst, %0, %arg0) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK:  %2 = "tf.Fill"(%1, %cst_0) : (tensor<4xi32>, tensor<f32>) -> tensor<*xf32>
// CHECK:  %3 = "tf.Gather"(%2, %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<3xf32>
// CHECK:  return %3 : tensor<3xf32>
}
