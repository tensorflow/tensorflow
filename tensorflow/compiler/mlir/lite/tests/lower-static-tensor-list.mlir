// RUN: tf-opt -tfl-lower-static-tensor-list %s | FileCheck %s --dump-input-on-failure

func @tensorlistGetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<10xf32>, tensor<3x10xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<10xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %1, %2 : tensor<10xf32>, tensor<3x10xf32>

// CHECK-LABEL: tensorlistGetItem
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<3x10xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK: return %0, %arg0 : tensor<10xf32>, tensor<3x10xf32>
}

func @tensorlistGetItemWithUnknownRank(%arg0: tensor<*xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<*xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<*xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<*xf32>
  return %1, %2 : tensor<*xf32>, tensor<*xf32>

// CHECK-LABEL: tensorlistGetItemWithUnknownRank
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %0, %arg0 : tensor<*xf32>, tensor<*xf32>
}

func @tensorlistSetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>) -> tensor<3x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %2 : tensor<3x10xf32>

// CHECK-LABEL: tensorlistSetItem
// CHECK:  %0 = "tf.Rank"(%arg0) : (tensor<3x10xf32>) -> tensor<i32>
// CHECK:  %1 = "tf.Rank"(%arg3) : (tensor<10xf32>) -> tensor<i32>
// CHECK:  %cst = constant dense<0> : tensor<i32>
// CHECK:  %2 = "tf.ExpandDims"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_0 = constant dense<0> : tensor<i32>
// CHECK:  %3 = "tf.Fill"(%2, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %cst_1 = constant dense<1> : tensor<1xi32>
// CHECK:  %4 = "tf.Add"(%arg2, %cst_1) : (tensor<i32>, tensor<1xi32>) -> tensor<*xi32>
// CHECK:  %5 = "tf.ExpandDims"(%1, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_2 = constant dense<0> : tensor<i32>
// CHECK:  %6 = "tf.Fill"(%5, %cst_2) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %7 = "tf.Concat"(%cst, %4, %6) {N = 2 : i64} : (tensor<i32>, tensor<*xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %8 = "tf.ExpandDims"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_3 = constant dense<-1> : tensor<i32>
// CHECK:  %9 = "tf.Fill"(%5, %cst_3) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %10 = "tf.Concat"(%cst, %8, %9) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %cst_4 = constant dense<-1> : tensor<i32>
// CHECK:  %11 = "tf.Fill"(%2, %cst_4) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %12 = "tf.Slice"(%arg0, %3, %10) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %13 = "tf.Slice"(%arg0, %7, %11) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %14 = "tf.ExpandDims"(%arg3, %cst) : (tensor<10xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  %15 = "tf.Concat"(%cst, %12, %14, %13) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<3x10xf32>
// CHECK:  return %15 : tensor<3x10xf32>
}

func @tensorlistSetItemWithScalarElements(%arg0: tensor<5xf32>, %arg1: tensor<0xi32>, %arg2: tensor<i32>, %arg3: tensor<f32>) -> tensor<5xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<5xf32>, tensor<0xi32>) -> tensor<!tf.variant<tensor<f32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf.variant<tensor<f32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf.variant<tensor<f32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<f32>>>, tensor<0xi32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>

// CHECK-LABEL: tensorlistSetItemWithScalarElements
// CHECK:  %0 = "tf.Rank"(%arg0) : (tensor<5xf32>) -> tensor<i32>
// CHECK:  %1 = "tf.Rank"(%arg3) : (tensor<f32>) -> tensor<i32>
// CHECK:  %cst = constant dense<0> : tensor<i32>
// CHECK:  %2 = "tf.ExpandDims"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_0 = constant dense<0> : tensor<i32>
// CHECK:  %3 = "tf.Fill"(%2, %cst_0) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %cst_1 = constant dense<1> : tensor<1xi32>
// CHECK:  %4 = "tf.Add"(%arg2, %cst_1) : (tensor<i32>, tensor<1xi32>) -> tensor<*xi32>
// CHECK:  %5 = "tf.ExpandDims"(%1, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_2 = constant dense<0> : tensor<i32>
// CHECK:  %6 = "tf.Fill"(%5, %cst_2) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %7 = "tf.Concat"(%cst, %4, %6) {N = 2 : i64} : (tensor<i32>, tensor<*xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %8 = "tf.ExpandDims"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  %cst_3 = constant dense<-1> : tensor<i32>
// CHECK:  %9 = "tf.Fill"(%5, %cst_3) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %10 = "tf.Concat"(%cst, %8, %9) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  %cst_4 = constant dense<-1> : tensor<i32>
// CHECK:  %11 = "tf.Fill"(%2, %cst_4) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  %12 = "tf.Slice"(%arg0, %3, %10) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %13 = "tf.Slice"(%arg0, %7, %11) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK:  %14 = "tf.ExpandDims"(%arg3, %cst) : (tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  %15 = "tf.Concat"(%cst, %12, %14, %13) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<5xf32>
// CHECK:  return %15 : tensor<5xf32>
}

func @tensorlistReserve(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-LABEL: tensorlistReserve
// CHECK-DAG:  [[ZERO1:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ZERO2:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[DIM0:%.*]] = "tf.ExpandDims"(%arg1, [[ZERO1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO2]], [[DIM0]], %arg0) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], %arg2) {validate_indices = true} : (tensor<?x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

func @tensorlistReserveUnrankedElements(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<?xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>, tensor<?xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL: tensorlistReserveUnrankedElements
// CHECK:  [[RESULT:%[0-9]+]] = "tf.Fill"{{.*}}(tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
// CHECK:  [[RESULT2:%[0-9]+]] = "tf.Gather"{{.*}}{validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  return [[RESULT2]] : tensor<*xf32>
}

func @EmptyTensorList(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.EmptyTensorList"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-LABEL: EmptyTensorList
// CHECK-SAME:  ([[ELEM_SHAPE:%.*]]: tensor<3xi32>, [[MAX_ELEMS:%.*]]: tensor<i32>, [[IDX:%.*]]: tensor<i32>)
// CHECK-DAG:  [[DIM0:%cst.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:  [[ZERO:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO]], [[DIM0]], [[ELEM_SHAPE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], [[IDX]]) {validate_indices = true} : (tensor<?x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

func @tensorlistPushBack(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<10xf32>) -> tensor<?x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListPushBack"(%0, %arg2) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<10xf32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>

// CHECK-LABEL: tensorlistPushBack
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK:   [[ZERO:%.*]] = constant dense<0> : tensor<i32>
// CHECK:   [[EXP_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) {{.*}} -> tensor<1x10xf32>
// CHECK:   [[RESULT:%.*]] = "tf.Concat"(%cst, [[INPUT]], [[EXP_ITEM]]) {N = 2 : i64} : {{.*}} -> tensor<?x10xf32>
// CHECK:   return [[RESULT]] : tensor<?x10xf32>
}

func @tensorlistWhileLoop(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %cst = constant dense<3> : tensor<1xi32>
  %cst_0 = constant dense<0> : tensor<i32>
  %cst_1 = constant dense<-1> : tensor<i32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<2x3xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<3xf32>>>
  %1:2 = "tf.While"(%cst_0, %0) {T = ["tfdtype$DT_INT32", "tfdtype$DT_VARIANT"], body = @tensorlistWhileBody, cond = @tensorlistWhileCond, is_stateless = false} : (tensor<i32>, tensor<!tf.variant<tensor<3xf32>>>) -> (tensor<i32>, tensor<!tf.variant<tensor<*xf32>>>)
  %2 = "tf.TensorListStack"(%1#1, %cst_1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>

// make sure the variant types in input/output have been updated, and `T` attribute
// is removed.
// CHECK-LABEL: func @tensorlistWhileLoop
// CHECK-NOT: "tf.While"{{.*}}T =
// CHECK: "tf.While"
// CHECK-SAME: (tensor<i32>, tensor<2x3xf32>) -> (tensor<i32>, tensor<*xf32>)
// CHECK:  return %0#1 : tensor<*xf32>
}

func @tensorlistWhileBody(%arg0: tensor<*xi32>, %arg1: tensor<!tf.variant>) -> (tensor<*xi32>, tensor<!tf.variant>) {
  %cst = constant dense<1> : tensor<i32>
  %0 = "tf.Add"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = "tf.Identity"(%arg1) : (tensor<!tf.variant>) -> tensor<!tf.variant>
  return %0, %1 : tensor<*xi32>, tensor<!tf.variant>

// verify `body` function's signature.
// CHECK: func @tensorlistWhileBody(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>)
// CHECK:  %0 = "tf.Add"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
// CHECK-NOT: tensor<!tf.variant>
// CHECK:  %1 = "tf.Identity"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:  return %0, %1 : tensor<*xi32>, tensor<*xf32>
}

func @tensorlistWhileCond(%arg0: tensor<*xi32>, %arg1: tensor<!tf.variant>) -> tensor<*xi1> {
  %cst = constant dense<2> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// verify `cond` function's signature.
// CHECK: func @tensorlistWhileCond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<*xi1>
// CHECK:  %0 = "tf.Less"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
// CHECK:  return %0 : tensor<*xi1>
}
