// RUN: litert-opt "-tfl-lower-static-tensor-list=allow-tensorlist-pass-through default-to-single-batch enable-dynamic-update-slice" -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: tensorlistSetItem
func.func @tensorlistSetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>) -> tensor<3x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  func.return %2 : tensor<3x10xf32>

// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[INDEX:%.*]]: tensor<i32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK-DAG:   [[ITEM_RANK:%.*]] = "tf.Rank"([[ITEM]]) : (tensor<10xf32>) -> tensor<i32>
// CHECK-DAG:   [[ZERO:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:   [[VECTOR_RANK:%.*]] = "tf.ExpandDims"([[ITEM_RANK]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:   [[ZERO_1:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:   [[PARTIAL_INDEX:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK-DAG:   [[EXPANDED_INDIX:%.*]] = "tf.ExpandDims"([[INDEX]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:   [[ITEM_INDEX:%.*]] = "tf.Concat"([[ZERO]], [[EXPANDED_INDIX]], [[PARTIAL_INDEX]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK-DAG:   [[EXPANDED_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) : (tensor<10xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK-DAG:   [[RESULT:%.*]] = "tf.XlaDynamicUpdateSlice"([[INPUT]], [[EXPANDED_ITEM]], [[ITEM_INDEX]]) : (tensor<3x10xf32>, tensor<*xf32>, tensor<?xi32>) -> tensor<3x10xf32>
// CHECK-DAG:  return [[RESULT]] : tensor<3x10xf32>
}

// -----
