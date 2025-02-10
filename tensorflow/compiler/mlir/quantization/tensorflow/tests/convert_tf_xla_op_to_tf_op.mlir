// RUN: tf-quant-opt %s -quant-convert-tf-xla-op-to-tf-op -split-input-file | FileCheck %s

func.func @xla_dot_v2(%arg0: tensor<?x2x3xf32>, %arg1: tensor<3x4x5xf32>) -> (tensor<?x2x4x5xf32>) {
  %0 = "tf.XlaDotV2"(%arg0, %arg1) {device = "", dimension_numbers = "\0A\01\02\12\01\00", precision_config = ""} : (tensor<?x2x3xf32>, tensor<3x4x5xf32>) -> tensor<?x2x4x5xf32>
  func.return %0 : tensor<?x2x4x5xf32>
}

// CHECK: func @xla_dot_v2
// CHECK: %[[einsum:.*]] = "tf.Einsum"(%arg0, %arg1) <{equation = "abc,cde->abde"}> : (tensor<?x2x3xf32>, tensor<3x4x5xf32>) -> tensor<?x2x4x5xf32>
// CHECK: return %[[einsum]] : tensor<?x2x4x5xf32>

// -----

// dimension_numbers: {
//   offset_dims: 0
//   collapsed_slice_dims: 1
//   start_index_map: 1
// }
func.func @xla_gather(%arg0: tensor<?x2xf32>, %arg1: tensor<1xi32>, %arg2: tensor<2xi32>) -> tensor<*xf32> {
  %0 = "tf.XlaGather"(%arg0, %arg1, %arg2) {device = "", dimension_numbers = "\0A\01\00\12\01\01\1A\01\01", indices_are_sorted = true} : (tensor<?x2xf32>, tensor<1xi32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK: func @xla_gather
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<1> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<-1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK: %[[arg1_i64:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[tensor_scatter_update:.*]] = "tf.TensorScatterUpdate"(%[[cst]], %[[cst_0]], %[[arg1_i64]]) <{bad_indices_policy = ""}> : (tensor<2xi64>, tensor<1x1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK: %[[arg2_i64:.*]] = "tf.Cast"(%arg2) <{Truncate = false}> : (tensor<2xi32>) -> tensor<2xi64>
// CHECK: %[[slice:.*]] = "tf.Slice"(%arg0, %[[tensor_scatter_update]], %[[arg2_i64]]) : (tensor<?x2xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[slice]], %[[cst_1]]) : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK: return %[[reshape]] : tensor<*xf32>

// -----

// Tests that the converted `tf.Slice` has the correct number of dimensions
// when the output shape is known (`tensor<i32>` instead of `tensor<*xi32>`).

func.func @xla_gather_known_output_shape(%arg0: tensor<5xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<i32> {
  // dimension_numbers: {
  //   collapsed_slice_dims: 0
  //   start_index_map: 0
  // }
  %0 = "tf.XlaGather"(%arg0, %arg1, %arg2) {device = "", dimension_numbers = "\12\01\00\1A\01\00", indices_are_sorted = true} : (tensor<5xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK: func @xla_gather_known_output_shape
// CHECK-DAG: %[[cst:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG: %[[cst_0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() <{value = dense<> : tensor<0xi64>}> : () -> tensor<0xi64>
// CHECK: %[[arg1_i64:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[tensor_scatter_update:.*]] = "tf.TensorScatterUpdate"(%[[cst]], %[[cst_0]], %[[arg1_i64]]) <{bad_indices_policy = ""}> : (tensor<1xi64>, tensor<1x1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK: %[[arg2_i64:.*]] = "tf.Cast"(%arg2) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[slice:.*]] = "tf.Slice"(%arg0, %[[tensor_scatter_update]], %[[arg2_i64]]) : (tensor<5xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi32>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[slice]], %[[cst_1]]) : (tensor<1xi32>, tensor<0xi64>) -> tensor<i32>
// CHECK: return %[[reshape]] : tensor<i32>
