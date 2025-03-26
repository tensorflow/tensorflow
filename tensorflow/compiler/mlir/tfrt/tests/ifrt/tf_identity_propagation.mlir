// RUN: tf-tfrt-opt %s -tf-identity-propagation -canonicalize | FileCheck %s

// CHECK-LABEL: func @identity
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>)
func.func @identity(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NOT: "tf.Identity"
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[ARG0]]
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @identity_terminator
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>)
func.func @identity_terminator(%arg0: tensor<i32>) -> (tensor<*xi32>, tensor<i32>) {
  // CHECK: %[[IDENTITY:.*]] = "tf.Identity"
  %0 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<*xi32>
  // CHECK-NOT: "tf.Identity"
  %1 = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[IDENTITY]], %[[ARG0]]
  func.return %0, %1 : tensor<*xi32>, tensor<i32>
}

// CHECK-LABEL: func @xla_sharding
func.func @xla_sharding(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: %[[OUTPUT:.*]] = "tf.Identity"
  %0 = "tf.Identity"(%arg0) {_XlaSharding = ""} : (tensor<i32>) -> tensor<i32>
  // CHECK: return %[[OUTPUT]]
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @identity_n
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<f32>)
func.func @identity_n(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
  // CHECK-NOT: "tf.IdentityN"
  %0:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  // CHECK: return %[[ARG0]], %[[ARG1]]
  func.return %0#0, %0#1 : tensor<i32>, tensor<f32>
}
