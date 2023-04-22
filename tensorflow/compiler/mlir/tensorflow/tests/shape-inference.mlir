// RUN: tf-opt -split-input-file -verify-diagnostics --tf-shape-inference %s | FileCheck %s

module attributes {tf.versions = {producer = 888 : i32}} {

func @set_dynamic_dimension_size_static_dim(%input: tensor<4x5xf32>, %size: tensor<i32>) -> tensor<*xf32> {
  %dimension = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>

  // CHECK: (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<?x5xf32>
  %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @set_dynamic_dimension_size_dynamic_dim(%input: tensor<4x5xf32>, %dimension: tensor<i32>, %size: tensor<i32>) -> tensor<*xf32> {

  // CHECK: (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<?x?xf32>
  %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4x5xf32>, tensor<i32>, tensor<i32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @set_dynamic_dimension_size_dynamic_input(%input: tensor<*xf32>, %size: tensor<i32>) -> tensor<*xf32> {
  %dimension = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>

  // CHECK: (tensor<*xf32>, tensor<i32>, tensor<i32>) -> tensor<*xf32>
  %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<*xf32>, tensor<i32>, tensor<i32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
}

