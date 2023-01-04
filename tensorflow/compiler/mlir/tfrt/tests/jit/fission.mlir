// RUN: tf-tfrt-opt %s -tf-jitrt-fission | FileCheck %s --dump-input=always

// CHECK-LABEL: @matmul_bias_add
func.func @matmul_bias_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>)
    -> tensor<?x?xf32> {
  // CHECK:      %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %arg0)
  // CHECK-SAME:   {transpose_a = false, transpose_b = false}
  // CHECK:      %[[BIASED:.*]] = "tf.BiasAdd"(%[[MATMUL]]
  // CHECK:      return %[[BIASED]]
  %0 = "tf._FusedMatMul"(%arg0, %arg0, %arg1)
       {
         fused_ops = ["BiasAdd"],
         transpose_a = false,
         transpose_b = false
       }
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  func.return %0: tensor<?x?xf32>
}

// CHECK-LABEL: @matmul_bias_add_relu
func.func @matmul_bias_add_relu(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>)
    -> tensor<?x?xf32> {
  // CHECK:      %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %arg0)
  // CHECK-SAME:   {transpose_a = false, transpose_b = false}
  // CHECK:      %[[BIASED:.*]] = "tf.BiasAdd"(%[[MATMUL]]
  // CHECK:      %[[RELU:.*]] = "tf.Relu"(%[[BIASED]]
  // CHECK:      return %[[RELU]]
  %0 = "tf._FusedMatMul"(%arg0, %arg0, %arg1)
       {
         fused_ops = ["BiasAdd", "Relu"],
         transpose_a = false,
         transpose_b = false
       }
       : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  func.return %0: tensor<?x?xf32>
}
