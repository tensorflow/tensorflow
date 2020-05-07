// RUN: tf-opt %s -tf-to-tfd-lowering | FileCheck %s

// CHECK: func @inference_call(
// CHECK-SAME: %arg0: !hex.chain,
// CHECK-SAME: %arg1: !tfd.tf_tensor,
// CHECK-SAME: %arg2: !tfd.tf_tensor,
// CHECK-SAME: %arg3: !tfd.tf_tensor,
// CHECK-SAME: %arg4: !tfd.tf_tensor,
// CHECK-SAME: %arg5: !tfd.tf_tensor
// CHECK-SAME: ) -> (!hex.chain, !tfd.tf_tensor)
func @inference_call(
  %arg0: tensor<?x784xf32>,
  %arg1: tensor<*x!tf.resource>,
  %arg2: tensor<*x!tf.resource>,
  %arg3: tensor<*x!tf.resource>,
  %arg4: tensor<*x!tf.resource>
  )-> tensor<?x10xf32> {
    // CHECK: %0:2 = "tfd.delegate_kernel"(%arg0, %arg5)
    // CHECK-SAME: _name = "tf.ReadVariableOp"
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: (!hex.chain, !tfd.tf_tensor) -> (!hex.chain, !tfd.tf_tensor)
    %0 = "tf.ReadVariableOp"(%arg4) {
      dtype = "tfdtype$DT_FLOAT"
      } : (tensor<*x!tf.resource>) -> tensor<10xf32>

    // CHECK: %1:2 = "tfd.delegate_kernel"(%0#0, %arg3) {
    // CHECK-SAME: _name = "tf.ReadVariableOp"
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %1 = "tf.ReadVariableOp"(%arg2) {
      dtype = "tfdtype$DT_FLOAT"
      } : (tensor<*x!tf.resource>) -> tensor<512xf32>

    // CHECK: %2:2 = "tfd.delegate_kernel"(%1#0, %arg4) {
    // CHECK-SAME: _name = "tf.ReadVariableOp",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %2 = "tf.ReadVariableOp"(%arg3) {
      dtype = "tfdtype$DT_FLOAT"
      } : (tensor<*x!tf.resource>) -> tensor<512x10xf32>

    // CHECK: %3:2 = "tfd.delegate_kernel"(%2#0, %arg2) {
    // CHECK-SAME: _name = "tf.ReadVariableOp",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %3 = "tf.ReadVariableOp"(%arg1) {
      dtype = "tfdtype$DT_FLOAT"
      } : (tensor<*x!tf.resource>) -> tensor<784x512xf32>

    // CHECK: %4:2 = "tfd.delegate_kernel"(%3#0, %arg1, %3#1) {
    // CHECK-SAME: _name = "tf.MatMul",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT",
    // CHECK-SAME: attr1_name = "transpose_a", attr1_value = false,
    // CHECK-SAME: attr2_name = "transpose_b", attr2_value = false
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %4 = "tf.MatMul"(%arg0, %3) {
      dtype = "tfdtype$DT_FLOAT", transpose_a = false, transpose_b = false
    } : (tensor<?x784xf32>, tensor<784x512xf32>) -> tensor<?x512xf32>

    // CHECK: %5:2 = "tfd.delegate_kernel"(%4#0, %4#1, %1#1) {
    // CHECK-SAME: _name = "tf.AddV2"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %5 = "tf.AddV2"(%4, %1)
      : (tensor<?x512xf32>, tensor<512xf32>)-> tensor<?x512xf32>

    // CHECK: %6:2 = "tfd.delegate_kernel"(%5#0, %5#1) {
    // CHECK-SAME: _name = "tf.Relu",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %6 = "tf.Relu"(%5) {
      dtype = "tfdtype$DT_FLOAT"
    } : (tensor<?x512xf32>) -> tensor<?x512xf32>

    // CHECK: %7:2 = "tfd.delegate_kernel"(%6#0, %6#1, %2#1) {
    // CHECK-SAME: _name = "tf.MatMul",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT",
    // CHECK-SAME: attr1_name = "transpose_a", attr1_value = false,
    // CHECK-SAME: attr2_name = "transpose_b", attr2_value = false
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %7 = "tf.MatMul"(%6, %2) {
      dtype = "tfdtype$DT_FLOAT", transpose_a = false, transpose_b = false
    } : (tensor<?x512xf32>, tensor<512x10xf32>) -> tensor<?x10xf32>

    // CHECK: %8:2 = "tfd.delegate_kernel"(%7#0, %7#1, %0#1) {
    // CHECK-SAME: _name = "tf.AddV2",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %8 = "tf.AddV2"(%7, %0) {
      dtype = "tfdtype$DT_FLOAT"
    } : (tensor<?x10xf32>, tensor<10xf32>) -> tensor<?x10xf32>

    // CHECK: %9:2 = "tfd.delegate_kernel"(%8#0, %8#1) {
    // CHECK-SAME: _name = "tf.Identity",
    // CHECK-SAME: attr0_name = "dtype", attr0_value = "tfdtype$DT_FLOAT"
    // CHECK-SAME: } : (!hex.chain, !tfd.tf_tensor)
    // CHECK-SAME: -> (!hex.chain, !tfd.tf_tensor)
    %9 = "tf.Identity"(%8) {
      dtype = "tfdtype$DT_FLOAT"
    } : (tensor<?x10xf32>) -> tensor<?x10xf32>

    // CHECK: hex.return %9#0, %9#1 : !hex.chain, !tfd.tf_tensor
    return %9 : tensor<?x10xf32>
}
