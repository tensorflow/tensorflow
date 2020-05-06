// RUN: tf-opt %s -tf-move-transposes=direction=end -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @move_across_single_op
func @move_across_single_op(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%arg0) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[TANH]], %[[RES_PERM]]) {{.*}} tensor<1x8x4x4xf32>
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Tanh"(%1) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %2 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multiple_ops
func @move_across_multiple_ops(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[TANH:[0-9]*]] = "tf.Tanh"(%arg0) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RELU:[0-9]*]] = "tf.Relu"(%[[TANH]]) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[RELU]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Tanh"(%1) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Relu"(%2) : (tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @move_across_multi_operand_op
func @move_across_multi_operand_op(%arg0: tensor<1x4x4x8xf32>, %arg1: tensor<1x4x4x8xf32>) -> tensor<1x8x4x4xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%arg0, %arg1) {{.*}} tensor<1x4x4x8xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[ADD]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %2 = "tf.Transpose"(%arg1, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.AddV2"(%1, %2) : (tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>

  return %3 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func @fold_into_max_pool
func @fold_into_max_pool(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x56x56x64xf32> {

  // MaxPool operand transpose must be folded into the op and MaxPool
  // must use NCHW data format with updated kernel size and strides.

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>}
  // CHECK: %[[MAX_POOL:[0-9]*]] = "tf.MaxPool"(%arg0) {data_format = "NCHW", ksize = [1, 1, 3, 3], padding = "SAME", strides = [1, 1, 2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%[[MAX_POOL]], %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  // Transpose NCHW -> NHWC
  %0 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>

  // Compute MaxPool in NHWC format
  %2 = "tf.MaxPool"(%1)
       {
         data_format = "NHWC", ksize = [1, 3, 3, 1],
         padding = "SAME", strides = [1, 2, 2, 1]
       } : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>

  return %2 : tensor<1x56x56x64xf32>
}

// CHECK-LABEL: func @fold_into_mean
func @fold_into_mean(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64xf32> {

  // CHECK: %[[RED_IDX:[0-9]*]] = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>}
  // CHECK: %[[MEAN:[0-9]*]] = "tf.Mean"(%arg0, %[[RED_IDX]])
  // CHECK-SAME: (tensor<1x64x112x112xf32>, tensor<2xi32>) -> tensor<1x64xf32>
  // CHECK: return %[[MEAN]]

  // Transpose NCHW -> NHWC
  %0 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>

  // Compute Mean over spatial dimensions in NHWC format.
  %2 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.Mean"(%1, %2) : (tensor<1x112x112x64xf32>, tensor<2xi32>) -> tensor<1x64xf32>

  return %3 : tensor<1x64xf32>
}

// CHECK-LABEL: func @fold_into_fused_batch_norm
func @fold_into_fused_batch_norm(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<64xf32>) -> tensor<1x112x112x64xf32> {

  // CHECK: %[[RES_PERM:[0-9]*]] = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>}
  // CHECK: "tf.FusedBatchNormV3"(%arg0, {{.*}} {data_format = "NCHW"
  // CHECK: %[[RES_TRANSPOSE:[0-9]*]] = "tf.Transpose"(%y, %[[RES_PERM]])
  // CHECK: return %[[RES_TRANSPOSE]]

  // Transpose NCHW -> NHWC
  %0 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>

  // Compute FusedBatchNormV3 in NHWC format
  %2, %batch_mean, %batch_var, %reserve_1, %reserve_2, %reserve_3
    = "tf.FusedBatchNormV3"(%1, %arg1, %arg1, %arg1, %arg1)
       {
         data_format = "NHWC",
         epsilon = 1.001 : f32,
         exponential_avg_factor = 1.0 : f32,
         is_training = false
       }
        : (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
       -> (tensor<1x112x112x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)

  return %2#0 : tensor<1x112x112x64xf32>
}
