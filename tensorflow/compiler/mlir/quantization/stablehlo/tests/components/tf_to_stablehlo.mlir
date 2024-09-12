// RUN: stablehlo-quant-opt %s -split-input-file -verify-diagnostics -stablehlo-test-tf-to-stablehlo | FileCheck %s

// TODO(b/330759552): Fix the msan issue and enable this test.
// func.func @fused_batchnorm_no_training() -> tensor<1x1x2x8xf32> {
//   %cst_0 = "tf.Const"() {value = dense<[[[[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2], [0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4]]]]> : tensor<1x1x2x8xf32>} : () -> tensor<1x1x2x8xf32>
//   %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]> : tensor<8xf32>} : () -> tensor<8xf32>
//   %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4]> : tensor<8xf32>} : () -> tensor<8xf32>
//   %0:6 = "tf.FusedBatchNormV3"(%cst_0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x1x2x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<1x1x2x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
//   func.return %0#0 : tensor<1x1x2x8xf32>
// }
// COM: CHECK: func.func @main() -> tensor<1x1x2x8xf32>
// COM: CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<{{.*}}> : tensor<1x1x2x8xf32>
// COM: CHECK: return %[[CONST]] : tensor<1x1x2x8xf32>

func.func @fused_batchnorm_no_training_arg_input(%arg_0: tensor<1x1x2x8xf32>) -> (tensor<1x1x2x8xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]> : tensor<8xf32>} : () -> tensor<8xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.3, 0.4, 0.3, 0.4, 0.3, 0.4, 0.3, 0.4]> : tensor<8xf32>} : () -> tensor<8xf32>
  %0:6 = "tf.FusedBatchNormV3"(%arg_0, %cst_0, %cst_1, %cst_0, %cst_1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x1x2x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<1x1x2x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<1x1x2x8xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<{{.*}}> : tensor<8xf32>
// CHECK: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<8xf32>) -> tensor<1x1x2x8xf32>
// CHECK: %[[MUL:.*]] = stablehlo.multiply %[[ARG]], %[[BROADCAST_0]] : tensor<1x1x2x8xf32>
// CHECK: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[CONST_1]], dims = [3] : (tensor<8xf32>) -> tensor<1x1x2x8xf32>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[BROADCAST_1]] : tensor<1x1x2x8xf32>
// CHECK: return %[[ADD]] : tensor<1x1x2x8xf32>

// -----

func.func @fuse_conv_batchnorm(%arg_0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg_0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %1#0 : tensor<1x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[CONST_1]]) {{.*}} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST]] : tensor<1x3x2x2xf32>
// CHECK: return %[[ADD]] : tensor<1x3x2x2xf32>

// -----

func.func @fuse_conv_batchnorm_dynamic(%arg_0: tensor<?x3x4x3xf32>) -> (tensor<?x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg_0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<?x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<?x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %1#0 : tensor<?x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[CONST_1]]) {{.*}} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x2x2xf32>
// CHECK-DAG: %[[SHAPE_OF:.*]] = shape.shape_of %[[CONV]] : tensor<?x3x2x2xf32> -> tensor<4xindex>
// CHECK-DAG: %[[BROADCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST_0]], %[[SHAPE_OF]], dims = [3] : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST]] : tensor<?x3x2x2xf32>
// CHECK: return %[[ADD]] : tensor<?x3x2x2xf32>

// -----

func.func @func_conv_batchnorm_relu6(%arg_0: tensor<1x3x4x3xf32>) -> (tensor<1x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg_0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<1x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %2 = "tf.Relu6"(%1#0) : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
  func.return %2 : tensor<1x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CONST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CONST_3:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[CONST_3]]) {{.*}} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[CONST_0]], dims = [3] : (tensor<2xf32>) -> tensor<1x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST]] : tensor<1x3x2x2xf32>
// CHECK-DAG: %[[RELU6:.*]] = stablehlo.clamp %[[CONST_2]], %[[ADD]], %[[CONST_1]] : (tensor<f32>, tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
// CHECK: return %[[RELU6]] : tensor<1x3x2x2xf32>

// -----

func.func @func_conv_batchnorm_relu6_dynamic(%arg_0: tensor<?x3x4x3xf32>) -> (tensor<?x3x2x2xf32>) {
  %cst_0 = "tf.Const"() {value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.1, 0.2]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_2 = "tf.Const"() {value = dense<[0.3, 0.4]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg_0, %cst_0) {data_format = "NHWC", dilations = [1, 1, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x2x2xf32>
  %1:6 = "tf.FusedBatchNormV3"(%0, %cst_1, %cst_2, %cst_1, %cst_2) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<?x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<?x3x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %2 = "tf.Relu6"(%1#0) : (tensor<?x3x2x2xf32>) -> tensor<?x3x2x2xf32>
  func.return %2 : tensor<?x3x2x2xf32>
}
// CHECK: func.func @main(%[[ARG:.+]]: tensor<?x3x4x3xf32>) -> tensor<?x3x2x2xf32> {
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CONST_2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[CONST_3:.*]] = stablehlo.constant dense<[{{.*}}]> : tensor<2x3x3x2xf32>
// CHECK-DAG: %[[CONV:.*]] = stablehlo.convolution(%[[ARG]], %[[CONST_3]]) {{.*}} : (tensor<?x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<?x3x2x2xf32>
// CHECK-DAG: %[[SHAPE_OF:.*]] = shape.shape_of %[[CONV]] : tensor<?x3x2x2xf32> -> tensor<4xindex>
// CHECK-DAG: %[[BROADCAST:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST_0]], %[[SHAPE_OF]], dims = [3] : (tensor<2xf32>, tensor<4xindex>) -> tensor<?x3x2x2xf32>
// CHECK-DAG: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST]] : tensor<?x3x2x2xf32>
// CHECK-DAG: %[[RELU6:.*]] = stablehlo.clamp %[[CONST_2]], %[[ADD]], %[[CONST_1]] : (tensor<f32>, tensor<?x3x2x2xf32>, tensor<f32>) -> tensor<?x3x2x2xf32>
// CHECK: return %[[RELU6]] : tensor<?x3x2x2xf32>

// -----

// This test makes sure functions with tf._noinline=true is not inlined.

module {
  func.func @stateful_partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<1x2x2x3xf32>) {
    %0 = "tf.StatefulPartitionedCall"(%arg0) <{
      config = "", config_proto = "", executor_type = "", f = @some_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    func.return %0: tensor<1x2x2x3xf32>
  }

  func.func private @some_func(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> attributes {tf._noinline = true} {
    return %arg0 : tensor<1x2x2x3xf32>
  }
}

// CHECK: module
// CHECK: tf.StatefulPartitionedCall
// CHECK: func.func private @some_func
// CHECK-NOT: func.call

// -----

// This test makes sure functions without tf._noinline=true is inlined.

module {
  func.func @partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<1x2x2x3xf32>) {
    %0 = "tf.PartitionedCall"(%arg0) <{
      config = "", config_proto = "", executor_type = "", f = @some_func
    }> {
      _collective_manager_ids = [], device = ""
    } : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    func.return %0: tensor<1x2x2x3xf32>
  }

  func.func private @some_func(%arg0: tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32> {
    return %arg0 : tensor<1x2x2x3xf32>
  }
}

// CHECK: module
// CHECK-NOT: tf.PartitionedCall
// CHECK-NOT: some_func
// CHECK-NOT: func.call
