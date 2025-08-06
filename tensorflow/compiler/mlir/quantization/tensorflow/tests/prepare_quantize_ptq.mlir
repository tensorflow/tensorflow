// RUN: tf-quant-opt %s -split-input-file -quant-prepare-quantize='post-training-quantize=true' | FileCheck %s

// -----

module {
  func.func @same_scale_ptq_test(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %cst = arith.constant dense<[-1, 144]> : tensor<2xi32>
    %cst_1 = arith.constant dense<1.0> : tensor<144x10xf32>
    %cst_2 = arith.constant dense<0.1> : tensor<10xf32>
    %0 = "quantization.stats"(%arg0) {
      layerStats = dense<[-1.28, 1.28]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "tf.MaxPool"(%0) {
      data_format = "NHWC", device = "", explicit_paddings = [],
      ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 2, 2, 1]
    } : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "quantization.stats"(%1) {
      layerStats = dense<[-0.9, 1.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.Reshape"(%2, %cst) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
    %4 = "quantization.stats"(%3) {
      layerStats = dense<[-1.0, 0.8]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "tf.PartitionedCall"(%4, %cst_1, %cst_2) {
      _tfl_quant_trait = "fully_quantizable", config = "", config_proto = "",
      executor_type = "", f = @composite_matmul_with_bias_fn_1
    } : (tensor<*xf32>, tensor<144x10xf32>, tensor<10xf32>) -> tensor<*xf32>
    %6 = "quantization.stats"(%5) {
      layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.Identity"(%6) : (tensor<*xf32>) -> tensor<*xf32>
    %8 = "quantization.stats"(%7) {
      layerStats = dense<[-2.0, 2.0]> : tensor<2xf32>
    } : (tensor<*xf32>) -> tensor<*xf32>
    func.return %8 : tensor<*xf32>
  }

  func.func private @composite_matmul_with_bias_fn_1(%a: tensor<*xf32>, %b: tensor<*xf32>, %c: tensor<*xf32>) -> tensor<*xf32> {
    func.return %a: tensor<*xf32>
  }
}

// CHECK-LABEL: same_scale_ptq_test

// CHECK: %[[q0:.*]] = "quantization.qcast"(%arg0)
// CHECK: %[[dq0:.*]] = "quantization.dcast"(%[[q0]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[maxpool:.*]] = "tf.MaxPool"(%[[dq0]])
// CHECK: %[[q1:.*]] = "quantization.qcast"(%[[maxpool]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[dq1:.*]] = "quantization.dcast"(%[[q1]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[dq1]]
// CHECK: %[[q2:.*]] = "quantization.qcast"(%[[reshape]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[dq2:.*]] = "quantization.dcast"(%[[q2]])
// CHECK-SAME: quant.uniform<i8:f32, 0.010039215461880554:-1>
// CHECK: %[[call:.*]] = "tf.PartitionedCall"(%[[dq2]]
// CHECK-SAME: f = @composite_matmul_with_bias_fn_1
// CHECK: %[[q3:.*]] = "quantization.qcast"(%[[call]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[dq3:.*]] = "quantization.dcast"(%[[q3]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>
// CHECK: %[[identity:.*]] = "tf.Identity"(%[[dq3]])
// CHECK: "quantization.qcast"(%[[identity]])
// CHECK-SAME: quant.uniform<i8:f32, 0.015686274509803921:-1>

// -----

module {
  func.func private @conv_with_bias_and_relu(%arg0: tensor<1x3x4x3xf32>) -> tensor<*xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[7.11401462, 7.05456924]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [9.54742622, -1.43770897], [-7.96835279, 8.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [9.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[1.27501142, 149.824783]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.PartitionedCall"(%0, %cst_0, %cst) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", device = "", executor_type = "", f = @composite_conv2d_with_bias_and_relu6_fn_10} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<*xf32>
    %2 = "quantization.stats"(%1) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    return %2 : tensor<*xf32>
  }

  func.func private @composite_conv2d_with_bias_and_relu6_fn_10(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> attributes {tf.tf_quant.composite_function} {
    %0 = "quantization.stats"(%arg1) {layerStats = dense<[-9.54062747, 9.54742622]> : tensor<2xf32>} : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2xf32>
    %1 = "quantization.stats"(%arg0) {layerStats = dense<[1.27501142, 149.824783]> : tensor<2xf32>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %2 = "tf.Conv2D"(%1, %0) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
    %3 = "quantization.stats"(%arg2) {layerStats = dense<[7.05456924, 7.11401462]> : tensor<2xf32>} : (tensor<2xf32>) -> tensor<2xf32>
    %4 = "quantization.stats"(%2) {layerStats = dense<[-2795.36523, 4609.57373]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "tf.BiasAdd"(%4, %3) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    %6 = "quantization.stats"(%5) {layerStats = dense<[-2788.31055, 4616.62842]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %7 = "tf.Relu6"(%6) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %8 = "quantization.stats"(%7) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    return %8 : tensor<*xf32>
  }

// CHECK-LABEL: conv_with_bias_and_relu
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<[7.11401462, 7.05456924]> : tensor<2xf32>
// CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<{{.*}}> : tensor<2x3x3x2xf32>

// CHECK: %[[q0:.*]] = "quantization.qcast"(%[[cst]]) {volatile}
// CHECK-SAME: tensor<2x!quant.uniform<i32:f32, 0.044169864606680966>>
// CHECK: %[[dq0:.*]] = "quantization.dcast"(%[[q0]])

// CHECK: %[[q1:.*]] = "quantization.qcast"(%[[cst_1]]) {volatile}
// CHECK-SAME: tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32, 0.075176584439014829>>
// CHECK: %[[dq1:.*]] = "quantization.dcast"(%[[q1]])

// CHECK: %[[q2:.*]] = "quantization.qcast"(%arg0)
// CHECK-SAME: tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58754816990272674:-128>>
// CHECK: %[[dq2:.*]] = "quantization.dcast"(%[[q2]])

// CHECK: %[[call:.*]] = "tf.PartitionedCall"(%[[dq2]], %[[dq1]], %[[dq0]])
// CHECK-SAME: f = @composite_conv2d_with_bias_and_relu6_fn_10
// CHECK: %[[q3:.*]] = "quantization.qcast"(%[[call]]) {volatile}
// CHECK-SAME: tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>
// CHECK: %[[dq3:.*]] = "quantization.dcast"(%[[q3]])
}

// -----

module {
  func.func @value_used_in_multiple_same_scale_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
    %cst = "tf.Const"() {device = "", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %0 = "quantization.stats"(%arg0) {layerStats = dense<[-0.83811146, 62.4960899]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "quantization.stats"(%arg1) {layerStats = dense<[-0.835039615, 1.000000e+00]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.ConcatV2"(%0, %1, %cst) {device = ""} : (tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
    %3 = "quantization.stats"(%2) {layerStats = dense<[-0.83811146, 62.4960899]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>

    %4 = "quantization.stats"(%arg2) {layerStats = dense<[-1.5726943, 1.07351148]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %5 = "quantization.stats"(%arg3) {layerStats = dense<[0.000000e+00, 10.6875381]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    %6 = "tf.ConcatV2"(%4, %5, %1, %cst) {device = ""} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
    %7 = "quantization.stats"(%6) {layerStats = dense<[-1.5726943, 10.6875381]> : tensor<2xf32>} : (tensor<*xf32>) -> tensor<*xf32>
    func.return  %3, %7 : tensor<*xf32>, tensor<*xf32>
  }
// CHECK-LABEL: value_used_in_multiple_same_scale_ops

// Check if operands of concat ops are quantized separately.
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<2> : tensor<i32>
// CHECK-DAG: %[[q1:.*]] = "quantization.qcast"(%arg1) {volatile} : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.048079342935599532:-95>>
// CHECK-DAG: %[[q2:.*]] = "quantization.qcast"(%arg1) {volatile} : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.24836941723730049:-125>>
// CHECK-DAG: %[[dq1:.*]] = "quantization.dcast"(%[[q1]])
// CHECK-DAG: %[[dq2:.*]] = "quantization.dcast"(%[[q2]])

// CHECK-DAG: %[[concat1:.*]] = "tf.ConcatV2"({{.*}}, %[[dq2]], %[[cst]])
// CHECK-DAG: %[[concat2:.*]] = "tf.ConcatV2"({{.*}}, {{.*}}, %[[dq1]], %[[cst]])
}
