// RUN: tf-quant-opt %s -split-input-file -quant-prepare-quantize | FileCheck %s

module {
  func.func @same_scale_test(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %cst = arith.constant dense<[-1, 144]> : tensor<2xi32>
    %cst_1 = arith.constant dense<1.0> : tensor<144x10xf32>
    %cst_2 = arith.constant dense<0.1> : tensor<10xf32>
    %0 = "quantfork.qcast"(%arg0) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.05:-10>>
    %1 = "quantfork.dcast"(%0) : (tensor<*x!quant.uniform<i8:f32, 0.05:-10>>) -> tensor<*xf32>
    %2 = "tf.MaxPool"(%1) {
      data_format = "NHWC", device = "", explicit_paddings = [],
      ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 2, 2, 1]
    } : (tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.Reshape"(%2, %cst) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
    %4 = "tf.PartitionedCall"(%3, %cst_1, %cst_2) {
      _tfl_quant_trait = "fully_quantizable", config = "", config_proto = "",
      executor_type = "", f = @composite_matmul_with_bias_fn_1
    } : (tensor<*xf32>, tensor<144x10xf32>, tensor<10xf32>) -> tensor<*xf32>
    %5 = "quantfork.qcast"(%4) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
    %6 = "quantfork.dcast"(%5) : (tensor<*x!quant.uniform<i8:f32, 0.1>>) -> tensor<*xf32>
    func.return %6 : tensor<*xf32>
  }

  func.func private @composite_matmul_with_bias_fn_1(%a: tensor<*xf32>, %b: tensor<*xf32>, %c: tensor<*xf32>) -> tensor<*xf32> {
    func.return %a: tensor<*xf32>
  }

// CHECK-LABEL: same_scale_test
// CHECK: %[[maxpool:.*]] = "tf.MaxPool"
// CHECK: %[[q1:.*]] = "quantfork.qcast"(%[[maxpool]])
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-02:-10>
// CHECK: %[[dq1:.*]] = "quantfork.dcast"(%[[q1]])
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-02:-10>
// CHECK: %[[reshape:.*]] = "tf.Reshape"(%[[dq1]]
// CHECK: %[[q2:.*]] = "quantfork.qcast"(%[[reshape]])
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-02:-10>
// CHECK: %[[dq2:.*]] = "quantfork.dcast"(%[[q2]])
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-02:-10>
// CHECK: "tf.PartitionedCall"(%[[dq2]]
// CHECK-SAME: f = @composite_matmul_with_bias_fn_1
}

