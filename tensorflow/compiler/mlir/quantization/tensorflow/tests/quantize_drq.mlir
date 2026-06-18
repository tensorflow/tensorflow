// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions -quant-prepare-quantize-drq -quant-quantize='weight-quantization=true' -verify-each=false | FileCheck %s

// -----

module {
  func.func @matmul(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }
  func.func private @composite_matmul_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x1024xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

// CHECK: %[[cst:.*]] = "arith.constant"() <{value = dense<0.000000e+00> : tensor<2x1024xf32>}> : () -> tensor<2x1024xf32>
// CHECK: %[[q_cst:.*]] = "quantization.qcast"(%[[cst]]) : (tensor<2x1024xf32>) -> tensor<2x1024x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>
// CHECK: %[[out:.*]] = "tf.PartitionedCall"([[ARG0:%arg[0-9]+]], %[[q_cst]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x2x2x3xf32>, tensor<2x1024x!quant.uniform<i8<-127:127>:f32, 3.9370078740157481E-9>>) -> tensor<*xf32>
// CHECK: "func.return"(%[[out]]) : (tensor<*xf32>) -> ()
}

// -----
