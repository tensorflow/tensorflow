// RUN: tf-quant-opt %s -split-input-file -quant-propagate-quantize-type | FileCheck %s

module {
  func.func @not_propagate_matmul(%arg0: tensor<1x2x2x2xf32>) -> tensor<*xf32> {
    %cst = "tf.Const"() {value = dense<127> : tensor<2x1024xi8>} : () -> tensor<2x1024xi8>
    %cst_0 = "tf.Const"() {value = dense<0.0157480314> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Identity"(%cst) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
    %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
    %2 = "tf.MatMul"(%arg0, %1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    %3 = "tf.Mul"(%2, %cst_0) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    return %3 : tensor<*xf32>
  }

// CHECK-LABEL: func @not_propagate_matmul
// CHECK: %[[CASTED_W:.*]] = "tf.Cast"(%0) <{Truncate = false}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %2 = "tf.MatMul"(%arg0, %[[CASTED_W]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
}

// -----

module {
  func.func @propagate_xladotv2_bf16(%arg0: tensor<1x2x2x2xbf16>) -> tensor<1x2x2x1024xbf16> {
    %cst = "tf.Const"() {value = dense<127> : tensor<2x1024xi8>} : () -> tensor<2x1024xi8>
    %0 = "tf.Identity"(%cst) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
    %1 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform} : (tensor<2x1024xi8>) -> tensor<2x1024xbf16>
    %2 = "tf.XlaDotV2"(%arg0, %1) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<1x2x2x2xbf16>, tensor<2x1024xbf16>) -> tensor<1x2x2x1024xbf16>
    %3 = "tf.Identity"(%2) : (tensor<1x2x2x1024xbf16>) -> tensor<1x2x2x1024xbf16>
    return %3 : tensor<1x2x2x1024xbf16>
  }

  func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xbf16> {
    %cst = "tf.Const"() {value = dense<1.574710e-02> : tensor<bf16>} : () -> tensor<bf16>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<*xi8>) -> tensor<*xbf16>
    %1 = "tf.Mul"(%0, %cst) : (tensor<*xbf16>, tensor<bf16>) -> tensor<*xbf16>
    return %1 : tensor<*xbf16>
  }

// CHECK-LABEL: func @propagate_xladotv2_bf16
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%cst) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[MATMUL:.*]] = "tf.XlaDotV2"(%arg0, %[[IDENTITY]]) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<1x2x2x2xbf16>, tensor<2x1024xi8>) -> tensor<1x2x2x1024xbf16>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[MATMUL]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1x2x2x1024xbf16>) -> tensor<1x2x2x1024xbf16>
}

// -----

module {
  func.func @not_propagate_last_op(%arg0: tensor<10x2xi32>) -> tensor<1x300x10xf32> {
    %cst = "tf.Const"() {value = dense<[1, 1, 300]> : tensor<3xi64>} : () -> tensor<3xi64>
    %cst_0 = "tf.Const"() {value = dense<127> : tensor<200x100x300xi8>} : () -> tensor<200x100x300xi8>
    %0 = "tf.Identity"(%cst_0) : (tensor<200x100x300xi8>) -> tensor<200x100x300xi8>
    %1 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform} : (tensor<200x100x300xi8>) -> tensor<200x100x300xf32>
    %2 = "tf.XlaGather"(%1, %arg0, %cst) {dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01 \01", indices_are_sorted = true} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
    return %2 : tensor<1x300x10xf32>
  }

  func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32> {
    %cst = "tf.Const"() {value = dense<0.0787401571> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

}

// CHECK-LABEL: func @not_propagate_last_op
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%cst_0) : (tensor<200x100x300xi8>) -> tensor<200x100x300xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[IDENTITY]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<200x100x300xi8>) -> tensor<200x100x300xf32>
// CHECK: %[[GATHER:.*]] = "tf.XlaGather"(%[[DEQUANTIZED]], %arg0, %cst) <{dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01 \01", indices_are_sorted = true}> : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
// CHECK: return %[[GATHER]] : tensor<1x300x10xf32>

// -----

module {
  func.func @propagate_xlagather(%arg0: tensor<10x2xi32>) -> tensor<1x300x10xf32> {
    %cst = "tf.Const"() {value = dense<[1, 1, 300]> : tensor<3xi64>} : () -> tensor<3xi64>
    %cst_0 = "tf.Const"() {value = dense<127> : tensor<200x100x300xi8>} : () -> tensor<200x100x300xi8>
    %0 = "tf.Identity"(%cst_0) : (tensor<200x100x300xi8>) -> tensor<200x100x300xi8>
    %1 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform} : (tensor<200x100x300xi8>) -> tensor<200x100x300xf32>
    %2 = "tf.XlaGather"(%1, %arg0, %cst) {dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01 \01", indices_are_sorted = true} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
    %3 = "tf.Identity"(%2) : (tensor<1x300x10xf32>) -> tensor<1x300x10xf32>
    return %3 : tensor<1x300x10xf32>
  }

  func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32> {
    %cst = "tf.Const"() {value = dense<0.0787401571> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %1 = "tf.Mul"(%0, %cst) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }
}

// CHECK-LABEL: func @propagate_xlagather
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%cst_0) : (tensor<200x100x300xi8>) -> tensor<200x100x300xi8>
// CHECK: %[[GATHER:.*]] = "tf.XlaGather"(%[[IDENTITY]], %arg0, %cst) <{dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01 \01", indices_are_sorted = true}> : (tensor<200x100x300xi8>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[GATHER]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1x300x10xi8>) -> tensor<1x300x10xf32>
// CHECK: %[[ORIGINAL_IDENTITY:.*]] = "tf.Identity"(%[[DEQUANTIZED]]) : (tensor<1x300x10xf32>) -> tensor<1x300x10xf32>
// CHECK: return %[[ORIGINAL_IDENTITY]] : tensor<1x300x10xf32>
