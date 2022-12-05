// RUN: tf-quant-opt %s -split-input-file -inline -quant-replace-cast-hacks-with-tf-xla-ops | FileCheck %s

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1177 : i32}} {
  func.func @conv_with_bias_and_relu(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x3x2x2xf32> {
    %cst = "tf.Const"() {value = dense<[162, 160]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_0 = "tf.Const"() {value = dense<[[[[-85, 72], [23, -103], [-29, -96]], [[-128, -83], [81, -57], [67, 119]], [[44, 10], [-90, -107], [77, 122]]], [[[18, 61], [127, -20], [-107, 119]], [[12, -66], [-98, 15], [124, 9]], [[68, 119], [20, -52], [48, 123]]]]> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
    %cst_1 = "tf.Const"() {value = dense<0.587548196> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_3 = "tf.Const"() {value = dense<18.1044273> : tensor<f32>} : () -> tensor<f32>
    %cst_4 = "tf.Const"() {value = dense<0.0748551115> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {value = dense<0.0439809859> : tensor<f32>} : () -> tensor<f32>
    %cst_7 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_1, %cst_2) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x3x4x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x3x4x3xi8>
    %1 = "tf.PartitionedCall"(%0, %cst_0, %cst, %cst_1, %cst_2, %cst_4, %cst_5, %cst_6, %cst_7, %cst_3, %cst_2) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_bias_and_relu_fn_0} : (tensor<1x3x4x3xi8>, tensor<2x3x3x2xi8>, tensor<2xi32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x3x2x2xi8>
    %2 = "tf.PartitionedCall"(%1, %cst_3, %cst_2) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<1x3x2x2xi8>, tensor<f32>, tensor<i32>) -> tensor<1x3x2x2xf32>
    return %2 : tensor<1x3x2x2xf32>
  }
  func.func private @quantize_i8(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x3x4x3xi8> {
    %0 = "tf.Div"(%arg0, %arg1) : (tensor<1x3x4x3xf32>, tensor<f32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Round"(%0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %2 = "tf.Cast"(%1) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xi32>
    %3 = "tf.AddV2"(%2, %arg2) : (tensor<1x3x4x3xi32>, tensor<i32>) -> tensor<1x3x4x3xi32>
    %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x3x4x3xi32>) -> tensor<1x3x4x3xi8>
    return %4 : tensor<1x3x4x3xi8>
  }
  func.func private @dequantize_i8(%arg0: tensor<1x3x2x2xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x3x2x2xf32> {
    %0 = "tf.Cast"(%arg0) : (tensor<1x3x2x2xi8>) -> tensor<1x3x2x2xi32>
    %1 = "tf.Sub"(%0, %arg2) : (tensor<1x3x2x2xi32>, tensor<i32>) -> tensor<1x3x2x2xi32>
    %2 = "tf.Cast"(%1) : (tensor<1x3x2x2xi32>) -> tensor<1x3x2x2xf32>
    %3 = "tf.Mul"(%2, %arg1) : (tensor<1x3x2x2xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
    return %3 : tensor<1x3x2x2xf32>
  }
  func.func private @quantized_conv2d_with_bias_and_relu_fn_0(%arg0: tensor<1x3x4x3xi8>, %arg1: tensor<2x3x3x2xi8>, %arg2: tensor<2xi32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<f32>, %arg6: tensor<i32>, %arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<f32>, %arg10: tensor<i32>) -> tensor<1x3x2x2xi8> {
    %cst = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x3x4x3xi8>) -> tensor<1x3x4x3xi32>
    %1 = "tf.Sub"(%0, %arg4) : (tensor<1x3x4x3xi32>, tensor<i32>) -> tensor<1x3x4x3xi32>
    %identity = "tf.Identity"(%arg1) : (tensor<2x3x3x2xi8>) -> tensor<2x3x3x2xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<2x3x3x2xi8>) -> tensor<2x3x3x2xi32>
    %3 = "tf.Sub"(%2, %arg6) : (tensor<2x3x3x2xi32>, tensor<i32>) -> tensor<2x3x3x2xi32>
    %4 = "tf.Conv2D"(%1, %3) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xi32>, tensor<2x3x3x2xi32>) -> tensor<1x3x2x2xi32>
    %5 = "tf.AddV2"(%4, %arg2) : (tensor<1x3x2x2xi32>, tensor<2xi32>) -> tensor<1x3x2x2xi32>
    %6 = "tf.Mul"(%arg3, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = "tf.Div"(%6, %arg9) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = "tf.Cast"(%5) {Truncate = false} : (tensor<1x3x2x2xi32>) -> tensor<1x3x2x2xf32>
    %9 = "tf.Mul"(%7, %8) : (tensor<f32>, tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %10 = "tf.Round"(%9) : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xi32>
    %12 = "tf.AddV2"(%11, %arg10) : (tensor<1x3x2x2xi32>, tensor<i32>) -> tensor<1x3x2x2xi32>
    %13 = "tf.Maximum"(%cst_0, %arg10) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = "tf.ClipByValue"(%12, %13, %cst) : (tensor<1x3x2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<1x3x2x2xi32>
    %15 = "tf.Cast"(%14) {Truncate = false} : (tensor<1x3x2x2xi32>) -> tensor<1x3x2x2xi8>
    return %15 : tensor<1x3x2x2xi8>
  }

// CHECK-LABEL: func @conv_with_bias_and_relu
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[CONST_3:.*]] = "tf.Const"() {value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
// CHECK-DAG: %[[CONST_4:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
// CHECK-DAG-SAME{LITERAL}: value = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]>
// CHECK-DAG: %[[CONST_5:.*]] = "tf.Const"() {value = dense<-128> : tensor<i8>} : () -> tensor<i8>
// CHECK-DAG: %[[CONST_6:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// CHECK-DAG: %[[CONST_7:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<1x1x1x2xi32>} : () -> tensor<1x1x1x2xi32>
// CHECK-DAG-SAME{LITERAL}: value = dense<[[[[-22016, -23680]]]]>
// CHECK-DAG: %[[CONST_8:.*]] = "tf.Const"() {value = dense<[162, 160]> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK: %[[PADV2_0:.*]] = "tf.PadV2"({{.*}}, %[[CONST_4]], %[[CONST_5]]) : (tensor<1x3x4x3xi8>, tensor<4x2xi32>, tensor<i8>) -> tensor<1x4x5x3xi8>
// CHECK: %[[XLACONVV2_0:.*]] = "tf.XlaConvV2"(%[[PADV2_0]], %[[CONST_6]], %[[CONST_0]], %[[CONST_3]], %[[CONST_1]], %[[CONST_1]], %[[CONST_2]])
// CHECK-SAME: (tensor<1x4x5x3xi8>, tensor<2x3x3x2xi8>, tensor<2xi32>, tensor<2x2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<i32>) -> tensor<1x3x2x2xi32>
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[XLACONVV2_0]], %[[CONST_7]]) : (tensor<1x3x2x2xi32>, tensor<1x1x1x2xi32>) -> tensor<1x3x2x2xi32>
// CHECK: %[[ADDV2_1:.*]] = "tf.AddV2"(%[[SUB_0]], %[[CONST_8]]) : (tensor<1x3x2x2xi32>, tensor<2xi32>) -> tensor<1x3x2x2xi32>
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1195 : i32}} {
  func.func @depthwise_conv_with_bias_and_relu6(%arg0: tensor<1x3x4x3xf32>) -> tensor<1x2x2x3xf32> {
    %cst = "tf.Const"() {value = dense<[129, 166, 221]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_0 = "tf.Const"() {value = dense<[[[[-84], [73], [24]], [[-102], [-28], [-94]], [[-127], [-82], [82]]], [[[-56], [67], [120]], [[45], [11], [-88]], [[-106], [77], [123]]]]> : tensor<2x3x3x1xi8>} : () -> tensor<2x3x3x1xi8>
    %cst_1 = "tf.Const"() {value = dense<0.587548196> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_3 = "tf.Const"() {value = dense<0.0235294122> : tensor<f32>} : () -> tensor<f32>
    %cst_4 = "tf.Const"() {value = dense<0.0751230493> : tensor<1xf32>} : () -> tensor<1xf32>
    %cst_5 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %cst_6 = "tf.Const"() {value = dense<0.0441384129> : tensor<f32>} : () -> tensor<f32>
    %cst_7 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_1, %cst_2) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x3x4x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x3x4x3xi8>
    %1 = "tf.PartitionedCall"(%0, %cst_0, %cst, %cst_1, %cst_2, %cst_4, %cst_5, %cst_6, %cst_7, %cst_3, %cst_2) {config = "", config_proto = "", executor_type = "", f = @quantized_depthwise_conv2d_with_bias_and_relu6_fn_0} : (tensor<1x3x4x3xi8>, tensor<2x3x3x1xi8>, tensor<3xi32>, tensor<f32>, tensor<i32>, tensor<1xf32>, tensor<1xi32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x3xi8>
    %2 = "tf.PartitionedCall"(%1, %cst_3, %cst_2) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<1x2x2x3xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x3xf32>
    return %2 : tensor<1x2x2x3xf32>
  }
  func.func private @quantize_i8(%arg0: tensor<1x3x4x3xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x3x4x3xi8> {
    %0 = "tf.Div"(%arg0, %arg1) : (tensor<1x3x4x3xf32>, tensor<f32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Round"(%0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %2 = "tf.Cast"(%1) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xi32>
    %3 = "tf.AddV2"(%2, %arg2) : (tensor<1x3x4x3xi32>, tensor<i32>) -> tensor<1x3x4x3xi32>
    %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x3x4x3xi32>) -> tensor<1x3x4x3xi8>
    return %4 : tensor<1x3x4x3xi8>
  }
  func.func private @dequantize_i8(%arg0: tensor<1x2x2x3xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x2x2x3xf32> {
    %0 = "tf.Cast"(%arg0) : (tensor<1x2x2x3xi8>) -> tensor<1x2x2x3xi32>
    %1 = "tf.Sub"(%0, %arg2) : (tensor<1x2x2x3xi32>, tensor<i32>) -> tensor<1x2x2x3xi32>
    %2 = "tf.Cast"(%1) : (tensor<1x2x2x3xi32>) -> tensor<1x2x2x3xf32>
    %3 = "tf.Mul"(%2, %arg1) : (tensor<1x2x2x3xf32>, tensor<f32>) -> tensor<1x2x2x3xf32>
    return %3 : tensor<1x2x2x3xf32>
  }
  func.func private @quantized_depthwise_conv2d_with_bias_and_relu6_fn_0(%arg0: tensor<1x3x4x3xi8>, %arg1: tensor<2x3x3x1xi8>, %arg2: tensor<3xi32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<1xf32>, %arg6: tensor<1xi32>, %arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<f32>, %arg10: tensor<i32>) -> tensor<1x2x2x3xi8> {
    %cst = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x3x4x3xi8>) -> tensor<1x3x4x3xi32>
    %1 = "tf.Sub"(%0, %arg4) : (tensor<1x3x4x3xi32>, tensor<i32>) -> tensor<1x3x4x3xi32>
    %identity = "tf.Identity"(%arg1) : (tensor<2x3x3x1xi8>) -> tensor<2x3x3x1xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<2x3x3x1xi8>) -> tensor<2x3x3x1xi32>
    %3 = "tf.Sub"(%2, %arg6) : (tensor<2x3x3x1xi32>, tensor<1xi32>) -> tensor<2x3x3x1xi32>
    %5 = "tf.Cast"(%1) {Truncate = false} : (tensor<1x3x4x3xi32>) -> tensor<1x3x4x3xf32>
    %6 = "tf.Cast"(%3) {Truncate = false} : (tensor<2x3x3x1xi32>) -> tensor<2x3x3x1xf32>
    %7 = "tf.DepthwiseConv2dNative"(%5, %6) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1xf32>) -> tensor<1x2x2x3xf32>
    %8 = "tf.Cast"(%7) : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xi32>
    %9 = "tf.AddV2"(%8, %arg2) : (tensor<1x2x2x3xi32>, tensor<3xi32>) -> tensor<1x2x2x3xi32>
    %10 = "tf.Mul"(%arg3, %arg5) : (tensor<f32>, tensor<1xf32>) -> tensor<1xf32>
    %11 = "tf.Div"(%10, %arg9) : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
    %12 = "tf.Cast"(%9) {Truncate = false} : (tensor<1x2x2x3xi32>) -> tensor<1x2x2x3xf32>
    %13 = "tf.Mul"(%11, %12) : (tensor<1xf32>, tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %14 = "tf.Round"(%13) : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xf32>
    %15 = "tf.Cast"(%14) {Truncate = false} : (tensor<1x2x2x3xf32>) -> tensor<1x2x2x3xi32>
    %16 = "tf.AddV2"(%15, %arg10) : (tensor<1x2x2x3xi32>, tensor<i32>) -> tensor<1x2x2x3xi32>
    %17 = "tf.Div"(%cst_1, %arg9) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %18 = "tf.Round"(%17) : (tensor<f32>) -> tensor<f32>
    %19 = "tf.Cast"(%18) : (tensor<f32>) -> tensor<i32>
    %20 = "tf.AddV2"(%19, %arg10) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %21 = "tf.Cast"(%20) : (tensor<i32>) -> tensor<i8>
    %22 = "tf.Cast"(%21) {Truncate = false} : (tensor<i8>) -> tensor<i8>
    %23 = "tf.Cast"(%22) {Truncate = false} : (tensor<i8>) -> tensor<i32>
    %24 = "tf.Maximum"(%cst_0, %arg10) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %25 = "tf.Minimum"(%cst, %23) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %26 = "tf.ClipByValue"(%16, %24, %25) : (tensor<1x2x2x3xi32>, tensor<i32>, tensor<i32>) -> tensor<1x2x2x3xi32>
    %27 = "tf.Cast"(%26) {Truncate = false} : (tensor<1x2x2x3xi32>) -> tensor<1x2x2x3xi8>
    return %27 : tensor<1x2x2x3xi8>
  }

// CHECK-LABEL: func @depthwise_conv_with_bias_and_relu6
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<-128> : tensor<i8>} : () -> tensor<i8>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<2x3x1x3xi8>} : () -> tensor<2x3x1x3xi8>
// CHECK-DAG: %[[CONST_3:.*]] = "tf.Const"() {value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG: %[[CONST_4:.*]] = "tf.Const"() {value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
// CHECK-DAG: %[[CONST_5:.*]] = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG: %[[CONST_6:.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[CONST_7:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<1x1x1x3xi32>} : () -> tensor<1x1x1x3xi32>
// CHECK-DAG-SAME{LITERAL}: value = dense<[[[[55040, -15104, -21376]]]]>
// CHECK-DAG: %[[CONST_8:.*]] = "tf.Const"() {value = dense<[129, 166, 221]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK: %[[PADV2_0:.*]] = "tf.PadV2"({{.*}}, %[[CONST_0]], %[[CONST_1]]) : (tensor<1x3x4x3xi8>, tensor<4x2xi32>, tensor<i8>) -> tensor<1x4x5x3xi8>
// CHECK: %[[XLACONVV2_0:.*]] = "tf.XlaConvV2"(%[[PADV2_0]], %[[CONST_2]], %[[CONST_3]], %[[CONST_4]], %[[CONST_5]], %[[CONST_5]], %[[CONST_6]])
// CHECK-SAME: (tensor<1x4x5x3xi8>, tensor<2x3x1x3xi8>, tensor<2xi32>, tensor<2x2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<i32>) -> tensor<1x2x2x3xi32>
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[XLACONVV2_0]], %[[CONST_7]]) : (tensor<1x2x2x3xi32>, tensor<1x1x1x3xi32>) -> tensor<1x2x2x3xi32>
// CHECK: %[[ADDV2_1:.*]] = "tf.AddV2"(%[[SUB_0]], %[[CONST_8]]) : (tensor<1x2x2x3xi32>, tensor<3xi32>) -> tensor<1x2x2x3xi32>
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1205 : i32}} {
  func.func @dynamic_shaped_conv2d_with_bias_and_relu6_inlined(%arg0: tensor<?x?x?x3xf32>) -> tensor<?x?x?x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {device = "", value = dense<[1.8772192, 1.82187414]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_2 = "tf.Const"() {device = "", value = dense<2> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
    %cst_3 = "tf.Const"() {device = "", value = dense<[161, 165]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_4 = "tf.Const"() {device = "", value = dense<0.587548196> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {device = "", value = dense<0.0235294122> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_4) {device = ""} : (tensor<?x?x?x3xf32>, tensor<f32>) -> tensor<?x?x?x3xf32>
    %1 = "tf.Round"(%0) {device = ""} : (tensor<?x?x?x3xf32>) -> tensor<?x?x?x3xf32>
    %2 = "tf.Cast"(%1) {device = ""} : (tensor<?x?x?x3xf32>) -> tensor<?x?x?x3xi32>
    %3 = "tf.AddV2"(%2, %cst_0) {device = ""} : (tensor<?x?x?x3xi32>, tensor<i32>) -> tensor<?x?x?x3xi32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<?x?x?x3xi32>) -> tensor<?x?x?x3xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<?x?x?x3xi8>) -> tensor<?x?x?x3xi32>
    %6 = "tf.Sub"(%5, %cst_0) {device = ""} : (tensor<?x?x?x3xi32>, tensor<i32>) -> tensor<?x?x?x3xi32>
    %identity = "tf.Identity"(%cst_2) : (tensor<2x3x3x2xi8>) -> tensor<2x3x3x2xi8>
    %cast_filter = "tf.Cast"(%identity) {Truncate = false} : (tensor<2x3x3x2xi8>) -> tensor<2x3x3x2xi32>
    %7 = "tf.Conv2D"(%6, %cast_filter) {device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<?x?x?x3xi32>, tensor<2x3x3x2xi32>) -> tensor<?x?x?x2xi32>
    %8 = "tf.AddV2"(%7, %cst_3) {device = ""} : (tensor<?x?x?x2xi32>, tensor<2xi32>) -> tensor<?x?x?x2xi32>
    %9 = "tf.Cast"(%8) {Truncate = false, device = ""} : (tensor<?x?x?x2xi32>) -> tensor<?x?x?x2xf32>
    %10 = "tf.Mul"(%9, %cst_1) {device = ""} : (tensor<?x?x?x2xf32>, tensor<2xf32>) -> tensor<?x?x?x2xf32>
    %11 = "tf.Round"(%10) {device = ""} : (tensor<?x?x?x2xf32>) -> tensor<?x?x?x2xf32>
    %12 = "tf.Cast"(%11) {Truncate = false, device = ""} : (tensor<?x?x?x2xf32>) -> tensor<?x?x?x2xi32>
    %13 = "tf.AddV2"(%12, %cst_0) {device = ""} : (tensor<?x?x?x2xi32>, tensor<i32>) -> tensor<?x?x?x2xi32>
    %14 = "tf.ClipByValue"(%13, %cst_0, %cst) {device = ""} : (tensor<?x?x?x2xi32>, tensor<i32>, tensor<i32>) -> tensor<?x?x?x2xi32>
    %15 = "tf.Cast"(%14) {Truncate = false, device = ""} : (tensor<?x?x?x2xi32>) -> tensor<?x?x?x2xi8>
    %16 = "tf.Cast"(%15) {device = ""} : (tensor<?x?x?x2xi8>) -> tensor<?x?x?x2xi32>
    %17 = "tf.Sub"(%16, %cst_0) {device = ""} : (tensor<?x?x?x2xi32>, tensor<i32>) -> tensor<?x?x?x2xi32>
    %18 = "tf.Cast"(%17) {device = ""} : (tensor<?x?x?x2xi32>) -> tensor<?x?x?x2xf32>
    %19 = "tf.Mul"(%18, %cst_5) {device = ""} : (tensor<?x?x?x2xf32>, tensor<f32>) -> tensor<?x?x?x2xf32>
    return %19 : tensor<?x?x?x2xf32>
  }

// CHECK-LABEL: func @dynamic_shaped_conv2d_with_bias_and_relu6_inlined
// CHECK-DAG: %[[filter:.*]] = "tf.Const"() {device = "", value = dense<2> : tensor<2x3x3x2xi8>} : () -> tensor<2x3x3x2xi8>
// CHECK-DAG: %[[input_shape:.*]] = "tf.Shape"({{.*}}) : (tensor<?x?x?x3xi8>) -> tensor<4xi32>
// CHECK-DAG: %[[input_dim_1:.*]] = "tf.StridedSlice"(%[[input_shape]], {{.*}}, {{.*}}, {{.*}}) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK-DAG: %[[input_dim_2:.*]] = "tf.StridedSlice"(%[[input_shape]], {{.*}}, {{.*}}, {{.*}}) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK-DAG: %[[padding_rank_1:.*]] = "tf.Concat"({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (tensor<i32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<8xi32>
// CHECK-DAG: %[[padding_rank_2:.*]] = "tf.Reshape"(%[[padding_rank_1]], {{.*}}) : (tensor<8xi32>, tensor<2xi64>) -> tensor<4x2xi32>
// CHECK-DAG: %[[input_padded:.*]] = "tf.PadV2"(%{{.*}}, %[[padding_rank_2]], {{.*}}) : (tensor<?x?x?x3xi8>, tensor<4x2xi32>, tensor<i8>) -> tensor<?x?x?x3xi8>
// CHECK: %[[conv_output:.*]] = "tf.XlaConvV2"(%[[input_padded]], %[[filter]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) {batch_group_count = 1 : i64, dimension_numbers = "{{.*}}", precision_config = ""} : (tensor<?x?x?x3xi8>, tensor<2x3x3x2xi8>, tensor<2xi32>, tensor<2x2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<i32>) -> tensor<?x?x?x2xi32>
// CHECK: %[[conv_output_sub:.*]] = "tf.Sub"(%[[conv_output]], {{.*}}) : (tensor<?x?x?x2xi32>, tensor<1x1x1x2xi32>) -> tensor<?x?x?x2xi32>
// CHECK: %[[conv_output_add:.*]] = "tf.AddV2"(%[[conv_output_sub]], {{.*}}) {device = ""} : (tensor<?x?x?x2xi32>, tensor<2xi32>) -> tensor<?x?x?x2xi32>
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @conv_with_filter_larger_than_1MB(%arg0: tensor<1x224x224x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<1x224x112x512xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {value = dense<2> : tensor<32x32x3x512xi8>} : () -> tensor<32x32x3x512xi8>
    %cst_0 = "tf.Const"() {value = dense<0.00117647066> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_2 = "tf.Const"() {value = dense<0.0027450982> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {value = dense<-19> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {value = dense<0.01> : tensor<512xf32>} : () -> tensor<512xf32>
    %cst_5 = "tf.Const"() {value = dense<0> : tensor<512xi32>} : () -> tensor<512xi32>
    %0 = "tf.PartitionedCall"(%arg0, %cst_0, %cst_1) {config = "", config_proto = "", executor_type = "", f = @quantize_i8} : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<i32>) -> tensor<1x224x224x3xi8>
    %1 = "tf.PartitionedCall"(%0, %cst, %cst_0, %cst_1, %cst_4, %cst_5, %cst_2, %cst_3) {config = "", config_proto = "", executor_type = "", f = @quantized_conv2d_with_relu_fn_0} : (tensor<1x224x224x3xi8>, tensor<32x32x3x512xi8>, tensor<f32>, tensor<i32>, tensor<512xf32>, tensor<512xi32>, tensor<f32>, tensor<i32>) -> tensor<1x224x112x512xi8>
    %2 = "tf.PartitionedCall"(%1, %cst_2, %cst_3) {config = "", config_proto = "", executor_type = "", f = @dequantize_i8} : (tensor<1x224x112x512xi8>, tensor<f32>, tensor<i32>) -> tensor<1x224x112x512xf32>
    return %2 : tensor<1x224x112x512xf32>
  }
  func.func private @quantize_i8(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x224x224x3xi8> {
    %0 = "tf.Div"(%arg0, %arg1) : (tensor<1x224x224x3xf32>, tensor<f32>) -> tensor<1x224x224x3xf32>
    %1 = "tf.Round"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
    %2 = "tf.Cast"(%1) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xi32>
    %3 = "tf.AddV2"(%2, %arg2) : (tensor<1x224x224x3xi32>, tensor<i32>) -> tensor<1x224x224x3xi32>
    %4 = "tf.Cast"(%3) {Truncate = false} : (tensor<1x224x224x3xi32>) -> tensor<1x224x224x3xi8>
    return %4 : tensor<1x224x224x3xi8>
  }
  func.func private @dequantize_i8(%arg0: tensor<1x224x112x512xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<1x224x112x512xf32> {
    %0 = "tf.Cast"(%arg0) : (tensor<1x224x112x512xi8>) -> tensor<1x224x112x512xi32>
    %1 = "tf.Sub"(%0, %arg2) : (tensor<1x224x112x512xi32>, tensor<i32>) -> tensor<1x224x112x512xi32>
    %2 = "tf.Cast"(%1) : (tensor<1x224x112x512xi32>) -> tensor<1x224x112x512xf32>
    %3 = "tf.Mul"(%2, %arg1) : (tensor<1x224x112x512xf32>, tensor<f32>) -> tensor<1x224x112x512xf32>
    return %3 : tensor<1x224x112x512xf32>
  }
  func.func private @quantized_conv2d_with_relu_fn_0(%arg0: tensor<1x224x224x3xi8>, %arg1: tensor<32x32x3x512xi8>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<512xf32>, %arg5: tensor<512xi32>, %arg6: tensor<f32>, %arg7: tensor<i32>) -> tensor<1x224x112x512xi8> {
    %cst = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x224x224x3xi8>) -> tensor<1x224x224x3xi32>
    %1 = "tf.Sub"(%0, %arg3) : (tensor<1x224x224x3xi32>, tensor<i32>) -> tensor<1x224x224x3xi32>
    %2 = "tf.Identity"(%arg1) : (tensor<32x32x3x512xi8>) -> tensor<32x32x3x512xi8>
    %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<32x32x3x512xi8>) -> tensor<32x32x3x512xi32>
    %4 = "tf.Sub"(%3, %arg5) : (tensor<32x32x3x512xi32>, tensor<512xi32>) -> tensor<32x32x3x512xi32>
    %5 = "tf.Conv2D"(%1, %4) {dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x224x224x3xi32>, tensor<32x32x3x512xi32>) -> tensor<1x224x112x512xi32>
    %6 = "tf.Mul"(%arg2, %arg4) : (tensor<f32>, tensor<512xf32>) -> tensor<512xf32>
    %7 = "tf.Div"(%6, %arg6) : (tensor<512xf32>, tensor<f32>) -> tensor<512xf32>
    %8 = "tf.Cast"(%5) {Truncate = false} : (tensor<1x224x112x512xi32>) -> tensor<1x224x112x512xf32>
    %9 = "tf.Mul"(%7, %8) : (tensor<512xf32>, tensor<1x224x112x512xf32>) -> tensor<1x224x112x512xf32>
    %10 = "tf.Round"(%9) : (tensor<1x224x112x512xf32>) -> tensor<1x224x112x512xf32>
    %11 = "tf.Cast"(%10) {Truncate = false} : (tensor<1x224x112x512xf32>) -> tensor<1x224x112x512xi32>
    %12 = "tf.AddV2"(%11, %arg7) : (tensor<1x224x112x512xi32>, tensor<i32>) -> tensor<1x224x112x512xi32>
    %13 = "tf.Maximum"(%cst_0, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %14 = "tf.ClipByValue"(%12, %13, %cst) : (tensor<1x224x112x512xi32>, tensor<i32>, tensor<i32>) -> tensor<1x224x112x512xi32>
    %15 = "tf.Cast"(%14) {Truncate = false} : (tensor<1x224x112x512xi32>) -> tensor<1x224x112x512xi8>
    return %15 : tensor<1x224x112x512xi8>
  }

// CHECK-LABEL: func @conv_with_filter_larger_than_1MB
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<-264192> : tensor<1x1x1x512xi32>} : () -> tensor<1x1x1x512xi32>
// CHECK: %[[PADV2_0:.*]] = "tf.PadV2"
// CHECK: %[[XLACONVV2_0:.*]] = "tf.XlaConvV2"(%[[PADV2_0]]
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[XLACONVV2_0]], %[[CONST]])
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @matmul_with_relu(%arg0: tensor<1x1024xf32> {tf_saved_model.index_path = ["serving_default_input_tensor:0"]}) -> (tensor<1x3xf32> {tf_saved_model.index_path = ["tf.PartitionedCall:0"]}) attributes {tf.entry_function = {inputs = "serving_default_input_tensor:0", outputs = "tf.PartitionedCall:0"}, tf_saved_model.exported_names = ["main"]} {
    %cst = "tf.Const"() {device = "", value = dense<3.08643539E-5> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-1.275000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {device = "", value = dense<1> : tensor<1024x3xi8>} : () -> tensor<1024x3xi8>
    %cst_3 = "tf.Const"() {device = "", value = dense<0.00392156653> : tensor<f32>} : () -> tensor<f32>
    %cst_4 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_5 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_3) {device = ""} : (tensor<1x1024xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<1x1024xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %3 = "tf.ClipByValue"(%2, %cst_1, %cst_5) {device = ""} : (tensor<1x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<1x1024xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<1x1024xi8>) -> tensor<1x1024xi32>
    %6 = "tf.Sub"(%5, %cst_4) {device = ""} : (tensor<1x1024xi32>, tensor<i32>) -> tensor<1x1024xi32>
    %7 = "tf.Identity"(%cst_2) {device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi32>
    %9 = "tf.MatMul"(%6, %8) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x1024xi32>, tensor<1024x3xi32>) -> tensor<1x3xi32>
    %10 = "tf.Cast"(%9) {Truncate = false, device = ""} : (tensor<1x3xi32>) -> tensor<1x3xf32>
    %11 = "tf.Mul"(%10, %cst) {device = ""} : (tensor<1x3xf32>, tensor<f32>) -> tensor<1x3xf32>
    %12 = "tf.Relu"(%11) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    return %12 : tensor<1x3xf32>
  }
// CHECK-LABEL: func @matmul_with_relu
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {device = "", value = dense<1> : tensor<1024x3xi8>} : () -> tensor<1024x3xi8>
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<-131072> : tensor<1x3xi32>} : () -> tensor<1x3xi32>
// CHECK: %[[MATMUL:.*]] = "tf.XlaDotV2"({{.*}}, %[[WEIGHT]])
// CHECK-SAME: (tensor<1x1024xi8>, tensor<1024x3xi8>) -> tensor<1x3xi32>
// CHECK: %[[SUB:.*]] = "tf.Sub"(%[[MATMUL]], %[[CONST]]) : (tensor<1x3xi32>, tensor<1x3xi32>) -> tensor<1x3xi32>
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @conv3d_with_static_shape(%arg0: tensor<1x3x4x3x3xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<1x3x2x3x2xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "tf.PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[4.57413898E-6, 4.56899261E-6]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-4.250000e+01> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<1> : tensor<2x3x3x3x2xi8>} : () -> tensor<2x3x3x3x2xi8>
    %cst_2 = "tf.Const"() {device = "", value = dense<0.00117643911> : tensor<f32>} : () -> tensor<f32>
    %cst_3 = "tf.Const"() {device = "", value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_4 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_2) {device = ""} : (tensor<1x3x4x3x3xf32>, tensor<f32>) -> tensor<1x3x4x3x3xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<1x3x4x3x3xf32>, tensor<f32>) -> tensor<1x3x4x3x3xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<1x3x4x3x3xf32>) -> tensor<1x3x4x3x3xf32>
    %3 = "tf.ClipByValue"(%2, %cst_4, %cst_5) {device = ""} : (tensor<1x3x4x3x3xf32>, tensor<f32>, tensor<f32>) -> tensor<1x3x4x3x3xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<1x3x4x3x3xf32>) -> tensor<1x3x4x3x3xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<1x3x4x3x3xi8>) -> tensor<1x3x4x3x3xi32>
    %6 = "tf.Sub"(%5, %cst_3) {device = ""} : (tensor<1x3x4x3x3xi32>, tensor<i32>) -> tensor<1x3x4x3x3xi32>
    %7 = "tf.Identity"(%cst_1) {device = ""} : (tensor<2x3x3x3x2xi8>) -> tensor<2x3x3x3x2xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<2x3x3x3x2xi8>) -> tensor<2x3x3x3x2xi32>
    %9 = "tf.Cast"(%6) {Truncate = false, device = ""} : (tensor<1x3x4x3x3xi32>) -> tensor<1x3x4x3x3xf32>
    %10 = "tf.Cast"(%8) {Truncate = false, device = ""} : (tensor<2x3x3x3x2xi32>) -> tensor<2x3x3x3x2xf32>
    %11 = "tf.Conv3D"(%9, %10) {device = "", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1, 1]} : (tensor<1x3x4x3x3xf32>, tensor<2x3x3x3x2xf32>) -> tensor<1x3x2x3x2xf32>
    %12 = "tf.Cast"(%11) {device = ""} : (tensor<1x3x2x3x2xf32>) -> tensor<1x3x2x3x2xi32>
    %13 = "tf.Cast"(%12) {Truncate = false, device = ""} : (tensor<1x3x2x3x2xi32>) -> tensor<1x3x2x3x2xf32>
    %14 = "tf.Mul"(%13, %cst) {device = ""} : (tensor<1x3x2x3x2xf32>, tensor<2xf32>) -> tensor<1x3x2x3x2xf32>
    %15 = "tf.Identity"(%14) {device = ""} : (tensor<1x3x2x3x2xf32>) -> tensor<1x3x2x3x2xf32>
    return %15 : tensor<1x3x2x3x2xf32>
  }

// CHECK-LABEL: func @conv3d_with_static_shape
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {device = "", value = dense<1> : tensor<2x3x3x3x2xi8>} : () -> tensor<2x3x3x3x2xi8>
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {{.*}} : () -> tensor<5x2xi32>
// CHECK-DAG-SAME{LITERAL}: value = dense<[[0, 0], [0, 1], [0, 1], [1, 1], [0, 0]]> : tensor<5x2xi32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<-43> : tensor<i8>} : () -> tensor<i8>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<-2322> : tensor<1x1x1x1x2xi32>} : () -> tensor<1x1x1x1x2xi32>

// CHECK: %[[PAD:.*]] = "tf.PadV2"({{.*}}, %[[CONST]], %[[CONST_1]])
// CHECK: %[[CONV:.*]] = "tf.XlaConvV2"(%[[PAD]], %[[WEIGHT]]
// CHECK-SAME: (tensor<1x4x5x5x3xi8>, tensor<2x3x3x3x2xi8>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<1x3x2x3x2xi32>
// CHECK: %[[SUB:.*]] = "tf.Sub"(%[[CONV]], %[[CONST_2]])
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @conv3d_with_dynamic_shape(%arg0: tensor<?x?x?x?x3xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x?x?x?x2xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "tf.PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[4.57413898E-6, 4.56899261E-6]> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-4.250000e+01> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<[4987, 41620]> : tensor<2xi32>} : () -> tensor<2xi32>
    %cst_2 = "tf.Const"() {device = "", value = dense<1> : tensor<2x3x3x3x2xi8>} : () -> tensor<2x3x3x3x2xi8>
    %cst_3 = "tf.Const"() {device = "", value = dense<0.00117643911> : tensor<f32>} : () -> tensor<f32>
    %cst_4 = "tf.Const"() {device = "", value = dense<-43> : tensor<i32>} : () -> tensor<i32>
    %cst_5 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_6 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_3) {device = ""} : (tensor<?x?x?x?x3xf32>, tensor<f32>) -> tensor<?x?x?x?x3xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<?x?x?x?x3xf32>, tensor<f32>) -> tensor<?x?x?x?x3xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<?x?x?x?x3xf32>) -> tensor<?x?x?x?x3xf32>
    %3 = "tf.ClipByValue"(%2, %cst_5, %cst_6) {device = ""} : (tensor<?x?x?x?x3xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x?x?x3xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<?x?x?x?x3xf32>) -> tensor<?x?x?x?x3xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<?x?x?x?x3xi8>) -> tensor<?x?x?x?x3xi32>
    %6 = "tf.Sub"(%5, %cst_4) {device = ""} : (tensor<?x?x?x?x3xi32>, tensor<i32>) -> tensor<?x?x?x?x3xi32>
    %7 = "tf.Identity"(%cst_2) {device = ""} : (tensor<2x3x3x3x2xi8>) -> tensor<2x3x3x3x2xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<2x3x3x3x2xi8>) -> tensor<2x3x3x3x2xi32>
    %9 = "tf.Cast"(%6) {Truncate = false, device = ""} : (tensor<?x?x?x?x3xi32>) -> tensor<?x?x?x?x3xf32>
    %10 = "tf.Cast"(%8) {Truncate = false, device = ""} : (tensor<2x3x3x3x2xi32>) -> tensor<2x3x3x3x2xf32>
    %11 = "tf.Conv3D"(%9, %10) {device = "", dilations = [1, 1, 1, 1, 1], padding = "SAME", strides = [1, 1, 2, 1, 1]} : (tensor<?x?x?x?x3xf32>, tensor<2x3x3x3x2xf32>) -> tensor<?x?x?x?x2xf32>
    %12 = "tf.Cast"(%11) {device = ""} : (tensor<?x?x?x?x2xf32>) -> tensor<?x?x?x?x2xi32>
    %13 = "tf.AddV2"(%12, %cst_1) {device = ""} : (tensor<?x?x?x?x2xi32>, tensor<2xi32>) -> tensor<?x?x?x?x2xi32>
    %14 = "tf.Cast"(%13) {Truncate = false, device = ""} : (tensor<?x?x?x?x2xi32>) -> tensor<?x?x?x?x2xf32>
    %15 = "tf.Mul"(%14, %cst) {device = ""} : (tensor<?x?x?x?x2xf32>, tensor<2xf32>) -> tensor<?x?x?x?x2xf32>
    %16 = "tf.Identity"(%15) {device = ""} : (tensor<?x?x?x?x2xf32>) -> tensor<?x?x?x?x2xf32>
    return %16 : tensor<?x?x?x?x2xf32>
  }

// CHECK-LABEL: func @conv3d_with_dynamic_shape
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {device = "", value = dense<1> : tensor<2x3x3x3x2xi8>} : () -> tensor<2x3x3x3x2xi8>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<-43> : tensor<i8>} : () -> tensor<i8>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<-2322> : tensor<1x1x1x1x2xi32>} : () -> tensor<1x1x1x1x2xi32>

// CHECK: %[[CONCAT:.*]] = "tf.Concat"({{.*}})
// CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%[[CONCAT]], {{.*}}) : (tensor<10xi32>, tensor<2xi64>) -> tensor<5x2xi32>
// CHECK: %[[PAD:.*]] = "tf.PadV2"({{.*}}, %[[RESHAPE]], %[[CONST_1]])
// CHECK: %[[CONV:.*]] = "tf.XlaConvV2"(%[[PAD]], %[[WEIGHT]]
// CHECK-SAME: (tensor<?x?x?x?x3xi8>, tensor<2x3x3x3x2xi8>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<?x?x?x?x2xi32>
// CHECK: %[[SUB:.*]] = "tf.Sub"(%[[CONV]], %[[CONST_2]])
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @batch_matmul(%arg0: tensor<20x30x64x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<20x30x64x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "tf.PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<3.08784583E-5> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-1.275000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {device = "", value = dense<1> : tensor<20x30x1024x3xi8>} : () -> tensor<20x30x1024x3xi8>
    %cst_3 = "tf.Const"() {device = "", value = dense<0.00392156886> : tensor<f32>} : () -> tensor<f32>
    %cst_4 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_5 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_3) {device = ""} : (tensor<20x30x64x1024xf32>, tensor<f32>) -> tensor<20x30x64x1024xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<20x30x64x1024xf32>, tensor<f32>) -> tensor<20x30x64x1024xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<20x30x64x1024xf32>) -> tensor<20x30x64x1024xf32>
    %3 = "tf.ClipByValue"(%2, %cst_1, %cst_5) {device = ""} : (tensor<20x30x64x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<20x30x64x1024xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<20x30x64x1024xf32>) -> tensor<20x30x64x1024xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<20x30x64x1024xi8>) -> tensor<20x30x64x1024xi32>
    %6 = "tf.Sub"(%5, %cst_4) {device = ""} : (tensor<20x30x64x1024xi32>, tensor<i32>) -> tensor<20x30x64x1024xi32>
    %7 = "tf.Identity"(%cst_2) {device = ""} : (tensor<20x30x1024x3xi8>) -> tensor<20x30x1024x3xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<20x30x1024x3xi8>) -> tensor<20x30x1024x3xi32>
    %9 = "tf.BatchMatMulV2"(%6, %8) {adj_x = false, adj_y = false, device = ""} : (tensor<20x30x64x1024xi32>, tensor<20x30x1024x3xi32>) -> tensor<20x30x64x3xi32>
    %10 = "tf.Cast"(%9) {Truncate = false, device = ""} : (tensor<20x30x64x3xi32>) -> tensor<20x30x64x3xf32>
    %11 = "tf.Mul"(%10, %cst) {device = ""} : (tensor<20x30x64x3xf32>, tensor<f32>) -> tensor<20x30x64x3xf32>
    %12 = "tf.Relu"(%11) {device = ""} : (tensor<20x30x64x3xf32>) -> tensor<20x30x64x3xf32>
    %13 = "tf.Identity"(%12) {device = ""} : (tensor<20x30x64x3xf32>) -> tensor<20x30x64x3xf32>
    return %13 : tensor<20x30x64x3xf32>
  }

// CHECK-LABEL: func @batch_matmul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<-131072> : tensor<20x30x1x3xi32>} : () -> tensor<20x30x1x3xi32>
// CHECK: %[[CAST:.*]] = "tf.Cast"
// CHECK: %[[XLADOTV2_0:.*]] = "tf.XlaDotV2"(%[[CAST]]
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[XLADOTV2_0]], %[[CONST]]) : (tensor<20x30x64x3xi32>, tensor<20x30x1x3xi32>) -> tensor<20x30x64x3xi32>
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @broadcasting_weight_batch_matmul(%arg0: tensor<2x1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<3.08762283E-5> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-1.275000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {device = "", value = dense<[-241, 5894, -3771]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_3 = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<1024x3xi8>} : () -> tensor<1024x3xi8>
    %cst_4 = "tf.Const"() {device = "", value = dense<0.00392156513> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_4) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<2x1x1024xf32>) -> tensor<2x1x1024xf32>
    %3 = "tf.ClipByValue"(%2, %cst_1, %cst_6) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<2x1x1024xf32>) -> tensor<2x1x1024xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<2x1x1024xi8>) -> tensor<2x1x1024xi32>
    %6 = "tf.Sub"(%5, %cst_5) {device = ""} : (tensor<2x1x1024xi32>, tensor<i32>) -> tensor<2x1x1024xi32>
    %7 = "tf.Identity"(%cst_3) {device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi32>
    %9 = "tf.BatchMatMulV2"(%6, %8) {adj_x = false, adj_y = false, device = ""} : (tensor<2x1x1024xi32>, tensor<1024x3xi32>) -> tensor<2x1x3xi32>
    %10 = "tf.AddV2"(%9, %cst_2) {device = ""} : (tensor<2x1x3xi32>, tensor<3xi32>) -> tensor<2x1x3xi32>
    %11 = "tf.Cast"(%10) {Truncate = false, device = ""} : (tensor<2x1x3xi32>) -> tensor<2x1x3xf32>
    %12 = "tf.Mul"(%11, %cst) {device = ""} : (tensor<2x1x3xf32>, tensor<f32>) -> tensor<2x1x3xf32>
    %13 = "tf.Identity"(%12) {device = ""} : (tensor<2x1x3xf32>) -> tensor<2x1x3xf32>
    %14 = "tf.Identity"(%13) {device = ""} : (tensor<2x1x3xf32>) -> tensor<2x1x3xf32>
    return %14 : tensor<2x1x3xf32>
  }

// CHECK-LABEL: func @broadcasting_weight_batch_matmul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<[2, 1024, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK: %[[CAST:.*]] = "tf.Cast"
// CHECK: %[[BROADCAST_TO:.*]] = "tf.BroadcastTo"({{.*}}, %[[CONST]]) : (tensor<1024x3xi8>, tensor<3xi64>) -> tensor<2x1024x3xi8>
// CHECK: %[[XLADOTV2_0:.*]] = "tf.XlaDotV2"(%[[CAST]], %[[BROADCAST_TO]])
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @broadcasting_input_batch_matmul(%arg0: tensor<2x1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<2x2x1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<3.08762283E-5> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-1.275000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {device = "", value = dense<[-241, 5894, -3771]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_3 = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<2x2x1024x3xi8>} : () -> tensor<2x2x1024x3xi8>
    %cst_4 = "tf.Const"() {device = "", value = dense<0.00392156513> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_4) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<2x1x1024xf32>) -> tensor<2x1x1024xf32>
    %3 = "tf.ClipByValue"(%2, %cst_1, %cst_6) {device = ""} : (tensor<2x1x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<2x1x1024xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<2x1x1024xf32>) -> tensor<2x1x1024xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<2x1x1024xi8>) -> tensor<2x1x1024xi32>
    %6 = "tf.Sub"(%5, %cst_5) {device = ""} : (tensor<2x1x1024xi32>, tensor<i32>) -> tensor<2x1x1024xi32>
    %7 = "tf.Identity"(%cst_3) {device = ""} : (tensor<2x2x1024x3xi8>) -> tensor<2x2x1024x3xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<2x2x1024x3xi8>) -> tensor<2x2x1024x3xi32>
    %9 = "tf.BatchMatMulV2"(%6, %8) {adj_x = false, adj_y = false, device = ""} : (tensor<2x1x1024xi32>, tensor<2x2x1024x3xi32>) -> tensor<2x2x1x3xi32>
    %10 = "tf.AddV2"(%9, %cst_2) {device = ""} : (tensor<2x2x1x3xi32>, tensor<3xi32>) -> tensor<2x2x1x3xi32>
    %11 = "tf.Cast"(%10) {Truncate = false, device = ""} : (tensor<2x2x1x3xi32>) -> tensor<2x2x1x3xf32>
    %12 = "tf.Mul"(%11, %cst) {device = ""} : (tensor<2x2x1x3xf32>, tensor<f32>) -> tensor<2x2x1x3xf32>
    %13 = "tf.Identity"(%12) {device = ""} : (tensor<2x2x1x3xf32>) -> tensor<2x2x1x3xf32>
    %14 = "tf.Identity"(%13) {device = ""} : (tensor<2x2x1x3xf32>) -> tensor<2x2x1x3xf32>
    return %14 : tensor<2x2x1x3xf32>
  }

// CHECK-LABEL: func @broadcasting_input_batch_matmul
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {device = "", value = {{.*}} : tensor<2x2x1024x3xi8>} : () -> tensor<2x2x1024x3xi8>
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<[2, 2, 1, 1024]> : tensor<4xi64>} : () -> tensor<4xi64>
// CHECK: %[[CAST:.*]] = "tf.Cast"
// CHECK: %[[BROADCAST_TO:.*]] = "tf.BroadcastTo"(%[[CAST]], %[[CONST]]) : (tensor<2x1x1024xi8>, tensor<4xi64>) -> tensor<2x2x1x1024xi8>
// CHECK: %[[XLADOTV2_0:.*]] = "tf.XlaDotV2"(%[[BROADCAST_TO]], %[[WEIGHT]])
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1213 : i32}, tf_saved_model.semantics} {
  func.func @dynamic_shape_batch_matmul(%arg0: tensor<?x1x1024xf32> {tf_saved_model.index_path = ["input_tensor"]}) -> (tensor<?x1x3xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input_tensor:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<3.08762283E-5> : tensor<f32>} : () -> tensor<f32>
    %cst_0 = "tf.Const"() {device = "", value = dense<-1.275000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_1 = "tf.Const"() {device = "", value = dense<-1.280000e+02> : tensor<f32>} : () -> tensor<f32>
    %cst_2 = "tf.Const"() {device = "", value = dense<[-241, 5894, -3771]> : tensor<3xi32>} : () -> tensor<3xi32>
    %cst_3 = "tf.Const"() {device = "", value = dense_resource<__elided__> : tensor<1024x3xi8>} : () -> tensor<1024x3xi8>
    %cst_4 = "tf.Const"() {device = "", value = dense<0.00392156513> : tensor<f32>} : () -> tensor<f32>
    %cst_5 = "tf.Const"() {device = "", value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %cst_6 = "tf.Const"() {device = "", value = dense<1.270000e+02> : tensor<f32>} : () -> tensor<f32>
    %0 = "tf.Div"(%arg0, %cst_4) {device = ""} : (tensor<?x1x1024xf32>, tensor<f32>) -> tensor<?x1x1024xf32>
    %1 = "tf.AddV2"(%0, %cst_0) {device = ""} : (tensor<?x1x1024xf32>, tensor<f32>) -> tensor<?x1x1024xf32>
    %2 = "tf.Floor"(%1) {device = ""} : (tensor<?x1x1024xf32>) -> tensor<?x1x1024xf32>
    %3 = "tf.ClipByValue"(%2, %cst_1, %cst_6) {device = ""} : (tensor<?x1x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<?x1x1024xf32>
    %4 = "tf.Cast"(%3) {Truncate = false, device = ""} : (tensor<?x1x1024xf32>) -> tensor<?x1x1024xi8>
    %5 = "tf.Cast"(%4) {Truncate = false, device = ""} : (tensor<?x1x1024xi8>) -> tensor<?x1x1024xi32>
    %6 = "tf.Sub"(%5, %cst_5) {device = ""} : (tensor<?x1x1024xi32>, tensor<i32>) -> tensor<?x1x1024xi32>
    %7 = "tf.Identity"(%cst_3) {device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi8>
    %8 = "tf.Cast"(%7) {Truncate = false, device = ""} : (tensor<1024x3xi8>) -> tensor<1024x3xi32>
    %9 = "tf.BatchMatMulV2"(%6, %8) {adj_x = false, adj_y = false, device = ""} : (tensor<?x1x1024xi32>, tensor<1024x3xi32>) -> tensor<?x1x3xi32>
    %10 = "tf.AddV2"(%9, %cst_2) {device = ""} : (tensor<?x1x3xi32>, tensor<3xi32>) -> tensor<?x1x3xi32>
    %11 = "tf.Cast"(%10) {Truncate = false, device = ""} : (tensor<?x1x3xi32>) -> tensor<?x1x3xf32>
    %12 = "tf.Mul"(%11, %cst) {device = ""} : (tensor<?x1x3xf32>, tensor<f32>) -> tensor<?x1x3xf32>
    %13 = "tf.Identity"(%12) {device = ""} : (tensor<?x1x3xf32>) -> tensor<?x1x3xf32>
    %14 = "tf.Identity"(%13) {device = ""} : (tensor<?x1x3xf32>) -> tensor<?x1x3xf32>
    return %14 : tensor<?x1x3xf32>
  }

// CHECK-LABEL: func @dynamic_shape_batch_matmul
// CHECK-DAG: %[[CONST:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK-DAG: %[[CONST_3:.*]] = "tf.Const"() {value = dense<[1024, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK-DAG: %[[CONST_4:.*]] = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
// CHECK-DAG: %[[CONST_5:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG: %[[WEIGHT:.*]] = "tf.Const"() {device = "", value = {{.*}} : tensor<1024x3xi8>} : () -> tensor<1024x3xi8>
// CHECK: %[[CAST:.*]] = "tf.Cast"({{.*}}) {Truncate = false, device = ""} : (tensor<?x1x1024xf32>) -> tensor<?x1x1024xi8>
// CHECK: %[[SHAPE:.*]] = "tf.Shape"(%[[CAST]]) : (tensor<?x1x1024xi8>) -> tensor<3xi64>
// CHECK: %[[SLICE_1:.*]] = "tf.Slice"(%[[SHAPE]], %[[CONST]], %[[CONST_2]]) : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
// CHECK: %[[SLICE_2:.*]] = "tf.Slice"(%[[SHAPE]], %[[CONST_2]], %[[CONST_1]]) : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi64>
// CHECK: %[[BROADCAST_ARGS:.*]] = "tf.BroadcastArgs"(%[[SLICE_1]], %[[CONST_4]]) : (tensor<1xi64>, tensor<0xi64>) -> tensor<1xi64>
// CHECK: %[[CONCAT_1:.*]] = "tf.Concat"(%[[CONST_5]], %[[BROADCAST_ARGS]], %[[SLICE_2]]) : (tensor<i32>, tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK: %[[CONCAT_2:.*]] = "tf.Concat"(%[[CONST_5]], %[[BROADCAST_ARGS]], %[[CONST_3]]) : (tensor<i32>, tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK: %[[BROADCAST_1:.*]] = "tf.BroadcastTo"(%[[CAST]], %[[CONCAT_1]]) : (tensor<?x1x1024xi8>, tensor<3xi64>) -> tensor<?x1x1024xi8>
// CHECK: %[[BROADCAST_2:.*]] = "tf.BroadcastTo"(%[[WEIGHT]], %[[CONCAT_2]]) : (tensor<1024x3xi8>, tensor<3xi64>) -> tensor<?x1024x3xi8>
// CHECK: %[[DOT:.*]] = "tf.XlaDotV2"(%[[BROADCAST_1]], %[[BROADCAST_2]])
}
