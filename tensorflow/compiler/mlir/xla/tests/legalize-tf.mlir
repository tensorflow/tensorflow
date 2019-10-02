// RUN: tf-opt -xla-legalize-tf %s | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// BatchNorm op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: fusedBatchNorm_notraining
func @fusedBatchNorm_notraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK-NEXT: "xla_hlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: fusedBatchNorm_training
func @fusedBatchNorm_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // TODO(riverriddle) Support training.
  // CHECK-NEXT: "tf.FusedBatchNorm"
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true}  : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

//===----------------------------------------------------------------------===//
// Bias op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @biasAdd_NHWC
func @biasAdd_NHWC(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK-NEXT: %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

// CHECK-LABEL: func @biasAdd_NCHW
func @biasAdd_NCHW(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  // CHECK-NEXT: %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW"} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

// In the next two tests, the replacement fails because the bias dimension does
// not have the same size as the feature dimension.

// CHECK-LABEL: func @biasAdd_NHWC_invalid
func @biasAdd_NHWC_invalid(%arg0: tensor<1x32x10x2xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x2xi32> {
  // CHECK-NOT: xla_hlo.add
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x10x2xi32>, tensor<32xi32>) -> tensor<1x32x10x2xi32>
  return %0 : tensor<1x32x10x2xi32>
}

// CHECK-LABEL: func @biasAdd_NCHW_invalid
func @biasAdd_NCHW_invalid(%arg0: tensor<1x10x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x10x10x32xi32> {
  // CHECK-NOT: xla_hlo.add
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NCHW"} : (tensor<1x10x10x32xi32>, tensor<32xi32>) -> tensor<1x10x10x32xi32>
  return %0 : tensor<1x10x10x32xi32>
}

//===----------------------------------------------------------------------===//
// Binary op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @add
func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %[[SUM0:.*]] = xla_hlo.add %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  %[[SUM1:.*]] = xla_hlo.add %[[SUM0]], %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %[[SUM1]] : tensor<2xi32>
  %0 = "tf.Add"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %1 = "tf.AddV2"(%0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %1: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_add
func @broadcast_add(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @broadcast_multi_dim_add
func @broadcast_multi_dim_add(%arg0: tensor<4x1x1xi32>, %arg1: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
  // CHECK-NEXT: "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>}
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
  return %0: tensor<4x4x4x4xi32>
}

// CHECK-LABEL: func @div
func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.div %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Div"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_div
func @broadcast_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: "xla_hlo.div"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @maximum
func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK:  xla_hlo.max %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @minimum
func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK:  xla_hlo.min %arg0, %arg1 : tensor<4xf32>
  %0 = "tf.Minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @mul
func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.mul %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Mul"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_mul
func @broadcast_mul(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: "xla_hlo.mul"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @real_div
func @real_div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.div %arg0, %arg0 : tensor<2xi32>
  %0 = "tf.RealDiv"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_real_div
func @broadcast_real_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: "xla_hlo.div"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.RealDiv"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}

// CHECK-LABEL: func @sub
func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT:  %0 = xla_hlo.sub %arg0, %arg0 : tensor<2xi32>
  // CHECK-NEXT:  return %0 : tensor<2xi32>
  %0 = "tf.Sub"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @broadcast_sub
func @broadcast_sub(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  // CHECK-NEXT: "xla_hlo.sub"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0: tensor<1x2xi32>
}


//===----------------------------------------------------------------------===//
// Compare op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @greater
func @greater(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GT"}
  %0 = "tf.Greater"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_greater
func @broadcast_greater(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "GT"}
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @greater_equal
func @greater_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GE"}
  %0 = "tf.GreaterEqual"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_greater_equal
func @broadcast_greater_equal(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "GE"}
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @less
func @less(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT"}
  %0 = "tf.Less"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_less
func @broadcast_less(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "LT"}
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

// CHECK-LABEL: func @less_equal
func @less_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LE"}
  %0 = "tf.LessEqual"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0: tensor<2xi1>
}

// CHECK-LABEL: func @broadcast_less_equal
func @broadcast_less_equal(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  // CHECK-NEXT: "xla_hlo.compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "LE"}
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0: tensor<1x2xi1>
}

//===----------------------------------------------------------------------===//
// Concat op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @concat_v2
func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) {N = 2 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  return %1 : tensor<6x3xf32>
}

// CHECK-LABEL: func @concat_v2_neg_axis
func @concat_v2_neg_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>

  %axis = "tf.Const"() { value = dense<-2> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) {N = 2 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  return %1 : tensor<6x3xf32>
}

// CHECK-LABEL: func @concat_v2_1d_axis
func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 1 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x6xf32>

  %axis = "tf.Const"() { value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) {N = 2 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<1xi64>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

// CHECK-LABEL: func @concat_v2_non_const_axis
func @concat_v2_non_const_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %axis: tensor<i64>) -> tensor<3x6xf32> {
  // CHECK: "tf.ConcatV2"

  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) {N = 2 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<3x6xf32>
  return %1 : tensor<3x6xf32>
}

// CHECK-LABEL: func @concat_v2_unranked
func @concat_v2_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.ConcatV2"

  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) {N = 2 : i64} : (tensor<*xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// Identity op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @identity
func @identity(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT:  return %arg0 : tensor<1xi32>
  %0 = "tf.Identity"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Nullary op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @const
func @const() -> tensor<2xi32> {
  // CHECK-NEXT: "xla_hlo.constant"() {value = dense<0> : tensor<2xi32>}
  %0 = "tf.Const"() {device = "", name = "", dtype = "tfdtype$DT_INT32", value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
  return %0: tensor<2xi32>
}

//===----------------------------------------------------------------------===//
// Matmul op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: matmul_notranspose
func @matmul_notranspose(%arg0: tensor<5x7xf32>, %arg1: tensor<7x11xf32>) -> tensor<5x11xf32> {
  // CHECK: "xla_hlo.dot"(%arg0, %arg1)
  %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<5x7xf32>, tensor<7x11xf32>) -> tensor<5x11xf32>

  return %0 : tensor<5x11xf32>
}

//===----------------------------------------------------------------------===//
// MaxPool op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: maxpool_valid_padding
// CHECK-SAME: %[[ARG:.*]]: tensor
func @maxpool_valid_padding(%arg0: tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32> {
  // CHECK: %[[INIT:.*]] = "xla_hlo.constant"() {value = dense<-2147483648> : tensor<i32>}
  // CHECK: "xla_hlo.reduce_window"(%[[ARG]], %[[INIT]])
  // CHECK: xla_hlo.max
  // CHECK: xla_hlo.return
  // CHECK: {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>}

  %0 = "tf.MaxPool"(%arg0) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 4, 4, 1]} : (tensor<2x12x20x7xi32>) -> tensor<2x3x5x7xi32>
  return %0 : tensor<2x3x5x7xi32>
}

//===----------------------------------------------------------------------===//
// Pack op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @pack
func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "xla_hlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "xla_hlo.reshape"({{.*}}) : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: "xla_hlo.concatenate"({{.*}}) {dimension = 0 : i64} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  %0 = "tf.Pack"(%arg0, %arg1) {N = 2 : i64} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

//===----------------------------------------------------------------------===//
// Relu op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @relu
func @relu(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: %[[ZERO:.*]] = "xla_hlo.constant"() {value = dense<0> : tensor<1xi32>}
  // CHECK-NEXT: xla_hlo.max %[[ZERO]], %arg0 : tensor<1xi32>
  %0 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

// CHECK-LABEL: func @relu_non_static_input
func @relu_non_static_input(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: tf.Relu
  %0 = "tf.Relu"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  return %0: tensor<?xi32>
}

// CHECK-LABEL: func @relu6
func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: %[[ZERO:.*]] = "xla_hlo.constant"() {value = dense<0> : tensor<1xi32>}
  // CHECK-NEXT: %[[SIX:.*]] = "xla_hlo.constant"() {value = dense<6> : tensor<1xi32>}
  // CHECK-NEXT: "xla_hlo.clamp"(%[[ZERO]], %arg0, %[[SIX]]) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Relu6"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  return %0: tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Select op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @select
func @select(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK-NEXT: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}

// CHECK-LABEL: func @select_float
func @select_float(%arg0: tensor<2xi1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-NEXT: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0: tensor<2xf32>
}

// CHECK-LABEL: func @select_multidimensional
func @select_multidimensional(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi32>, %arg2: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // CHECK-NEXT: "xla_hlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<3x2xi1>, tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  return %0: tensor<3x2xi32>
}

//===----------------------------------------------------------------------===//
// Softmax op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @simple_softmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>)
func @simple_softmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {

  // Verify reduce op for max computation and its body.
  // CHECK: %[[NEG_INF:.*]] = "xla_hlo.constant"() {value = dense<0xFF800000> : tensor<f32>}
  // CHECK: %[[MAX:.*]] = "xla_hlo.reduce"(%[[ARG0]], %[[NEG_INF]])
  // CHECK:  xla_hlo.max
  // CHECK: "xla_hlo.return"
  // CHECK: {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>

  // CHECK: %[[SHIFTED_INP:.*]] = "xla_hlo.sub"(%[[ARG0]], %[[MAX]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: %[[EXP:.*]] = "xla_hlo.exp"(%[[SHIFTED_INP]])
  // CHECK: %[[CASTED_EXP:.*]] = "xla_hlo.convert"(%[[EXP]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // Verify reduce op for summation and its body.
  // CHECK: %[[ZERO:.*]] = "xla_hlo.constant"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: %[[SUM:.*]] = "xla_hlo.reduce"(%[[CASTED_EXP]], %[[ZERO]])
  // CHECK:  xla_hlo.add
  // CHECK: "xla_hlo.return"
  // CHECK: {dimensions = dense<1> : tensor<1xi64>}
  // CHECK: %[[CASTED_SUM:.*]] = "xla_hlo.convert"(%[[SUM]]) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: %[[RESULT:.*]] = "xla_hlo.div"(%[[EXP]], %[[CASTED_SUM]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // return %[[RESULT]]

  %0 = "tf.Softmax"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>
}

// CHECK-LABEL: bf16_softmax
func @bf16_softmax(%arg0: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
  // Verify that conversion to f32 and then back to bf16 are introduced.

  // CHECK: "xla_hlo.convert"({{.*}}) : (tensor<2x3xbf16>) -> tensor<2x3xf32>
  // CHECK: "xla_hlo.convert"({{.*}}) : (tensor<2xf32>) -> tensor<2xbf16>

  %0 = "tf.Softmax"(%arg0) : (tensor<2x3xbf16>) -> tensor<2x3xbf16>
  return %0: tensor<2x3xbf16>
}

// CHECK-LABEL: rank4_softmax
func @rank4_softmax(%arg0: tensor<2x3x4x5xf16>) -> tensor<2x3x4x5xf16> {
  // Verify that reduce op dimensions and broadcast dimensions are correct.

  // CHECK: "xla_hlo.reduce"
  // CHECK: dimensions = dense<3>

  // CHECK: "xla_hlo.reduce"
  // CHECK: dimensions = dense<3>

  // CHECK: "xla_hlo.div"{{.*}} {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
  %0 = "tf.Softmax"(%arg0) : (tensor<2x3x4x5xf16>) -> tensor<2x3x4x5xf16>
  return %0: tensor<2x3x4x5xf16>
}

//===----------------------------------------------------------------------===//
// Transpose op legalization.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @transpose_noop
func @transpose_noop(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %permutation = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: @transpose_2d
func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: @transpose_3d
func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %permutation = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> (tensor<3xi64>)
  // CHECK: "xla_hlo.transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
  return %0 : tensor<3x2x1xf32>
}

// CHECK-LABEL: @transpose_dynamic_2d
func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "tf.Transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<4x?xf32>
  return %0 : tensor<4x?xf32>
}

// CHECK-LABEL: @transpose_rankless_2d
func @transpose_rankless_2d(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %permutation = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // CHECK: "tf.Transpose"
  %0 = "tf.Transpose"(%arg0, %permutation) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}


//===----------------------------------------------------------------------===//
// Unary op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @abs
func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @abs_dynamic
func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @abs_rankless
func @abs_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @cast_dynamic_i2f
func @cast_dynamic_i2f(%arg0: tensor<?xi32>) -> tensor<?xf32> {
  // CHECK: "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @cast_i2f
func @cast_i2f(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @cast_c2f
func @cast_c2f(%arg0: tensor<2x!tf.complex64>) -> tensor<2xf32> {
  // CHECK: "tf.Cast"(%arg0) : (tensor<2x!tf.complex64>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) : (tensor<2x!tf.complex64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: @ceil
func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @ceil_dynamic
func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @ceil_rankless
func @ceil_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @cos
func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @cos_dynamic
func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @cos_rankless
func @cos_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Cos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Cos"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @exp
func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @exp_dynamic
func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @exp_rankless
func @exp_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Exp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Exp"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @floor
func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @floor_dynamic
func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @floor_rankless
func @floor_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @neg
func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @neg_dynamic
func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @neg_rankless
func @neg_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Neg"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: @sigmoid
func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: [[R0:%.+]] = "xla_hlo.constant"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: [[R1:%.+]] = "xla_hlo.broadcast"([[R0]]) {broadcast_sizes = dense<2> : tensor<1xi64>} : (tensor<f32>) -> tensor<2xf32>
  // CHECK-DAG: [[R2:%.+]] =  xla_hlo.mul %arg0, [[R1]] : tensor<2xf32>
  // CHECK-DAG: [[R3:%.+]] =  "xla_hlo.tanh"([[R2]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK-DAG: [[R4:%.+]] =  xla_hlo.mul [[R3]], [[R1]] : tensor<2xf32>
  // CHECK-DAG: [[R5:%.+]] =  xla_hlo.add [[R4]], [[R1]] : tensor<2xf32>
  %0 = "tf.Sigmoid"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @rsqrt
func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @rsqrt_dynamic
func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rsqrt_rankless
func @rsqrt_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @tanh
func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "xla_hlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @tanh_dynamic
func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK:  "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @tanh_rankless
func @tanh_rankless(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  "tf.Tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}


// CHECK-LABEL: reshape
func @reshape(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) -> tensor<1x1xf32> {
  // CHECK:  %0 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x1xf32>
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xi32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: reshape_dynamic
func @reshape_dynamic(%arg0: tensor<*xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK:  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: squeeze
func @squeeze(%arg0: tensor<1x1x10xf32>) -> tensor<1x10xf32> {
  // CHECK-NEXT: %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
  %0 = "tf.Squeeze"(%arg0) : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}

// CHECK-LABEL: squeeze_dynamic
func @squeeze_dynamic(%arg0: tensor<?x10xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: %0 = "tf.Squeeze"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %0 = "tf.Squeeze"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: expand_dims
func @expand_dims(%arg0: tensor<2xf32>, %axis: tensor<i32>) -> tensor<1x2xf32> {
  // CHECK: "xla_hlo.reshape"{{.*}} : (tensor<2xf32>) -> tensor<1x2xf32>
  %0 = "tf.ExpandDims"(%arg0, %axis) : (tensor<2xf32>, tensor<i32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: simple_strided_slice
func @simple_strided_slice(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[3, 7]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %output : tensor<3x2xf32>
}

// CHECK-LABEL: strided_slice_negative_indices
func @strided_slice_negative_indices(%input: tensor<4x8xf32>) -> tensor<3x2xf32> {
  %begin = "tf.Const"() {value = dense<[-1, -2]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-4, -8]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, -3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "xla_hlo.reverse"(%arg0) {dimensions = dense<[0, 1]> : tensor<2xi64>}

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[0, 1]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 7]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<3x2xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %output : tensor<3x2xf32>
}

// CHECK-LABEL: strided_slice_range_clamping
func @strided_slice_range_clamping(%input: tensor<4x8xf32>) -> tensor<0x3xf32> {
  %begin = "tf.Const"() {value = dense<[-4, -10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[-1, 10]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[-1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: "xla_hlo.reverse"(%arg0) {dimensions = dense<0> : tensor<1xi64>}

  // CHECK: xla_hlo.slice
  // CHECK-DAG-SAME: start_indices = dense<[3, 0]>
  // CHECK-DAG-SAME: limit_indices = dense<[3, 8]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<0x3xf32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides)
      : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<0x3xf32>
  return %output : tensor<0x3xf32>
}

// CHECK-LABEL: strided_slice_shrink_axis
func @strided_slice_shrink_axis(%input: tensor<4x8xf32>) -> tensor<f32> {
  %begin = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %end = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %strides = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)

  // CHECK: %[[SLICED:.*]] = "xla_hlo.slice"
  // CHECK-DAG-SAME: start_indices = dense<[1, 3]>
  // CHECK-DAG-SAME: limit_indices = dense<[2, 4]>
  // CHECK-DAG-SAME: strides = dense<[1, 3]>
  // CHECK-SAME: -> tensor<1x1xf32>

  // CHECK: "xla_hlo.reshape"(%[[SLICED]]) : (tensor<1x1xf32>) -> tensor<f32>

  %output = "tf.StridedSlice"(%input, %begin, %end, %strides) {shrink_axis_mask = 3
      : i64} : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<f32>
  return %output : tensor<f32>
}

//===----------------------------------------------------------------------===//
// Reduction op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @mean
func @mean(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
    // CHECK: %[[INITIAL:.*]] = "xla_hlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
    // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
    // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.add %[[ARGA]], %[[ARGB]] : tensor<f32>
    // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: %[[DIVISOR:.*]] = "xla_hlo.constant"() {value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %[[MEAN:.*]] = "xla_hlo.div"(%[[REDUCED]], %[[DIVISOR]]) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[MEAN]]) : (tensor<4xf32>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Mean"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @sum
func @sum(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf32>
    // CHECK: %[[INITIAL:.*]] = "xla_hlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
    // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f32>, %[[ARGB:.*]]: tensor<f32>):
    // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.add %[[ARGA]], %[[ARGB]] : tensor<f32>
    // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f32>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf32>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Sum"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

// CHECK-LABEL: func @max
func @max(%arg0: tensor<4x8xf16>) -> tensor<4x1xf16> {
    // CHECK: %[[CAST:.*]] = "xla_hlo.convert"(%arg0) : (tensor<4x8xf16>) -> tensor<4x8xf16>
    // CHECK: %[[INITIAL:.*]] = "xla_hlo.constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    // CHECK: %[[REDUCED:.*]] = "xla_hlo.reduce"(%[[CAST]], %[[INITIAL]]) ( {
    // CHECK: ^bb0(%[[ARGA:.*]]: tensor<f16>, %[[ARGB:.*]]: tensor<f16>):
    // CHECK:  %[[REDUCE_BODY_RESULT:.*]] = xla_hlo.max %[[ARGA]], %[[ARGB]] : tensor<f16>
    // CHECK:  "xla_hlo.return"(%[[REDUCE_BODY_RESULT]]) : (tensor<f16>) -> ()
    // CHECK: }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf16>, tensor<f16>) -> tensor<4xf16>
    // CHECK: %[[CAST_BACK:.*]] = "xla_hlo.convert"(%[[REDUCED]]) : (tensor<4xf16>) -> tensor<4xf16>
    // CHECK: %[[RESULT:.*]] = "xla_hlo.reshape"(%[[CAST_BACK]]) : (tensor<4xf16>) -> tensor<4x1xf16>
    // CHECK: return %[[RESULT]] : tensor<4x1xf16>
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "tf.Max"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xf16>, tensor<1xi64>) -> tensor<4x1xf16>
  return %0 : tensor<4x1xf16>
}

