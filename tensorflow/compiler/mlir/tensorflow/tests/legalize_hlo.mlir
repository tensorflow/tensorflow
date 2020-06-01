// RUN: tf-opt -tf-legalize-hlo %s | FileCheck %s --dump-input-on-failure


func @biasAdd_NHWC(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  %0 = "xla_chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

func @biasAdd_NCHW(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  %0 = "xla_chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  return %0 : tensor<1x32x10x32xi32>
}

func @biasAdd_dynamic(%arg0: tensor<?x?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?x?xi32> {
  %0 = "xla_chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
  return %0 : tensor<?x?x?x?xi32>
}

func @add(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = xla_hlo.add %arg0, %arg0 : tensor<2xi32>
  %1 = xla_hlo.add %0, %arg0 : tensor<2xi32>
  return %1 : tensor<2xi32>
}

func @broadcast_add(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "xla_chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

func @broadcast_multi_dim_add(%arg0: tensor<4x1x1xi32>, %arg1: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
  %0 = "xla_chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
  return %0 : tensor<4x4x4x4xi32>
}

func @div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @broadcast_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

func @shift_left(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = xla_hlo.shift_left %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func @div_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @mul(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = xla_hlo.multiply %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @broadcast_mul(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "xla_chlo.broadcast_multiply"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

func @real_div(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = xla_hlo.divide %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @broadcast_real_div(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

func @sub(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = xla_hlo.subtract %arg0, %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @broadcast_sub(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "xla_chlo.broadcast_subtract"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

func @shift_right(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = xla_hlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func @broadcast_shift_right(%arg0: tensor<4xi32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %0 = "xla_chlo.broadcast_shift_right_arithmetic"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func @and(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = xla_hlo.and %arg0, %arg0 : tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @and_broadcast(%arg0: tensor<1xi1>, %arg1: tensor<1x2xi1>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_and"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @and_dynamic(%arg0: tensor<?xi1>, %arg1: tensor<1xi1>) -> tensor<?xi1> {
  %0 = "xla_chlo.broadcast_and"(%arg0, %arg1) : (tensor<?xi1>, tensor<1xi1>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @or(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = xla_hlo.or %arg0, %arg0 : tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @or_broadcast(%arg0: tensor<1xi1>, %arg1: tensor<1x2xi1>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_or"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @or_dynamic(%arg0: tensor<?xi1>, %arg1: tensor<1xi1>) -> tensor<?xi1> {
  %0 = "xla_chlo.broadcast_or"(%arg0, %arg1) : (tensor<?xi1>, tensor<1xi1>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @bitwise_or(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = xla_hlo.or %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func @bitwise_or_broadcast(%arg0: tensor<1xi8>, %arg1: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = "xla_chlo.broadcast_or"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi8>, tensor<1x4xi8>) -> tensor<1x4xi8>
  return %0 : tensor<1x4xi8>
}

func @bitwise_or_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi32> {
  %0 = "xla_chlo.broadcast_or"(%arg0, %arg1) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

func @bitwise_and(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = xla_hlo.and %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

func @bitwise_and_broadcast(%arg0: tensor<1xi8>, %arg1: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = "xla_chlo.broadcast_and"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1xi8>, tensor<1x4xi8>) -> tensor<1x4xi8>
  return %0 : tensor<1x4xi8>
}

func @bitwise_and_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi32> {
  %0 = "xla_chlo.broadcast_and"(%arg0, %arg1) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

func @pow(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = xla_hlo.power %arg0, %arg0 : tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @pow_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = xla_hlo.power %arg0, %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @floordiv_broadcast_i32(%arg0: tensor<2x3xi32>, %arg1: tensor<3xi32>) -> tensor<2x3xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<2x3xi32>
  %1 = "xla_chlo.broadcast_compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  %2 = xla_hlo.constant dense<0> : tensor<3xi32>
  %3 = "xla_chlo.broadcast_compare"(%arg1, %2) {comparison_direction = "LT"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  %4 = "xla_chlo.broadcast_compare"(%1, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"} : (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>
  %5 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  %6 = "xla_hlo.abs"(%arg0) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %7 = "xla_hlo.abs"(%arg1) : (tensor<3xi32>) -> tensor<3xi32>
  %8 = xla_hlo.constant dense<1> : tensor<3xi32>
  %9 = xla_hlo.subtract %7, %8 : tensor<3xi32>
  %10 = "xla_chlo.broadcast_add"(%6, %9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  %11 = "xla_hlo.negate"(%10) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %12 = "xla_hlo.abs"(%arg1) : (tensor<3xi32>) -> tensor<3xi32>
  %13 = "xla_chlo.broadcast_divide"(%11, %12) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  %14 = "xla_hlo.select"(%4, %5, %13) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %14 : tensor<2x3xi32>
}

func @floordiv_reverse_broadcast_i32(%arg0: tensor<3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<3xi32>
  %1 = "xla_hlo.compare"(%arg0, %0) {comparison_direction = "LT"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  %2 = xla_hlo.constant dense<0> : tensor<2x3xi32>
  %3 = "xla_chlo.broadcast_compare"(%arg1, %2) {comparison_direction = "LT"} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
  %4 = "xla_chlo.broadcast_compare"(%1, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"} : (tensor<3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
  %5 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %6 = "xla_hlo.abs"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  %7 = "xla_hlo.abs"(%arg1) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %8 = xla_hlo.constant dense<1> : tensor<2x3xi32>
  %9 = xla_hlo.subtract %7, %8 : tensor<2x3xi32>
  %10 = "xla_chlo.broadcast_add"(%6, %9) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %11 = "xla_hlo.negate"(%10) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %12 = "xla_hlo.abs"(%arg1) : (tensor<2x3xi32>) -> tensor<2x3xi32>
  %13 = xla_hlo.divide %11, %12 : tensor<2x3xi32>
  %14 = "xla_hlo.select"(%4, %5, %13) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %14 : tensor<2x3xi32>
}

func @floordiv_f32(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = xla_hlo.divide %arg0, %arg0 : tensor<2xf32>
  %1 = xla_hlo.divide %arg0, %arg0 : tensor<2xf32>
  %2 = "xla_hlo.floor"(%1) : (tensor<2xf32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}

func @floordiv_f16_broadcast(%arg0: tensor<2x3xf16>, %arg1: tensor<3xf16>) -> tensor<2x3xf16> {
  %0 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
  %1 = "xla_chlo.broadcast_divide"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
  %2 = "xla_hlo.floor"(%1) : (tensor<2x3xf16>) -> tensor<2x3xf16>
  return %2 : tensor<2x3xf16>
}

func @equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @equal_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @equal_broadcast(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @equal_broadcast_no_incompatible_shapes_error(%arg0: tensor<2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "EQ"} : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @equal_incompatible_shape_broadcastable(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @notequal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @notequal_broadcast(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @notequal_broadcast_no_incompatible_shapes_error(%arg0: tensor<2xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "NE"} : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @notequal_incompatible_shape_broadcastable(%arg0: tensor<?xi32>, %arg1: tensor<1xi32>) -> tensor<?xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @greater(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GT"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @broadcast_greater(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "GT"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @greater_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @broadcast_greater_equal(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "GE"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @less(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @broadcast_less(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "LT"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @less_equal(%arg0: tensor<2xi32>) -> tensor<2xi1> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @broadcast_less_equal(%arg0: tensor<1xi32>, %arg1: tensor<1x2xi32>) -> tensor<1x2xi1> {
  %0 = "xla_chlo.broadcast_compare"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, comparison_direction = "LE"} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
  return %0 : tensor<1x2xi1>
}

func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  %2 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 0 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  return %2 : tensor<6x3xf32>
}

func @concat_v2_1d_axis(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x6xf32> {
  %2 = "xla_hlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x6xf32>
  return %2 : tensor<3x6xf32>
}

func @const() -> tensor<2xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @relu(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = "xla_chlo.broadcast_maximum"(%0, %arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
  return %1 : tensor<1xi32>
}

func @relu_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = "xla_chlo.broadcast_maximum"(%0, %arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
  return %1 : tensor<?xi32>
}

func @relu6(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = xla_hlo.constant dense<6> : tensor<i32>
  %2 = "xla_chlo.broadcast_minimum"(%arg0, %1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  %3 = "xla_chlo.broadcast_maximum"(%2, %0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  return %3 : tensor<1xi32>
}

func @relu6_unranked(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %0 = xla_hlo.constant dense<0> : tensor<i32>
  %1 = xla_hlo.constant dense<6> : tensor<i32>
  %2 = "xla_chlo.broadcast_minimum"(%arg0, %1) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %3 = "xla_chlo.broadcast_maximum"(%2, %0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %3 : tensor<?xi32>
}

func @relu_grad(%arg0: tensor<4x8xf32>, %arg1: tensor<?x?xf32>) -> tensor<4x8xf32> {
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "xla_chlo.broadcast_compare"(%arg1, %0) {broadcast_dimensions = dense<[]> : tensor<0xi64>, comparison_direction = "GT"} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
  %2 = xla_hlo.constant dense<0.000000e+00> : tensor<4x8xf32>
  %3 = "xla_hlo.select"(%1, %arg0, %2) : (tensor<?x?xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %3 : tensor<4x8xf32>
}

func @select(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @select_float(%arg0: tensor<2xi1>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @select_multidimensional(%arg0: tensor<3x2xi1>, %arg1: tensor<3x2xi32>, %arg2: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<3x2xi1>, tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
  return %0 : tensor<3x2xi32>
}

func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @transpose_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %1 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %2 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %2 : tensor<3x2xf32>
}

func @transpose_3d_int32(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = xla_hlo.constant dense<[2, 1, 0]> : tensor<3xi32>
  %1 = xla_hlo.constant dense<[2, 1, 0]> : tensor<3xi64>
  %2 = "xla_hlo.transpose"(%arg0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  return %2 : tensor<3x2x1xf32>
}

func @transpose_3d(%arg0: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
  %0 = xla_hlo.constant dense<[2, 1, 0]> : tensor<3xi64>
  %1 = xla_hlo.constant dense<[2, 1, 0]> : tensor<3xi64>
  %2 = "xla_hlo.transpose"(%arg0) {permutation = dense<[2, 1, 0]> : tensor<3xi64>} : (tensor<1x2x3xf32>) -> tensor<3x2x1xf32>
  return %2 : tensor<3x2x1xf32>
}

func @transpose_dynamic_2d(%arg0: tensor<?x4xf32>) -> tensor<4x?xf32> {
  %0 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %1 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %2 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x4xf32>) -> tensor<4x?xf32>
  return %2 : tensor<4x?xf32>
}

func @transpose_unranked_2d(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %1 = xla_hlo.constant dense<[1, 0]> : tensor<2xi64>
  %2 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>
}

func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @abs_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @abs_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @ceil(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @ceil_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @ceil_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @complex_abs(%arg0: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @cos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.cosine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @cos_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.cosine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @cos_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.cosine"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @exp(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @exp_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.exponential"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @exp_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.exponential"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @floor(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @floor_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @floor_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.floor"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @is_finite(%arg0: tensor<2xf32>) -> tensor<2xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<2xf32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

func @is_finite_dynamic(%arg0: tensor<?xf32>) -> tensor<?xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<?xf32>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

func @is_finite_unranked(%arg0: tensor<*xf32>) -> tensor<*xi1> {
  %0 = "xla_hlo.is_finite"(%arg0) : (tensor<*xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

func @log(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @log_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @log_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.log"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @log1p(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @log1p_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @log1p_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.log_plus_one"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @neg(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @neg_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.negate"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @neg_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.negate"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sigmoid(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = xla_hlo.constant dense<5.000000e-01> : tensor<f32>
  %1 = xla_hlo.constant dense<2> : tensor<1xi64>
  %2 = xla_hlo.constant dense<5.000000e-01> : tensor<2xf32>
  %3 = xla_hlo.multiply %arg0, %2 : tensor<2xf32>
  %4 = "xla_hlo.tanh"(%3) : (tensor<2xf32>) -> tensor<2xf32>
  %5 = xla_hlo.multiply %4, %2 : tensor<2xf32>
  %6 = xla_hlo.add %5, %2 : tensor<2xf32>
  return %6 : tensor<2xf32>
}

func @sin(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.sine"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @sin_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.sine"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @sin_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.sine"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @rsqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @rsqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @rsqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.rsqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @sqrt(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @sqrt_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @sqrt_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.sqrt"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @tanh(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @tanh_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @tanh_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @bitcast(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "xla_hlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @bitcast_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "xla_hlo.bitcast_convert"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

func @bitcast_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "xla_hlo.bitcast_convert"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @bitcast_same_widths(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  %0 = "xla_hlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

func @sign(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi1>
  %1 = xla_hlo.constant dense<0.000000e+00> : tensor<1x2x3x4xf32>
  %2 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi1>
  %3 = xla_hlo.constant dense<0.000000e+00> : tensor<1x2x3x4xf32>
  %4 = "xla_hlo.sign"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %5 = "xla_hlo.select"(%2, %3, %4) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  %6 = "xla_hlo.select"(%0, %1, %5) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %6 : tensor<1x2x3x4xf32>
}

func @size_rank_one_i32(%arg0: tensor<f32>) -> tensor<i32> {
  %0 = xla_hlo.constant dense<1> : tensor<i32>
  return %0 : tensor<i32>
}

func @size_rank_one_i64(%arg0: tensor<f32>) -> tensor<i64> {
  %0 = xla_hlo.constant dense<1> : tensor<i64>
  return %0 : tensor<i64>
}

func @complex(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xcomplex<f32>> {
  %0 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
  return %0 : tensor<3xcomplex<f32>>
}

func @convert_i32_f32(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

func @convert_slice(%arg0: tensor<1x4672xf32>) -> tensor<1x519xf32> {
  %0 = "xla_hlo.slice"(%arg0) {limit_indices = dense<[1, 4672]> : tensor<2xi64>, start_indices = dense<[0, 4153]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x4672xf32>) -> tensor<1x519xf32>
  return %0 : tensor<1x519xf32>
}

func @reshape(%arg0: tensor<4x6xf32>) -> tensor<2x2x6xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<4x6xf32>) -> tensor<2x2x6xf32>
  return %0 : tensor<2x2x6xf32>

}

func @convert_dot_1d_2d(%arg0: tensor<256xf32>, %arg1: tensor<256x1xf32>) -> tensor<1xf32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256xf32>, tensor<256x1xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

func @convert_dot_2d_1d(%arg0: tensor<1x256xf32>, %arg1: tensor<256xf32>) -> tensor<1xf32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

func @convert_dot_1d_1d(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<f32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<256xf32>, tensor<256xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @convert_dot_2d_2d(%arg0: tensor<1x256xf32>, %arg1: tensor<256x1xf32>) -> tensor<1x1xf32> {
  %0 = "xla_hlo.dot"(%arg0, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// CHECK-LABEL:   func @biasAdd_NHWC(
// CHECK-SAME:                       [[VAL_0:%.*]]: tensor<1x32x10x32xi32>, [[VAL_1:%.*]]: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
// CHECK:           [[VAL_2:%.*]] = "tf.AddV2"([[VAL_0]], [[VAL_1]]) : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
// CHECK:           return [[VAL_2]] : tensor<1x32x10x32xi32>
// CHECK:         }

// CHECK-LABEL:   func @biasAdd_NCHW(
// CHECK-SAME:                       [[VAL_3:%.*]]: tensor<1x32x10x32xi32>, [[VAL_4:%.*]]: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
// CHECK:           [[VAL_5:%.*]] = "tf.AddV2"([[VAL_3]], [[VAL_4]]) : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
// CHECK:           return [[VAL_5]] : tensor<1x32x10x32xi32>
// CHECK:         }

// CHECK-LABEL:   func @biasAdd_dynamic(
// CHECK-SAME:                          [[VAL_6:%.*]]: tensor<?x?x?x?xi32>, [[VAL_7:%.*]]: tensor<?xi32>) -> tensor<?x?x?x?xi32> {
// CHECK:           [[VAL_8:%.*]] = "tf.AddV2"([[VAL_6]], [[VAL_7]]) : (tensor<?x?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?x?xi32>
// CHECK:           return [[VAL_8]] : tensor<?x?x?x?xi32>
// CHECK:         }

// CHECK-LABEL:   func @add(
// CHECK-SAME:              [[VAL_9:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_10:%.*]] = "tf.AddV2"([[VAL_9]], [[VAL_9]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           [[VAL_11:%.*]] = "tf.AddV2"([[VAL_10]], [[VAL_9]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_11]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_add(
// CHECK-SAME:                        [[VAL_12:%.*]]: tensor<1xi32>, [[VAL_13:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_14:%.*]] = "tf.AddV2"([[VAL_12]], [[VAL_13]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_14]] : tensor<1x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_multi_dim_add(
// CHECK-SAME:                                  [[VAL_15:%.*]]: tensor<4x1x1xi32>, [[VAL_16:%.*]]: tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32> {
// CHECK:           [[VAL_17:%.*]] = "tf.AddV2"([[VAL_15]], [[VAL_16]]) : (tensor<4x1x1xi32>, tensor<4x4x4x4xi32>) -> tensor<4x4x4x4xi32>
// CHECK:           return [[VAL_17]] : tensor<4x4x4x4xi32>
// CHECK:         }

// CHECK-LABEL:   func @div(
// CHECK-SAME:              [[VAL_18:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_19:%.*]] = "tf.Div"([[VAL_18]], [[VAL_18]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_19]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_div(
// CHECK-SAME:                        [[VAL_20:%.*]]: tensor<1xi32>, [[VAL_21:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_22:%.*]] = "tf.Div"([[VAL_20]], [[VAL_21]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_22]] : tensor<1x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @shift_left(
// CHECK-SAME:                     [[VAL_23:%.*]]: tensor<4xi32>, [[VAL_24:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_25:%.*]] = "tf.LeftShift"([[VAL_23]], [[VAL_24]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_25]] : tensor<4xi32>
// CHECK:         }

// CHECK-LABEL:   func @div_dynamic(
// CHECK-SAME:                      [[VAL_26:%.*]]: tensor<?xi32>, [[VAL_27:%.*]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:           [[VAL_28:%.*]] = "tf.Div"([[VAL_26]], [[VAL_27]]) : (tensor<?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:           return [[VAL_28]] : tensor<?x?xi32>
// CHECK:         }

// CHECK-LABEL:   func @maximum(
// CHECK-SAME:                  [[VAL_29:%.*]]: tensor<4xf32>, [[VAL_30:%.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           [[VAL_31:%.*]] = "tf.Maximum"([[VAL_29]], [[VAL_30]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:           return [[VAL_31]] : tensor<4xf32>
// CHECK:         }

// CHECK-LABEL:   func @minimum(
// CHECK-SAME:                  [[VAL_32:%.*]]: tensor<4xf32>, [[VAL_33:%.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           [[VAL_34:%.*]] = "tf.Minimum"([[VAL_32]], [[VAL_33]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:           return [[VAL_34]] : tensor<4xf32>
// CHECK:         }

// CHECK-LABEL:   func @mul(
// CHECK-SAME:              [[VAL_35:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_36:%.*]] = "tf.Mul"([[VAL_35]], [[VAL_35]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_36]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_mul(
// CHECK-SAME:                        [[VAL_37:%.*]]: tensor<1xi32>, [[VAL_38:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_39:%.*]] = "tf.Mul"([[VAL_37]], [[VAL_38]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_39]] : tensor<1x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @real_div(
// CHECK-SAME:                   [[VAL_40:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_41:%.*]] = "tf.Div"([[VAL_40]], [[VAL_40]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_41]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_real_div(
// CHECK-SAME:                             [[VAL_42:%.*]]: tensor<1xi32>, [[VAL_43:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_44:%.*]] = "tf.Div"([[VAL_42]], [[VAL_43]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_44]] : tensor<1x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @sub(
// CHECK-SAME:              [[VAL_45:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_46:%.*]] = "tf.Sub"([[VAL_45]], [[VAL_45]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_46]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_sub(
// CHECK-SAME:                        [[VAL_47:%.*]]: tensor<1xi32>, [[VAL_48:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi32> {
// CHECK:           [[VAL_49:%.*]] = "tf.Sub"([[VAL_47]], [[VAL_48]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
// CHECK:           return [[VAL_49]] : tensor<1x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @shift_right(
// CHECK-SAME:                      [[VAL_50:%.*]]: tensor<4xi32>, [[VAL_51:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_52:%.*]] = "tf.RightShift"([[VAL_50]], [[VAL_51]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_52]] : tensor<4xi32>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_shift_right(
// CHECK-SAME:                                [[VAL_53:%.*]]: tensor<4xi32>, [[VAL_54:%.*]]: tensor<2x4xi32>) -> tensor<2x4xi32> {
// CHECK:           [[VAL_55:%.*]] = "tf.RightShift"([[VAL_53]], [[VAL_54]]) : (tensor<4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK:           return [[VAL_55]] : tensor<2x4xi32>
// CHECK:         }

// CHECK-LABEL:   func @and(
// CHECK-SAME:              [[VAL_56:%.*]]: tensor<2xi1>) -> tensor<2xi1> {
// CHECK:           [[VAL_57:%.*]] = "tf.LogicalAnd"([[VAL_56]], [[VAL_56]]) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
// CHECK:           return [[VAL_57]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @and_broadcast(
// CHECK-SAME:                        [[VAL_58:%.*]]: tensor<1xi1>, [[VAL_59:%.*]]: tensor<1x2xi1>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_60:%.*]] = "tf.LogicalAnd"([[VAL_58]], [[VAL_59]]) : (tensor<1xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_60]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @and_dynamic(
// CHECK-SAME:                      [[VAL_61:%.*]]: tensor<?xi1>, [[VAL_62:%.*]]: tensor<1xi1>) -> tensor<?xi1> {
// CHECK:           [[VAL_63:%.*]] = "tf.LogicalAnd"([[VAL_61]], [[VAL_62]]) : (tensor<?xi1>, tensor<1xi1>) -> tensor<?xi1>
// CHECK:           return [[VAL_63]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @or(
// CHECK-SAME:             [[VAL_64:%.*]]: tensor<2xi1>) -> tensor<2xi1> {
// CHECK:           [[VAL_65:%.*]] = "tf.LogicalOr"([[VAL_64]], [[VAL_64]]) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
// CHECK:           return [[VAL_65]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @or_broadcast(
// CHECK-SAME:                       [[VAL_66:%.*]]: tensor<1xi1>, [[VAL_67:%.*]]: tensor<1x2xi1>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_68:%.*]] = "tf.LogicalOr"([[VAL_66]], [[VAL_67]]) : (tensor<1xi1>, tensor<1x2xi1>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_68]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @or_dynamic(
// CHECK-SAME:                     [[VAL_69:%.*]]: tensor<?xi1>, [[VAL_70:%.*]]: tensor<1xi1>) -> tensor<?xi1> {
// CHECK:           [[VAL_71:%.*]] = "tf.LogicalOr"([[VAL_69]], [[VAL_70]]) : (tensor<?xi1>, tensor<1xi1>) -> tensor<?xi1>
// CHECK:           return [[VAL_71]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_or(
// CHECK-SAME:                     [[VAL_72:%.*]]: tensor<4xi32>, [[VAL_73:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_74:%.*]] = "tf.BitwiseOr"([[VAL_72]], [[VAL_73]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_74]] : tensor<4xi32>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_or_broadcast(
// CHECK-SAME:                               [[VAL_75:%.*]]: tensor<1xi8>, [[VAL_76:%.*]]: tensor<1x4xi8>) -> tensor<1x4xi8> {
// CHECK:           [[VAL_77:%.*]] = "tf.BitwiseOr"([[VAL_75]], [[VAL_76]]) : (tensor<1xi8>, tensor<1x4xi8>) -> tensor<1x4xi8>
// CHECK:           return [[VAL_77]] : tensor<1x4xi8>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_or_dynamic(
// CHECK-SAME:                             [[VAL_78:%.*]]: tensor<?xi32>, [[VAL_79:%.*]]: tensor<1xi32>) -> tensor<?xi32> {
// CHECK:           [[VAL_80:%.*]] = "tf.BitwiseOr"([[VAL_78]], [[VAL_79]]) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK:           return [[VAL_80]] : tensor<?xi32>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_and(
// CHECK-SAME:                      [[VAL_81:%.*]]: tensor<4xi32>, [[VAL_82:%.*]]: tensor<4xi32>) -> tensor<4xi32> {
// CHECK:           [[VAL_83:%.*]] = "tf.BitwiseAnd"([[VAL_81]], [[VAL_82]]) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
// CHECK:           return [[VAL_83]] : tensor<4xi32>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_and_broadcast(
// CHECK-SAME:                                [[VAL_84:%.*]]: tensor<1xi8>, [[VAL_85:%.*]]: tensor<1x4xi8>) -> tensor<1x4xi8> {
// CHECK:           [[VAL_86:%.*]] = "tf.BitwiseAnd"([[VAL_84]], [[VAL_85]]) : (tensor<1xi8>, tensor<1x4xi8>) -> tensor<1x4xi8>
// CHECK:           return [[VAL_86]] : tensor<1x4xi8>
// CHECK:         }

// CHECK-LABEL:   func @bitwise_and_dynamic(
// CHECK-SAME:                              [[VAL_87:%.*]]: tensor<?xi32>, [[VAL_88:%.*]]: tensor<1xi32>) -> tensor<?xi32> {
// CHECK:           [[VAL_89:%.*]] = "tf.BitwiseAnd"([[VAL_87]], [[VAL_88]]) : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK:           return [[VAL_89]] : tensor<?xi32>
// CHECK:         }

// CHECK-LABEL:   func @pow(
// CHECK-SAME:              [[VAL_90:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_91:%.*]] = "tf.Pow"([[VAL_90]], [[VAL_90]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_91]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @pow_dynamic(
// CHECK-SAME:                      [[VAL_92:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_93:%.*]] = "tf.Pow"([[VAL_92]], [[VAL_92]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_93]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @floordiv_broadcast_i32(
// CHECK-SAME:                                 [[VAL_94:%.*]]: tensor<2x3xi32>, [[VAL_95:%.*]]: tensor<3xi32>) -> tensor<2x3xi32> {
// CHECK:           [[VAL_96:%.*]] = "tf.Const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
// CHECK:           [[VAL_97:%.*]] = "tf.Less"([[VAL_94]], [[VAL_96]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
// CHECK:           [[VAL_98:%.*]] = "tf.Const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           [[VAL_99:%.*]] = "tf.Less"([[VAL_95]], [[VAL_98]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
// CHECK:           [[VAL_100:%.*]] = "tf.Equal"([[VAL_97]], [[VAL_99]]) {incompatible_shape_error = true} : (tensor<2x3xi1>, tensor<3xi1>) -> tensor<2x3xi1>
// CHECK:           [[VAL_101:%.*]] = "tf.Div"([[VAL_94]], [[VAL_95]]) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_102:%.*]] = "tf.Abs"([[VAL_94]]) : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_103:%.*]] = "tf.Abs"([[VAL_95]]) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK:           [[VAL_104:%.*]] = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           [[VAL_105:%.*]] = "tf.Sub"([[VAL_103]], [[VAL_104]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
// CHECK:           [[VAL_106:%.*]] = "tf.AddV2"([[VAL_102]], [[VAL_105]]) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_107:%.*]] = "tf.Neg"([[VAL_106]]) : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_108:%.*]] = "tf.Abs"([[VAL_95]]) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK:           [[VAL_109:%.*]] = "tf.Div"([[VAL_107]], [[VAL_108]]) : (tensor<2x3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_110:%.*]] = "tf.Select"([[VAL_100]], [[VAL_101]], [[VAL_109]]) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           return [[VAL_110]] : tensor<2x3xi32>
// CHECK:         }

// CHECK-LABEL:   func @floordiv_reverse_broadcast_i32(
// CHECK-SAME:                                         [[VAL_111:%.*]]: tensor<3xi32>, [[VAL_112:%.*]]: tensor<2x3xi32>) -> tensor<2x3xi32> {
// CHECK:           [[VAL_113:%.*]] = "tf.Const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           [[VAL_114:%.*]] = "tf.Less"([[VAL_111]], [[VAL_113]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
// CHECK:           [[VAL_115:%.*]] = "tf.Const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
// CHECK:           [[VAL_116:%.*]] = "tf.Less"([[VAL_112]], [[VAL_115]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
// CHECK:           [[VAL_117:%.*]] = "tf.Equal"([[VAL_114]], [[VAL_116]]) {incompatible_shape_error = true} : (tensor<3xi1>, tensor<2x3xi1>) -> tensor<2x3xi1>
// CHECK:           [[VAL_118:%.*]] = "tf.Div"([[VAL_111]], [[VAL_112]]) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_119:%.*]] = "tf.Abs"([[VAL_111]]) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK:           [[VAL_120:%.*]] = "tf.Abs"([[VAL_112]]) : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_121:%.*]] = "tf.Const"() {value = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
// CHECK:           [[VAL_122:%.*]] = "tf.Sub"([[VAL_120]], [[VAL_121]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_123:%.*]] = "tf.AddV2"([[VAL_119]], [[VAL_122]]) : (tensor<3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_124:%.*]] = "tf.Neg"([[VAL_123]]) : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_125:%.*]] = "tf.Abs"([[VAL_112]]) : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_126:%.*]] = "tf.Div"([[VAL_124]], [[VAL_125]]) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           [[VAL_127:%.*]] = "tf.Select"([[VAL_117]], [[VAL_118]], [[VAL_126]]) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK:           return [[VAL_127]] : tensor<2x3xi32>
// CHECK:         }

// CHECK-LABEL:   func @floordiv_f32(
// CHECK-SAME:                       [[VAL_128:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_129:%.*]] = "tf.Div"([[VAL_128]], [[VAL_128]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           [[VAL_130:%.*]] = "tf.Div"([[VAL_128]], [[VAL_128]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           [[VAL_131:%.*]] = "tf.FloorDiv"([[VAL_128]], [[VAL_128]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_131]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @floordiv_f16_broadcast(
// CHECK-SAME:                                 [[VAL_132:%.*]]: tensor<2x3xf16>, [[VAL_133:%.*]]: tensor<3xf16>) -> tensor<2x3xf16> {
// CHECK:           [[VAL_134:%.*]] = "tf.Div"([[VAL_132]], [[VAL_133]]) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
// CHECK:           [[VAL_135:%.*]] = "tf.Div"([[VAL_132]], [[VAL_133]]) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
// CHECK:           [[VAL_136:%.*]] = "tf.FloorDiv"([[VAL_132]], [[VAL_133]]) : (tensor<2x3xf16>, tensor<3xf16>) -> tensor<2x3xf16>
// CHECK:           return [[VAL_136]] : tensor<2x3xf16>
// CHECK:         }

// CHECK-LABEL:   func @equal(
// CHECK-SAME:                [[VAL_137:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_138:%.*]] = "tf.Equal"([[VAL_137]], [[VAL_137]]) {incompatible_shape_error = true} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_138]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @equal_dynamic(
// CHECK-SAME:                        [[VAL_139:%.*]]: tensor<?xi32>, [[VAL_140:%.*]]: tensor<1xi32>) -> tensor<?xi1> {
// CHECK:           [[VAL_141:%.*]] = "tf.Equal"([[VAL_139]], [[VAL_140]]) {incompatible_shape_error = true} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
// CHECK:           return [[VAL_141]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @equal_broadcast(
// CHECK-SAME:                          [[VAL_142:%.*]]: tensor<1xi32>, [[VAL_143:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_144:%.*]] = "tf.Equal"([[VAL_142]], [[VAL_143]]) {incompatible_shape_error = true} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_144]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @equal_broadcast_no_incompatible_shapes_error(
// CHECK-SAME:                                                       [[VAL_145:%.*]]: tensor<2xi32>, [[VAL_146:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_147:%.*]] = "tf.Equal"([[VAL_145]], [[VAL_146]]) {incompatible_shape_error = true} : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_147]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @equal_incompatible_shape_broadcastable(
// CHECK-SAME:                                                 [[VAL_148:%.*]]: tensor<?xi32>, [[VAL_149:%.*]]: tensor<1xi32>) -> tensor<?xi1> {
// CHECK:           [[VAL_150:%.*]] = "tf.Equal"([[VAL_148]], [[VAL_149]]) {incompatible_shape_error = true} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
// CHECK:           return [[VAL_150]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @notequal(
// CHECK-SAME:                   [[VAL_151:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_152:%.*]] = "tf.NotEqual"([[VAL_151]], [[VAL_151]]) {incompatible_shape_error = true} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_152]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @notequal_broadcast(
// CHECK-SAME:                             [[VAL_153:%.*]]: tensor<1xi32>, [[VAL_154:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_155:%.*]] = "tf.NotEqual"([[VAL_153]], [[VAL_154]]) {incompatible_shape_error = true} : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_155]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @notequal_broadcast_no_incompatible_shapes_error(
// CHECK-SAME:                                                          [[VAL_156:%.*]]: tensor<2xi32>, [[VAL_157:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_158:%.*]] = "tf.NotEqual"([[VAL_156]], [[VAL_157]]) {incompatible_shape_error = true} : (tensor<2xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_158]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @notequal_incompatible_shape_broadcastable(
// CHECK-SAME:                                                    [[VAL_159:%.*]]: tensor<?xi32>, [[VAL_160:%.*]]: tensor<1xi32>) -> tensor<?xi1> {
// CHECK:           [[VAL_161:%.*]] = "tf.NotEqual"([[VAL_159]], [[VAL_160]]) {incompatible_shape_error = true} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi1>
// CHECK:           return [[VAL_161]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @greater(
// CHECK-SAME:                  [[VAL_162:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_163:%.*]] = "tf.Greater"([[VAL_162]], [[VAL_162]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_163]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_greater(
// CHECK-SAME:                            [[VAL_164:%.*]]: tensor<1xi32>, [[VAL_165:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_166:%.*]] = "tf.Greater"([[VAL_164]], [[VAL_165]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_166]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @greater_equal(
// CHECK-SAME:                        [[VAL_167:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_168:%.*]] = "tf.GreaterEqual"([[VAL_167]], [[VAL_167]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_168]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_greater_equal(
// CHECK-SAME:                                  [[VAL_169:%.*]]: tensor<1xi32>, [[VAL_170:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_171:%.*]] = "tf.GreaterEqual"([[VAL_169]], [[VAL_170]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_171]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @less(
// CHECK-SAME:               [[VAL_172:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_173:%.*]] = "tf.Less"([[VAL_172]], [[VAL_172]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_173]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_less(
// CHECK-SAME:                         [[VAL_174:%.*]]: tensor<1xi32>, [[VAL_175:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_176:%.*]] = "tf.Less"([[VAL_174]], [[VAL_175]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_176]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @less_equal(
// CHECK-SAME:                     [[VAL_177:%.*]]: tensor<2xi32>) -> tensor<2xi1> {
// CHECK:           [[VAL_178:%.*]] = "tf.LessEqual"([[VAL_177]], [[VAL_177]]) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
// CHECK:           return [[VAL_178]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @broadcast_less_equal(
// CHECK-SAME:                               [[VAL_179:%.*]]: tensor<1xi32>, [[VAL_180:%.*]]: tensor<1x2xi32>) -> tensor<1x2xi1> {
// CHECK:           [[VAL_181:%.*]] = "tf.LessEqual"([[VAL_179]], [[VAL_180]]) : (tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x2xi1>
// CHECK:           return [[VAL_181]] : tensor<1x2xi1>
// CHECK:         }

// CHECK-LABEL:   func @concat_v2(
// CHECK-SAME:                    [[VAL_182:%.*]]: tensor<3x3xf32>, [[VAL_183:%.*]]: tensor<3x3xf32>) -> tensor<6x3xf32> {
// CHECK:           [[VAL_184:%.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
// CHECK:           [[VAL_185:%.*]] = "tf.ConcatV2"([[VAL_182]], [[VAL_183]], [[VAL_184]]) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
// CHECK:           return [[VAL_185]] : tensor<6x3xf32>
// CHECK:         }

// CHECK-LABEL:   func @concat_v2_1d_axis(
// CHECK-SAME:                            [[VAL_186:%.*]]: tensor<3x3xf32>, [[VAL_187:%.*]]: tensor<3x3xf32>) -> tensor<3x6xf32> {
// CHECK:           [[VAL_188:%.*]] = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
// CHECK:           [[VAL_189:%.*]] = "tf.ConcatV2"([[VAL_186]], [[VAL_187]], [[VAL_188]]) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<3x6xf32>
// CHECK:           return [[VAL_189]] : tensor<3x6xf32>
// CHECK:         }

// CHECK-LABEL:   func @const() -> tensor<2xi32> {
// CHECK:           [[VAL_190:%.*]] = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           return [[VAL_190]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @relu(
// CHECK-SAME:               [[VAL_192:%.*]]: tensor<1xi32>) -> tensor<1xi32> {
// CHECK:           [[VAL_193:%.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_194:%.*]] = "tf.Maximum"([[VAL_193]], [[VAL_192]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:           return [[VAL_194]] : tensor<1xi32>
// CHECK:         }

// CHECK-LABEL:   func @relu_unranked(
// CHECK-SAME:                        [[VAL_195:%.*]]: tensor<?xi32>) -> tensor<?xi32> {
// CHECK:           [[VAL_196:%.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_197:%.*]] = "tf.Maximum"([[VAL_196]], [[VAL_195]]) : (tensor<i32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:           return [[VAL_197]] : tensor<?xi32>
// CHECK:         }

// CHECK-LABEL:   func @relu6(
// CHECK-SAME:                [[VAL_198:%.*]]: tensor<1xi32>) -> tensor<1xi32> {
// CHECK:           [[VAL_199:%.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_200:%.*]] = "tf.Const"() {value = dense<6> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_201:%.*]] = "tf.Minimum"([[VAL_198]], [[VAL_200]]) : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK:           [[VAL_202:%.*]] = "tf.Maximum"([[VAL_201]], [[VAL_199]]) : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
// CHECK:           return [[VAL_202]] : tensor<1xi32>
// CHECK:         }

// CHECK-LABEL:   func @relu6_unranked(
// CHECK-SAME:                         [[VAL_203:%.*]]: tensor<?xi32>) -> tensor<?xi32> {
// CHECK:           [[VAL_204:%.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_205:%.*]] = "tf.Const"() {value = dense<6> : tensor<i32>} : () -> tensor<i32>
// CHECK:           [[VAL_206:%.*]] = "tf.Minimum"([[VAL_203]], [[VAL_205]]) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:           [[VAL_207:%.*]] = "tf.Maximum"([[VAL_206]], [[VAL_204]]) : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:           return [[VAL_207]] : tensor<?xi32>
// CHECK:         }

// CHECK-LABEL:   func @relu_grad(
// CHECK-SAME:                    [[VAL_208:%.*]]: tensor<4x8xf32>, [[VAL_209:%.*]]: tensor<?x?xf32>) -> tensor<4x8xf32> {
// CHECK:           [[VAL_210:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK:           [[VAL_211:%.*]] = "tf.Greater"([[VAL_209]], [[VAL_210]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
// CHECK:           [[VAL_212:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x8xf32>} : () -> tensor<4x8xf32>
// CHECK:           [[VAL_213:%.*]] = "tf.Select"([[VAL_211]], [[VAL_208]], [[VAL_212]]) : (tensor<?x?xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:           return [[VAL_213]] : tensor<4x8xf32>
// CHECK:         }

// CHECK-LABEL:   func @select(
// CHECK-SAME:                 [[VAL_214:%.*]]: tensor<2xi1>, [[VAL_215:%.*]]: tensor<2xi32>, [[VAL_216:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_217:%.*]] = "tf.Select"([[VAL_214]], [[VAL_215]], [[VAL_216]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_217]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @select_float(
// CHECK-SAME:                       [[VAL_218:%.*]]: tensor<2xi1>, [[VAL_219:%.*]]: tensor<2xf32>, [[VAL_220:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_221:%.*]] = "tf.Select"([[VAL_218]], [[VAL_219]], [[VAL_220]]) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_221]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @select_multidimensional(
// CHECK-SAME:                                  [[VAL_222:%.*]]: tensor<3x2xi1>, [[VAL_223:%.*]]: tensor<3x2xi32>, [[VAL_224:%.*]]: tensor<3x2xi32>) -> tensor<3x2xi32> {
// CHECK:           [[VAL_225:%.*]] = "tf.Select"([[VAL_222]], [[VAL_223]], [[VAL_224]]) : (tensor<3x2xi1>, tensor<3x2xi32>, tensor<3x2xi32>) -> tensor<3x2xi32>
// CHECK:           return [[VAL_225]] : tensor<3x2xi32>
// CHECK:         }

// CHECK-LABEL:   func @selectv2(
// CHECK-SAME:                   [[VAL_226:%.*]]: tensor<2xi1>, [[VAL_227:%.*]]: tensor<2xi32>, [[VAL_228:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_229:%.*]] = "tf.Select"([[VAL_226]], [[VAL_227]], [[VAL_228]]) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_229]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @selectv2_pred_scalar(
// CHECK-SAME:                               [[VAL_230:%.*]]: tensor<i1>, [[VAL_231:%.*]]: tensor<2xi32>, [[VAL_232:%.*]]: tensor<2xi32>) -> tensor<2xi32> {
// CHECK:           [[VAL_233:%.*]] = "tf.Select"([[VAL_230]], [[VAL_231]], [[VAL_232]]) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
// CHECK:           return [[VAL_233]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @transpose_2d(
// CHECK-SAME:                       [[VAL_234:%.*]]: tensor<2x3xf32>) -> tensor<3x2xf32> {
// CHECK:           [[VAL_235:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_236:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_237:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_238:%.*]] = "tf.Transpose"([[VAL_234]], [[VAL_237]]) : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<3x2xf32>
// CHECK:           return [[VAL_238]] : tensor<3x2xf32>
// CHECK:         }

// CHECK-LABEL:   func @transpose_3d_int32(
// CHECK-SAME:                             [[VAL_239:%.*]]: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           [[VAL_240:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           [[VAL_241:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK:           [[VAL_242:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK:           [[VAL_243:%.*]] = "tf.Transpose"([[VAL_239]], [[VAL_242]]) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
// CHECK:           return [[VAL_243]] : tensor<3x2x1xf32>
// CHECK:         }

// CHECK-LABEL:   func @transpose_3d(
// CHECK-SAME:                       [[VAL_244:%.*]]: tensor<1x2x3xf32>) -> tensor<3x2x1xf32> {
// CHECK:           [[VAL_245:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK:           [[VAL_246:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK:           [[VAL_247:%.*]] = "tf.Const"() {value = dense<[2, 1, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
// CHECK:           [[VAL_248:%.*]] = "tf.Transpose"([[VAL_244]], [[VAL_247]]) : (tensor<1x2x3xf32>, tensor<3xi64>) -> tensor<3x2x1xf32>
// CHECK:           return [[VAL_248]] : tensor<3x2x1xf32>
// CHECK:         }

// CHECK-LABEL:   func @transpose_dynamic_2d(
// CHECK-SAME:                               [[VAL_249:%.*]]: tensor<?x4xf32>) -> tensor<4x?xf32> {
// CHECK:           [[VAL_250:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_251:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_252:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_253:%.*]] = "tf.Transpose"([[VAL_249]], [[VAL_252]]) : (tensor<?x4xf32>, tensor<2xi64>) -> tensor<4x?xf32>
// CHECK:           return [[VAL_253]] : tensor<4x?xf32>
// CHECK:         }

// CHECK-LABEL:   func @transpose_unranked_2d(
// CHECK-SAME:                                [[VAL_254:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_255:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_256:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_257:%.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_258:%.*]] = "tf.Transpose"([[VAL_254]], [[VAL_257]]) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
// CHECK:           return [[VAL_258]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @abs(
// CHECK-SAME:              [[VAL_259:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_260:%.*]] = "tf.Abs"([[VAL_259]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_260]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @abs_dynamic(
// CHECK-SAME:                      [[VAL_261:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_262:%.*]] = "tf.Abs"([[VAL_261]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_262]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @abs_unranked(
// CHECK-SAME:                       [[VAL_263:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_264:%.*]] = "tf.Abs"([[VAL_263]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_264]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @ceil(
// CHECK-SAME:               [[VAL_265:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_266:%.*]] = "tf.Ceil"([[VAL_265]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_266]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @ceil_dynamic(
// CHECK-SAME:                       [[VAL_267:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_268:%.*]] = "tf.Ceil"([[VAL_267]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_268]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @ceil_unranked(
// CHECK-SAME:                        [[VAL_269:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_270:%.*]] = "tf.Ceil"([[VAL_269]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_270]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @complex_abs(
// CHECK-SAME:                      [[VAL_271:%.*]]: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
// CHECK:           [[VAL_272:%.*]] = "tf.ComplexAbs"([[VAL_271]]) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// CHECK:           return [[VAL_272]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @cos(
// CHECK-SAME:              [[VAL_273:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_274:%.*]] = "tf.Cos"([[VAL_273]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_274]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @cos_dynamic(
// CHECK-SAME:                      [[VAL_275:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_276:%.*]] = "tf.Cos"([[VAL_275]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_276]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @cos_unranked(
// CHECK-SAME:                       [[VAL_277:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_278:%.*]] = "tf.Cos"([[VAL_277]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_278]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @exp(
// CHECK-SAME:              [[VAL_279:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_280:%.*]] = "tf.Exp"([[VAL_279]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_280]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @exp_dynamic(
// CHECK-SAME:                      [[VAL_281:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_282:%.*]] = "tf.Exp"([[VAL_281]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_282]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @exp_unranked(
// CHECK-SAME:                       [[VAL_283:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_284:%.*]] = "tf.Exp"([[VAL_283]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_284]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @floor(
// CHECK-SAME:                [[VAL_285:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_286:%.*]] = "tf.Floor"([[VAL_285]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_286]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @floor_dynamic(
// CHECK-SAME:                        [[VAL_287:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_288:%.*]] = "tf.Floor"([[VAL_287]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_288]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @floor_unranked(
// CHECK-SAME:                         [[VAL_289:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_290:%.*]] = "tf.Floor"([[VAL_289]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_290]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @is_finite(
// CHECK-SAME:                    [[VAL_291:%.*]]: tensor<2xf32>) -> tensor<2xi1> {
// CHECK:           [[VAL_292:%.*]] = "tf.IsFinite"([[VAL_291]]) : (tensor<2xf32>) -> tensor<2xi1>
// CHECK:           return [[VAL_292]] : tensor<2xi1>
// CHECK:         }

// CHECK-LABEL:   func @is_finite_dynamic(
// CHECK-SAME:                            [[VAL_293:%.*]]: tensor<?xf32>) -> tensor<?xi1> {
// CHECK:           [[VAL_294:%.*]] = "tf.IsFinite"([[VAL_293]]) : (tensor<?xf32>) -> tensor<?xi1>
// CHECK:           return [[VAL_294]] : tensor<?xi1>
// CHECK:         }

// CHECK-LABEL:   func @is_finite_unranked(
// CHECK-SAME:                             [[VAL_295:%.*]]: tensor<*xf32>) -> tensor<*xi1> {
// CHECK:           [[VAL_296:%.*]] = "tf.IsFinite"([[VAL_295]]) : (tensor<*xf32>) -> tensor<*xi1>
// CHECK:           return [[VAL_296]] : tensor<*xi1>
// CHECK:         }

// CHECK-LABEL:   func @log(
// CHECK-SAME:              [[VAL_297:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_298:%.*]] = "tf.Log"([[VAL_297]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_298]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @log_dynamic(
// CHECK-SAME:                      [[VAL_299:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_300:%.*]] = "tf.Log"([[VAL_299]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_300]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @log_unranked(
// CHECK-SAME:                       [[VAL_301:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_302:%.*]] = "tf.Log"([[VAL_301]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_302]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @log1p(
// CHECK-SAME:                [[VAL_303:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_304:%.*]] = "tf.Log1p"([[VAL_303]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_304]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @log1p_dynamic(
// CHECK-SAME:                        [[VAL_305:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_306:%.*]] = "tf.Log1p"([[VAL_305]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_306]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @log1p_unranked(
// CHECK-SAME:                         [[VAL_307:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_308:%.*]] = "tf.Log1p"([[VAL_307]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_308]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @neg(
// CHECK-SAME:              [[VAL_309:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_310:%.*]] = "tf.Neg"([[VAL_309]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_310]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @neg_dynamic(
// CHECK-SAME:                      [[VAL_311:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_312:%.*]] = "tf.Neg"([[VAL_311]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_312]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @neg_unranked(
// CHECK-SAME:                       [[VAL_313:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_314:%.*]] = "tf.Neg"([[VAL_313]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_314]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @sigmoid(
// CHECK-SAME:                  [[VAL_315:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_316:%.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
// CHECK:           [[VAL_317:%.*]] = "tf.Const"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAL_318:%.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK:           [[VAL_319:%.*]] = "tf.Mul"([[VAL_315]], [[VAL_318]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           [[VAL_320:%.*]] = "tf.Tanh"([[VAL_319]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           [[VAL_321:%.*]] = "tf.Mul"([[VAL_320]], [[VAL_318]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           [[VAL_322:%.*]] = "tf.AddV2"([[VAL_321]], [[VAL_318]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_322]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @sin(
// CHECK-SAME:              [[VAL_323:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_324:%.*]] = "tf.Sin"([[VAL_323]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_324]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @sin_dynamic(
// CHECK-SAME:                      [[VAL_325:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_326:%.*]] = "tf.Sin"([[VAL_325]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_326]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @sin_unranked(
// CHECK-SAME:                       [[VAL_327:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_328:%.*]] = "tf.Sin"([[VAL_327]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_328]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @rsqrt(
// CHECK-SAME:                [[VAL_329:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_330:%.*]] = "tf.Rsqrt"([[VAL_329]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_330]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @rsqrt_dynamic(
// CHECK-SAME:                        [[VAL_331:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_332:%.*]] = "tf.Rsqrt"([[VAL_331]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_332]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @rsqrt_unranked(
// CHECK-SAME:                         [[VAL_333:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_334:%.*]] = "tf.Rsqrt"([[VAL_333]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_334]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @sqrt(
// CHECK-SAME:               [[VAL_335:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_336:%.*]] = "tf.Sqrt"([[VAL_335]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_336]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @sqrt_dynamic(
// CHECK-SAME:                       [[VAL_337:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_338:%.*]] = "tf.Sqrt"([[VAL_337]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_338]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @sqrt_unranked(
// CHECK-SAME:                        [[VAL_339:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_340:%.*]] = "tf.Sqrt"([[VAL_339]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_340]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @tanh(
// CHECK-SAME:               [[VAL_341:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_342:%.*]] = "tf.Tanh"([[VAL_341]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_342]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @tanh_dynamic(
// CHECK-SAME:                       [[VAL_343:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_344:%.*]] = "tf.Tanh"([[VAL_343]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_344]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @tanh_unranked(
// CHECK-SAME:                        [[VAL_345:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_346:%.*]] = "tf.Tanh"([[VAL_345]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_346]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @bitcast(
// CHECK-SAME:                  [[VAL_347:%.*]]: tensor<2xf32>) -> tensor<2xf32> {
// CHECK:           [[VAL_348:%.*]] = "tf.Bitcast"([[VAL_347]]) : (tensor<2xf32>) -> tensor<2xf32>
// CHECK:           return [[VAL_348]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @bitcast_dynamic(
// CHECK-SAME:                          [[VAL_349:%.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAL_350:%.*]] = "tf.Bitcast"([[VAL_349]]) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_350]] : tensor<?xf32>
// CHECK:         }

// CHECK-LABEL:   func @bitcast_unranked(
// CHECK-SAME:                           [[VAL_351:%.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAL_352:%.*]] = "tf.Bitcast"([[VAL_351]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_352]] : tensor<*xf32>
// CHECK:         }

// CHECK-LABEL:   func @bitcast_same_widths(
// CHECK-SAME:                              [[VAL_353:%.*]]: tensor<2xf32>) -> tensor<2xi32> {
// CHECK:           [[VAL_354:%.*]] = "tf.Bitcast"([[VAL_353]]) : (tensor<2xf32>) -> tensor<2xi32>
// CHECK:           return [[VAL_354]] : tensor<2xi32>
// CHECK:         }

// CHECK-LABEL:   func @sign(
// CHECK-SAME:               [[VAL_355:%.*]]: tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32> {
// CHECK:           [[VAL_356:%.*]] = "tf.NotEqual"([[VAL_355]], [[VAL_355]]) {incompatible_shape_error = true} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi1>
// CHECK:           [[VAL_357:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1x2x3x4xf32>} : () -> tensor<1x2x3x4xf32>
// CHECK:           [[VAL_358:%.*]] = "tf.NotEqual"([[VAL_355]], [[VAL_355]]) {incompatible_shape_error = true} : (tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xi1>
// CHECK:           [[VAL_359:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1x2x3x4xf32>} : () -> tensor<1x2x3x4xf32>
// CHECK:           [[VAL_360:%.*]] = "tf.Sign"([[VAL_355]]) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK:           [[VAL_361:%.*]] = "tf.Select"([[VAL_358]], [[VAL_359]], [[VAL_360]]) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK:           [[VAL_362:%.*]] = "tf.Select"([[VAL_356]], [[VAL_357]], [[VAL_361]]) : (tensor<1x2x3x4xi1>, tensor<1x2x3x4xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
// CHECK:           return [[VAL_362]] : tensor<1x2x3x4xf32>
// CHECK:         }

// CHECK-LABEL:   func @size_rank_one_i32(
// CHECK-SAME:                            [[VAL_363:%.*]]: tensor<f32>) -> tensor<i32> {
// CHECK:           [[VAL_364:%.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK:           return [[VAL_364]] : tensor<i32>
// CHECK:         }

// CHECK-LABEL:   func @size_rank_one_i64(
// CHECK-SAME:                            [[VAL_365:%.*]]: tensor<f32>) -> tensor<i64> {
// CHECK:           [[VAL_366:%.*]] = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
// CHECK:           return [[VAL_366]] : tensor<i64>
// CHECK:         }

// CHECK-LABEL:   func @complex(
// CHECK-SAME:                  [[VAL_367:%.*]]: tensor<3xf32>, [[VAL_368:%.*]]: tensor<3xf32>) -> tensor<3xcomplex<f32>> {
// CHECK:           [[VAL_369:%.*]] = "tf.Complex"([[VAL_367]], [[VAL_368]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
// CHECK:           return [[VAL_369]] : tensor<3xcomplex<f32>>
// CHECK:         }

// CHECK-LABEL:   func @convert_i32_f32(
// CHECK-SAME:                          [[VAL_370:%.*]]: tensor<2xi32>) -> tensor<2xf32> {
// CHECK:           [[VAL_371:%.*]] = "tf.Cast"([[VAL_370]]) {Truncate = false} : (tensor<2xi32>) -> tensor<2xf32>
// CHECK:           return [[VAL_371]] : tensor<2xf32>
// CHECK:         }

// CHECK-LABEL:   func @convert_slice(
// CHECK-SAME:                          [[VAL_372:%.*]]: tensor<1x4672xf32>) -> tensor<1x519xf32> {
// CHECK:           [[VAL_373:%.*]] = "tf.Const"() {value = dense<[0, 4153]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_374:%.*]] = "tf.Const"() {value = dense<[1, 519]> : tensor<2xi64>} : () -> tensor<2xi64>
// CHECK:           [[VAL_375:%.*]] = "tf.Slice"([[VAL_372]], [[VAL_373]], [[VAL_374]]) : (tensor<1x4672xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x519xf32>
// CHECK:           return [[VAL_375]] : tensor<1x519xf32>
// CHECK:         }

// CHECK-LABEL:   func @reshape(
// CHECK-SAME:                  [[VAL_372:%.*]]: tensor<4x6xf32>) -> tensor<2x2x6xf32> {
// CHECK:           [[VAL_373:%.*]] = constant dense<[2, 2, 6]> : tensor<3xi64>
// CHECK:           [[VAL_374:%.*]] = "tf.Reshape"([[VAL_372]], [[VAL_373]]) : (tensor<4x6xf32>, tensor<3xi64>) -> tensor<2x2x6xf32>
// CHECK:           return [[VAL_374]] : tensor<2x2x6xf32>
// CHECK:         }

// CHECK-LABEL:   func @convert_dot_1d_2d(
// CHECK-SAME:                            [[VAL_376:%.*]]: tensor<256xf32>, [[VAL_377:%.*]]: tensor<256x1xf32>) -> tensor<1xf32> {
// CHECK:           [[VAL_378:%.*]] = "tf.Reshape"([[VAL_376]], {{.*}}) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:           [[VAL_379:%.*]] = "tf.MatMul"([[VAL_378]], [[VAL_377]]) {transpose_a = false, transpose_b = false} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:           [[VAL_380:%.*]] = "tf.Reshape"([[VAL_379]], {{.*}}) : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           return [[VAL_380]] : tensor<1xf32>
// CHECK:         }

// CHECK-LABEL:   func @convert_dot_2d_1d(
// CHECK-SAME:                            [[VAL_381:%.*]]: tensor<1x256xf32>, [[VAL_382:%.*]]: tensor<256xf32>) -> tensor<1xf32> {
// CHECK:           [[VAL_383:%.*]] = "tf.Reshape"([[VAL_382]], {{.*}}) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:           [[VAL_384:%.*]] = "tf.MatMul"([[VAL_381]], [[VAL_383]]) {transpose_a = false, transpose_b = true} : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x1xf32>
// CHECK:           [[VAL_385:%.*]] = "tf.Reshape"([[VAL_384]], {{.*}}) : (tensor<1x1xf32>, tensor<1xi64>) -> tensor<1xf32>
// CHECK:           return [[VAL_385]] : tensor<1xf32>
// CHECK:         }

// CHECK-LABEL:   func @convert_dot_1d_1d(
// CHECK-SAME:                            [[VAL_386:%.*]]: tensor<256xf32>, [[VAL_387:%.*]]: tensor<256xf32>) -> tensor<f32> {
// CHECK-DAG:       [[VAL_388:%.*]] = "tf.Reshape"([[VAL_386]], {{.*}}) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK-DAG:       [[VAL_389:%.*]] = "tf.Reshape"([[VAL_387]], {{.*}}) : (tensor<256xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:           [[VAL_390:%.*]] = "tf.MatMul"([[VAL_388]], [[VAL_389]]) {transpose_a = false, transpose_b = true} : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x1xf32>
// CHECK:           [[VAL_391:%.*]] = "tf.Reshape"([[VAL_390]], {{.*}}) : (tensor<1x1xf32>, tensor<0xi64>) -> tensor<f32>
// CHECK:           return [[VAL_391]] : tensor<f32>
// CHECK:         }

// CHECK-LABEL:   func @convert_dot_2d_2d(
// CHECK-SAME:                            [[VAL_392:%.*]]: tensor<1x256xf32>, [[VAL_393:%.*]]: tensor<256x1xf32>) -> tensor<1x1xf32> {
// CHECK:           [[VAL_394:%.*]] = "tf.MatMul"([[VAL_392]], [[VAL_393]]) {transpose_a = false, transpose_b = false} : (tensor<1x256xf32>, tensor<256x1xf32>) -> tensor<1x1xf32>
// CHECK:           return [[VAL_394]] : tensor<1x1xf32>
// CHECK:         }
