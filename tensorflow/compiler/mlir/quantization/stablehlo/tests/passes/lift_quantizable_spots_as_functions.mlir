// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-lift-quantizable-spots-as-functions | FileCheck %s

// CHECK-LABEL: @conv_fn(
// CHECK-SAME:          %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  func.return %1: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: return %[[CONV]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: return %[[DOT_GENERAL:.*]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_same_shape_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x2xf32>
func.func @dot_general_with_bias_same_shape_fn(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<2x3xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x3xf32>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  %3 = stablehlo.add %2, %1 : tensor<1x3xf32>
  func.return %3: tensor<1x3xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_same_shape_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK: return %[[ADD]] : tensor<1x3xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_bias_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %3 = stablehlo.broadcast_in_dim %1, dims = [3] : (tensor<4xf32>) -> tensor<1x3x3x4xf32>
  %4 = stablehlo.add %2, %3 : tensor<1x3x3x4xf32>
  func.return %4: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST_IN_DIM]]
// CHECK: return %[[ADD]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_bias_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<64xf32>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %3 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
  %4 = stablehlo.add %2, %3 : tensor<1x1x64xf32>
  func.return %4: tensor<1x1x64xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[BROADCAST_IN_DIM]]
// CHECK: return %[[ADD]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_dynamic_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %3 = shape.shape_of %2 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %5 = stablehlo.add %2, %4 : tensor<?x28x28x16xf32>
  func.return %5: tensor<?x28x28x16xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x28x28x16xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_dynamic_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[SHAPE_OF:.*]] = shape.shape_of %[[CONV]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF]]
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[DYNAMIC_BROADCAST_IN_DIM]]
// CHECK: return %[[ADD]] : tensor<?x28x28x16xf32>
// CHECK: }

// -----

// Because the operand of shape_of is other than the target conv,
// should not match conv bias pattern.

// CHECK-LABEL: @conv_with_bias_dynamic_shape_not_same_op_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_dynamic_shape_not_same_op_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %4 = shape.shape_of %3 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %6 = stablehlo.add %2, %5 : tensor<?x28x28x16xf32>
  func.return %6: tensor<?x28x28x16xf32>
}
// CHECK-NOT: @composite_conv_with_bias_dynamic_fn_1

// -----

// CHECK-LABEL: @dot_general_with_bias_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x12544xf32>
func.func @dot_general_with_bias_dynamic_fn(%arg0: tensor<?x12544xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<12544x10xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<10xf32>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x12544xf32>, tensor<12544x10xf32>) -> tensor<?x10xf32>
  %3 = shape.shape_of %2 : tensor<?x10xf32> -> tensor<2xindex>
  %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [1] : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
  %5 = stablehlo.add %2, %4 : tensor<?x10xf32>
  func.return %5: tensor<?x10xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x10xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_dynamic_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[SHAPE_OF_0:.*]] = shape.shape_of %[[DOT_GENERAL]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF_0]]
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[DYNAMIC_BROADCAST_IN_DIM_0]]
// CHECK: return %[[ADD]] : tensor<?x10xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_relu_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %3 = stablehlo.maximum %2, %1 : tensor<1x3x3x4xf32>
  func.return %3: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[CONV]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_relu_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>,
func.func @dot_general_with_relu_fn(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %3 = stablehlo.maximum %2, %1 : tensor<1x1x64xf32>
  return %3 : tensor<1x1x64xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu_fn_1
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[DOT_GENERAL]], %[[CONST]]
// CHECK: return %[[MAX:.*]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_relu_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_relu_dynamic_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %3 = shape.shape_of %2 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %5 = stablehlo.maximum %2, %4 : tensor<?x28x28x16xf32>
  func.return %5: tensor<?x28x28x16xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x28x28x16xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu_dynamic_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[SHAPE_OF:.*]] = shape.shape_of %[[CONV]]
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST]], %[[SHAPE_OF]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[CONV]], %[[DYNAMIC_BROADCAST_IN_DIM]]
// CHECK: return %[[MAX]] : tensor<?x28x28x16xf32>
// CHECK: }

// -----

// Because the operand of shape_of is other than the target conv,
// should not match conv relu dynamic pattern.

// CHECK-LABEL: @conv_with_relu_dynamic_shape_not_same_op_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_relu_dynamic_shape_not_same_op_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %4 = shape.shape_of %3 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %6 = stablehlo.maximum %2, %5 : tensor<?x28x28x16xf32>
  func.return %6: tensor<?x28x28x16xf32>
}
// CHECK-NOT: private @composite_conv_with_relu_dynamic_fn_1

// -----

// CHECK-LABEL: @dot_general_with_relu_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x12544xf32>
func.func @dot_general_with_relu_dynamic_fn(%arg0: tensor<?x12544xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<12544x10xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x12544xf32>, tensor<12544x10xf32>) -> tensor<?x10xf32>
  %3 = shape.shape_of %2 : tensor<?x10xf32> -> tensor<2xindex>
  %4 = stablehlo.dynamic_broadcast_in_dim %1, %3, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
  %5 = stablehlo.maximum %2, %4 : tensor<?x10xf32>
  func.return %5: tensor<?x10xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x10xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu_dynamic_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[SHAPE_OF:.*]] = shape.shape_of %[[DOT_GENERAL]]
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST]], %[[SHAPE_OF]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[DOT_GENERAL]], %[[DYNAMIC_BROADCAST_IN_DIM]]
// CHECK: return %[[MAX]] : tensor<?x10xf32>
// CHECK: }

// -----

// The pattern should not match when the const value for relu is not 0.

// CHECK-LABEL: @conv_with_relu_wrong_const_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_relu_wrong_const_fn(%arg0: tensor<1x3x3x4xf32>, %arg1: tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x3x3x4xf32>
  %2 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %3 = stablehlo.maximum %2, %1 : tensor<1x3x3x4xf32>
  func.return %3: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]])
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[XLA_CALL_MODULE]], %[[CONST_1]]
// CHECK: return %[[MAX]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_fn_1
// CHECK-NOT: private @composite_conv_with_relu_fn_1

// -----

// CHECK-LABEL: @conv_with_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_relu6_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %2 = stablehlo.constant dense<6.000000e+00> : tensor<1x3x3x4xf32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %4 = stablehlo.clamp %1, %3, %2 : tensor<1x3x3x4xf32>
  func.return %4: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu6_fn_1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[CONV]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_relu6_fn(
// CHECK-SAME:                 %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_relu6_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.constant dense<6.000000e+00> : tensor<1x1x64xf32>
  %3 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %4 = stablehlo.clamp %1, %3, %2 : tensor<1x1x64xf32>
  return %4 : tensor<1x1x64xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu6_fn_1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[DOT_GENERAL]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_relu6_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_relu6_dynamic_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %4 = stablehlo.clamp %1, %3, %2 : (tensor<f32>, tensor<?x28x28x16xf32>, tensor<f32>) -> tensor<?x28x28x16xf32>
  func.return %4: tensor<?x28x28x16xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x28x28x16xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_relu6_fn_1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[CONV]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<?x28x28x16xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_relu6_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x12544xf32>
func.func @dot_general_with_relu6_dynamic_fn(%arg0: tensor<?x12544xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<12544x10xf32>
  %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %3 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x12544xf32>, tensor<12544x10xf32>) -> tensor<?x10xf32>
  %4 = stablehlo.clamp %1, %3, %2 : (tensor<f32>, tensor<?x10xf32>, tensor<f32>) -> tensor<?x10xf32>
  func.return %4: tensor<?x10xf32>
}
// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x10xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_relu6_fn_1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[DOT_GENERAL]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<?x10xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_same_shape_and_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_bias_same_shape_and_relu_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %3 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %4 = stablehlo.add %3, %1 : tensor<1x1x64xf32>
  %5 = stablehlo.maximum %4, %2 : tensor<1x1x64xf32>
  func.return %5: tensor<1x1x64xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_same_shape_and_relu_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_bias_and_relu_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %4 = stablehlo.broadcast_in_dim %1, dims = [3] : (tensor<4xf32>) -> tensor<1x3x3x4xf32>
  %5 = stablehlo.add %3, %4 : tensor<1x3x3x4xf32>
  %6 = stablehlo.maximum %5, %2 : tensor<1x3x3x4xf32>
  func.return %6: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST_IN_DIM]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_bias_and_relu_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<64xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %3 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %4 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
  %5 = stablehlo.add %3, %4 : tensor<1x1x64xf32>
  %6 = stablehlo.maximum %5, %2 : tensor<1x1x64xf32>
  func.return %6: tensor<1x1x64xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[BROADCAST_IN_DIM]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CONST]]
// CHECK: return %[[MAX]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_and_relu_dynamic_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %4 = shape.shape_of %3 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %6 = stablehlo.add %3, %5 : tensor<?x28x28x16xf32>
  %7 = shape.shape_of %6 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %8 = stablehlo.dynamic_broadcast_in_dim %2, %7, dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %9 = stablehlo.maximum %6, %8 : tensor<?x28x28x16xf32>
  func.return %9: tensor<?x28x28x16xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x28x28x16xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu_dynamic_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[SHAPE_OF_0:.*]] = shape.shape_of %[[CONV]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF_0]]
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[DYNAMIC_BROADCAST_IN_DIM_0]]
// CHECK: %[[SHAPE_OF_1:.*]] = shape.shape_of %[[ADD]]
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_1:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST]], %[[SHAPE_OF_1]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[DYNAMIC_BROADCAST_IN_DIM_1]]
// CHECK: return %[[MAX]] : tensor<?x28x28x16xf32>
// CHECK: }

// -----

// Because the operand of shape_of is other than the target conv,
// should not match conv bias relu dynamic pattern.

// CHECK-LABEL: @conv_with_bias_and_relu_dynamic_shape_not_same_op_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_and_relu_dynamic_shape_not_same_op_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %4 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %5 = shape.shape_of %4 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %7 = stablehlo.add %3, %6 : tensor<?x28x28x16xf32>
  %8 = shape.shape_of %7 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %9 = stablehlo.dynamic_broadcast_in_dim %2, %8, dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %10 = stablehlo.maximum %7, %9 : tensor<?x28x28x16xf32>
  func.return %10: tensor<?x28x28x16xf32>
}
// CHECK-NOT: private @composite_conv_with_bias_and_relu_dynamic_fn_1

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x12544xf32>
func.func @dot_general_with_bias_and_relu_dynamic_fn(%arg0: tensor<?x12544xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<12544x10xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<10xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x12544xf32>, tensor<12544x10xf32>) -> tensor<?x10xf32>
  %4 = shape.shape_of %3 : tensor<?x10xf32> -> tensor<2xindex>
  %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [1] : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
  %6 = stablehlo.add %3, %5 : tensor<?x10xf32>
  %7 = shape.shape_of %6 : tensor<?x10xf32> -> tensor<2xindex>
  %8 = stablehlo.dynamic_broadcast_in_dim %2, %7, dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
  %9 = stablehlo.maximum %6, %8 : tensor<?x10xf32>
  func.return %9: tensor<?x10xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x10xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu_dynamic_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[SHAPE_OF_0:.*]] = shape.shape_of %[[DOT_GENERAL]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF_0]]
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[DYNAMIC_BROADCAST_IN_DIM_0]]
// CHECK: %[[SHAPE_OF_1:.*]] = shape.shape_of %[[ADD]]
// CHECK-DAG: %[[CONST:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_1:.*]] = stablehlo.dynamic_broadcast_in_dim %[[CONST]], %[[SHAPE_OF_1]]
// CHECK: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[DYNAMIC_BROADCAST_IN_DIM_1]]
// CHECK: return %[[MAX]] : tensor<?x10xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_same_shape_and_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_bias_same_shape_and_relu6_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<1x1x64xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<1x1x64xf32>
  %4 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %5 = stablehlo.add %4, %1 : tensor<1x1x64xf32>
  %6 = stablehlo.clamp %2, %5, %3 : tensor<1x1x64xf32>
  func.return %6: tensor<1x1x64xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_same_shape_and_relu6_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg2
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x3x3x4xf32>
func.func @conv_with_bias_and_relu6_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x3x3x4xf32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<1x3x3x4xf32>
  %4 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
  %5 = stablehlo.broadcast_in_dim %1, dims = [3] : (tensor<4xf32>) -> tensor<1x3x3x4xf32>
  %6 = stablehlo.add %4, %5 : tensor<1x3x3x4xf32>
  %7 = stablehlo.clamp %2, %6, %3 : tensor<1x3x3x4xf32>
  func.return %7: tensor<1x3x3x4xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x3x3x4xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu6_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[BROADCAST_IN_DIM]]
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x3x3x4xf32>
// CHECK: }

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu6_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<1x1x167xf32>
func.func @dot_general_with_bias_and_relu6_fn(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<64xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<1x1x64xf32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<1x1x64xf32>
  %4 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  %5 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<64xf32>) -> tensor<1x1x64xf32>
  %6 = stablehlo.add %4, %5 : tensor<1x1x64xf32>
  %7 = stablehlo.clamp %2, %6, %3 : tensor<1x1x64xf32>
  func.return %7: tensor<1x1x64xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu6_fn_1
// CHECK: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg2
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[BROADCAST_IN_DIM]]
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<1x1x64xf32>
// CHECK: }

// -----

// CHECK-LABEL: @conv_with_bias_and_relu6_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_and_relu6_dynamic_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %4 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %5 = shape.shape_of %4 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %7 = stablehlo.add %4, %6 : tensor<?x28x28x16xf32>
  %8 = stablehlo.clamp %2, %7, %3 : (tensor<f32>, tensor<?x28x28x16xf32>, tensor<f32>) -> tensor<?x28x28x16xf32>
  func.return %8: tensor<?x28x28x16xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x28x28x16xf32>
// CHECK: }

// CHECK-LABEL: private @composite_conv_with_bias_and_relu6_dynamic_fn_1
// CHECK: %[[CONV:.*]] = stablehlo.convolution(%arg0, %arg1)
// CHECK: %[[SHAPE_OF_0:.*]] = shape.shape_of %[[CONV]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF_0]]
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[CONV]], %[[DYNAMIC_BROADCAST_IN_DIM_0]]
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<?x28x28x16xf32>
// CHECK: }

// -----

// Because the operand of shape_of is other than the target conv,
// should not match conv bias relu6 dynamic pattern.

// CHECK-LABEL: @conv_with_bias_and_relu6_dynamic_shape_not_same_op_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x28x28x1xf32>
func.func @conv_with_bias_and_relu6_dynamic_shape_not_same_op_fn(%arg0: tensor<?x28x28x1xf32>) -> tensor<?x28x28x16xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x1x16xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<16xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %4 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %5 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x28x28x1xf32>, tensor<3x3x1x16xf32>) -> tensor<?x28x28x16xf32>
  %6 = shape.shape_of %5 : tensor<?x28x28x16xf32> -> tensor<4xindex>
  %7 = stablehlo.dynamic_broadcast_in_dim %1, %6, dims = [3] : (tensor<16xf32>, tensor<4xindex>) -> tensor<?x28x28x16xf32>
  %8 = stablehlo.add %4, %7 : tensor<?x28x28x16xf32>
  %9 = stablehlo.clamp %2, %8, %3 : (tensor<f32>, tensor<?x28x28x16xf32>, tensor<f32>) -> tensor<?x28x28x16xf32>
  func.return %9: tensor<?x28x28x16xf32>
}
// CHECK-NOT: private @composite_conv_with_bias_and_relu6_dynamic_fn_1

// -----

// CHECK-LABEL: @dot_general_with_bias_and_relu6_dynamic_fn(
// CHECK-SAME:                    %[[ARG_0:.*]]: tensor<?x12544xf32>
func.func @dot_general_with_bias_and_relu6_dynamic_fn(%arg0: tensor<?x12544xf32>) -> tensor<?x10xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<12544x10xf32>
  %1 = stablehlo.constant dense<2.000000e+00> : tensor<10xf32>
  %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %3 = stablehlo.constant dense<6.000000e+00> : tensor<f32>
  %4 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x12544xf32>, tensor<12544x10xf32>) -> tensor<?x10xf32>
  %5 = shape.shape_of %4 : tensor<?x10xf32> -> tensor<2xindex>
  %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [1] : (tensor<10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
  %7 = stablehlo.add %4, %6 : tensor<?x10xf32>
  %8 = stablehlo.clamp %2, %7, %3 : (tensor<f32>, tensor<?x10xf32>, tensor<f32>) -> tensor<?x10xf32>
  func.return %8: tensor<?x10xf32>
}
// CHECK: %[[CONST_0:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[CONST_1:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST_0]], %[[CONST_1]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<?x10xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_with_bias_and_relu6_dynamic_fn_1
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: %[[SHAPE_OF_0:.*]] = shape.shape_of %[[DOT_GENERAL]]
// CHECK: %[[DYNAMIC_BROADCAST_IN_DIM_0:.*]] = stablehlo.dynamic_broadcast_in_dim %arg2, %[[SHAPE_OF_0]]
// CHECK-DAG: %[[CONST_1:.*]] = stablehlo.constant dense<6.000000e+00>
// CHECK: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[DYNAMIC_BROADCAST_IN_DIM_0]]
// CHECK-DAG: %[[CONST_0:.*]] = stablehlo.constant dense<0.000000e+00>
// CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[CONST_0]], %[[ADD]], %[[CONST_1]]
// CHECK: return %[[CLAMP]] : tensor<?x10xf32>
// CHECK: }

// -----

// CHECK-LABEL: @gather_fn(
func.func @gather_fn() -> tensor<2x3x2x2xi32> {
  %0 = stablehlo.constant dense<1> : tensor<3x4x2xi32>
  %1 = stablehlo.constant dense<1> : tensor<2x3x2xi64>
  %2 = "stablehlo.gather"(%0, %1) {
  dimension_numbers = #stablehlo.gather<
    offset_dims = [2, 3],
    collapsed_slice_dims = [0],
    start_index_map = [1, 0],
    index_vector_dim = 2>,
  slice_sizes = array<i64: 1, 2, 2>,
  indices_are_sorted = false
} : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
  func.return %2: tensor<2x3x2x2xi32>
}
// CHECK: %[[OPERAND:.*]] = stablehlo.constant
// CHECK: %[[INDICES:.*]] = stablehlo.constant
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%[[OPERAND]], %[[INDICES]])
// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<2x3x2x2xi32>
// CHECK: }

// CHECK-LABEL: private @composite_gather_fn_1
// CHECK: %[[GATHER:.*]] = "stablehlo.gather"(%arg0, %arg1)
// CHECK: return %[[GATHER]] : tensor<2x3x2x2xi32>
// CHECK: }

// -----

// Test that the name of composite functions are deterministic. There are 3
// unsorted functions in this module and each function has 2 quantizable ops.
module {
  func.func @conv_3_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
    %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    func.return %2: tensor<1x3x3x4xf32>
  }

  func.func @conv_1_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
    %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    func.return %2: tensor<1x3x3x4xf32>
  }

  func.func @conv_2_fn(%arg0: tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32> {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<3x3x4x4xf32>
    %1 = stablehlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    %2 = stablehlo.convolution(%1, %0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x3x4xf32>, tensor<3x3x4x4xf32>) -> tensor<1x3x3x4xf32>
    func.return %2: tensor<1x3x3x4xf32>
  }
}

// CHECK-LABEL: @conv_3_fn
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_6, _original_entry_function = "composite_conv_fn_6"
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_5, _original_entry_function = "composite_conv_fn_5"

// CHECK-LABEL: @conv_1_fn
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_2, _original_entry_function = "composite_conv_fn_2"
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_1, _original_entry_function = "composite_conv_fn_1"

// CHECK-LABEL: @conv_2_fn
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_4, _original_entry_function = "composite_conv_fn_4"
// CHECK: tf.XlaCallModule
// CHECK-SAME: _entry_function = @composite_conv_fn_3, _original_entry_function = "composite_conv_fn_3"