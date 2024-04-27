// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-prepare-quantize=enable-per-channel-quantized-weight=true -verify-diagnostics | FileCheck %s

// -----

module {
  // CHECK-LABEL: conv_with_bias_and_relu
  func.func private @conv_with_bias_and_relu(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32> {
    %cst = "tf.Const"() {device = "", value = dense<[7.11401462, 7.05456924]> : tensor<2xf32>} : () -> tensor<2xf32>
    // CHECK: %[[q_weight_per_channel:.*]] = "quantfork.qcast"
    // CHECK-SAME: -> tensor<2x3x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.075123051020104109,0.072960192762960605}>>
    // CHECK: %[[dq_weight:.*]] = "quantfork.dcast"(%[[q_weight_per_channel]])
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    // CHECK: %[[q_act:.*]] = "quantfork.qcast"(%arg0)
    // CHECK-SAME: -> tensor<1x3x2x3x!quant.uniform<i8:f32, 0.018920717052384919:-128>>
    // CHECK: %[[dq_act:.*]] = "quantfork.dcast"(%[[q_act]])
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[1.27501142, 4.824783]> : tensor<2xf32>} : (tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
    // CHECK: "tf.XlaCallModule"(%[[dq_act]], %[[dq_weight]]
    %1 = "tf.XlaCallModule"(%0, %cst_0, %cst) {
      Sout = [#tf_type.shape<1x2x2x2>], config = "",
      module = "composite_conv2d_with_bias_and_relu6_fn_10",
      _entry_function = @composite_conv2d_with_bias_and_relu6_fn_10,
      // Represents a per-channel quantization for the operand index 1 with
      // quantization dimension of 3
      _quantization_method = "static_range_ptq {input_quantized_types {key: 1, value {dimension_specs {dimension: 3}}}}",
      platforms = [], version = 4 : i64
    } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %2 : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: composite_conv2d_with_bias_and_relu6_fn_10
  func.func private @composite_conv2d_with_bias_and_relu6_fn_10(%arg0: tensor<1x3x2x3xf32>, %arg1: tensor<2x3x3x2xf32>, %arg2: tensor<2xf32>) -> tensor<1x2x2x2xf32> attributes {tf.tf_quant.composite_function} {
    %0 = "quantfork.stats"(%arg1) {layerStats = dense<[-3.54062747, 0.54742622]> : tensor<2xf32>} : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2xf32>
    %1 = "quantfork.stats"(%arg0) {layerStats = dense<[1.27501142, 2.824783]> : tensor<2xf32>} : (tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
    %2 = stablehlo.convolution(%1, %0)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {
        stride = [1, 1], pad = [[0, 0], [1, 1]],
        lhs_dilate = [1, 1],
        rhs_dilate = [1, 1]
      }
      {
        batch_group_count = 1 : i64,
        feature_group_count = 1 : i64
      } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>)
      -> tensor<1x2x2x2xf32>
    %3 = "quantfork.stats"(%arg2) {layerStats = dense<[7.05456924, 7.11401462]> : tensor<2xf32>} : (tensor<2xf32>) -> tensor<2xf32>
    %4 = "quantfork.stats"(%2) {layerStats = dense<[-1.36523, 3.57373]> : tensor<2xf32>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %5 = "chlo.broadcast_add"(%4, %3) : (tensor<1x2x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    %6 = "quantfork.stats"(%5) {layerStats = dense<[-1.31055, 2.62842]> : tensor<2xf32>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %cst_min = stablehlo.constant dense<0.0> : tensor<f32>
    %cst_max = stablehlo.constant dense<6.0> : tensor<f32>
    %7 = "stablehlo.clamp"(%cst_min, %6, %cst_max) {device = ""} : (tensor<f32>, tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2x2x2xf32>
    %8 = "quantfork.stats"(%7) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %8 : tensor<1x2x2x2xf32>
  }
}

// -----

module {
  // CHECK-LABEL: dot_general
  func.func private @dot_general(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    // CHECK: %[[q_weight:.*]] = "quantfork.qcast"
    // CHECK-SAME: -> tensor<2x2x!quant.uniform<i8<-127:127>:f32:1, {0.049663885371891529,0.060200210631363035}>>
    // CHECK: %[[dq_weight:.*]] = "quantfork.dcast"(%[[q_weight]])
    %cst = "tf.Const"() {device = "", value = dense<[[-6.30731344, 5.4962182], [1.80364347, -7.64542675]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
    // CHECK: %[[q_act:.*]] = "quantfork.qcast"(%arg0)
    // CHECK-SAME: -> tensor<2x2x!quant.uniform<i8:f32, 0.018920717052384919:-128>>
    // CHECK: %[[dq_act:.*]] = "quantfork.dcast"(%[[q_act]])
    %0 = "quantfork.stats"(%arg0) {layerStats = dense<[1.27501142, 4.824783]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: "tf.XlaCallModule"(%[[dq_act]], %[[dq_weight]]
    %1 = "tf.XlaCallModule"(%0, %cst) {
      Sout = [#tf_type.shape<2x2>], config = "",
      _entry_function = @composite_dot_general,
      module = "composite_dot_general",
      platforms = [], version = 4 : i64
    } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = "quantfork.stats"(%1) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %2 : tensor<2x2xf32>
  }

  // CHECK-LABEL: composite_dot_general
  func.func private @composite_dot_general(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
      >
    } : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----

// Tests that the `PrepareQuantizePass` prepares for per-tensor quantization for
// the weight of convolution. This is based on the `_quantization_method` that
// does not have a `input_quantized_types` with a specified `dimension_specs`.

// CHECK-LABEL: conv_per_tensor_quantized_method
func.func private @conv_per_tensor_quantized_method(%arg0: tensor<1x3x2x3xf32>) -> tensor<1x2x2x2xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[7.11401462, 7.05456924]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_0 = "tf.Const"() {device = "", value = dense<[[[[-6.30731344, 5.4962182], [1.80364347, -7.64542675], [-2.11145878, -7.08605719]], [[-9.54062747, -6.14013147], [6.12640238, -4.18223286], [5.05738974, 8.99269962]], [[3.3535192, 0.84816426], [-6.64676809, -7.95477629], [5.81315517, 9.21566581]]], [[[1.38622558, 4.63866329], [4.54742622, -1.43770897], [-3.96835279, 2.99996852]], [[0.989735424, -4.83384752], [-7.27702999, 1.17216611], [1.33735656, 0.728900194]], [[5.1286211, 8.98645591], [1.55008793, -3.85491467], [3.7003777, 9.26594448]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[1.27501142, 4.824783]> : tensor<2xf32>} : (tensor<1x3x2x3xf32>) -> tensor<1x3x2x3xf32>
  %1 = "tf.XlaCallModule"(%0, %cst_0, %cst) {
    Sout = [#tf_type.shape<1x2x2x2>], config = "",
    module = "composite_conv_fn_1",
    _entry_function = @composite_conv_fn_1,
    _quantization_method = "static_range_ptq {}",
    platforms = [], version = 4 : i64
  } : (tensor<1x3x2x3xf32>, tensor<2x3x3x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  %2 = "quantfork.stats"(%1) {layerStats = dense<[0.000000e+00, 6.000000e+00]> : tensor<2xf32>} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %2 : tensor<1x2x2x2xf32>
}
// CHECK-SAME: %[[ARG_0:.+]]: tensor<1x3x2x3xf32>

// Test that the weight is prepared for per-tensor quantization, based on the
// `_quantization_method` attribute without a `dimension_specs` field in
// `QuantizedType`.
// CHECK-DAG: %[[WEIGHT_CONST:.+]] = stablehlo.constant {{.*}} tensor<2x3x3x2xf32>
// CHECK: %[[Q_WEIGHT_PER_TENSOR:.*]] = "quantfork.qcast"(%[[WEIGHT_CONST]]) {{.*}} (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, {{.*}}>>
// CHECK: %[[DQ_WEIGHT:.*]] = "quantfork.dcast"(%[[Q_WEIGHT_PER_TENSOR]])

// CHECK: %[[Q_ACTIVATION:.*]] = "quantfork.qcast"(%[[ARG_0]])
// CHECK-SAME: -> tensor<1x3x2x3x!quant.uniform<i8:f32, 0.018920717052384919:-128>>
// CHECK: %[[DQ_ACTIVATION:.*]] = "quantfork.dcast"(%[[Q_ACTIVATION]])
// CHECK: "tf.XlaCallModule"(%[[DQ_ACTIVATION]], %[[DQ_WEIGHT]]
