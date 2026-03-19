// RUN: tf-quant-opt %s -split-input-file -quant-convert-fake-quant-to-qdq -quant-lift-quantizable-spots-as-functions='target-opset=XLA' -quant-insert-quantized-functions -quant-quantize-composite-functions='target-opset=XLA' -symbol-dce -inline -tf-shape-inference -canonicalize -quant-replace-cast-hacks-with-tf-xla-ops -cse -quant-optimize | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1219 : i32}, tf_saved_model.semantics} {
  func.func @conv_with_multiple_uses(%arg0: tensor<1x3x4x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<1x3x2x2xf32> {tf_saved_model.index_path = ["output"]}, tensor<1x3x2x1xf32> {tf_saved_model.index_path = ["output2"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input:0", outputs = "PartitionedCall:0, Sum:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[[[[-0.315365672, 0.27481091], [0.0901821703, -0.382271349], [-0.105572946, -0.354302853]], [[-0.47703138, -0.307006568], [0.306320101, -0.209111646], [0.252869487, 0.449634969]], [[0.167675957, 0.042408213], [-0.332338423, -0.397738814], [0.290657759, 0.460783273]]], [[[0.0693112761, 0.231933162], [0.477371335, -0.0718854442], [-0.398417652, 0.449998438]], [[0.0494867712, -0.241692379], [-0.363851488, 0.0586083047], [0.466867805, 0.0364450105]], [[0.256431073, 0.44932279], [0.0775043964, -0.192745745], [0.185018882, 0.463297218]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %cst_0 = "tf.Const"() {device = "", value = dense<[[[[0.211401448, 0.205456927], [0.418355644, -0.314615548]], [[0.493921608, -0.101286061], [-0.16083248, -0.0546654463]], [[-0.157245964, 0.419805884], [-0.0499645844, 3.726310e-01]]], [[[-0.353424132, 0.361233443], [0.391344249, -0.364820778]], [[-0.476781279, -0.180014133], [-0.302823931, 0.199466437]], [[-0.385851651, 0.0372837223], [-0.0986057966, -0.0732412189]]]]> : tensor<2x3x2x2xf32>} : () -> tensor<2x3x2x2xf32>
    %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {device = "", max = 2.000000e-01 : f32, min = -1.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<1x3x2x2xf32>
    %2 = "tf.Relu"(%1) {device = ""} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %3 = "tf.FakeQuantWithMinMaxArgs"(%2) {device = "", max = 4.000000e-01 : f32, min = -3.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %4 = "tf.Conv2D"(%3, %cst_0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<1x3x2x2xf32>, tensor<2x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %5 = "tf.FakeQuantWithMinMaxArgs"(%4) {device = "", max = 8.000000e-01 : f32, min = -6.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    %dimension = "tf.Const"() { value = dense<3> : tensor<1xi64> } : () -> tensor<1xi64>
    %6 = "tf.Sum"(%3, %dimension) { keep_dims = true }: (tensor<1x3x2x2xf32>, tensor<1xi64>) -> tensor<1x3x2x1xf32>
    return %5, %6 : tensor<1x3x2x2xf32>, tensor<1x3x2x1xf32>
  }

// CHECK-LABEL: func @conv_with_multiple_uses
// CHECK: %[[div:.*]] = "tf.Div"(%arg0
// CHECK: %[[add:.*]] = "tf.AddV2"(%[[div]]
// CHECK: %[[maximum:.*]] = "tf.Maximum"(%[[add]]
// CHECK: %[[minimum:.*]] = "tf.Minimum"(%[[maximum]]
// CHECK: %[[round:.*]] = "tf.Round"(%[[minimum]]
// CHECK: %[[quant:.*]] = "tf.Cast"(%[[round]]) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3xi8>
// CHECK: %[[pad:.*]] = "tf.PadV2"(%[[quant]]
// CHECK: %[[xlaconv:.*]] = "tf.XlaConvV2"(%[[pad]]
// CHECK: %[[sub:.*]] = "tf.Sub"(%[[xlaconv]]
// CHECK: %[[cast:.*]] = "tf.Cast"(%[[sub]]) <{Truncate = false}> : (tensor<1x3x2x2xi32>) -> tensor<1x3x2x2xf32>
// CHECK: %[[dequant1:.*]] = "tf.Mul"(%[[cast]]
// CHECK: %[[relu:.*]] = "tf.Relu"(%[[dequant1]]
// CHECK: %[[clamped:.*]] = "tf.Minimum"(%[[relu]]

// CHECK: %[[rescale1:.*]] = "tf.Mul"(%[[cast]]
// CHECK: %[[add2:.*]] = "tf.AddV2"(%[[rescale1]]
// CHECK: %[[maximum2:.*]] = "tf.Maximum"(%[[add2]]
// CHECK: %[[minimum2:.*]] = "tf.Minimum"(%[[maximum2]]
// CHECK: %[[round2:.*]] = "tf.Round"(%[[minimum2]]
// CHECK: %[[quant2:.*]] = "tf.Cast"(%[[round2]]) <{Truncate = false}> : (tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xi8>

// CHECK: %[[pad2:.*]] = "tf.PadV2"(%[[quant2]]
// CHECK: %[[xlaconv2:.*]] = "tf.XlaConvV2"(%[[pad2]]
// CHECK: %[[sub2:.*]] = "tf.Sub"(%[[xlaconv2]]
// CHECK: %[[cast2:.*]] = "tf.Cast"(%[[sub2]]) <{Truncate = false}> : (tensor<1x3x2x2xi32>) -> tensor<1x3x2x2xf32>
// CHECK: %[[rescale2:.*]] = "tf.Mul"(%[[cast2]]
// CHECK: %[[rescale2_maxclamped:.*]] = "tf.Maximum"(%[[rescale2]]
// CHECK: %[[rescale2_minclamped:.*]] = "tf.Minimum"(%[[rescale2_maxclamped]]

// CHECK: %[[sum:.*]] = "tf.Sum"(%[[clamped]]
// CHECK: return %[[rescale2_minclamped]], %[[sum]]
}

// -----

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1219 : i32}, tf_saved_model.semantics} {
  func.func @conv_with_dynamic_shape(%arg0: tensor<?x?x?x?xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<?x?x?x2xf32> {tf_saved_model.index_path = ["output"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %cst = "tf.Const"() {device = "", value = dense<[[[[-0.315365672, 0.27481091], [0.0901821703, -0.382271349], [-0.105572946, -0.354302853]], [[-0.47703138, -0.307006568], [0.306320101, -0.209111646], [0.252869487, 0.449634969]], [[0.167675957, 0.042408213], [-0.332338423, -0.397738814], [0.290657759, 0.460783273]]], [[[0.0693112761, 0.231933162], [0.477371335, -0.0718854442], [-0.398417652, 0.449998438]], [[0.0494867712, -0.241692379], [-0.363851488, 0.0586083047], [0.466867805, 0.0364450105]], [[0.256431073, 0.44932279], [0.0775043964, -0.192745745], [0.185018882, 0.463297218]]]]> : tensor<2x3x3x2xf32>} : () -> tensor<2x3x3x2xf32>
    %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {device = "", max = 2.000000e-01 : f32, min = -1.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %1 = "tf.Conv2D"(%0, %cst) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<?x?x?x?xf32>, tensor<2x3x3x2xf32>) -> tensor<?x?x?x2xf32>
    %2 = "tf.Relu"(%1) {device = ""} : (tensor<?x?x?x2xf32>) -> tensor<?x?x?x2xf32>
    %3 = "tf.FakeQuantWithMinMaxArgs"(%2) {device = "", max = 4.000000e-01 : f32, min = -3.000000e-01 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<?x?x?x2xf32>) -> tensor<?x?x?x2xf32>
    return %3 : tensor<?x?x?x2xf32>
  }

// CHECK-LABEL: func @conv_with_dynamic_shape
// The Conv2D should not be quantized since it has dynamic channel.
// CHECK: "tf.Conv2D"
// CHECK-SAME: (tensor<?x?x?x?xf32>, tensor<2x3x3x2xf32>) -> tensor<?x?x?x2xf32>
}
