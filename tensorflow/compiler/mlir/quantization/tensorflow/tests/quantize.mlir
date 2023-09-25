// RUN: tf-quant-opt %s -split-input-file -quant-lift-quantizable-spots-as-functions -quant-quantize -verify-each=false | FileCheck %s

func.func private @conv(%input: tensor<1x3x4x3xf32> {tf._user_specified_name = "input_tensor"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x3x4x3>]} {
  %weight = arith.constant dense_resource<__elided__> : tensor<2x3x3x2xf32>
  %bias = arith.constant dense<[7.11401462, 7.05456924]> : tensor<2xf32>

  %q_input= "quantfork.qcast"(%input) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58810077742034317:-128>>
  %dq_input= "quantfork.dcast"(%q_input) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58810077742034317:-128>>) -> tensor<1x3x4x3xf32>
  %q_weight = "quantfork.qcast"(%weight) : (tensor<2x3x3x2xf32>) -> tensor<2x3x3x2x!quant.uniform<i8:f32, 0.074855112561992565:-1>>
  %dq_weight = "quantfork.dcast"(%q_weight) : (tensor<2x3x3x2x!quant.uniform<i8:f32, 0.074855112561992565:-1>>) -> tensor<2x3x3x2xf32>
  %q_bias = "quantfork.qcast"(%bias) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i32:f32, 0.044022349891595126>>
  %dq_bias = "quantfork.dcast"(%q_bias) : (tensor<2x!quant.uniform<i32:f32, 0.044022349891595126>>) -> tensor<2xf32>
  %conv = "tf.Conv2D"(%dq_input, %dq_weight) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x2xf32>) -> tensor<*xf32>
  %biasadd = "tf.BiasAdd"(%conv, %dq_bias) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
  %res = "tf.Relu6"(%biasadd) : (tensor<*xf32>) -> tensor<*xf32>
  %q_res = "quantfork.qcast"(%res) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>
  %dq_res = "quantfork.dcast"(%q_res) : (tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>) -> tensor<*xf32>

  func.return %dq_res : tensor<*xf32>
}

// CHECK-DAG: [[bias:%.+]] = "arith.constant"() <{value = dense<[7.11401462, 7.05456924]> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-DAG: [[weight:%.+]] = "arith.constant"() <{value = dense_resource<__elided__> : tensor<2x3x3x2xf32>}> : () -> tensor<2x3x3x2x!quant.uniform<i8:f32, 0.074855112561992565:-1>>
// CHECK: [[q_input:%.+]] = "quantfork.qcast"(%arg0) : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58810077742034317:-128>>
// CHECK-NEXT: [[q_bias:%.+]] = "quantfork.qcast"([[bias]]) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i32:f32, 0.044022349891595126>>
// CHECK-NEXT: [[conv:%.+]] = "tf.PartitionedCall"([[q_input]], [[weight]], [[q_bias]]) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @[[composite_fn:composite_conv2d_with_bias_and_relu6_fn.*]]} : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.58810077742034317:-128>>, tensor<2x3x3x2x!quant.uniform<i8:f32, 0.074855112561992565:-1>>, tensor<2x!quant.uniform<i32:f32, 0.044022349891595126>>) -> tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>
// CHECK-NEXT: [[res:%.+]] = "quantfork.dcast"([[conv]]) : (tensor<*x!quant.uniform<i8:f32, 0.023529411764705882:-128>>) -> tensor<*xf32>
// CHECK-NEXT: "func.return"([[res]]) : (tensor<*xf32>) -> ()


// -----

// CHECK-LABEL: same_scale_test
func.func @same_scale_test(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[-1, 144]> : tensor<2xi32>
  %0 = "quantfork.qcast"(%arg0) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>
  %1 = "quantfork.dcast"(%0) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>) -> tensor<*xf32>
  %2 = "tf.MaxPool"(%1) {data_format = "NHWC", device = "", explicit_paddings = [], ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 2, 2, 1]} : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "quantfork.qcast"(%2) {volatile} : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>
  %4 = "quantfork.dcast"(%3) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>) -> tensor<*xf32>
  %5 = "tf.Reshape"(%4, %cst) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
  %6 = "quantfork.qcast"(%5) {volatile} : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>
  %7 = "quantfork.dcast"(%6) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>) -> tensor<*xf32>
  func.return %7 : tensor<*xf32>
}

// CHECK: %[[q:.*]] = "quantfork.qcast"(%arg0)
// CHECK: %[[sc1:.*]] = "quantfork.scast"(%[[q]]) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>)
// CHECK: %[[maxpool_i8:.*]] = "tf.MaxPool"(%[[sc1]])
// CHECK-SAME: (tensor<*xi8>) -> tensor<*xi8>
// CHECK: %[[reshape_i8:.*]] = "tf.Reshape"(%[[maxpool_i8]]
// CHECK-SAME: (tensor<*xi8>, tensor<2xi32>) -> tensor<*xi8>
// CHECK: %[[sc2:.*]] = "quantfork.scast"(%[[reshape_i8]])
// CHECK: %[[dq:.*]] = "quantfork.dcast"(%[[sc2]]) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>)
// CHECK: return %[[dq]]

// -----

// CHECK-LABEL: avgpool_test
func.func @avgpool_test(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[-1, 144]> : tensor<2xi32>
  %0 = "quantfork.qcast"(%arg0) : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>
  %1 = "quantfork.dcast"(%0) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>) -> tensor<*xf32>
  %2 = "tf.AvgPool"(%1) {data_format = "NHWC", ksize = [1, 2, 2, 1], padding = "VALID", strides = [1, 2, 2, 1]} : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "quantfork.qcast"(%2) {volatile} : (tensor<*xf32>) -> tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>
  %4 = "quantfork.dcast"(%3) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>) -> tensor<*xf32>
  func.return %4 : tensor<*xf32>
}

// CHECK: %[[q:.*]] = "quantfork.qcast"(%arg0)
// CHECK: %[[sc1:.*]] = "quantfork.scast"(%[[q]]) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>)
// CHECK: %[[fcast:.*]] = "tf.Cast"(%[[sc1]]) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
// CHECK: %[[avgpool_f32:.*]] = "tf.AvgPool"(%[[fcast]])
// CHECK-SAME: (tensor<*xf32>) -> tensor<*xf32>
// CHECK: %[[round:.*]] = "tf.Round"(%[[avgpool_f32]])
// CHECK: %[[icast:.*]] = "tf.Cast"(%[[round]]) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi8>
// CHECK: %[[sc2:.*]] = "quantfork.scast"(%[[icast]])
// CHECK: %[[dq:.*]] = "quantfork.dcast"(%[[sc2]]) : (tensor<*x!quant.uniform<i8:f32, 5.000000e-02:-10>>)
// CHECK: return %[[dq]]
