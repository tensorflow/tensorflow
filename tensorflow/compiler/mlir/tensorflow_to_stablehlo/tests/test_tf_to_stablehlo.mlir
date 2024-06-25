// RUN: tf-to-stablehlo-translate %s --input-arg-shapes=1 -o - | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: %[[UQ:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<1xf32>) -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
// CHECK: %[[BITCAST_CONVERT_0:.*]] = stablehlo.bitcast_convert %[[UQ]] : (tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1xi8>
// CHECK: %[[BITCAST_CONVERT_1:.*]] = stablehlo.bitcast_convert %[[BITCAST_CONVERT_0]] : (tensor<1xi8>) -> tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>
// CHECK: %[[UDQ:.*]] = stablehlo.uniform_dequantize %[[BITCAST_CONVERT_1]] : (tensor<1x!quant.uniform<i8:f32, 1.000000e+00:3>>) -> tensor<1xf32>
// CHECK: return %[[UDQ]] : tensor<1xf32>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
      %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
      %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

      %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?xf32>, tensor<f32>, tensor<i32>) -> tensor<?x!tf_type.qint8>
      %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
        quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
      } : (tensor<?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?xf32>
      func.return %1 : tensor<?xf32>
    }
}
