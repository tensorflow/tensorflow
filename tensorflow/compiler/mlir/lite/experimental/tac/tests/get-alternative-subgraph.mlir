// RUN: tac-opt-all-backends -tfl-get-alternative-subgraph='device-specs=GPU' %s -split-input-file -verify-diagnostics | FileCheck %s

module {
  func.func @simpleTest(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> {
    %0 = func.call @func_0_GPU_FLOAT(%arg0, %arg1, %arg2) {tac.interface_name = "func_0"} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = func.call @func_1_GPU_FLOAT(%arg0, %arg3) {tac.interface_name = "func_1"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = func.call @func_2_CPU_FLOAT(%0, %1) {tac.interface_name = "func_2"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    func.return %2 : tensor<2x1xf32>
  }

  func.func private @func_2_CPU_FLOAT(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<2x1xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    func.return %0 : tensor<2x1xf32>
  }

  func.func private @func_0_GPU_FLOAT(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
    %1 = tfl.mul %0, %arg2 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
    func.return %1 : tensor<1xf32>
  }

  func.func private @func_1_GPU_FLOAT(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
    func.return %0 : tensor<1xf32>
  }

// CHECK:   func @simpleTest(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>, %[[VAL_3:.*]]: tensor<1xf32>) -> tensor<2x1xf32> {
// CHECK:           %[[VAL_4:.*]] = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {tac.interface_name = "func_0"} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[VAL_5:.*]] = call @func_1_GPU_FLOAT(%[[VAL_0]], %[[VAL_3]]) {tac.interface_name = "func_1"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = call @func_2_CPU_FLOAT(%[[VAL_4]], %[[VAL_5]]) {tac.interface_name = "func_2"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_6]] : tensor<2x1xf32>
// CHECK:         }

// CHECK:   func private @func_2_CPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<2x1xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
// CHECK:           %[[VAL_2:.*]] = "tfl.pack"(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x1xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_3:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = tfl.mul %[[VAL_3]], %[[VAL_2]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           return %[[VAL_4]] : tensor<1xf32>
// CHECK:         }

// CHECK:   func private @func_1_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_2:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           return %[[VAL_2]] : tensor<1xf32>
// CHECK:         }

// CHECK:   func private @func_2_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<2x1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
// CHECK-DAG:       %[[VAL_2:.*]] = "tfl.pseudo_const"(){{.*}} dense<1> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = "tfl.pseudo_const"(){{.*}}dense<2> : tensor<1xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"(){{.*}}dense<[2, 1]> : tensor<2xi32>
// CHECK:           %[[VAL_5:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_6:.*]] = "tfl.reshape"(%[[VAL_1]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<4xi32>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.concatenation"(%[[VAL_5]], %[[VAL_6]]) {axis = 3 : i32, fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x2xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.reshape"(%[[VAL_7]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x1x2xf32>, tensor<1xi32>) -> tensor<2xf32>
// CHECK:           %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_8]], %[[VAL_4]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<2xf32>, tensor<2xi32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_9]] : tensor<2x1xf32>
// CHECK:         }

// CHECK:   func private @func_0_CPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_3:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = tfl.mul %[[VAL_3]], %[[VAL_2]] {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           return %[[VAL_4]] : tensor<1xf32>
// CHECK:         }

// CHECK:   func private @func_1_CPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_2:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           return %[[VAL_2]] : tensor<1xf32>
// CHECK:         }

}

// -----

module {
func.func private @func_10_CPU_FLOAT(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<*xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_10"} {
  %0 = "tfl.one_hot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i32, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK:   func private @func_10_CPU_FLOAT(%[[VAL_0:.*]]: tensor<3xi32>, %[[VAL_1:.*]]: tensor<i32>, %[[VAL_2:.*]]: tensor<f32>, %[[VAL_3:.*]]: tensor<f32>) -> tensor<*xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_10"} {
// CHECK:           %[[VAL_4:.*]] = "tfl.one_hot"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) {axis = -1 : i32, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
// CHECK:           return %[[VAL_4]] : tensor<*xf32>
// CHECK:         }

}

// -----

module {
func.func private @func_20_GPU_FLOAT(%arg0: tensor<128x128xf32>, %arg1: tensor<3xi32>) -> tensor<1x128x128xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_20"} {
  %0 = "tfl.reshape"(%arg0, %arg1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
  func.return %0 : tensor<1x128x128xf32>
}

// CHECK:   func private @func_20_GPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<3xi32>) -> tensor<1x128x128xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_20"} {
// CHECK:           %[[VAL_2:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_1]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func private @func_20_CPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<3xi32>) -> tensor<1x128x128xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_20"} {
// CHECK:           %[[VAL_2:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_1]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x128x128xf32>
// CHECK:         }
}

// -----

module {
func.func private @quantize_ops_CPU_QUANTIZED_INT8(%arg0: tensor<384x512x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.1>>, %arg2: tensor<128x!quant.uniform<i8:f32, 0.2:-128>>, %arg3: tensor<128x!quant.uniform<i8:f32, 0.2:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>  attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "quantize_ops"} {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<128x!quant.uniform<i32:f32, 0.7>>, value = dense<0> : tensor<128xi32>} : () -> tensor<128x!quant.uniform<i32:f32, 0.7>>
  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 0.1>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.1>>, tensor<128x!quant.uniform<i32:f32, 0.7>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.9:-4>>
  %2 = "arith.constant"() {value = dense<[1, 384, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tfl.reshape"(%1, %2) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<384x128x!quant.uniform<i8:f32, 0.9:-4>>, tensor<3xi32>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.9:-4>>
  %4 = "tfl.mul"(%3, %arg2) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x384x128x!quant.uniform<i8:f32, 0.9:-4>>, tensor<128x!quant.uniform<i8:f32, 0.2:-128>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:3>>
  %5 = "tfl.add"(%4, %arg3) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x384x128x!quant.uniform<i8:f32, 0.3:3>>, tensor<128x!quant.uniform<i8:f32, 0.2:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>
  func.return %5 : tensor<1x384x128x!quant.uniform<i8:f32, 0.3:-3>>
}

// CHECK:   func private @quantize_ops_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, %[[VAL_1:.*]]: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>, %[[VAL_2:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>, %[[VAL_3:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "quantize_ops"} {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<[1, 384, 128]> : tensor<3xi32>
// CHECK-DAG:       %[[VAL_5:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>, value = dense<0> : tensor<128xi32>} : () -> tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>
// CHECK:           %[[VAL_6:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_5]]) {fused_activation_function = "NONE", keep_num_dims = false, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>, tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>) -> tensor<384x128x!quant.uniform<i8:f32, 9.000000e-01:-4>>
// CHECK:           %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_6]], %[[VAL_4]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<384x128x!quant.uniform<i8:f32, 9.000000e-01:-4>>, tensor<3xi32>) -> tensor<1x384x128x!quant.uniform<i8:f32, 9.000000e-01:-4>>
// CHECK:           %[[VAL_8:.*]] = tfl.mul(%[[VAL_7]], %[[VAL_2]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x384x128x!quant.uniform<i8:f32, 9.000000e-01:-4>>, tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:3>>
// CHECK:           %[[VAL_9:.*]] = tfl.add(%[[VAL_8]], %[[VAL_3]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:3>>, tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:           return %[[VAL_9]] : tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:         }

// CHECK:   func private @quantize_ops_GPU_FLOAT(%[[VAL_0:.*]]: tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, %[[VAL_1:.*]]: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>, %[[VAL_2:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>, %[[VAL_3:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "quantize_ops"} {
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"(){{.*}}dense<0.000000e+00> : tensor<128xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant dense<[1, 384, 128]> : tensor<3xi32>
// CHECK-DAG:       %[[VAL_6:.*]] = "tfl.pseudo_const"(){{.*}}dense<[1, 1, 384, 512]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_7:.*]] = "tfl.pseudo_const"(){{.*}}dense<[128, 1, 1, 512]> : tensor<4xi32>
// CHECK-DAG:       %[[VAL_8:.*]] = "tfl.pseudo_const"(){{.*}}dense<[384, 128]> : tensor<2xi32>
// CHECK:           %[[VAL_9:.*]] = "tfl.dequantize"(%[[VAL_0]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<384x512xf32>
// CHECK:           %[[VAL_10:.*]] = "tfl.dequantize"(%[[VAL_1]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>) -> tensor<128x512xf32>
// CHECK:           %[[VAL_11:.*]] = "tfl.reshape"(%[[VAL_9]], %[[VAL_6]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<384x512xf32>, tensor<4xi32>) -> tensor<1x1x384x512xf32>
// CHECK:           %[[VAL_12:.*]] = "tfl.reshape"(%[[VAL_10]], %[[VAL_7]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x512xf32>, tensor<4xi32>) -> tensor<128x1x1x512xf32>
// CHECK:           %[[VAL_13:.*]] = "tfl.conv_2d"(%[[VAL_11]], %[[VAL_12]], %[[VAL_4]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x384x512xf32>, tensor<128x1x1x512xf32>, tensor<128xf32>) -> tensor<1x1x384x128xf32>
// CHECK:           %[[VAL_14:.*]] = "tfl.reshape"(%[[VAL_13]], %[[VAL_8]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x384x128xf32>, tensor<2xi32>) -> tensor<384x128xf32>
// CHECK:           %[[VAL_15:.*]] = "tfl.reshape"(%[[VAL_14]], %[[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<384x128xf32>, tensor<3xi32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_16:.*]] = "tfl.dequantize"(%[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>) -> tensor<128xf32>
// CHECK:           %[[VAL_17:.*]] = tfl.mul(%[[VAL_15]], %[[VAL_16]]) {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>, tensor<128xf32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_18:.*]] = "tfl.dequantize"(%[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<128xf32>
// CHECK:           %[[VAL_19:.*]] = tfl.add(%[[VAL_17]], %[[VAL_18]]) {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>, tensor<128xf32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_20:.*]] = "tfl.quantize"(%[[VAL_19]]) {qtype = tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:           return %[[VAL_20]] : tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:         }

// CHECK:   func private @quantize_ops_CPU_FLOAT(%[[VAL_0:.*]]: tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, %[[VAL_1:.*]]: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>, %[[VAL_2:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>, %[[VAL_3:.*]]: tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "quantize_ops"} {
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"(){{.*}}dense<0.000000e+00> : tensor<128xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant dense<[1, 384, 128]> : tensor<3xi32>
// CHECK:           %[[VAL_6:.*]] = "tfl.dequantize"(%[[VAL_0]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<384x512xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.dequantize"(%[[VAL_1]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-01>>) -> tensor<128x512xf32>
// CHECK:           %[[VAL_8:.*]] = "tfl.fully_connected"(%[[VAL_6]], %[[VAL_7]], %[[VAL_4]]) {fused_activation_function = "NONE", keep_num_dims = false, tac.device = "CPU", tac.inference_type = "FLOAT", weights_format = "DEFAULT"} : (tensor<384x512xf32>, tensor<128x512xf32>, tensor<128xf32>) -> tensor<384x128xf32>
// CHECK:           %[[VAL_9:.*]] = "tfl.reshape"(%[[VAL_8]], %[[VAL_5]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<384x128xf32>, tensor<3xi32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_10:.*]] = "tfl.dequantize"(%[[VAL_2]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-128>>) -> tensor<128xf32>
// CHECK:           %[[VAL_11:.*]] = tfl.mul(%[[VAL_9]], %[[VAL_10]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>, tensor<128xf32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_12:.*]] = "tfl.dequantize"(%[[VAL_3]]) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<128x!quant.uniform<i8:f32, 2.000000e-01:-4>>) -> tensor<128xf32>
// CHECK:           %[[VAL_13:.*]] = tfl.add(%[[VAL_11]], %[[VAL_12]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>, tensor<128xf32>) -> tensor<1x384x128xf32>
// CHECK:           %[[VAL_14:.*]] = "tfl.quantize"(%[[VAL_13]]) {qtype = tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1x384x128xf32>) -> tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:           return %[[VAL_14]] : tensor<1x384x128x!quant.uniform<i8:f32, 3.000000e-01:-3>>
// CHECK:         }

}
