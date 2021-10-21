// RUN: tac-opt-all-backends -tfl-raise-target-subgraphs %s -split-input-file -verify-diagnostics | FileCheck %s

module {
func @simpleTest(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%0, %arg2) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.add"(%arg0, %arg3) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.pack"(%1, %2) {tac.device = "CPU", tac.inference_type = "FLOAT", axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  return %3 : tensor<2x1xf32>
}
}

// CHECK:   func @simpleTest(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>, %[[VAL_3:.*]]: tensor<1xf32>) -> tensor<2x1xf32> {
// CHECK:           %[[VAL_4:.*]]:2 = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_0]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>)
// CHECK:           %[[VAL_5:.*]] = call @func_1_CPU_FLOAT(%[[VAL_4]]#0, %[[VAL_4]]#1) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x1xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>, %[[VAL_3:.*]]: tensor<1xf32>, %[[VAL_4:.*]]: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_5:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           %[[VAL_6:.*]] = tfl.mul %[[VAL_5]], %[[VAL_2]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           %[[VAL_7:.*]] = tfl.add %[[VAL_0]], %[[VAL_4]] {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:           return %[[VAL_6]], %[[VAL_7]] : tensor<1xf32>, tensor<1xf32>
// CHECK:         }

// CHECK:   func private @func_1_CPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> tensor<2x1xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_2:.*]] = "tfl.pack"(%[[VAL_0]], %[[VAL_1]]) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_2]] : tensor<2x1xf32>
// CHECK:         }

// -----

module {
func @constWeight(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
  %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {tac.device = "GPU", tac.inference_type = "FLOAT", dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %3 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x16xf32>} : () -> tensor<16x3x3x16xf32>
  %4 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  %5 = "tfl.conv_2d"(%2, %3, %4) {tac.device = "GPU", tac.inference_type = "FLOAT", dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x30x30x16xf32>, tensor<16x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %5 : tensor<256x30x30x16xf32>
}

// CHECK:   func @constWeight(%[[VAL_0:.*]]: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
// CHECK-DAG:       %[[VAL_2:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
// CHECK-DAG:       %[[VAL_3:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x16xf32>} : () -> tensor<16x3x3x16xf32>
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
// CHECK:           %[[VAL_5:.*]] = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>, tensor<16x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK:           return %[[VAL_5]] : tensor<256x30x30x16xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<256x32x32x3xf32>, %[[VAL_1:.*]]: tensor<16x3x3x3xf32>, %[[VAL_2:.*]]: tensor<16xf32>, %[[VAL_3:.*]]: tensor<16x3x3x16xf32>, %[[VAL_4:.*]]: tensor<16xf32>) -> tensor<256x30x30x16xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_5:.*]] = "tfl.conv_2d"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK:           %[[VAL_6:.*]] = "tfl.conv_2d"(%[[VAL_5]], %[[VAL_3]], %[[VAL_4]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<256x30x30x16xf32>, tensor<16x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// CHECK:           return %[[VAL_6]] : tensor<256x30x30x16xf32>
// CHECK:         }

}

// -----

module {
func @norm1(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
  %1 = "tfl.add"(%arg0, %0) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[128, 128]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.reshape"(%1, %2) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
  %4 = "tfl.relu"(%3) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %6 = "tfl.reshape"(%4, %5) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
  %7 = "tfl.add"(%1, %6) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  return %7 : tensor<1x128x128xf32>
}

// CHECK:   func @norm1(%[[VAL_0:.*]]: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
// CHECK-DAG:       %[[VAL_2:.*]] = "tfl.pseudo_const"() {value = dense<128> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_4:.*]] = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<2xi32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1x128x128xf32>, %[[VAL_1:.*]]: tensor<128xf32>, %[[VAL_2:.*]]: tensor<2xi32>, %[[VAL_3:.*]]: tensor<3xi32>) -> tensor<1x128x128xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_4:.*]] = tfl.add(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.reshape"(%[[VAL_4]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_6:.*]] = "tfl.relu"(%[[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.reshape"(%[[VAL_6]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_8:.*]] = tfl.add %[[VAL_4]], %[[VAL_7]] {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1x128x128xf32>
// CHECK:           return %[[VAL_8]] : tensor<1x128x128xf32>
// CHECK:         }

}

// -----

module {
func @norm2(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
  %1 = "tfl.add"(%arg0, %0) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[128, 128]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.reshape"(%1, %2) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
  %4 = "tfl.relu"(%3) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  %5 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128x128xf32>} : () -> tensor<128x128xf32>
  %6 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
  %7 = "tfl.fully_connected"(%4, %5, %6) {tac.device = "CPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
  %8 = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %9 = "tfl.reshape"(%7, %8) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
  %10 = "tfl.add"(%1, %9) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  return %10 : tensor<1x128x128xf32>
}

// CHECK:   func @norm2(%[[VAL_0:.*]]: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
// CHECK-DAG:       %[[VAL_2:.*]] = "tfl.pseudo_const"() {value = dense<128> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           %[[VAL_3:.*]]:2 = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<2xi32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>)
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128x128xf32>} : () -> tensor<128x128xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
// CHECK:           %[[VAL_6:.*]] = call @func_1_CPU_FLOAT(%[[VAL_3]]#1, %[[VAL_4]], %[[VAL_5]]) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_8:.*]] = call @func_2_GPU_FLOAT(%[[VAL_6]], %[[VAL_7]], %[[VAL_3]]#0) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<128x128xf32>, tensor<3xi32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
// CHECK:           return %[[VAL_8]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1x128x128xf32>, %[[VAL_1:.*]]: tensor<128xf32>, %[[VAL_2:.*]]: tensor<2xi32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>) attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_3:.*]] = tfl.add(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = "tfl.reshape"(%[[VAL_3]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.relu"(%[[VAL_4]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:           return %[[VAL_3]], %[[VAL_5]] : tensor<1x128x128xf32>, tensor<128x128xf32>
// CHECK:         }

// CHECK:   func private @func_2_GPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<3xi32>, %[[VAL_2:.*]]: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
// CHECK:           %[[VAL_3:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_1]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = tfl.add %[[VAL_2]], %[[VAL_3]] {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1x128x128xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func private @func_1_CPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<128x128xf32>, %[[VAL_2:.*]]: tensor<128xf32>) -> tensor<128x128xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_3:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {fused_activation_function = "NONE", keep_num_dims = false, tac.device = "CPU", tac.inference_type = "FLOAT", weights_format = "DEFAULT"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK:           return %[[VAL_3]] : tensor<128x128xf32>
// CHECK:         }

}

// -----

module {

func @quantizedOpOnly(%arg0: tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, %arg1: tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %2 = "tfl.mul"(%arg0, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %3 = "tfl.add"(%2, %1) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %4 = "tfl.add"(%arg1, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %5 = "tfl.pack"(%3, %4) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", axis = 0 : i32, values_count = 2 : i32} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>>
  return %5: tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>>
}

// CHECK:   func @quantizedOpOnly(%[[VAL_0:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>> {
// CHECK:           %[[VAL_2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_3:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_4:.*]] = call @func_0_CPU_QUANTIZED_INT8(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_1]], %[[VAL_2]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} : (tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_4]] : tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_0_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_2:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_3:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_4:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_5:.*]] = tfl.mul %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_6:.*]] = tfl.add %[[VAL_5]], %[[VAL_2]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_7:.*]] = tfl.add %[[VAL_3]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_8:.*]] = "tfl.pack"(%[[VAL_6]], %[[VAL_7]]) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", values_count = 2 : i32} : (tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_8]] : tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

}

// -----

module {
func @quantizationWithFloat(%arg0: tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, %arg1: tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1x384x1xi8>} : () -> tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.mul"(%arg0, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384xf32>
  %3 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x1x384xf32>} : () -> tensor<1x384x384xf32>
  %4 = "tfl.add"(%2, %3) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x384x384xf32>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  %6 = "tfl.mul"(%arg1, %5) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  return %6: tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
}

// CHECK:   func @quantizationWithFloat(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> {
// CHECK:           %[[VAL_2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1x384x1xi8>} : () -> tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_3:.*]] = call @func_0_CPU_QUANTIZED_INT8(%[[VAL_0]], %[[VAL_2]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_4:.*]] = "tfl.dequantize"(%[[VAL_3]]) : (tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x1x384xf32>} : () -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_6:.*]] = call @func_1_GPU_FLOAT(%[[VAL_4]], %[[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.quantize"(%[[VAL_6]]) {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>} : (tensor<1x384x384xf32>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_8:.*]] = call @func_2_CPU_QUANTIZED_INT8(%[[VAL_1]], %[[VAL_7]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_8]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_0_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_2:.*]] = tfl.mul(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_2_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} {
// CHECK:           %[[VAL_2:.*]] = tfl.mul(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_1_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1x384x384xf32>, %[[VAL_1:.*]]: tensor<1x384x384xf32>) -> tensor<1x384x384xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_2:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1x384x384xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384xf32>
// CHECK:         }

}
