// Test for partial folding: only fold i32 constants.
// RUN: tac-opt-all-backends -tfl-fold-constants-to-subgraph='fold-all-constants=false' %s -split-input-file -verify-diagnostics | FileCheck --check-prefix=PARTIAL %s

// Test for fold all constants.
// RUN: tac-opt-all-backends -tfl-fold-constants-to-subgraph='fold-all-constants=true' %s -split-input-file -verify-diagnostics | FileCheck --check-prefix=ALL %s

module {

func @main(%arg0: tensor<4x384x32xf32>) -> tensor<1x384x32xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = call @simple_test(%arg0, %0, %1) {tac.interface_name = "func1"} : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %2 : tensor<1x384x32xf32>
}

// PARTIAL-LABEL: @simple_test
func @simple_test(%arg0: tensor<4x384x32xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func1"} {
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %0 : tensor<1x384x32xf32>
}

// PARTIAL:       func @simple_test(%[[VAL_0:.*]]: tensor<4x384x32xf32>, %[[VAL_1:.*]]: tensor<3xi32>, %[[VAL_2:.*]]: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func1"} {
// PARTIAL:           %[[VAL_3:.*]] = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_5:.*]] = "tfl.slice"(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]]) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
// PARTIAL:           return %[[VAL_5]] : tensor<1x384x32xf32>
// PARTIAL:         }
}

// -----

module {

func @main(%arg0: tensor<4x384x32xf32>) -> (tensor<1x384x32xf32>, tensor<1x384x32xf32>) {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = call @arg_reuse_test_1(%arg0, %0, %1) {tac.interface_name = "func1"} : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  %3 = call @arg_reuse_test_2(%arg0, %0, %1) {tac.interface_name = "func2"} : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %2, %3 : tensor<1x384x32xf32>, tensor<1x384x32xf32>
}

// PARTIAL-LABEL: @arg_reuse_test_1
func @arg_reuse_test_1(%arg0: tensor<4x384x32xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func1"} {
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %0 : tensor<1x384x32xf32>
}

// PARTIAL-LABEL: @arg_reuse_test_2
func @arg_reuse_test_2(%arg0: tensor<4x384x32xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func2"} {
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
  return %0 : tensor<1x384x32xf32>
}

// PARTIAL:       func @arg_reuse_test_1(%[[VAL_0:.*]]: tensor<4x384x32xf32>, %[[VAL_1:.*]]: tensor<3xi32>, %[[VAL_2:.*]]: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func1"} {
// PARTIAL:           %[[VAL_3:.*]] = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_5:.*]] = "tfl.slice"(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]]) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
// PARTIAL:           return %[[VAL_5]] : tensor<1x384x32xf32>
// PARTIAL:         }

// PARTIAL:       func @arg_reuse_test_2(%[[VAL_6:.*]]: tensor<4x384x32xf32>, %[[VAL_7:.*]]: tensor<3xi32>, %[[VAL_8:.*]]: tensor<3xi32>) -> tensor<1x384x32xf32> attributes {tac.interface_name = "func2"} {
// PARTIAL:           %[[VAL_9:.*]] = "tfl.pseudo_const"() {value = dense<[1, 384, 32]> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_10:.*]] = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
// PARTIAL:           %[[VAL_11:.*]] = "tfl.slice"(%[[VAL_6]], %[[VAL_10]], %[[VAL_9]]) : (tensor<4x384x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x384x32xf32>
// PARTIAL:           return %[[VAL_11]] : tensor<1x384x32xf32>
// PARTIAL:         }

}

// -----

module {
func @main(%arg0: tensor<384x512x!quant.uniform<i8:f32, 0.1>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.01>>, value = dense<0> : tensor<128x512xi8>} : () -> tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.01>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<128x!quant.uniform<i32:f32, 0.7>>, value = dense<0> : tensor<128xi32>} : () -> tensor<128x!quant.uniform<i32:f32, 0.7>>
  %2 = call @quantization_test(%arg0, %0, %1) {tac.interface_name = "func1"} : (tensor<384x512x!quant.uniform<i8:f32, 0.1>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.01>>, tensor<128x!quant.uniform<i32:f32, 0.7>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>>
  return %2 : tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>>
}

// PARTIAL-LABEL: @quantization_test
func @quantization_test(%arg0: tensor<384x512x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.01>>, %arg2: tensor<128x!quant.uniform<i32:f32, 0.7>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>> {
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 0.1>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 0.01>>, tensor<128x!quant.uniform<i32:f32, 0.7>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>>
  return %0 : tensor<384x128x!quant.uniform<i8:f32, 0.09:-4>>
}

// PARTIAL:   func @quantization_test(%[[VAL_0:.*]]: tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, %[[VAL_1:.*]]: tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-02>>, %[[VAL_2:.*]]: tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.089999999999999996:-4>> {
// PARTIAL:           %[[VAL_3:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>, value = dense<0> : tensor<128xi32>} : () -> tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>
// PARTIAL:           %[[VAL_4:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_3]]) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<384x512x!quant.uniform<i8:f32, 1.000000e-01>>, tensor<128x512x!quant.uniform<i8<-127:127>:f32, 1.000000e-02>>, tensor<128x!quant.uniform<i32:f32, 0.69999999999999996>>) -> tensor<384x128x!quant.uniform<i8:f32, 0.089999999999999996:-4>>
// PARTIAL:           return %[[VAL_4]] : tensor<384x128x!quant.uniform<i8:f32, 0.089999999999999996:-4>>
// PARTIAL:         }

}

// -----

module {
func @main(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
  %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  %2 = call @fold_all_test(%arg0, %0, %1) : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %2 : tensor<256x30x30x16xf32>
}

// ALL-LABEL: @fold_all_test
func @fold_all_test(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {tac.device = "GPU", tac.inference_type = "FLOAT", dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// ALL: func @fold_all_test(%[[VAL_0:.*]]: tensor<256x32x32x3xf32>, %[[VAL_1:.*]]: tensor<16x3x3x3xf32>, %[[VAL_2:.*]]: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
// ALL:           %[[VAL_3:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
// ALL:           %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
// ALL:           %[[VAL_5:.*]] = "tfl.conv_2d"(%[[VAL_0]], %[[VAL_4]], %[[VAL_3]]) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
// ALL:           return %[[VAL_5]] : tensor<256x30x30x16xf32>
// ALL:         }
}
