// RUN: tac-opt-all-backends -tfl-raise-target-subgraphs %s -split-input-file | FileCheck %s

module {
func.func @simpleWhile(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tfl.while"(%arg0) ({
  ^bb0(%block: tensor<i32>):
    "tfl.yield"(%block) : (tensor<i32>) -> ()
  },{
  ^bb0(%block: tensor<i32>):
    "tfl.yield"(%block) : (tensor<i32>) -> ()
  }) {tac.device = "CPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
}

// CHECK:     func.func @simpleWhile(%arg0: tensor<i32>) -> tensor<i32> {
// CHECK:       %0 = call @func_0_CPU_FLOAT(%arg0) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<i32>) -> tensor<i32>
// CHECK:       return %0 : tensor<i32>
// CHECK:     }
// CHECK:     func.func private @func_0_CPU_FLOAT(%arg0: tensor<i32>) -> tensor<i32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:       %0 = "tfl.while"(%arg0) ({
// CHECK:       ^bb0(%arg1: tensor<i32>):
// CHECK:         "tfl.yield"(%arg1) : (tensor<i32>) -> ()
// CHECK:       }, {
// CHECK:       ^bb0(%arg1: tensor<i32>):
// CHECK:         "tfl.yield"(%arg1) : (tensor<i32>) -> ()
// CHECK:       }) {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<i32>) -> tensor<i32>
// CHECK:       return %0 : tensor<i32>
// CHECK:     }

// -----

module {
func.func @whileWithNested(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tfl.while"(%arg0) ({
  ^bb0(%block: tensor<i32>):
    %1 = "tfl.add"(%arg0, %arg0) { fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    %2 = "tfl.add"(%1, %1) { fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
    "tfl.yield"(%2) : (tensor<i32>) -> ()
  },{
  ^bb0(%block: tensor<i32>):
    "tfl.yield"(%block) : (tensor<i32>) -> ()
  }) {tac.device = "CPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
}

// CHECK:     func.func @whileWithNested(%arg0: tensor<i32>) -> tensor<i32> {
// CHECK:       %0 = call @func_0_CPU_FLOAT(%arg0) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<i32>) -> tensor<i32>
// CHECK:       return %0 : tensor<i32>
// CHECK:     }
// CHECK:     func.func private @func_0_CPU_FLOAT(%arg0: tensor<i32>) -> tensor<i32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:       %0 = "tfl.while"(%arg0) ({
// CHECK:       ^bb0(%arg1: tensor<i32>):
// CHECK:         %1 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<i32>
// CHECK:         %2 = func.call @func_1_GPU_FLOAT(%1) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<i32>) -> tensor<i32>
// CHECK:         "tfl.yield"(%2) : (tensor<i32>) -> ()
// CHECK:       }, {
// CHECK:       ^bb0(%arg1: tensor<i32>):
// CHECK:         "tfl.yield"(%arg1) : (tensor<i32>) -> ()
// CHECK:       }) {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<i32>) -> tensor<i32>
// CHECK:       return %0 : tensor<i32>
// CHECK:     }
// CHECK:     func.func private @func_1_GPU_FLOAT(%arg0: tensor<i32>) -> tensor<i32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:       %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<i32>
// CHECK:       return %0 : tensor<i32>
// CHECK:     }





// -----

module {
func.func @degenerateCase(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tfl.add"(%arg0, %arg0) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}
}

// CHECK:     func.func @degenerateCase(%arg0: tensor<1xf32>) -> tensor<1xf32> {
// CHECK:       %0 = call @func_0_GPU_FLOAT(%arg0) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:       return %0 : tensor<1xf32>
// CHECK:     }
// CHECK:     func.func private @func_0_GPU_FLOAT(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:       %0 = tfl.add %arg0, %arg0  {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1xf32>
// CHECK:       return %0 : tensor<1xf32>
// CHECK:     }

// -----

module {
func.func @simpleTest(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<2x1xf32> {
  %0 = "tfl.add"(%arg0, %arg1) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "tfl.mul"(%0, %arg2) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %2 = "tfl.add"(%arg0, %arg3) {tac.device = "GPU", fused_activation_function = "RELU6", tac.inference_type = "FLOAT"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %3 = "tfl.pack"(%1, %2) {tac.device = "CPU", tac.inference_type = "FLOAT", axis = 0 : i32, values_count = 2 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  func.return %3 : tensor<2x1xf32>
}
}

// CHECK:   func @simpleTest(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>, %[[VAL_3:.*]]: tensor<1xf32>) -> tensor<2x1xf32> {
// CHECK:           %[[VAL_4:.*]]:2 = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>)
// CHECK:           %[[VAL_5:.*]] = call @func_1_CPU_FLOAT(%[[VAL_4]]#0, %[[VAL_4]]#1) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x1xf32>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1xf32>, %[[VAL_1:.*]]: tensor<1xf32>, %[[VAL_2:.*]]: tensor<1xf32>, %[[VAL_4:.*]]: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
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
func.func @constWeight(%arg0: tensor<256x32x32x3xf32>) -> tensor<256x30x30x16xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x3xf32>} : () -> tensor<16x3x3x3xf32>
  %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  %2 = "tfl.conv_2d"(%arg0, %0, %1) {tac.device = "GPU", tac.inference_type = "FLOAT", dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  %3 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x3x3x16xf32>} : () -> tensor<16x3x3x16xf32>
  %4 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
  %5 = "tfl.conv_2d"(%2, %3, %4) {tac.device = "GPU", tac.inference_type = "FLOAT", dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x30x30x16xf32>, tensor<16x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %5 : tensor<256x30x30x16xf32>
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
func.func @norm1(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
  %1 = "tfl.add"(%arg0, %0) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
  %2 = "tfl.pseudo_const"() {value = dense<[128, 128]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tfl.reshape"(%1, %2) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
  %4 = "tfl.relu"(%3) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  %5 = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
  %6 = "tfl.reshape"(%4, %5) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
  %7 = "tfl.add"(%1, %6) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  func.return %7 : tensor<1x128x128xf32>
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
func.func @norm2(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
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
  func.return %10 : tensor<1x128x128xf32>
}
}

// CHECK:   func @norm2(%[[VAL_0:.*]]: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
// CHECK-DAG:       %[[VAL_1:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
// CHECK-DAG:       %[[VAL_2:.*]] = "tfl.pseudo_const"() {value = dense<128> : tensor<2xi32>} : () -> tensor<2xi32>
// CHECK:           %[[VAL_3:.*]]:2 = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<2xi32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>)
// CHECK-DAG:       %[[VAL_4:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128x128xf32>} : () -> tensor<128x128xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<128xf32>} : () -> tensor<128xf32>
// CHECK:           %[[VAL_6:.*]] = call @func_2_CPU_FLOAT(%[[VAL_3]]#1, %[[VAL_4]], %[[VAL_5]]) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.pseudo_const"() {value = dense<[1, 128, 128]> : tensor<3xi32>} : () -> tensor<3xi32>
// CHECK:           %[[VAL_8:.*]] = call @func_1_GPU_FLOAT(%[[VAL_6]], %[[VAL_7]], %[[VAL_3]]#0) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<128x128xf32>, tensor<3xi32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
// CHECK:           return %[[VAL_8]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func.func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1x128x128xf32>, %[[VAL_1:.*]]: tensor<128xf32>, %[[VAL_2:.*]]: tensor<2xi32>) -> (tensor<1x128x128xf32>, tensor<128x128xf32>) attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_3:.*]] = tfl.add(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<128xf32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = "tfl.reshape"(%[[VAL_3]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.relu"(%[[VAL_4]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK:           return %[[VAL_3]], %[[VAL_5]] : tensor<1x128x128xf32>, tensor<128x128xf32>
// CHECK:         }

// CHECK:   func.func private @func_1_GPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<3xi32>, %[[VAL_2:.*]]: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_3:.*]] = "tfl.reshape"(%[[VAL_0]], %[[VAL_1]]) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<128x128xf32>, tensor<3xi32>) -> tensor<1x128x128xf32>
// CHECK:           %[[VAL_4:.*]] = tfl.add %[[VAL_2]], %[[VAL_3]] {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1x128x128xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x128x128xf32>
// CHECK:         }

// CHECK:   func.func private @func_2_CPU_FLOAT(%[[VAL_0:.*]]: tensor<128x128xf32>, %[[VAL_1:.*]]: tensor<128x128xf32>, %[[VAL_2:.*]]: tensor<128xf32>) -> tensor<128x128xf32> attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
// CHECK:           %[[VAL_3:.*]] = "tfl.fully_connected"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {fused_activation_function = "NONE", keep_num_dims = false, tac.device = "CPU", tac.inference_type = "FLOAT", weights_format = "DEFAULT"} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>
// CHECK:           return %[[VAL_3]] : tensor<128x128xf32>
// CHECK:         }

// -----

module {

func.func @quantizedOpOnly(%arg0: tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, %arg1: tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %2 = "tfl.mul"(%arg0, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %3 = "tfl.add"(%2, %1) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %4 = "tfl.add"(%arg1, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.003:-128>>
  %5 = "tfl.pack"(%3, %4) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", axis = 0 : i32, values_count = 2 : i32} : (tensor<1x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>>
  func.return %5: tensor<2x1x!quant.uniform<i8:f32, 0.003:-128>>
}

// CHECK:   func @quantizedOpOnly(%[[VAL_0:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>> {
// CHECK:           %[[VAL_2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_3:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_4:.*]] = call @func_0_CPU_QUANTIZED_INT8(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_1]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} : (tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_4]] : tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_0_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_2:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_3:.*]]: tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_5:.*]] = tfl.mul %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_6:.*]] = tfl.add %[[VAL_5]], %[[VAL_2]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_7:.*]] = tfl.add %[[VAL_3]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_8:.*]] = "tfl.pack"(%[[VAL_6]], %[[VAL_7]]) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", values_count = 2 : i32} : (tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_8]] : tensor<2x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

}

// -----

module {
func.func @quantizationWithFloat(%arg0: tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, %arg1: tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>> {
  %0 = "tfl.pseudo_qconst"() {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>, value = dense<127> : tensor<1x384x1xi8>} : () -> tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>
  %1 = "tfl.mul"(%arg0, %0) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  %2 = "tfl.dequantize"(%1) : (tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384xf32>
  %3 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x1x384xf32>} : () -> tensor<1x384x384xf32>
  %4 = "tfl.add"(%2, %3) {tac.device = "GPU", tac.inference_type = "FLOAT", fused_activation_function = "NONE"} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %5 = "tfl.quantize"(%4) {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 0.003:-128>>} : (tensor<1x384x384xf32>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  %6 = "tfl.mul"(%arg1, %5) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", fused_activation_function = "NONE"} : (tensor<1x1x384x!quant.uniform<i8:f32, 0.003:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
  func.return %6: tensor<1x384x384x!quant.uniform<i8:f32, 0.003:-128>>
}
}

// CHECK:   func @quantizationWithFloat(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> {
// CHECK:           %[[VAL_2:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>, value = dense<127> : tensor<1x384x1xi8>} : () -> tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_3:.*]] = call @func_1_CPU_QUANTIZED_INT8(%[[VAL_0]], %[[VAL_2]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_1"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_4:.*]] = "tfl.dequantize"(%[[VAL_3]]) : (tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_5:.*]] = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<1x1x384xf32>} : () -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_6:.*]] = call @func_0_GPU_FLOAT(%[[VAL_4]], %[[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
// CHECK:           %[[VAL_7:.*]] = "tfl.quantize"(%[[VAL_6]]) {qtype = tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>} : (tensor<1x384x384xf32>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           %[[VAL_8:.*]] = call @func_2_CPU_QUANTIZED_INT8(%[[VAL_1]], %[[VAL_7]]) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_8]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_0_GPU_FLOAT(%[[VAL_0:.*]]: tensor<1x384x384xf32>, %[[VAL_1:.*]]: tensor<1x384x384xf32>) -> tensor<1x384x384xf32> attributes {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:           %[[VAL_2:.*]] = tfl.add %[[VAL_0]], %[[VAL_1]] {fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<1x384x384xf32>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384xf32>
// CHECK:         }

// CHECK:   func private @func_1_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_1"} {
// CHECK:           %[[VAL_2:.*]] = tfl.mul(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x1x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// CHECK:   func private @func_2_CPU_QUANTIZED_INT8(%[[VAL_0:.*]]: tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, %[[VAL_1:.*]]: tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>> attributes {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} {
// CHECK:           %[[VAL_2:.*]] = tfl.mul(%[[VAL_0]], %[[VAL_1]]) {fused_activation_function = "NONE", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : (tensor<1x1x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>, tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>) -> tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:           return %[[VAL_2]] : tensor<1x384x384x!quant.uniform<i8:f32, 3.000000e-03:-128>>
// CHECK:         }

// -----

func.func @cond_false_72730(%arg0: tensor<?x?x!tf_type.string>, %arg1: tensor<?x?x!tf_type.string>, %arg2: tensor<?x?xi32>, %arg3: tensor<?x!tf_type.string>, %arg4: tensor<?x!tf_type.string>, %arg5: tensor<?xi32>, %arg6: tensor<?xi32>) -> (tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>) {
    %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
    %cst_0 = arith.constant dense<0> : tensor<i32>
    %cst_1 = arith.constant dense<-1> : tensor<1xi32>
    %cst_2 = arith.constant dense<1> : tensor<1xi32>
    %cst_3 = arith.constant dense<0> : tensor<1xi32>
    %0 = "tfl.shape"(%arg2) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?xi32>) -> tensor<2xi32>
    %1 = "tfl.strided_slice"(%0, %cst_3, %cst_2, %cst_2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    %2 = "tfl.custom"(%cst_1, %1) {custom_code = "FlexTensorListReserve", custom_option = #tfl<const_bytes : "0x1154656E736F724C697374526573657276650040121154656E736F724C697374526573657276651A001A002A130A0D656C656D656E745F6474797065120230072A100A0A73686170655F74797065120230033200000255431414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
    %3 = "tfl.custom"(%cst_1, %1) {custom_code = "FlexTensorListReserve", custom_option = #tfl<const_bytes : "0x1154656E736F724C697374526573657276650040121154656E736F724C697374526573657276651A001A002A130A0D656C656D656E745F6474797065120230032A100A0A73686170655F74797065120230033200000255431414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
    %4:8 = "tfl.while"(%cst_0, %cst_0, %arg5, %arg6, %2, %2, %3, %3) ({
    ^bb0(%arg7: tensor<i32>, %arg8: tensor<i32>, %arg9: tensor<?xi32>, %arg10: tensor<?xi32>, %arg11: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg12: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg13: tensor<!tf_type.variant<tensor<?xi32>>>, %arg14: tensor<!tf_type.variant<tensor<?xi32>>>):
      %9 = tfl.less(%arg8, %1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %10 = tfl.less(%arg7, %1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %11 = tfl.logical_and %10, %9 {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i1>
      "tfl.yield"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg7: tensor<i32>, %arg8: tensor<i32>, %arg9: tensor<?xi32>, %arg10: tensor<?xi32>, %arg11: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg12: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg13: tensor<!tf_type.variant<tensor<?xi32>>>, %arg14: tensor<!tf_type.variant<tensor<?xi32>>>):
      %cst_4 = arith.constant dense<[0, 0, 1, 1, 1]> : tensor<5xi32>
      %cst_5 = arith.constant dense<[0, 1, 0, 1, 1]> : tensor<5xi32>
      %cst_6 = arith.constant dense<2> : tensor<i32>
      %cst_7 = arith.constant dense<"*"> : tensor<!tf_type.string>
      %cst_8 = arith.constant dense<-1> : tensor<1xi32>
      %cst_9 = arith.constant dense<-1> : tensor<i32>
      %cst_10 = arith.constant dense<2> : tensor<1xi32>
      %cst_11 = arith.constant dense<1> : tensor<i32>
      %cst_12 = arith.constant dense<0> : tensor<i32>
      %cst_13 = arith.constant dense<1> : tensor<1xi32>
      %cst_14 = arith.constant dense<0> : tensor<1xi32>
      %9 = "tfl.shape"(%arg1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>) -> tensor<2xi32>
      %10 = "tfl.strided_slice"(%9, %cst_14, %cst_13, %cst_13) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %11 = "tfl.range"(%cst_12, %10, %cst_11) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      %12 = "tfl.pack"(%10, %cst_11) {axis = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
      %13 = "tfl.strided_slice"(%9, %cst_13, %cst_10, %cst_13) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %14 = tfl.mul(%11, %13) {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
      %15 = "tfl.reshape"(%14, %12) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
      %16 = "tfl.strided_slice"(%9, %cst_14, %cst_10, %cst_13) {begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %17 = "tfl.reduce_prod"(%16, %cst_14) {keep_dims = true, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<1xi32>
      %18 = "tfl.reshape"(%arg1, %17) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
      %19 = "tfl.shape"(%arg0) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>) -> tensor<2xi32>
      %20 = "tfl.strided_slice"(%19, %cst_14, %cst_13, %cst_13) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %21 = "tfl.range"(%cst_12, %20, %cst_11) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
      %22 = "tfl.pack"(%20, %cst_11) {axis = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
      %23 = "tfl.strided_slice"(%19, %cst_13, %cst_10, %cst_13) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
      %24 = tfl.mul(%21, %23) {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
      %25 = "tfl.reshape"(%24, %22) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
      %26 = "tfl.strided_slice"(%19, %cst_14, %cst_10, %cst_13) {begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
      %27 = "tfl.reduce_prod"(%26, %cst_14) {keep_dims = true, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<1xi32>
      %28 = "tfl.reshape"(%arg0, %27) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
      %29 = tfl.add %arg8, %cst_11 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i32>
      %30 = "tfl.expand_dims"(%arg9, %cst_9) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
      %31 = tfl.add %30, %25 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?x1xi32>
      %32 = "tfl.reshape"(%31, %cst_8) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32>
      %33 = "tfl.gather"(%28, %32) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<?xi32>) -> tensor<?x!tf_type.string>
      %34 = "tfl.reshape"(%33, %cst_8) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
      %35 = "tfl.shape"(%34) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
      %36 = "tfl.fill"(%35, %cst_7) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<!tf_type.string>) -> tensor<?x!tf_type.string>
      %37 = "tfl.expand_dims"(%arg10, %cst_9) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
      %38 = tfl.add %37, %15 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?x1xi32>
      %39 = "tfl.reshape"(%38, %cst_8) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32>
      %40 = "tfl.gather"(%18, %39) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<?xi32>) -> tensor<?x!tf_type.string>
      %41 = "tfl.reshape"(%40, %cst_8) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
      %42 = "tfl.shape"(%41) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
      %43 = "tfl.fill"(%42, %cst_7) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<!tf_type.string>) -> tensor<?x!tf_type.string>
      %44 = "tfl.gather"(%arg2, %arg8) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
      %45 = "tfl.equal"(%44, %cst_6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
      %46 = "tfl.custom"(%45, %36, %34) {custom_code = "FlexSelect", custom_option = #tfl<const_bytes : "0x0653656C6563740031120653656C6563741A001A001A002A070A01541202300732180A052E31323131120F7768696C655F626F64795F3733303200023B341414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi1>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
      %47 = "tfl.custom"(%arg11, %arg8, %46) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300732170A042E326333120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<i32>, tensor<?x!tf_type.string>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
      %48 = "tfl.equal"(%44, %cst_11) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
      %49 = "tfl.custom"(%48, %43, %41) {custom_code = "FlexSelect", custom_option = #tfl<const_bytes : "0x0653656C6563740031120653656C6563741A001A001A002A070A01541202300732180A052E31323166120F7768696C655F626F64795F3733303200023B341414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi1>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
      %50 = "tfl.custom"(%arg12, %arg8, %49) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300732170A042E326335120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<i32>, tensor<?x!tf_type.string>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
      %51 = "tfl.gather"(%cst_5, %44) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<5xi32>, tensor<?xi32>) -> tensor<?xi32>
      %52 = tfl.add %arg9, %51 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?xi32>
      %53 = "tfl.custom"(%arg13, %arg8, %52) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300332170A042E326337120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?xi32>>>, tensor<i32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
      %54 = "tfl.gather"(%cst_4, %44) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<5xi32>, tensor<?xi32>) -> tensor<?xi32>
      %55 = tfl.add %arg10, %54 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?xi32>
      %56 = "tfl.custom"(%arg14, %arg8, %55) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300332170A042E343139120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?xi32>>>, tensor<i32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
      %57 = tfl.add %arg7, %cst_11 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i32>
      "tfl.yield"(%57, %29, %52, %55, %47, %50, %53, %56) : (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>) -> ()
    }) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>) -> (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>)
    %5 = "tfl.custom"(%4#4, %cst_1) {custom_code = "FlexTensorListStack", custom_option = #tfl<const_bytes : "0x0F54656E736F724C697374537461636B0049120F54656E736F724C697374537461636B1A001A002A130A0D656C656D656E745F6474797065120230072A1B0A0C6E756D5F656C656D656E7473120B18FFFFFFFFFFFFFFFFFF01320000025C4C1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<1xi32>) -> tensor<?x?x!tf_type.string>
    %6 = "tfl.custom"(%5, %cst) {custom_code = "FlexTranspose", custom_option = #tfl<const_bytes : "0x095472616E73706F7365002712095472616E73706F73651A001A002A0B0A05547065726D120230032A070A01541202300732000002342A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<2xi32>) -> tensor<?x?x!tf_type.string>
    %7 = "tfl.custom"(%4#5, %cst_1) {custom_code = "FlexTensorListStack", custom_option = #tfl<const_bytes : "0x0F54656E736F724C697374537461636B0049120F54656E736F724C697374537461636B1A001A002A130A0D656C656D656E745F6474797065120230072A1B0A0C6E756D5F656C656D656E7473120B18FFFFFFFFFFFFFFFFFF01320000025C4C1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<1xi32>) -> tensor<?x?x!tf_type.string>
    %8 = "tfl.custom"(%7, %cst) {custom_code = "FlexTranspose", custom_option = #tfl<const_bytes : "0x095472616E73706F7365002712095472616E73706F73651A001A002A070A0154120230072A0B0A05547065726D1202300332000002342A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<2xi32>) -> tensor<?x?x!tf_type.string>
    return %6, %8 : tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>
  }

// CHECK:   func.func @cond_false_72730(%arg0: tensor<?x?x!tf_type.string>, %arg1: tensor<?x?x!tf_type.string>, %arg2: tensor<?x?xi32>, %arg3: tensor<?x!tf_type.string>, %arg4: tensor<?x!tf_type.string>, %arg5: tensor<?xi32>, %arg6: tensor<?xi32>) -> (tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>) {
// CHECK:     %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK:     %cst_0 = arith.constant dense<0> : tensor<i32>
// CHECK:     %cst_1 = arith.constant dense<-1> : tensor<1xi32>
// CHECK:     %cst_2 = arith.constant dense<1> : tensor<1xi32>
// CHECK:     %cst_3 = arith.constant dense<0> : tensor<1xi32>
// CHECK:     %0 = call @func_0_DARWINN_FLOAT(%arg2, %cst_3, %cst_2) {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<?x?xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     %1:2 = call @func_1_CPU_FLOAT(%cst_1, %0, %cst_0, %arg5, %arg6, %arg1, %arg0, %arg2, %cst) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>, tensor<?x?xi32>, tensor<2xi32>) -> (tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>)
// CHECK:     return %1#0, %1#1 : tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>
// CHECK:   }
// CHECK:   func.func private @func_0_DARWINN_FLOAT(%arg0: tensor<?x?xi32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>) -> tensor<i32> attributes {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
// CHECK:     %0 = "tfl.shape"(%arg0) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?xi32>) -> tensor<2xi32>
// CHECK:     %1 = "tfl.strided_slice"(%0, %arg1, %arg2, %arg2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     return %1 : tensor<i32>
// CHECK:   }
// CHECK:   func.func private @func_1_CPU_FLOAT(%arg0: tensor<1xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<?x?x!tf_type.string>, %arg6: tensor<?x?x!tf_type.string>, %arg7: tensor<?x?xi32>, %arg8: tensor<2xi32>) -> (tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>) attributes {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
// CHECK:     %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexTensorListReserve", custom_option = #tfl<const_bytes : "0x1154656E736F724C697374526573657276650040121154656E736F724C697374526573657276651A001A002A130A0D656C656D656E745F6474797065120230072A100A0A73686170655F74797065120230033200000255431414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
// CHECK:     %1 = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexTensorListReserve", custom_option = #tfl<const_bytes : "0x1154656E736F724C697374526573657276650040121154656E736F724C697374526573657276651A001A002A130A0D656C656D656E745F6474797065120230032A100A0A73686170655F74797065120230033200000255431414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
// CHECK:     %2:8 = "tfl.while"(%arg2, %arg2, %arg3, %arg4, %0, %0, %1, %1) ({
// CHECK:     ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<?xi32>, %arg12: tensor<?xi32>, %arg13: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg14: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg15: tensor<!tf_type.variant<tensor<?xi32>>>, %arg16: tensor<!tf_type.variant<tensor<?xi32>>>):
// CHECK:       %7 = func.call @func_2_DARWINN_FLOAT(%arg10, %arg1, %arg9) {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:       "tfl.yield"(%7) : (tensor<i1>) -> ()
// CHECK:     }, {
// CHECK:     ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<?xi32>, %arg12: tensor<?xi32>, %arg13: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg14: tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, %arg15: tensor<!tf_type.variant<tensor<?xi32>>>, %arg16: tensor<!tf_type.variant<tensor<?xi32>>>):
// CHECK:       %cst = arith.constant dense<[0, 0, 1, 1, 1]> : tensor<5xi32>
// CHECK:       %cst_0 = arith.constant dense<[0, 1, 0, 1, 1]> : tensor<5xi32>
// CHECK:       %cst_1 = arith.constant dense<2> : tensor<i32>
// CHECK:       %cst_2 = arith.constant dense<"*"> : tensor<!tf_type.string>
// CHECK:       %cst_3 = arith.constant dense<-1> : tensor<1xi32>
// CHECK:       %cst_4 = arith.constant dense<-1> : tensor<i32>
// CHECK:       %cst_5 = arith.constant dense<2> : tensor<1xi32>
// CHECK:       %cst_6 = arith.constant dense<1> : tensor<i32>
// CHECK:       %cst_7 = arith.constant dense<0> : tensor<i32>
// CHECK:       %cst_8 = arith.constant dense<1> : tensor<1xi32>
// CHECK:       %cst_9 = arith.constant dense<0> : tensor<1xi32>
// CHECK:       %7:2 = func.call @func_3_DARWINN_FLOAT(%arg5, %cst_9, %cst_8, %cst_7, %cst_6, %cst_5) {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<1xi32>) -> (tensor<?x1xi32>, tensor<2xi32>)
// CHECK:       %8 = "tfl.reduce_prod"(%7#1, %cst_9) {keep_dims = true, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:       %9:3 = func.call @func_4_DARWINN_FLOAT(%arg5, %8, %arg6, %cst_9, %cst_8, %cst_7, %cst_6, %cst_5) {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_4"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>, tensor<?x?x!tf_type.string>, tensor<1xi32>, tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<1xi32>) -> (tensor<?x!tf_type.string>, tensor<?x1xi32>, tensor<2xi32>)
// CHECK:       %10 = "tfl.reduce_prod"(%9#2, %cst_9) {keep_dims = true, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:       %11 = "tfl.expand_dims"(%arg11, %cst_4) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
// CHECK:       %12 = "tfl.expand_dims"(%arg12, %cst_4) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
// CHECK:       %13:10 = func.call @func_5_DARWINN_FLOAT(%arg6, %10, %arg10, %cst_6, %11, %9#1, %cst_3, %cst_2, %12, %7#0, %9#0, %arg7, %cst_1, %cst_0, %arg11, %cst, %arg12, %arg9) {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_5"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<?x1xi32>, tensor<?x1xi32>, tensor<1xi32>, tensor<!tf_type.string>, tensor<?x1xi32>, tensor<?x1xi32>, tensor<?x!tf_type.string>, tensor<?x?xi32>, tensor<i32>, tensor<5xi32>, tensor<?xi32>, tensor<5xi32>, tensor<?xi32>, tensor<i32>) -> (tensor<i32>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?xi1>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>, tensor<i32>)
// CHECK:       %14 = "tfl.custom"(%13#5, %13#2, %13#1) {custom_code = "FlexSelect", custom_option = #tfl<const_bytes : "0x0653656C6563740031120653656C6563741A001A001A002A070A01541202300732180A052E31323131120F7768696C655F626F64795F3733303200023B341414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi1>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK:       %15 = "tfl.custom"(%arg13, %arg10, %14) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300732170A042E326333120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<i32>, tensor<?x!tf_type.string>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
// CHECK:       %16 = "tfl.custom"(%13#6, %13#4, %13#3) {custom_code = "FlexSelect", custom_option = #tfl<const_bytes : "0x0653656C6563740031120653656C6563741A001A001A002A070A01541202300732180A052E31323166120F7768696C655F626F64795F3733303200023B341414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?xi1>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK:       %17 = "tfl.custom"(%arg14, %arg10, %16) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300732170A042E326335120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<i32>, tensor<?x!tf_type.string>) -> tensor<!tf_type.variant<tensor<?x!tf_type.string>>>
// CHECK:       %18 = "tfl.custom"(%arg15, %arg10, %13#7) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300332170A042E326337120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?xi32>>>, tensor<i32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
// CHECK:       %19 = "tfl.custom"(%arg16, %arg10, %13#8) {custom_code = "FlexTensorListSetItem", custom_option = #tfl<const_bytes : "0x1154656E736F724C6973745365744974656D0047121154656E736F724C6973745365744974656D1A001A001A002A130A0D656C656D656E745F64747970651202300332170A042E343139120F7768696C655F626F64795F3733303200025C4A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?xi32>>>, tensor<i32>, tensor<?xi32>) -> tensor<!tf_type.variant<tensor<?xi32>>>
// CHECK:       "tfl.yield"(%13#9, %13#0, %13#7, %13#8, %15, %17, %18, %19) : (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>) -> ()
// CHECK:     }) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>) -> (tensor<i32>, tensor<i32>, tensor<?xi32>, tensor<?xi32>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<!tf_type.variant<tensor<?xi32>>>, tensor<!tf_type.variant<tensor<?xi32>>>)
// CHECK:     %3 = "tfl.custom"(%2#4, %arg0) {custom_code = "FlexTensorListStack", custom_option = #tfl<const_bytes : "0x0F54656E736F724C697374537461636B0049120F54656E736F724C697374537461636B1A001A002A130A0D656C656D656E745F6474797065120230072A1B0A0C6E756D5F656C656D656E7473120B18FFFFFFFFFFFFFFFFFF01320000025C4C1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<1xi32>) -> tensor<?x?x!tf_type.string>
// CHECK:     %4 = "tfl.custom"(%3, %arg8) {custom_code = "FlexTranspose", custom_option = #tfl<const_bytes : "0x095472616E73706F7365002712095472616E73706F73651A001A002A0B0A05547065726D120230032A070A01541202300732000002342A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<2xi32>) -> tensor<?x?x!tf_type.string>
// CHECK:     %5 = "tfl.custom"(%2#5, %arg0) {custom_code = "FlexTensorListStack", custom_option = #tfl<const_bytes : "0x0F54656E736F724C697374537461636B0049120F54656E736F724C697374537461636B1A001A002A130A0D656C656D656E745F6474797065120230072A1B0A0C6E756D5F656C656D656E7473120B18FFFFFFFFFFFFFFFFFF01320000025C4C1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<!tf_type.variant<tensor<?x!tf_type.string>>>, tensor<1xi32>) -> tensor<?x?x!tf_type.string>
// CHECK:     %6 = "tfl.custom"(%5, %arg8) {custom_code = "FlexTranspose", custom_option = #tfl<const_bytes : "0x095472616E73706F7365002712095472616E73706F73651A001A002A070A0154120230072A0B0A05547065726D1202300332000002342A1414042801">, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<2xi32>) -> tensor<?x?x!tf_type.string>
// CHECK:     return %4, %6 : tensor<?x?x!tf_type.string>, tensor<?x?x!tf_type.string>
// CHECK:   }
// CHECK:   func.func private @func_2_DARWINN_FLOAT(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i1> attributes {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
// CHECK:     %0 = tfl.less(%arg0, %arg1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:     %1 = tfl.less(%arg2, %arg1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:     %2 = tfl.logical_and %1, %0 {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i1>
// CHECK:     return %2 : tensor<i1>
// CHECK:   }
// CHECK:   func.func private @func_3_DARWINN_FLOAT(%arg0: tensor<?x?x!tf_type.string>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<1xi32>) -> (tensor<?x1xi32>, tensor<2xi32>) attributes {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} {
// CHECK:     %0 = "tfl.shape"(%arg0) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>) -> tensor<2xi32>
// CHECK:     %1 = "tfl.strided_slice"(%0, %arg1, %arg2, %arg2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     %2 = "tfl.range"(%arg3, %1, %arg4) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
// CHECK:     %3 = "tfl.pack"(%1, %arg4) {axis = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %4 = "tfl.strided_slice"(%0, %arg2, %arg5, %arg2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     %5 = tfl.mul(%2, %4) {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:     %6 = "tfl.reshape"(%5, %3) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
// CHECK:     %7 = "tfl.strided_slice"(%0, %arg1, %arg5, %arg2) {begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
// CHECK:     return %6, %7 : tensor<?x1xi32>, tensor<2xi32>
// CHECK:   }
// CHECK:   func.func private @func_4_DARWINN_FLOAT(%arg0: tensor<?x?x!tf_type.string>, %arg1: tensor<1xi32>, %arg2: tensor<?x?x!tf_type.string>, %arg3: tensor<1xi32>, %arg4: tensor<1xi32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<1xi32>) -> (tensor<?x!tf_type.string>, tensor<?x1xi32>, tensor<2xi32>) attributes {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_4"} {
// CHECK:     %0 = "tfl.reshape"(%arg0, %arg1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %1 = "tfl.shape"(%arg2) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>) -> tensor<2xi32>
// CHECK:     %2 = "tfl.strided_slice"(%1, %arg3, %arg4, %arg4) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     %3 = "tfl.range"(%arg5, %2, %arg6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
// CHECK:     %4 = "tfl.pack"(%2, %arg6) {axis = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
// CHECK:     %5 = "tfl.strided_slice"(%1, %arg4, %arg7, %arg4) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 1 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
// CHECK:     %6 = tfl.mul(%3, %5) {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:     %7 = "tfl.reshape"(%6, %4) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<2xi32>) -> tensor<?x1xi32>
// CHECK:     %8 = "tfl.strided_slice"(%1, %arg3, %arg7, %arg4) {begin_mask = 1 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
// CHECK:     return %0, %7, %8 : tensor<?x!tf_type.string>, tensor<?x1xi32>, tensor<2xi32>
// CHECK:   }
// CHECK:   func.func private @func_5_DARWINN_FLOAT(%arg0: tensor<?x?x!tf_type.string>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<?x1xi32>, %arg5: tensor<?x1xi32>, %arg6: tensor<1xi32>, %arg7: tensor<!tf_type.string>, %arg8: tensor<?x1xi32>, %arg9: tensor<?x1xi32>, %arg10: tensor<?x!tf_type.string>, %arg11: tensor<?x?xi32>, %arg12: tensor<i32>, %arg13: tensor<5xi32>, %arg14: tensor<?xi32>, %arg15: tensor<5xi32>, %arg16: tensor<?xi32>, %arg17: tensor<i32>) -> (tensor<i32>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?xi1>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>, tensor<i32>) attributes {tac.device = "DARWINN", tac.inference_type = "FLOAT", tac.interface_name = "func_5"} {
// CHECK:     %0 = "tfl.reshape"(%arg0, %arg1) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %1 = tfl.add %arg2, %arg3 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i32>
// CHECK:     %2 = tfl.add %arg4, %arg5 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?x1xi32>
// CHECK:     %3 = "tfl.reshape"(%2, %arg6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK:     %4 = "tfl.gather"(%0, %3) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<?xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %5 = "tfl.reshape"(%4, %arg6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %6 = "tfl.shape"(%5) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
// CHECK:     %7 = "tfl.fill"(%6, %arg7) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK:     %8 = tfl.add %arg8, %arg9 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?x1xi32>
// CHECK:     %9 = "tfl.reshape"(%8, %arg6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x1xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK:     %10 = "tfl.gather"(%arg10, %9) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<?xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %11 = "tfl.reshape"(%10, %arg6) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
// CHECK:     %12 = "tfl.shape"(%11) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
// CHECK:     %13 = "tfl.fill"(%12, %arg7) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<1xi32>, tensor<!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK:     %14 = "tfl.gather"(%arg11, %arg2) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:     %15 = "tfl.equal"(%14, %arg12) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
// CHECK:     %16 = "tfl.equal"(%14, %arg3) {tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi1>
// CHECK:     %17 = "tfl.gather"(%arg13, %14) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<5xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:     %18 = tfl.add %arg14, %17 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?xi32>
// CHECK:     %19 = "tfl.gather"(%arg15, %14) {axis = 0 : i32, batch_dims = 0 : i32, tac.device = "DARWINN", tac.inference_type = "FLOAT"} : (tensor<5xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:     %20 = tfl.add %arg16, %19 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<?xi32>
// CHECK:     %21 = tfl.add %arg17, %arg3 {fused_activation_function = "NONE", tac.device = "DARWINN", tac.inference_type = "FLOAT"} : tensor<i32>
// CHECK:     return %1, %5, %7, %11, %13, %15, %16, %18, %20, %21 : tensor<i32>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?x!tf_type.string>, tensor<?xi1>, tensor<?xi1>, tensor<?xi32>, tensor<?xi32>, tensor<i32>
// CHECK:   }
