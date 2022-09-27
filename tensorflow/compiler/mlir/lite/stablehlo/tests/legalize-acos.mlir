// RUN: tf-mhlo-tfl-opt %s -tf-mhlo-tfl | FileCheck %s

// Convert Acos to TFL via MHLO, but leave the unsupported tf.Cos untouched.
func.func @convertAcos(%arg0: tensor<3xf32>) -> tensor<3xf32>  {
  %1 = "tf.Acos"(%arg0) {device = ""} : (tensor<3xf32>) -> tensor<3xf32>
  %2 = "tf.Cos"(%1) : (tensor<3xf32>) -> tensor<3xf32>
  func.return %2: tensor<3xf32>
}

// CHECK-LABEL: @convertAcos
// CHECK-SAME: %arg0: tensor<3xf32>
// CHECK: %[[CST:.*]] = arith.constant dense<-1.000000e+00> : tensor<3xf32>
// CHECK: %[[TMP1:.*]] = tfl.not_equal(%arg0, %[[CST]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
// CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : tensor<3xf32>
// CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK: %[[TMP2:.*]] = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<3xf32>
// CHECK: %[[TMP3:.*]] = tfl.sub %[[CST1]], %[[TMP2]] {fused_activation_function = "NONE"} : tensor<3xf32>
// CHECK: %[[TMP4:.*]] = "tfl.sqrt"(%[[TMP3]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK: %[[CST2:.*]] = arith.constant dense<1.000000e+00> : tensor<3xf32>
// CHECK: %[[TMP5:.*]] = tfl.add %[[CST2]], %arg0 {fused_activation_function = "NONE"} : tensor<3xf32>
// CHECK: %[[TMP6:.*]] = "tfl.custom"(%[[TMP4]], %[[TMP5]]) {custom_code = "atan2", custom_option = #tfl<const_bytes : "0x">} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK: %[[TMP7:.*]] = tfl.mul %[[CST0]], %[[TMP6]] {fused_activation_function = "NONE"} : tensor<3xf32>
// CHECK: %[[CST3:.*]] = arith.constant dense<3.14159274> : tensor<3xf32>
// CHECK: %[[RES1:.*]] = "tfl.select"(%[[TMP1]], %[[TMP7]], %[[CST3]]) : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK: %[[RES2:.*]] = "tf.Cos"(%[[RES1]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK: return %[[RES2]] : tensor<3xf32>


// Leave unsupported tf.Cos untouched in TF dialect.
func.func @cosUnconverted(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "tf.Cos"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK-LABEL: @cosUnconverted
// CHECK-SAME: %arg0: tensor<3xf32>
// CHECK: %[[RES:.*]] = "tf.Cos"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK: return %[[RES]] : tensor<3xf32>
