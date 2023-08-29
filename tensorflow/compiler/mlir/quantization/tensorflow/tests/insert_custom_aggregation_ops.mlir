// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=MIN_MAX'  | FileCheck --check-prefix=MIN-MAX-CHECK %s
// RUN: tf-quant-opt %s -quant-insert-custom-aggregation-ops='test-case=AVERAGE_MIN_MAX'  | FileCheck --check-prefix=AVERAGE-MIN-MAX-CHECK %s

module {
  func.func @add_custom_ops(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func @no_custom_ops_on_non_f32_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %add = "tf.AddV2"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %add : tensor<*xi32>
  }

}

// "\08\01" represents serialized CalibrationOptions(calibration_method=CALIBRATION_METHOD_MIN_MAX)
// MIN-MAX-CHECK: func @add_custom_ops
// MIN-MAX-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = "", serialized_calibration_options = "\08\01"} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = "", serialized_calibration_options = "\08\01"} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = "", serialized_calibration_options = "\08\01"} : (tensor<*xf32>) -> tensor<*xf32>
// MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// MIN-MAX-CHECK: func @no_custom_ops_on_non_f32_type
// MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// MIN-MAX-CHECK-NEXT:  return

// "\08\02" represents serialized CalibrationOptions(calibration_method=CALIBRATION_METHOD_AVERAGE_MIN_MAX)
// AVERAGE-MIN-MAX-CHECK: func @add_custom_ops
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[rhs:%.*]] = "tf.CustomAggregator"(%arg1) {id = "", serialized_calibration_options = "\08\02"} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[lhs:%.*]] = "tf.CustomAggregator"(%arg0) {id = "", serialized_calibration_options = "\08\02"} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[add:%.*]] = "tf.AddV2"([[lhs]], [[rhs]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  [[res:%.*]] = "tf.CustomAggregator"([[add]]) {id = "", serialized_calibration_options = "\08\02"} : (tensor<*xf32>) -> tensor<*xf32>
// AVERAGE-MIN-MAX-CHECK-NEXT:  return [[res]] : tensor<*xf32>

// AVERAGE-MIN-MAX-CHECK: func @no_custom_ops_on_non_f32_type
// AVERAGE-MIN-MAX-CHECK-NEXT:  "tf.AddV2"
// AVERAGE-MIN-MAX-CHECK-NEXT:  return
