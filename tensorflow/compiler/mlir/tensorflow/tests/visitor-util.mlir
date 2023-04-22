// RUN: tf-opt -split-input-file -verify-diagnostics -tf-test-visitor-util %s

// Test simple operations with no regions. They should be visited with stage
// = before all regions.

// expected-remark@below {{0: before all regions}}
// expected-remark@below {{4: after all regions}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  %0 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
  // expected-remark@below {{3: before all regions}}
  return %0 : tensor<f32>
}

// -----
// Test operation with empty regions.
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{5: after all regions}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  // expected-remark@below {{3: after all regions}}
  %0 = "tf.unknownop"(%arg0) ({
  }) : (tensor<f32>) -> tensor<f32>
  // expected-remark@below {{4: before all regions}}
  return %0 : tensor<f32>
}

// -----
// Test operation with non empty regions.
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{7: after all regions}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  // expected-remark@below {{5: after all regions}}
  %0 = "tf.unknownop"(%arg0) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>
  // expected-remark@below {{6: before all regions}}
  return %0 : tensor<f32>
}

// -----
// Test operation with multiple regions.
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{10: after all regions}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  // expected-remark@below {{5: before region #1}}
  // expected-remark@below {{8: after all regions}}
  %0 = "tf.unknownop"(%arg0) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }, {
    // expected-remark@below {{6: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{7: before all regions}}
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }) : (tensor<f32>) -> tensor<f32>
  // expected-remark@below {{9: before all regions}}
  return %0 : tensor<f32>
}

// -----
// Test static filtering
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{10: after all regions}}
func @foo(%arg0: tensor<f32>, %arg1: tensor<i1>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  // expected-remark@below {{5: before region #1}}
  // expected-remark@below {{8: after all regions}}
  // expected-remark@below {{11: before all regions}}
  // expected-remark@below {{12: before region #1}}
  // expected-remark@below {{13: after all regions}}
  %0 = "tf.IfRegion"(%arg1) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.Yield"(%1) : (tensor<f32>) -> ()
  }, {
    // expected-remark@below {{6: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{7: before all regions}}
    "tf.Yield"(%1) : (tensor<f32>) -> ()
  }) {is_stateless = true}: (tensor<i1>) -> tensor<f32>
  // expected-remark@below {{9: before all regions}}
  return %0 : tensor<f32>
}
