// RUN: tf-opt -split-input-file -verify-diagnostics -tf-test-visitor-util-interrupt %s

// Test simple operations with no regions and no interrupts. They should be
// visited with stage "before all regions".

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

// Test simple operations with no regions and interrupts. No remarks after
// the interrupting operation is visited.

// expected-remark@below {{0: before all regions}}
// expected-remark@below {{2: walk was interrupted}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  %0 = "tf.Identity"(%arg0)  {interrupt_before_all = true} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test operation with non empty regions.
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{5: walk was interrupted}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  %0 = "tf.unknownop"(%arg0) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }) {interrupt_after_all = true} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test operation with multiple regions.
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{5: walk was interrupted}}
func @foo(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  %0 = "tf.unknownop"(%arg0) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }, {
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    "tf.yield"(%1) : (tensor<f32>) -> ()
  }) {interrupt_after_region = 0} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Test static filtering
// expected-remark@below {{0: before all regions}}
// expected-remark@below {{7: walk was interrupted}}
func @foo(%arg0: tensor<f32>, %arg1: tensor<i1>) -> tensor<f32> {
  // expected-remark@below {{1: before all regions}}
  %cst = constant dense<1.0> : tensor<f32>
  // expected-remark@below {{2: before all regions}}
  // expected-remark@below {{5: before region #1}}
  // expected-remark@below {{8: before all regions}}
  // expected-remark@below {{9: before region #1}}
  // expected-remark@below {{10: after all regions}}
  %0 = "tf.IfRegion"(%arg1) ({
    // expected-remark@below {{3: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    // expected-remark@below {{4: before all regions}}
    "tf.Yield"(%1) : (tensor<f32>) -> ()
  }, {
    // expected-remark@below {{6: before all regions}}
    %1 = "tf.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
    "tf.Yield"(%1) { interrupt_after_all = true } : (tensor<f32>) -> ()
  }) {is_stateless = true}: (tensor<i1>) -> tensor<f32>
  return %0 : tensor<f32>
}
