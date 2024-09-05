// RUN: tac-opt-all-backends -tfl-tac-filter='use-test-setting=true' %s -split-input-file -verify-diagnostics | FileCheck %s

// expected-remark@below {{Tac filter (0): filter type: function filter SKIP_TARGET_ANNOTATION, filter_pattern: "^testFunction"}}
// expected-remark@below {{Tac filter (1): filter type: function filter INCLUDE_TARGET_ANNOTATION, filter_pattern: "testFunctionInclude"}}
// expected-remark@below {{Tac filter (1) specified but not applied to any op}}
// expected-remark@below {{Tac filter (2): filter type: op filter, filter_pattern: "^test_op"}}
// expected-remark@below {{Tac filter (2) specified but not applied to any op}}
module {
  // CHECK-LABEL: testFunctionSkiped
  // expected-remark@+1 {{filtered by tac filter (0)}}
  func.func @testFunctionSkiped(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
    // CHECK: tfl.add
    // CHECK-SAME: tac.skip_target_annotation
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    // CHECK: tfl.add
    // CHECK-SAME: tac.skip_target_annotation
    %1 = "tfl.add"(%arg0, %0) {fused_activation_function = "RELU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    // CHECK: tfl.relu
    // CHECK-SAME: tac.skip_target_annotation
    %2 = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
    func.return
  }
}

// -----

// expected-remark@below {{Tac filter (0): filter type: function filter SKIP_TARGET_ANNOTATION, filter_pattern: "^testFunction"}}
// expected-remark@below {{Tac filter (1): filter type: function filter INCLUDE_TARGET_ANNOTATION, filter_pattern: "testFunctionInclude"}}
// expected-remark@below {{Tac filter (2): filter type: op filter, filter_pattern: "^test_op"}}
// expected-remark@below {{Tac filter (2) specified but not applied to any op}}
module {
  // CHECK-LABEL: testFunctionInclude
  // CHECK-NOT: tac.skip_target_annotation
  // expected-remark@+2 {{filtered by tac filter (0)}}
  // expected-remark@+1 {{filtered by tac filter (1)}}
  func.func @testFunctionInclude(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    func.return
  }
}

// -----

// expected-remark@below {{Tac filter (0): filter type: function filter SKIP_TARGET_ANNOTATION, filter_pattern: "^testFunction"}}
// expected-remark@below {{Tac filter (0) specified but not applied to any op}}
// expected-remark@below {{Tac filter (1): filter type: function filter INCLUDE_TARGET_ANNOTATION, filter_pattern: "testFunctionInclude"}}
// expected-remark@below {{Tac filter (1) specified but not applied to any op}}
// expected-remark@below {{Tac filter (2): filter type: op filter, filter_pattern: "^test_op"}}
module {
  // CHECK-LABEL: testOpFilter
  // expected-remark@+1 {{all ops filtered by tac filter (2): "tfl.add", "tfl.relu"}}
  func.func @testOpFilter(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) {
    // CHECK: tfl.add
    // CHECK-SAME: tac.skip_target_annotation
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc("test_op_0")
    // CHECK: tfl.add
    // CHECK-NOT: tac.skip_target_annotation
    %1 = "tfl.add"(%arg0, %0) {fused_activation_function = "RELU"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc("non_test_op")
    // CHECK: tfl.relu
    // CHECK-SAME: tac.skip_target_annotation
    %2 = "tfl.relu"(%arg0) : (tensor<1xf32>) -> tensor<1xf32> loc("test_op_1")
    func.return
  }
}
