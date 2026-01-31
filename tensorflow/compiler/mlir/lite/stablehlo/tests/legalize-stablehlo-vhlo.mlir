// RUN: odml-to-stablehlo-opt %s --stablehlo-legalize-vhlo -reconcile-unrealized-casts -split-input-file | FileCheck %s
// RUN: odml-to-stablehlo-opt --stablehlo-legalize-vhlo -reconcile-unrealized-casts %s | odml-to-stablehlo-opt --vhlo-legalize-stablehlo -reconcile-unrealized-casts > %t.0
// RUN: odml-to-stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

// CHECK-LABEL: op_tfl
func.func @op_tfl(%arg0 : tensor<f32>) -> (tensor<f32>) {
  // CHECK: %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<f32>
  %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<f32>
  return %0 :  tensor<f32>
}

// -----

// CHECK-LABEL: op_shlo
func.func @op_shlo(%arg0 : tensor<f32>) -> (tensor<f32>) {
  // CHECK: %0 = "vhlo.add_v1"(%arg0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
  return %0 :  tensor<f32>
}

// -----

// CHECK-LABEL: mixed_shlo_tfl_shlo
func.func @mixed_shlo_tfl_shlo(%arg0 : tensor<f32>) -> (tensor<f32>) {
  // CHECK: %0 = "vhlo.abs_v1"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<f32>
  // CHECK-NEXT: %2 = "vhlo.abs_v1"(%1) : (tensor<f32>) -> tensor<f32>
  %0 = stablehlo.abs %arg0 : tensor<f32>
  %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<f32>
  %2 = stablehlo.abs %1 : tensor<f32>
  return %2 :  tensor<f32>
}

// -----

// CHECK-LABEL: mixed_tfl_shlo_tfl
func.func @mixed_tfl_shlo_tfl(%arg0 : tensor<f32>) -> (tensor<f32>) {
  %0 = "tfl.abs"(%arg0) {fused_activation_function = "NONE"} : (tensor<f32>) -> tensor<f32>
  // CHECK: %1 = "vhlo.add_v1"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = stablehlo.add %0, %arg0 : tensor<f32>
  %2 = "tfl.abs"(%1) {fused_activation_function = "NONE"} : (tensor<f32>) -> tensor<f32>
  return %2 :  tensor<f32>
}

// -----

// CHECK-LABEL: op_with_region
func.func @op_with_region(%arg0: tensor<1x16x16x320xf32>, %arg1: tensor<f32>) -> tensor<1x320xf32> {
  // CHECK:      %0 = "vhlo.reduce_v1"(%arg0, %arg1) <{{.*}}> ({
  // CHECK-NEXT:  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  // CHECK-NEXT:    %1 = "vhlo.add_v1"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:    "vhlo.return_v1"(%1) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
  %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1, 2] : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
  return %0 : tensor<1x320xf32>
}

// -----

// CHECK-LABEL: op_with_region_mixed_tfl_shlo_tfl
func.func @op_with_region_mixed_tfl_shlo_tfl(%arg0: tensor<7x5xf32>, %arg1 : tensor<5xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<5xf32>, %arg3: tensor<5xf32>):
    // CHECK:      %1 = "tfl.abs"(%arg2) {fused_activation_function = "NONE"} : (tensor<5xf32>) -> tensor<5xf32>
    // CHECK-NEXT: %2 = "vhlo.add_v1"(%1, %arg2) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    // CHECK-NEXT: %3 = "tfl.abs"(%2) {fused_activation_function = "NONE"} : (tensor<5xf32>) -> tensor<5xf32>
    %1 = "tfl.abs"(%arg2) {fused_activation_function = "NONE"} : (tensor<5xf32>) -> tensor<5xf32>
    %2 = stablehlo.add %1, %arg2 : tensor<5xf32>
    %3 = "tfl.abs"(%2) {fused_activation_function = "NONE"} : (tensor<5xf32>) -> tensor<5xf32>
    "stablehlo.return"(%3) : (tensor<5xf32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<7x5xf32>, tensor<5xf32>) -> tensor<5xf32>
  func.return %0: tensor<5xf32>
}

// -----

// CHECK-LABEL: op_with_region_mixed_shlo_tfl_shlo
func.func @op_with_region_mixed_shlo_tfl_shlo(%arg0: tensor<7x5xf32>, %arg1 : tensor<5xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<5xf32>, %arg3: tensor<5xf32> ):
    // CHECK:      %1 = "vhlo.abs_v1"(%arg2) : (tensor<5xf32>) -> tensor<5xf32>
    // CHECK-NEXT: %2 = tfl.add %1, %arg2 {fused_activation_function = "NONE"} : tensor<5xf32>
    // CHECK-NEXT: %3 = "vhlo.abs_v1"(%2) : (tensor<5xf32>) -> tensor<5xf32>
    %1 = stablehlo.abs %arg2 : tensor<5xf32>
    %2 = tfl.add %1, %arg2 {fused_activation_function = "NONE"} : tensor<5xf32>
    %3 = stablehlo.abs %2 : tensor<5xf32>
    "stablehlo.return"(%3) : (tensor<5xf32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<7x5xf32>, tensor<5xf32>) -> tensor<5xf32>
  func.return %0: tensor<5xf32>
}

// -----

// CHECK-LABEL: op_with_tfl_control_flow
func.func @op_with_tfl_control_flow() -> (tensor<1xf32>, !tfl.control) {
  // CHECK: vhlo.constant_v1
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: tfl.control_node
  %outputs, %control = tfl.control_node {
    %1 = "tfl.neg"(%0) : (tensor<1xf32>) -> tensor<1xf32>
    "tfl.yield"(%1) : (tensor<1xf32>) -> ()
  }
  return %outputs, %control : tensor<1xf32>, !tfl.control
}

// -----

// CHECK-LABEL: func_with_tfl_attrs
func.func @func_with_tfl_attrs(%arg0: tensor<!tf_type.variant<tensor<2xi32>>>, %arg1: tensor<!tf_type.variant<tensor<*xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>> attributes {tf.entry_function = {inputs = "arg0,arg1", outputs = "arg0"}} {
  return %arg0 : tensor<!tf_type.variant<tensor<2xi32>>>
}

// -----

// There are cases where ODML converter relies on constants not being folded or
// CSE'ed. This test ensures that StableHLO<->ODML conversion does not fold.

// CHECK-LABEL: mixed_no_constant_folding
func.func @mixed_no_constant_folding() -> (tensor<f32>) {
  // CHECK:      %[[CST0:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-NEXT: %[[CST1:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-NEXT: "vhlo.add_v1"(%[[CST0]], %[[CST1]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.add %cst_0, %cst_1 : tensor<f32>
  return %0 : tensor<f32>
}
