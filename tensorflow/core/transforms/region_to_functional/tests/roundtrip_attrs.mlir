// RUN: tfg-transforms-opt --tfg-functional-to-region --tfg-region-to-functional %s | FileCheck %s

// Check that the original functions aren't re-used because the last argument
// gets dropped.

// CHECK: tfg.func @case0
tfg.func @case0(%A: tensor<f32> {tf._attr_a, tfg.name = "A"},
                %B: tensor<f64> {tf._attr_b, tfg.name = "B"},
                %C: tensor<i32>)
    -> (tensor<i32> {tf._attr_q, tfg.name = "Q"})
    attributes {tf._case0} {
  %AB, %ctl = AB(%A, %B) : (tensor<f32>, tensor<f64>) -> (tensor<i32>)
  return(%AB) : tensor<i32>
}

// CHECK: tfg.func @case1
tfg.func @case1(%A: tensor<f32> {tf._attr_0, tfg.name = "A"},
                %B: tensor<f64> {tf._attr_b, tfg.name = "B"},
                %C: tensor<i32>)
    -> (tensor<i32> {tf._attr_9, tfg.name = "Q"})
    attributes {tf._case1} {
  %BA, %ctl = BA(%B, %A) : (tensor<f64>, tensor<f32>) -> (tensor<i32>)
  return(%BA) : tensor<i32>
}

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
tfg.func @test(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: %[[A:.*]], %[[CTLA:.*]] = A
  %A, %ctlA = A : () -> (tensor<f32>)
  // CHECK: %[[B:.*]], %[[CTLB:.*]] = B
  %B, %ctlB = B : () -> (tensor<f64>)
  // CHECK: Case(%[[ARG0]], %[[A]], %[[B]]) [%[[CTLA]], %[[CTLB]]]
  // Check op attributes are propragated.
  // CHECK-SAME: _attr_a
  // CHECK-SAME: _attr_b
  %Case, %ctlCase = Case(%arg0, %A, %B, %arg0) [%ctlA, %ctlB] {
    _attr_a, _attr_b, Tin = [f32, f64, i32], Tout = [i32], output_shapes = [#tf_type.shape<>],
    branches = [#tf_type.func<@case0, {}>, #tf_type.func<@case1, {}>],
    _mlir_name = "foo"
  } : (tensor<i32>, tensor<f32>, tensor<f64>, tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}

// Check argument attributes are dropped (converted to implicit capture).
// CHECK: tfg.func @case0_tfg_region_specialized_foo
// CHECK-SAME: (%[[A:.*]]: tensor<f32> {tfg.regenerate_output_shapes},
// CHECK-NEXT:  %[[B:.*]]: tensor<f64> {tfg.regenerate_output_shapes})
// Check result attributes are preserved.
// CHECK: -> (tensor<i32> {tf._attr_q, tfg.name = "Q", tfg.regenerate_output_shapes})
// Check function attributes are preserved.
// CHECK: attributes {tf._case0}
