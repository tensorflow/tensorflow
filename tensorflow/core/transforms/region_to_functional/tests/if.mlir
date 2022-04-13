// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that `IfRegion` is correctly converted back to functional form with
// argument names.

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[ARGS:.*]]:2, %[[CTL:.*]] = Value name("[[VALUE:.*]]") : () -> (tensor<[[TYPE:.*]]>, tensor<[[TYPE_0:.*]]>)
  %Value:2, %ctl = Value name("value") : () -> (tensor<i32>, tensor<f32>)
  // CHECK: %[[COND:.*]], %[[CTL_0:.*]] = Cond
  %Cond, %ctl_0 = Cond : () -> (tensor<i1>)
  // CHECK:      If(%[[COND]], %[[ARGS]]#0, %[[ARGS]]#1) [%[[CTL_0]]]
  // CHECK-SAME: {else_branch = #tf_type.func<@[[ELSE_FUNC:.*]], {}>,
  // CHECK-SAME:  then_branch = #tf_type.func<@[[THEN_FUNC:.*]], {}>}
  // CHECK-SAME: : (tensor<i1>, tensor<[[TYPE]]>, tensor<[[TYPE_0]]>) -> (tensor<[[DTYPE:.*]]>)
  %If:2 = IfRegion %Cond [%ctl_0] then {
    %A, %ctl_1 = A(%Value#0) name("A") : (tensor<i32>) -> (tensor<i64>)
    yield(%A) : tensor<i64>
  } else {
    %B, %ctl_1 = B(%Value#1) name("B") : (tensor<f32>) -> (tensor<i64>)
    yield(%B) : tensor<i64>
  } : (tensor<i1>) -> (tensor<i64>)
}

// CHECK: tfg.func @if_then_function
// CHECK-SAME: (%[[VALUE]]_tfg_result_0: tensor<[[TYPE]]> {tfg.name = "[[VALUE]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK-NEXT:  %[[VALUE]]_tfg_result_1: tensor<[[TYPE_0]]> {tfg.name = "[[VALUE]]_tfg_result_1", tfg.regenerate_output_shapes})
// CHECK-NEXT:      -> (tensor<[[DTYPE:.*]]> {tfg.name = "[[A:.*]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK-NEXT: {
// CHECK-NEXT:   %[[A]], %[[CTL:.*]] = A(%[[VALUE]]_tfg_result_0) name("[[A]]")
// CHECK-NEXT:   return(%[[A]])

// CHECK: tfg.func @if_else_function
// CHECK-SAME: (%[[VALUE]]_tfg_result_0: tensor<[[TYPE]]> {tfg.name = "[[VALUE]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK-NEXT:  %[[VALUE]]_tfg_result_1: tensor<[[TYPE_0]]> {tfg.name = "[[VALUE]]_tfg_result_1", tfg.regenerate_output_shapes})
// CHECK-NEXT:      -> (tensor<[[DTYPE]]> {tfg.name = "[[B:.*]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK-NEXT: {
// CHECK-NEXT:   %[[B]], %[[CTL:.*]] = B(%[[VALUE]]_tfg_result_1) name("[[B]]")
// CHECK-NEXT:   return(%[[B]])
