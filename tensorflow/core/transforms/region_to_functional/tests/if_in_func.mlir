// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that a region op in a function is correctly converted back to
// functional form.

// CHECK:      %[[ARG0:.*]]: tensor<*xi1>
// CHECK-NEXT: %[[ARG1:.*]]: tensor<*xi32>
// CHECK: If(%[[ARG0]], %[[ARG1]], %[[ARG0]])
// CHECK: else_branch = #tf_type.func<@[[ELSE:.*]], {}>,
// CHECK: then_branch = #tf_type.func<@[[THEN:.*]], {}>}
tfg.func @body(%arg0: tensor<*xi1> {tfg.name = "arg0"},
               %arg1: tensor<*xi32> {tfg.name = "arg1"})
     -> (tensor<*xf32>)
 {
  %0:2 = IfRegion %arg0 then  {
    %A, %ctl = A(%arg1) [%arg1.ctl] : (tensor<*xi32>) -> (tensor<*xf32>)
    yield(%A) [%arg0.ctl] : tensor<*xf32>
  } else  {
    %B, %ctl = B(%arg1) [%arg1.ctl] : (tensor<*xi32>) -> (tensor<*xf32>)
    yield(%B) [%arg0.ctl] : tensor<*xf32>
  } : (tensor<*xi1>) -> (tensor<*xf32>)
  return(%0#0) : tensor<*xf32>
}

// CHECK: tfg.func @[[THEN]]
// CHECK:      %[[ARG1]]: tensor<*xi32>
// CHECK-NEXT: %[[ARG0]]: tensor<*xi1>
// CHECK: %[[A:.*]], %[[CTL:.*]] = A(%[[ARG1]]) [%[[ARG1]].ctl]
// CHECK: return(%[[A]]) [%[[ARG0]].ctl {tfg.name = "[[ARG0]]_0"}]

// CHECK: tfg.func @[[ELSE]]
// CHECK:      %[[ARG1]]: tensor<*xi32>
// CHECK-NEXT: %[[ARG0]]: tensor<*xi1>
// CHECK: %[[B:.*]], %[[CTL:.*]] = B(%[[ARG1]]) [%[[ARG1]].ctl]
// CHECK: return(%[[B]]) [%[[ARG0]].ctl {tfg.name = "[[ARG0]]_0"}]
