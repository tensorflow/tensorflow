// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @then_function
tfg.func @then_function(%arg0: tensor<*xi32>) -> (tensor<*xf32>) attributes {} {
  %A, %ctl = A(%arg0) [%arg0.ctl] : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%A) : tensor<*xf32>
}

// CHECK: tfg.func @else_function
tfg.func @else_function(%arg0: tensor<*xi32>) -> (tensor<*xf32>) attributes {} {
  %B, %ctl = B(%arg0) [%arg0.ctl] : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%B) : tensor<*xf32>
}

// CHECK-LABEL: @body
// CHECK-SAME: %[[ARG0:.*]]: tensor<{{.*}}>
// CHECK-NEXT: %[[ARG1:.*]]: tensor<{{.*}}>
tfg.func @body(%arg0: tensor<*xi1> {tfg.name = "arg0"},
               %arg1: tensor<*xi32> {tfg.name = "arg1"}) -> (tensor<*xf32>) attributes {} {
  // CHECK:      %[[IF:.*]], %[[CTL:.*]] = IfRegion %[[ARG0]] then {
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_0:.*]] = A(%[[ARG1]]) [%[[ARG1]].ctl]
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[B:.*]], %[[CTL_0:.*]] = B(%[[ARG1]]) [%[[ARG1]].ctl]
  // CHECK-NEXT:   yield(%[[B]])
  // CHECK-NEXT: }
  %If, %ctl = If(%arg0, %arg1) {Tcond = i1, Tin = [i32], Tout = [f32], output_shapes = [#tf_type.shape<>],
                                then_branch = #tf_type.func<@then_function, {}>,
                                else_branch = #tf_type.func<@else_function, {}>}
                               : (tensor<*xi1>, tensor<*xi32>) -> (tensor<*xf32>)
  // CHECK: return(%[[IF]])
  return(%If) : tensor<*xf32>
}
