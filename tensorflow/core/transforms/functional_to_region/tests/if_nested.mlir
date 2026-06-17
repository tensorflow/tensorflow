// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @then_function_0
tfg.func @then_function_0(%arg0: tensor<*xi32>) -> (tensor<*xf32>) attributes {} {
  %A, %ctl = A(%arg0) : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%A) : tensor<*xf32>
}

// CHECK: tfg.func @else_function_0
tfg.func @else_function_0(%arg0: tensor<*xi32>) -> (tensor<*xf32>) attributes {} {
  %B, %ctl = B(%arg0) : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%B) : tensor<*xf32>
}

// CHECK: tfg.func @then_function_1
tfg.func @then_function_1(%arg0: tensor<*xi1>, %arg1: tensor<*xi32>)
     -> (tensor<*xf32>) attributes {} {
  %If, %ctl = If(%arg0, %arg1) {
    Tcond = i1, Tin = [i32], Tout = [f32], output_shapes = [#tf_type.shape<>],
    then_branch = #tf_type.func<@then_function_0, {}>,
    else_branch = #tf_type.func<@else_function_0, {}>}
  : (tensor<*xi1>, tensor<*xi32>) -> (tensor<*xf32>)
  return(%If) : tensor<*xf32>
}

// CHECK: tfg.func @else_function_1
tfg.func @else_function_1(%arg0: tensor<*xi1>, %arg1: tensor<*xi32>)
     -> (tensor<*xf32>) attributes {} {
  %C, %ctl = C(%arg1) : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%C) : tensor<*xf32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[COND:.*]]:2, %[[CTL:.*]] = Cond
  %Cond:2, %ctl = Cond : () -> (tensor<*xi1>, tensor<*xi1>)
  // CHECK: %[[ARG:.*]], %[[CTL_0:.*]] = Arg
  %Arg, %ctl_0 = Arg : () -> (tensor<*xi32>)
  // CHECK:      %[[IF:.*]], %[[CTLS:.*]] = IfRegion %[[COND]]#0 then {
  // CHECK-NEXT:   %[[IF_0:.*]], %[[CTLS_0:.*]] = IfRegion %[[COND]]#1 then {
  // CHECK-NEXT:     %[[A:.*]], %[[CTL_1:.*]] = A(%[[ARG]])
  // CHECK-NEXT:     yield(%[[A]])
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[B:.*]], %[[CTL_1:.*]] = B(%[[ARG]])
  // CHECK-NEXT:     yield(%[[B]])
  // CHECK-NEXT:   }
  // CHECK-NEXT:   yield(%[[IF_0]])
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[C:.*]], %[[CTL_1:.*]] = C(%[[ARG]])
  // CHECK-NEXT:   yield(%[[C]])
  // CHECK-NEXT: }
  %If, %ctl_1 = If(%Cond#0, %Cond#1, %Arg) {
    Tcond = i1, Tin = [i1, i32], Tout = [f32], output_shapes = [#tf_type.shape<>],
    then_branch = #tf_type.func<@then_function_1, {}>,
    else_branch = #tf_type.func<@else_function_1, {}>}
  : (tensor<*xi1>, tensor<*xi1>, tensor<*xi32>) -> (tensor<*xf32>)
}
