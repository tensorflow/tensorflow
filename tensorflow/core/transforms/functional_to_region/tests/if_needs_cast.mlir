// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @then_func
tfg.func @then_func(%arg0: tensor<?xi32>) -> (tensor<?xf32>) {
  %Bitcast, %ctl = Bitcast(%arg0) : (tensor<?xi32>) -> (tensor<?xf32>)
  return(%Bitcast) : tensor<?xf32>
}

// CHECK: tfg.func @else_func
tfg.func @else_func(%arg0: tensor<*xi32>) -> (tensor<*xf32>) {
  %Bitcast, %ctl = Bitcast(%arg0) : (tensor<*xi32>) -> (tensor<*xf32>)
  return(%Bitcast) : tensor<*xf32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[ARG:.*]], %{{.*}} = Arg : () -> (tensor<[[TYPE:.*]]>)
  %Arg, %ctl = Arg : () -> (tensor<4xi32>)
  // CHECK-NEXT: %[[COND:.*]], %{{.*}} = Cond
  %Cond, %ctl_0 = Cond : () -> (tensor<*xi1>)
  // CHECK-NEXT: %[[IF:.*]], %{{.*}} = IfRegion %[[COND]] then {
  // CHECK-NEXT:   %[[BITCAST:.*]], %{{.*}} = Bitcast(%[[ARG]]) : (tensor<[[TYPE]]>) -> (tensor<[[RET_TY:.*]]>)
  // CHECK-NEXT:   yield(%[[BITCAST]]) : tensor<[[RET_TY]]>
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[BITCAST:.*]], %{{.*}} = Bitcast(%[[ARG]]) : (tensor<[[TYPE]]>) -> (tensor<[[RET_TY:.*]]>)
  // CHECK-NEXT:   yield(%[[BITCAST]]) : tensor<[[RET_TY]]>
  // CHECK-NEXT: else_attrs = {}
  // CHECK-SAME: then_attrs = {}
  // CHECK-SAME: : (tensor<*xi1>) -> tensor<[[RET_TY_0:.*]]>
  %If, %ctl_1 = If(%Cond, %Arg) {Tcond = i1, Tin = [i32], Tout = [f32],
                                 else_branch = #tf_type.func<@else_func, {}>,
                                 output_shapes = [#tf_type.shape<>],
                                 then_branch = #tf_type.func<@then_func, {}>}
                                : (tensor<*xi1>, tensor<4xi32>) -> (tensor<4xf32>)
  // CHECK: Consume(%[[IF]]) : tensor<[[RET_TY_0]]>
  %ctl_2 = Consume(%If) : tensor<4xf32>
}
