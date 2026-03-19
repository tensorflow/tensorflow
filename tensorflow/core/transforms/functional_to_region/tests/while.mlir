// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @cond_func
tfg.func @cond_func(%arg: tensor<*xi32> {tfg.name = "arg"},
                    %other: tensor<*xi32> {tfg.name = "other"})
    -> (tensor<*xi1> {tf._some_attr})
   attributes {tf._some_attr} {
  %A, %ctl = A(%arg) : (tensor<*xi32>) -> (tensor<*xi1>)
  return(%A) [%arg.ctl] : tensor<*xi1>
}

// CHECK: tfg.func @body_func
tfg.func @body_func(%arg: tensor<*xi32> {tfg.name = "arg"},
                    %another: tensor<*xi32> {tfg.name = "another"})
    -> (tensor<*xi32> {tf._some_attr},
        tensor<*xi32> {tf._other}) {
  %B, %ctl = B(%arg) : (tensor<*xi32>) -> (tensor<*xi32>)
  return(%B, %another) [%arg.ctl] : tensor<*xi32>, tensor<*xi32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INIT:.*]]:2, %{{.*}} = Init
  %Init:2, %ctl = Init : () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK-NEXT: %[[OUTS:.*]]:2, %{{.*}} = WhileRegion(%[[INIT]]#0, %[[INIT]]#1)  {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<{{.*}}>, %[[ARG1:.*]]: tensor<{{.*}}>, %[[CTL0:.*]]: !tf_type.control, %[[CTL1:.*]]: !tf_type.control):
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_2:.*]] = A(%[[ARG0]])
  // CHECK-NEXT:   condition %[[A]] : tensor<{{.*}}xi1> (%[[ARG0]], %[[ARG1]]) [%[[CTL0]]]
  // CHECK-NEXT: } do  {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<{{.*}}>, %[[ARG1:.*]]: tensor<{{.*}}>, %[[CTL0:.*]]: !tf_type.control, %[[CTL1:.*]]: !tf_type.control):
  // CHECK-NEXT:   %[[B:.*]], %[[CTL_2:.*]] = B(%[[ARG0]])
  // CHECK-NEXT:   yield(%[[B]], %[[ARG1]]) [%[[CTL0]]]
  // CHECK-NEXT: } {_some_attr, body_attrs = {}, body_region_attrs = #tfg.region_attrs<{sym_name = "body_func"} [{tfg.name = "arg"}, {tfg.name = "another"}] [{tf._some_attr}, {tf._other}]>,
  // CHECK-SAME:    cond_attrs = {}, cond_region_attrs = #tfg.region_attrs<{sym_name = "cond_func", tf._some_attr} [{tfg.name = "arg"}, {tfg.name = "other"}] [{tf._some_attr}]>,
  // CHECK-SAME:    parallel_iterations = 10 : i64}
  %While:2, %ctl_0 = While(%Init#0, %Init#1) {
    T = [i32, i32], _some_attr,
    body = #tf_type.func<@body_func, {}>,
    cond = #tf_type.func<@cond_func, {}>,
    output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
    parallel_iterations = 10 : i64
  } : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK-NEXT: Consume(%[[OUTS]]#0)
  %ctl_1 = Consume(%While#0) : (tensor<*xi32>) -> ()
}
