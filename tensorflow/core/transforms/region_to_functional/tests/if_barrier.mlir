// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that a region that only uses a control token implicit capture is
// correctly converted to functional form.

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[VALUE:.*]], %[[CTL:.*]] = Value
  %Value, %ctl = Value name("value") : () -> (tensor<i32>)
  // CHECK: %[[COND:.*]], %[[CTL_0:.*]] = Cond
  %Cond, %ctl_0 = Cond : () -> (tensor<i1>)
  // CHECK: %[[BARRIER:.*]], %[[CTL_1:.*]] = Barrier
  %Barrier, %ctl_1 = Barrier name("barrier") : () -> (tensor<i32>)
  // CHECK: If(%[[COND]], %[[VALUE]], %[[BARRIER]])
  // CHECK-SAME: {else_branch = #tf_type.func<@[[ELSE:.*]], {}>,
  // CHECK-SAME:  then_branch = #tf_type.func<@[[THEN:.*]], {}>}
  %If, %ctl_2 = IfRegion %Cond then {
    yield(%Value) [%ctl_1] : tensor<i32>
  } else {
    yield(%Value) [%ctl_1] : tensor<i32>
  } : (tensor<i1>) -> (tensor<i32>)
}

// CHECK: tfg.func @[[THEN]]
// CHECK-SAME: (%[[VALUE:.*]]: tensor<{{.*}}>
// CHECK-NEXT:  %[[BARRIER:.*]]: tensor<{{.*}}>
// CHECK: return(%[[VALUE]]) [%[[BARRIER]].ctl {tfg.name = "barrier_tfg_result_0_0"}]

// CHECK: tfg.func @[[ELSE]]
// CHECK-SAME: (%[[VALUE:.*]]: tensor<{{.*}}>
// CHECK-NEXT:  %[[BARRIER:.*]]: tensor<{{.*}}>
// CHECK: return(%[[VALUE]]) [%[[BARRIER]].ctl {tfg.name = "barrier_tfg_result_0_0"}]
