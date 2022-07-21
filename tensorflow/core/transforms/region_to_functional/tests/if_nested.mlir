// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that nested region ops are correctly converted back to functional form.
// The conversion is expected to happen "inside out".

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[COND:.*]]:2, %[[CTL:.*]] = Cond
  %Cond:2, %ctl = Cond : () -> (tensor<*xi1>, tensor<*xi1>)
  // CHECK: %[[ARG:.*]], %[[CTL_0:.*]] = Arg
  %Arg, %ctl_0 = Arg : () -> (tensor<*xi32>)
  // CHECK: If(%[[COND]]#0, %[[COND]]#1, %[[ARG]])
  // CHECK-SAME: else_branch = #tf_type.func<@[[ELSE:.*]], {}>,
  // CHECK-SAME: then_branch = #tf_type.func<@[[THEN:.*]], {}>}
  %0:2 = IfRegion %Cond#0 then  {
    %1:2 = IfRegion %Cond#1 then  {
      %A, %ctl_1 = A(%Arg) : (tensor<*xi32>) -> (tensor<*xf32>)
      yield(%A) : tensor<*xf32>
    } else  {
      %B, %ctl_1 = B(%Arg) : (tensor<*xi32>) -> (tensor<*xf32>)
      yield(%B) : tensor<*xf32>
    } : (tensor<*xi1>) -> (tensor<*xf32>)
    yield(%1#0) : tensor<*xf32>
  } else  {
    %C, %ctl_1 = C(%Arg) : (tensor<*xi32>) -> (tensor<*xf32>)
    yield(%C) : tensor<*xf32>
  } : (tensor<*xi1>) -> (tensor<*xf32>)
}

// Regions with no nested regions are outlined first.

// CHECK: tfg.func @[[THEN0:.*]](%[[ARG:.*]]: tensor<{{.*}}>
// CHECK:      %[[A:.*]], %[[CTL:.*]] = A(%[[ARG]])
// CHECK-NEXT: return(%[[A]])

// CHECK: tfg.func @[[ELSE0:.*]](%[[ARG:.*]]: tensor<{{.*}}>
// CHECK:      %[[B:.*]], %[[CTL:.*]] = B(%[[ARG]])
// CHECK-NEXT: return(%[[B]])

// CHECK: tfg.func @[[THEN]]
// CHECK-SAME: (%[[COND:.*]]: tensor<*xi1>
// CHECK-NEXT:  %[[ARG:.*]]: tensor<{{.*}}>
// CHECK:      %[[IF:.*]], %[[CTL:.*]] = If(%[[COND]], %[[ARG]])
// CHECK-SAME: else_branch = #tf_type.func<@[[ELSE0]], {}>,
// CHECK-SAME: then_branch = #tf_type.func<@[[THEN0]], {}>}
// CHECK-NEXT: return(%[[IF]])

// CHECK: tfg.func @[[ELSE]]
// CHECK-SAME: (%[[COND:.*]]: tensor<*xi1>
// CHECK-NEXT:  %[[ARG:.*]]: tensor<{{.*}}>
// CHECK:      %[[C:.*]], %[[CTL:.*]] = C(%[[ARG]])
// CHECK-NEXT: return(%[[C]])
