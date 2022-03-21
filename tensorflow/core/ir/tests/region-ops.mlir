// RUN: tfg-opt-no-passes -verify-diagnostics %s | tfg-opt-no-passes | FileCheck %s

//===----------------------------------------------------------------------===//
// IfRegion
//===----------------------------------------------------------------------===//

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[$OP:.*]]:2, %[[$CTL:.*]] = Op
  %Op:2, %ctl = Op : () -> (tensor<*xf32>, tensor<*xf32>)
  // CHECK: %[[$OP_0:.*]]:2, %[[$CTL_0:.*]] = Op
  %Op0:2, %ctl_0 = Op : () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: %[[$COND:.*]], %[[$CTL_1:.*]] = Op
  %cond, %ctl_1 = Op : () -> (tensor<*xi1>)

  // CHECK:      IfRegion %[[$COND]] [%[[$CTL]], %[[$CTL_1]]] then {
  // CHECK-NEXT:   yield(%[[$OP]]#0, %[[$OP_0]]#0) [%[[$CTL_1]]] : tensor<{{.*}}>, tensor<{{.*}}>
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   yield(%[[$OP]]#1, %[[$OP_0]]#1) [%[[$CTL]], %[[$CTL_0]]] : tensor<{{.*}}>, tensor<{{.*}}>
  // CHECK-NEXT: } : (tensor<{{.*}}i1>) -> (tensor<{{.*}}>, tensor<{{.*}}>)
  %IfRegion:2, %ctl_2 = IfRegion %cond [%ctl, %ctl_1] then {
    yield(%Op#0, %Op0#0) [%ctl_1] : tensor<*xf32>, tensor<*xi32>
  } else {
    yield(%Op#1, %Op0#1) [%ctl, %ctl_0] : tensor<*xf32>, tensor<*xi32>
  } : (tensor<*xi1>) -> (tensor<*xf32>, tensor<*xi32>)
}

//===----------------------------------------------------------------------===//
// CaseRegion
//===----------------------------------------------------------------------===//

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INDEX:.*]], %[[CTL:.*]] = Index
  %Index, %ctl = Index : () -> (tensor<i32>)
  // CHECK-NEXT: %[[A:.*]], %[[CTL_0:.*]] = A
  %A, %ctl_0 = A : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[B:.*]], %[[CTL_1:.*]] = B
  %B, %ctl_1 = B : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[C:.*]], %[[CTL_2:.*]] = C
  %C, %ctl_2 = C : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[CASE:.*]], %[[CTL_3:.*]] = CaseRegion %[[INDEX]] [%[[CTL_1]]]  {
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: },  {
  // CHECK-NEXT:   yield(%[[B]])
  // CHECK-NEXT: },  {
  // CHECK-NEXT:   yield(%[[C]])
  // CHECK-NEXT: } {branch_attrs = [{}, {}, {}]} : (tensor<{{.*}}i32>) -> tensor<{{.*}}>
  %Case, %ctl_3 = CaseRegion %Index [%ctl_1] {
    yield(%A) : tensor<*xf32>
  }, {
    yield(%B) : tensor<*xf32>
  }, {
    yield(%C) : tensor<*xf32>
  } {branch_attrs = [{}, {}, {}]} : (tensor<i32>) -> (tensor<*xf32>)
}

//===----------------------------------------------------------------------===//
// WhileRegion
//===----------------------------------------------------------------------===//

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[$OP:.*]]:2, %[[$CTL:.*]] = Op
  %Op:2, %ctl = Op : () -> (tensor<*xf32>, tensor<*xi32>)

  // CHECK:      WhileRegion(%[[$OP]]#0, %[[$OP]]#1) [%[[$CTL]]] {
  // CHECK-NEXT: ^bb0(%[[$ARG0:.*]]: tensor<{{.*}}>, %[[$ARG1:.*]]: tensor<{{.*}}>, %[[$ARG2:.*]]: !tf_type.control, %[[$ARG3:.*]]):
  // CHECK-NEXT:   %[[$LESS:.*]], %[[$CTL_0:.*]] = Less(%[[$ARG0]], %[[$ARG1]]) [%[[$ARG2]]]
  // CHECK-NEXT:   condition %[[$LESS]] : tensor<{{.*}}xi1> (%[[$ARG0]], %[[$ARG1]]) : tensor<{{.*}}>, tensor<{{.*}}>
  // CHECK-NEXT: } do {
  // CHECK-NEXT: ^bb0(%[[$ARG0:.*]]: tensor<{{.*}}>, %[[$ARG1:.*]]: tensor<{{.*}}>, %[[$ARG2:.*]]: !tf_type.control, %[[$ARG3:.*]]: !tf_type.control):
  // CHECK-NEXT:   %[[$FWD:.*]]:2, %[[$CTL_0:.*]] = Fwd(%[[$ARG0]], %[[$ARG1]]) [%[[$ARG2]]]
  // CHECK-NEXT:   yield(%[[$FWD]]#0, %[[$FWD]]#1) [%[[$CTL_0]], %[[$ARG3]]] : tensor<{{.*}}>, tensor<{{.*}}>
  // CHECK-NEXT: } {parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xi32>) -> (tensor<f32>, tensor<i32>)
  %WhileRegion:2, %ctl_0 = WhileRegion(%Op#0, %Op#1) [%ctl] {
  ^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>,
       %arg2: !tf_type.control, %arg3: !tf_type.control):
    %Less, %ctl_0 = Less(%arg0, %arg1) [%arg2] : (tensor<*xf32>, tensor<*xi32>) -> (tensor<*xi1>)
    condition %Less : tensor<*xi1> (%arg0, %arg1) : tensor<*xf32>, tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>,
       %arg2: !tf_type.control, %arg3: !tf_type.control):
    %Fwd:2, %ctl_0 = Fwd(%arg0, %arg1) [%arg2] : (tensor<*xf32>, tensor<*xi32>) -> (tensor<*xf32>, tensor<*xi32>)
    yield(%Fwd#0, %Fwd#1) [%ctl_0, %arg3] : tensor<*xf32>, tensor<*xi32>
  } {parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xi32>) -> (tensor<f32>, tensor<i32>)
}

//===----------------------------------------------------------------------===//
// ForRegion
//===----------------------------------------------------------------------===//

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INDEX:.*]]:3, %[[CTL:.*]] = Index
  %Index:3, %ctl = Index : () -> (tensor<i32>, tensor<i32>, tensor<i32>)
  // CHECK-NEXT: %[[ARG:.*]], %[[CTL_0:.*]] = Arg
  %Arg, %ctl_0 = Arg : () -> (tensor<*xf32>)
  // CHECK-NEXT: %[[OUTS:.*]], %[[CTL_1:.*]] = ForRegion(%[[ARG]]) [%[[CTL]]]
  // CHECK-SAME: from %[[INDEX]]#0 to %[[INDEX]]#1 by %[[INDEX]]#2  {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<{{.*}}>, %[[ARG2:.*]]: !tf_type.control, %[[ARG3:.*]]: !tf_type.control):
  // CHECK-NEXT:   yield(%[[ARG1]]) [%[[ARG3]]] : tensor<{{.*}}>
  // CHECK-NEXT: } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<f32>)
  %For, %ctl_1 = ForRegion(%Arg) [%ctl] from %Index#0 to %Index#1 by %Index#2 {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg1) [%arg3] : tensor<*xf32>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<*xf32>) -> (tensor<f32>)
}
