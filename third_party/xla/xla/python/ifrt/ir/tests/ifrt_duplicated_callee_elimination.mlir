// RUN: ifrt-opt %s -ifrt-duplicated-callee-elimination | FileCheck %s

// CHECK-LABEL: @main
func.func @main(%arg0: !ifrt.array<tensor<2x2xi32>,
                                   #ifrt.sharding_param<1x1 to [0] on 1>, [0]>)
    attributes {ifrt.function} {
  // CHECK: %[[CTRL:.+]] = ifrt.Call @callee
  %ctrl_0 = ifrt.Call @callee() on devices [0,1] : () -> ()
  // CHECK: ifrt.Call @callee
  // CHECK-SAME: after %[[CTRL]]
  %ctrl_1 = ifrt.Call @callee_dup() after %ctrl_0 on devices [0,1] : () -> ()
  // CHECK-NOT: ifrt.Call @callee
  // CHECK: ifrt.Call @callee_different_body
  %ctrl_2 = ifrt.Call @callee_different_body() on devices [0,1] : () -> ()
  // CHECK-NOT: ifrt.Call @callee
  // CHECK: ifrt.Call @callee_different_attr
  %ctrl_3 = ifrt.Call @callee_different_attr() on devices [0,1] : () -> ()
  // CHECK-NOT: ifrt.Call @callee
  // CHECK: ifrt.Call @callee_different_signature
  %ctrl_4 = ifrt.Call @callee_different_signature(%arg0) on devices [0,1]
      : (!ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>) -> ()
  return
}

func.func private @callee() {
  return
}

func.func private @callee_dup() {
  return
}

func.func private @callee_different_body() {
  %0 = builtin.unrealized_conversion_cast to tensor<4x4xi32>
  return
}

func.func private @callee_different_attr() attributes {_ifrt.attr} {
  return
}

func.func private @callee_different_signature(%arg0: tensor<2x2xi32>) {
  return
}
