// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %Placeholder, %ctl = Placeholder name("x") {dtype = i1, shape = #tf_type.shape<>} : () -> (tensor<i1>)
    %Switch:2, %ctl_0 = Switch(%Placeholder, %Placeholder) name("switch") {T = i1} : (tensor<i1>, tensor<i1>) -> (tensor<*xi1>, tensor<*xi1>)
    // CHECK-DAG: , %[[CTRL1:.*]] = Identity(%[[SWITCH:.*]]#1) name("switch/ControlDependencyCtrl_1")
    // CHECK-DAG: , %[[CTRL0:.*]] = Identity(%[[SWITCH]]#0) name("switch/ControlDependencyCtrl_0")
    // CHECK-DAG: , %[[CTRL_SWITCH_1:.*]] = Const [%[[CTRL1]]] name("switch/_const_true")
    // CHECK-DAG: , %[[CTRL_SWITCH_0:.*]] = Const [%[[CTRL0]]] name("switch/_const_false")
    // CHECK-DAG: Const [%[[CTRL_SWITCH_0]]] name("id_false")
    %LogicalNot, %ctl_1 = LogicalNot(%Switch#0) name("id_false") : (tensor<*xi1>) -> (tensor<*xi1>)
    // CHECK-DAG: Const [%[[CTRL_SWITCH_1]]] name("id_true")
    %LogicalNot_2, %ctl_3 = LogicalNot(%Switch#1) name("id_true") : (tensor<*xi1>) -> (tensor<*xi1>)
    // CHECK-DAG: Const [%[[CTRL_SWITCH_0]]] name("id_false_1")
    %LogicalNot_3, %ctl_4 = LogicalNot(%Switch#0) name("id_false_1") : (tensor<*xi1>) -> (tensor<*xi1>)
    // CHECK-DAG: Const [%[[CTRL_SWITCH_1]]] name("id_true_1")
    %LogicalNot_4, %ctl_5 = LogicalNot(%Switch#1) name("id_true_1") : (tensor<*xi1>) -> (tensor<*xi1>)
    return
  }
}
