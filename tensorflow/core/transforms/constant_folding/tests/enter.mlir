// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 0, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    // CHECK: , %[[CTRL_C1:.*]] = Const {{.*}} name("c1")
    %Const, %ctl_0 = Const [%ctl] name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<f32>} : () -> (tensor<f32>)
    %Enter, %ctl_1 = Enter(%Placeholder) name("enter1") {T = f32, frame_name = "foo", is_constant = true, parallel_iterations = 10 : i64} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[CONST_0:.*]], %[[CTRL:.*]] = Const [%[[ENTER_CTRL:.*]]] name("c1/_enter")
    // CHECK: , %[[ENTER_CTRL]] = Enter{{.*}} name("enter2")
    %Enter_2, %ctl_3 = Enter(%Const) name("enter2") {T = f32, frame_name = "foo", is_constant = true, parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<*xf32>)
    %Enter_4, %ctl_5 = Enter(%Const) name("enter3") {T = f32, frame_name = "foo", is_constant = false, parallel_iterations = 10 : i64} : (tensor<f32>) -> (tensor<*xf32>)
    // CHECK: Identity{{.*}} name("id1")
    %Identity, %ctl_6 = Identity(%Enter) name("id1") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL]]] name("id2/eval_0/const_folded")
    %Identity_7, %ctl_8 = Identity(%Enter_2) name("id2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL]]] name("id3/eval_0/const_folded")
    %Identity_9, %ctl_10 = Identity(%Enter_2) name("id3") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Identity{{.*}} name("id4")
    %Identity_11, %ctl_12 = Identity(%Enter_4) name("id4") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  }
}
