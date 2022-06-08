// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    // CHECK: %[[VAR:.*]], {{.*}} = {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = i32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.int32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = i32, shape = #tf_type.shape<2x2>, shared_name = ""} : () -> (tensor<2x2x!tf_type.int32ref>)
    // CHECK: , %[[CTRL2:.*]] = Const name("paddings1")
    %Const, %ctl_2 = Const name("paddings1") {dtype = i32, value = dense<0> : tensor<2x2xi32>} : () -> (tensor<2x2xi32>)
    %Const_3, %ctl_4 = Const name("paddings2") {dtype = i32, value = dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>} : () -> (tensor<2x2xi32>)
    // CHECK: , %[[CTRL6:.*]] = Const name("c1")
    %Const_5, %ctl_6 = Const name("c1") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
    %Const_7, %ctl_8 = Const name("c2") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL2]], %[[CTRL6]]] name("p1")
    %PadV2, %ctl_9 = PadV2(%VariableV2, %Const, %Const_5) name("p1") {T = i32, Tpaddings = i32} : (tensor<4x6x!tf_type.int32ref>, tensor<2x2xi32>, tensor<i32>) -> (tensor<*xi32>)
    %PadV2_10, %ctl_11 = PadV2(%VariableV2_0, %Const_3, %Const_7) name("p2") {T = i32, Tpaddings = i32} : (tensor<2x2x!tf_type.int32ref>, tensor<2x2xi32>, tensor<i32>) -> (tensor<*xi32>)
    %Add, %ctl_12 = Add(%PadV2, %PadV2_10) name("out") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    return
  }
}
