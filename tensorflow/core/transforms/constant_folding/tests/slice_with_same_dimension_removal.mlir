// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: %[[VAR:.*]], {{.*}} = VariableV2 name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<3x5>, shared_name = ""} : () -> (tensor<3x5x!tf_type.f32ref>)
    // CHECK:, %[[CTRL:.*]] = Const name("begin")
    %Const, %ctl_0 = Const name("begin") {dtype = i32, value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK:, %[[CTRL2:.*]] = Const name("size")
    %Const_1, %ctl_2 = Const name("size") {dtype = i32, value = dense<[3, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %VariableV2_3, %ctl_4 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.f32ref>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL]], %[[CTRL2]]] name("s1")
    %Slice, %ctl_5 = Slice(%VariableV2, %Const, %Const_1) name("s1") {Index = i32, T = f32} : (tensor<3x5x!tf_type.f32ref>, tensor<2xi32>, tensor<2xi32>) -> (tensor<*xf32>)
    %Slice_6, %ctl_7 = Slice(%VariableV2_3, %Const, %Const_1) name("s2") {Index = i32, T = f32} : (tensor<4x6x!tf_type.f32ref>, tensor<2xi32>, tensor<2xi32>) -> (tensor<*xf32>)
    %Add, %ctl_8 = Add(%Slice, %Slice_6) name("out") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  }
}
