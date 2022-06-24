// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    // CHECK: , %[[CTRL:.*]] = VariableV2 name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<1x2x4x1>, shared_name = ""} : () -> (tensor<1x2x4x1x!tf_type.f32ref>)
    %Const, %ctl_0 = Const name("a1") {dtype = i32, value = dense<[3, 2, 1, 0]> : tensor<4xi32>} : () -> (tensor<4xi32>)
    // CHECK: %[[VAR:.*]], {{.*}} = VariableV2 name("in2")
    %VariableV2_1, %ctl_2 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<1x2x4x1>, shared_name = ""} : () -> (tensor<1x2x4x1x!tf_type.f32ref>)
    // CHECK: , %[[CTRL4:.*]] = Const name("a2")
    %Const_3, %ctl_4 = Const name("a2") {dtype = i32, value = dense<[0, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %ReverseV2, %ctl_5 = ReverseV2(%VariableV2, %Const) name("r1") {T = f32, Tidx = i32} : (tensor<1x2x4x1x!tf_type.f32ref>, tensor<4xi32>) -> (tensor<*xf32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL4]], %[[CTRL]]] name("r2")
    %ReverseV2_6, %ctl_7 = ReverseV2(%VariableV2_1, %Const_3) [%ctl] name("r2") {T = f32, Tidx = i32} : (tensor<1x2x4x1x!tf_type.f32ref>, tensor<2xi32>) -> (tensor<*xf32>)
    %Add, %ctl_8 = Add(%ReverseV2, %ReverseV2_6) name("out1") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
