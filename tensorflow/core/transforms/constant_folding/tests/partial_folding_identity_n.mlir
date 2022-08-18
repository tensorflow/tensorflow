// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module  {
  tfg.func @test() {
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<>} : () -> (tensor<f32>)
    // CHECK: %[[CONST:.*]], %[[CTRL1:.*]] = Const name("c1")
    %Const, %ctl_0 = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: , %[[CTRL2:.*]] = Const name("c2")
    %Const_1, %ctl_2 = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[IDENTITY:.*]]:3, %[[CTRL:.*]] = {{.*}} name("id_n")
    %IdentityN:3, %ctl_3 = IdentityN(%Const, %Placeholder, %Const_1) name("id_n") {T = [f32, f32, f32]} : (tensor<2x2xf32>, tensor<f32>, tensor<2x2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    %Identity, %ctl_4 = Identity(%IdentityN#0) name("id0") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Identity_5, %ctl_6 = Identity(%IdentityN#1) name("id1") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Add(%[[CONST]], %[[IDENTITY]]#1) {{.*}} name("add0")
    %Add, %ctl_7 = Add(%IdentityN#0, %IdentityN#1) name("add0") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL1]], %[[CTRL2]], %[[CTRL]]] name("add1")
    %Add_8, %ctl_9 = Add(%IdentityN#0, %IdentityN#2) name("add1") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
