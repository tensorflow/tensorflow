// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module  {
  tfg.func @test() {
    %Const, %ctl = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_0, %ctl_1 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("x")
    %Placeholder, %ctl_2 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[CONST:.*]], {{.*}} = Const {{.*}} name("child")
    %Add, %ctl_3 = Add(%Const, %Placeholder) name("child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Add(%[[PLACEHOLDER]], %[[CONST]]) name("parent")
    %Add_4, %ctl_5 = Add(%Const_0, %Add) name("parent") {T = f32} : (tensor<2xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    return
  }
}
