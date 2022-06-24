// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module  {
  tfg.func @test() {
    // CHECK: , %[[CTRL:.*]] = Const name("if")
    %Const, %ctl = Const name("if") {dtype = i1, value = dense<false> : tensor<2x2xi1>} : () -> (tensor<2x2xi1>)
    // CHECK: , %[[CTRL0:.*]] = Placeholder name("then")
    %Placeholder, %ctl_0 = Placeholder name("then") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[PLACEHOLDER1:.*]], {{.*}} = Placeholder name("else")
    %Placeholder_1, %ctl_2 = Placeholder name("else") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: Identity(%[[PLACEHOLDER1]]) [%[[CTRL]], %[[CTRL0]]] name("select")
    %SelectV2, %ctl_3 = SelectV2(%Const, %Placeholder, %Placeholder_1) name("select") {T = f32} : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    %Identity, %ctl_4 = Identity(%SelectV2) name("id") {T = f32} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    return
  }
}
