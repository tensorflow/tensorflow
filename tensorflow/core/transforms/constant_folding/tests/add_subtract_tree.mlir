// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s
module  {
  tfg.func @test() {
    // CHECK: %[[CONST:.*]], {{%.*}} = Const
    %Const, %ctl = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<1xf32>} : () -> (tensor<1xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{%.*}} = Placeholder
    %Placeholder, %ctl_0 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[SUB:.*]], {{%.*}} = Sub(%[[CONST]], %[[PLACEHOLDER]])
    %Sub, %ctl_1 = Sub(%Placeholder, %Placeholder) name("sub_child") {T = f32} : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Add(%[[PLACEHOLDER]], %[[SUB]])
    %Add, %ctl_2 = Add(%Sub, %Const) name("add_parent") {T = f32} : (tensor<*xf32>, tensor<1xf32>) -> (tensor<*xf32>)
    return
  }
}
