// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_0, %ctl_1 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} = Placeholder name("x")
    %Placeholder, %ctl_2 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[CONST:.*]], %[[CTRL:.*]] = Const{{.*}} name("child")
    %Sub, %ctl_3 = Sub(%Placeholder, %Const) name("child") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
    // CHECK: Sub(%[[PLACEHOLDER]], %[[CONST]]) name("parent")
    %Sub_4, %ctl_5 = Sub(%Sub, %Const_0) name("parent") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
  }
}
