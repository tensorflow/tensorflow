// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<*xf32>)
    %Const_0, %ctl_1 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<*xf32>)
    %Placeholder, %ctl_2 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<*xf32>)
    %Const_3, %ctl_4 = Const [%ctl_1, %ctl] name("child") {dtype = f32, value = dense<1.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: Add{{.*}} name("parent")
    %Add, %ctl_5 = Add(%Placeholder, %Const_3) name("parent") {T = f32} : (tensor<*xf32>, tensor<2xf32>) -> (tensor<*xf32>)
  }
}
