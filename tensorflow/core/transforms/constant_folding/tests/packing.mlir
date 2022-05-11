// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: , %[[CTRL:.*]] = Const name("c")
    %Const, %ctl = Const name("c") {dtype = f32, value = dense<3.140000e+00> : tensor<1000xf32>} : () -> (tensor<1000xf32>)
    // CHECK: Const [%[[CTRL]]] name("i1")
    %Identity, %ctl_0 = Identity(%Const) name("i1") {T = f32} : (tensor<1000xf32>) -> (tensor<*xf32>)
    // CHECK: Const [%[[CTRL]]] name("i2")
    %Identity_1, %ctl_2 = Identity(%Const) name("i2") {T = f32} : (tensor<1000xf32>) -> (tensor<*xf32>)
  }
}
