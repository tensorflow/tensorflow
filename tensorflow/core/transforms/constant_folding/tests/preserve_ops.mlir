// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %Const, %ctl_0 = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: Const {{.*}} name("id
    %Identity, %ctl_1 = Identity(%Const) name("id") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    // CHECK: Identity{{.*}} name("id_1")
    %Identity_1, %ctl_2 = Identity(%Identity) name("id_1") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    // CHECK: Identity{{.*}} name("id_2")
    %Identity_2, %ctl_3 = Identity(%Const) name("id_2") {T = f32} : (tensor<2xf32>) -> (tensor<2xf32>)
    return [%ctl_2, %ctl_3]
  }
}
