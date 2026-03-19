// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    // CHECK: %[[PLACEHOLDER_0:.*]], %[[CTRL_0:.*]] = Placeholder name("input")
    %Placeholder, %ctl = Placeholder name("input") {dtype = f32, shape = #tf_type.shape<?x?>} : () -> (tensor<?x?xf32>)
    // CHECK: %[[PLACEHOLDER_1:.*]], %[[CTRL_1:.*]] = Placeholder name("indices")
    %Placeholder_0, %ctl_1 = Placeholder name("indices") {dtype = i32, shape = #tf_type.shape<?>} : () -> (tensor<?xi32>)
    // CHECK: %[[CONST:.*]], %[[CONST_CTRL:.*]] = Const [%[[CTRL_1]]] name("sum/indices/const_folded") {dtype = i32, value = dense<[0, 1]> : tensor<2xi32>}
    // CHECK: Sum(%[[PLACEHOLDER_0]], %[[CONST]]) name("sum")
    %Sum, %ctl_2 = Sum(%Placeholder, %Placeholder_0) name("sum") {T = f32, Tidx = i32, keep_dims = false} : (tensor<?x?xf32>, tensor<?xi32>) -> (tensor<*xf32>)
    %Const, %ctl_3 = Const name("size") {dtype = i32, value = dense<1> : tensor<1xi32>} : () -> (tensor<1xi32>)
    // CHECK: Reshape{{.*}} name("reshape")
    %Reshape, %ctl_4 = Reshape(%Sum, %Const) name("reshape") {T = f32, Tshape = i32} : (tensor<*xf32>, tensor<1xi32>) -> (tensor<1xf32>)
    return
  }
}

