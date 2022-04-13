// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module  {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<1xf32>} : () -> (tensor<1xf32>)
    %Const_0, %ctl_1 = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_2, %ctl_3 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("x")
    %Placeholder, %ctl_4 = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[CONST:.*]], {{.*}} = Const {{.*}} name("add_child") {{.*}} value = dense<3.000000e+00> : tensor<2xf32>
    %Add, %ctl_5 = Add(%Const_0, %Placeholder) name("add_child") {T = f32} : (tensor<2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    // CHECK: Add(%[[PLACEHOLDER]], %[[CONST]]) name("add_parent")
    %Add_6, %ctl_7 = Add(%Const, %Add) name("add_parent") {T = f32} : (tensor<1xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  }
}
