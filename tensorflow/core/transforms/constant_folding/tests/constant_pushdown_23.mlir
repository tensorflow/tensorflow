// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const device("/job:localhost/replica:0/task:0/device:CPU:0") name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    %Const_0, %ctl_1 = Const device("/job:localhost/replica:0/task:0/device:CPU:0") name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2xf32>} : () -> (tensor<2xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} = Placeholder device({{.*}}) name("x")
    %Placeholder, %ctl_2 = Placeholder device("/job:localhost/replica:0/task:0/device:CPU:0") name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[CONST:.*]], %[[CTRL:.*]] = Const{{.*}} device([[DEVICE:.*]]) name("child")
    // CHECK-SAME: value = dense<1.500000e+00>
    %Div, %ctl_3 = Div(%Placeholder, %Const) device("/job:localhost/replica:0/task:0/device:CPU:0") name("child") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
    // CHECK: Mul(%[[PLACEHOLDER]], %[[CONST]]) device([[DEVICE]]) name("parent")
    %Mul, %ctl_4 = Mul(%Div, %Const_0) device("/job:localhost/replica:0/task:0/device:CPU:0") name("parent") {T = f32} : (tensor<2x2xf32>, tensor<2xf32>) -> (tensor<2x2xf32>)
  }
}
