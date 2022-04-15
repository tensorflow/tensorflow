// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: , %[[CTRL:.*]] = {{.*}} name("v1")
    %VariableV2, %ctl = VariableV2 name("v1") {container = "", dtype = f32, shape = #tf_type.shape<3>, shared_name = ""} : () -> (tensor<3x!tf_type.f32ref>)
    // CHECK: , %[[CTRL1:.*]] = {{.*}} name("v2")
    %VariableV2_0, %ctl_1 = VariableV2 name("v2") {container = "", dtype = f32, shape = #tf_type.shape<5x7>, shared_name = ""} : () -> (tensor<5x7x!tf_type.f32ref>)
    // CHECK: , %[[CTRL3:.*]] = {{.*}} name("v3")
    %VariableV2_2, %ctl_3 = VariableV2 name("v3") {container = "", dtype = f32, shape = #tf_type.shape<11x13>, shared_name = ""} : () -> (tensor<11x13x!tf_type.f32ref>)
    // CHECK: Const [%[[CTRL]]] name("rank")
    %Rank, %ctl_4 = Rank(%VariableV2) name("rank") {T = f32} : (tensor<3x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[SHAPE_CTRL:.*]] = Const [%[[CTRL1]]] name("shape")
    %Shape, %ctl_5 = Shape(%VariableV2_0) name("shape") {T = f32, out_type = i32} : (tensor<5x7x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[SIZE_CTRL:.*]] = Const [%[[CTRL3]]] name("size")
    %Size, %ctl_6 = Size(%VariableV2_2) name("size") {T = f32, out_type = i32} : (tensor<11x13x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[P1_CTRL:.*]] = Const{{.*}} name("p1")
    %Mul, %ctl_7 = Mul(%Size, %Rank) name("p1") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[SHAPE_CTRL]], %[[P1_CTRL]]] name("p2")
    %Mul_8, %ctl_9 = Mul(%Mul, %Shape) name("p2") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  }
}
