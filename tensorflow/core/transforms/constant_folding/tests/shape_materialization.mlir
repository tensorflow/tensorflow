// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test(%ArgWithShape: tensor<2x3xi32> {tfg.name = "ArgWithShape"}, %ArgWithoutShape: tensor<*xi32> {tfg.name = "ArgWithoutShape"}) {
    // CHECK: , %[[CTRL:.*]] = {{.*}} name("v1")
    %VariableV2, %ctl = VariableV2 name("v1") {container = "", dtype = f32, shape = #tf_type.shape<3>, shared_name = ""} : () -> (tensor<3x!tf_type.f32ref>)
    // CHECK: , %[[CTRL1:.*]] = {{.*}} name("v2")
    %VariableV2_0, %ctl_1 = VariableV2 name("v2") {container = "", dtype = f32, shape = #tf_type.shape<5x7>, shared_name = ""} : () -> (tensor<5x7x!tf_type.f32ref>)
    // CHECK: , %[[CTRL3:.*]] = {{.*}} name("v3")
    %VariableV2_2, %ctl_3 = VariableV2 name("v3") {container = "", dtype = f32, shape = #tf_type.shape<11x13>, shared_name = ""} : () -> (tensor<11x13x!tf_type.f32ref>)
    // CHECK: Const [%[[CTRL]]] name("rank/const_folded")
    %Rank, %ctl_4 = Rank(%VariableV2) name("rank") {T = f32} : (tensor<3x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[SHAPE_CTRL:.*]] = Const [%[[CTRL1]]] name("shape/const_folded")
    %Shape, %ctl_5 = Shape(%VariableV2_0) name("shape") {T = f32, out_type = i32} : (tensor<5x7x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[SIZE_CTRL:.*]] = Const [%[[CTRL3]]] name("size/const_folded")
    %Size, %ctl_6 = Size(%VariableV2_2) name("size") {T = f32, out_type = i32} : (tensor<11x13x!tf_type.f32ref>) -> (tensor<*xi32>)
    // CHECK: , %[[P1_CTRL:.*]] = Const{{.*}} name("p1/eval_0/const_folded")
    %Mul, %ctl_7 = Mul(%Size, %Rank) name("p1") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[SHAPE_CTRL]], %[[P1_CTRL]]] name("p2/eval_0/const_folded")
    %Mul_8, %ctl_9 = Mul(%Mul, %Shape) name("p2") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%ArgWithShape.ctl] name("arg_shape/const_folded") {{.*}} -> (tensor<2xi32>)
    %ArgShape, %ctl_10 = Shape(%ArgWithShape) name("arg_shape") {T = i32, out_type = i32} : (tensor<2x3xi32>) -> (tensor<*xi32>)
    // CHECK: Shape{{.*}} name("arg_without_shape")
    %ArgShape_1, %ctl_11 = Shape(%ArgWithoutShape) name("arg_without_shape") {T = i32, out_type = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%ArgWithShape.ctl] name("arg_size/const_folded") {{.*}} -> (tensor<i32>)
    %ArgSize, %ctl_12 = Size(%ArgWithShape) name("arg_size") {T = i32, out_type = i32} : (tensor<2x3xi32>) -> (tensor<*xi32>)
    // CHECK: Size{{.*}} name("arg_without_size")
    %ArgSize_1, %ctl_13 = Size(%ArgWithoutShape) name("arg_without_size") {T = i32, out_type = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%ArgWithShape.ctl] name("arg_rank/const_folded") {{.*}} -> (tensor<i32>)
    %ArgRank, %ctl_14 = Rank(%ArgWithShape) name("arg_rank") {T=f32} : (tensor<2x3xi32>) -> (tensor<*xi32>)
    // CHECK: Rank{{.*}} name("arg_without_rank")
    %ArgRank_1, %ctl_15 = Rank(%ArgWithoutShape) name("arg_without_rank") {T=f32} : (tensor<*xi32>) -> (tensor<*xi32>)
    tfg.return
  }
}
