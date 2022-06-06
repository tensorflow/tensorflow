// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
    // CHECK: (%[[ARG:.*]]: tensor<2x2xf32> {tfg.name = "name"}
  tfg.func @mlir_lifted_graph(%arg: tensor<2x2xf32> {tfg.name = "name"}) {
    %Const, %ctl = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %Const_0, %ctl_1 = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %Const_2, %ctl_3 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    %Const_4, %ctl_5 = Const name("c4") {dtype = f32, value = dense<4.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[P:.*]], {{.*}} = Placeholder name("ph")
    %Placeholder, %ctl_6 = Placeholder name("ph") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[AXIS:.*]], {{.*}} = Const name("axis")
    %Const_7, %ctl_8 = Const name("axis") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %ConcatV2, %ctl_9 = ConcatV2(%Const, %Const_0, %Placeholder, %Const_7) name("concat1") {N = 3 : i64, T = f32, Tidx = i32} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> (tensor<6x2xf32>)
    // CHECK: %[[CONST:.*]], {{.*}} = Const{{.*}} name("concat2/_partial_split_0/eval_0/const_folded")
    // CHECK: ConcatV2(%[[CONST]], %[[P]], %[[AXIS]]) name("concat2")
    %ConcatV2_10, %ctl_11 = ConcatV2(%Const_2, %Const_4, %ConcatV2, %Const_7) name("concat2") {N = 3 : i64, T = f32, Tidx = i32} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<6x2xf32>, tensor<i32>) -> (tensor<10x2xf32>)
    // CHECK: ConcatV2(%[[P]], %[[ARG]], %[[AXIS]]) name("concat3")
    %ConcatV2_11, %ctl_12 = ConcatV2(%Placeholder, %arg, %Const_7) name("concat3") {N = 3 : i64, T = f32, Tidx = i32} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<i32>) -> (tensor<10x2xf32>)
    return
  }
}

