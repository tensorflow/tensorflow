// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1142, min_consumer = 0> {
    // CHECK: , %[[CTRL_0:.*]] = VariableV2 name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<1x2x4x1>, shared_name = ""} : () -> (tensor<1x2x4x1xf32>)
    %Const, %ctl_0 = Const name("p1") {dtype = i32, value = dense<[3, 2, 1, 0]> : tensor<4xi32>} : () -> (tensor<4xi32>)
    // CHECK: %[[VAR:.*]], {{.*}} VariableV2 name("in2")
    %VariableV2_1, %ctl_2 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<1x4x2x1>, shared_name = ""} : () -> (tensor<1x4x2x1xf32>)
    // CHECK: , %[[CTRL_1:.*]] = Const name("p2")
    %Const_3, %ctl_4 = Const name("p2") {dtype = i32, value = dense<[3, 1, 2, 0]> : tensor<4xi32>} : () -> (tensor<4xi32>)
    %Transpose, %ctl_5 = Transpose(%VariableV2, %Const) name("t1") {T = f32, Tperm = i32} : (tensor<1x2x4x1xf32>, tensor<4xi32>) -> (tensor<1x4x2x1xf32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL_1]], %[[CTRL_0]]] name("t2")
    %Transpose_6, %ctl_7 = Transpose(%VariableV2_1, %Const_3) [%ctl] name("t2") {T = f32, Tperm = i32} : (tensor<1x4x2x1xf32>, tensor<4xi32>) -> (tensor<1x4x2x1xf32>)
    %Add, %ctl_8 = Add(%Transpose, %Transpose_6) name("out1") {T = f32} : (tensor<1x4x2x1xf32>, tensor<1x4x2x1xf32>) -> (tensor<1x4x2x1xf32>)
  }
}
