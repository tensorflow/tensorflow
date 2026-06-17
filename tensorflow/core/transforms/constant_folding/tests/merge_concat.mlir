// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    // CHECK: %[[V2:.*]], {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.f32ref>)
    // CHECK: %[[V2_0:.*]], {{.*}} name("in2")
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.f32ref>)
    // CHECK: %[[V2_2:.*]], {{.*}} name("in3")
    %VariableV2_2, %ctl_3 = VariableV2 name("in3") {container = "", dtype = f32, shape = #tf_type.shape<4x6>, shared_name = ""} : () -> (tensor<4x6x!tf_type.f32ref>)
    // CHECK: %[[CONST:.*]], {{.*}} name("axis")
    %Const, %ctl_4 = Const name("axis") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %ConcatV2, %ctl_5 = ConcatV2(%VariableV2, %VariableV2_0, %Const) name("c1") {N = 2 : i64, T = f32, Tidx = i32} : (tensor<4x6x!tf_type.f32ref>, tensor<4x6x!tf_type.f32ref>, tensor<i32>) -> (tensor<*xf32>)
    // CHECK: ConcatV2(%[[V2]], %[[V2_0]], %[[V2_2]], %[[CONST]]) name("c2")
    %ConcatV2_6, %ctl_7 = ConcatV2(%ConcatV2, %VariableV2_2, %Const) name("c2") {N = 2 : i64, T = f32, Tidx = i32} : (tensor<*xf32>, tensor<4x6x!tf_type.f32ref>, tensor<i32>) -> (tensor<*xf32>)
    return
  }
}
