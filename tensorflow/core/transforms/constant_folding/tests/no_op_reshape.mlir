// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    // CHECK:, %[[CTRL:.*]] = Const name("d1")
    %Const, %ctl = Const name("d1") {dtype = f32, value = dense<3.140000e+00> : tensor<17xf32>} : () -> (tensor<17xf32>)
    // CHECK: %[[VAR_0:.*]], %[[CTRL_0:.*]] = VariableV2 name("v1")
    %VariableV2, %ctl_0 = VariableV2 name("v1") {container = "", dtype = f32, shape = #tf_type.shape<17>, shared_name = ""} : () -> (tensor<17xf32>)
    // CHECK: , %[[CTRL_2:.*]] = Const {{.*}} name("c1")
    %Const_1, %ctl_2 = Const [%ctl_0] name("c1") {dtype = i32, value = dense<17> : tensor<1xi32>} : () -> (tensor<1xi32>)
    // CHECK: , %[[CTRL_4:.*]] = Const [%[[CTRL_2]]] name("i1")
    %Identity, %ctl_3 = Identity(%Const_1) name("i1") {T = i32} : (tensor<1xi32>) -> (tensor<1xi32>)
    // CHECK: Identity(%[[VAR_0]]) [%[[CTRL_4]], %[[CTRL]]] name("r1")
    %Reshape, %ctl_4 = Reshape(%VariableV2, %Identity) [%ctl] name("r1") {T = f32, Tshape = i32} : (tensor<17xf32>, tensor<1xi32>) -> (tensor<17xf32>)
    %Square, %ctl_5 = Square(%Reshape) name("s1") {T = f32} : (tensor<17xf32>) -> (tensor<17xf32>)
    // CHECK: %[[VAR_1:.*]], %[[CTRL_2:.*]] = VariableV2 name("v3")
    %VariableV2_6, %ctl_7 = VariableV2 name("v3") {container = "", dtype = f32, shape = #tf_type.shape<5x5x5>, shared_name = ""} : () -> (tensor<5x5x5xf32>)
    %Const_8, %ctl_9 = Const [%ctl_7] name("c3") {dtype = i32, value = dense<5> : tensor<3xi32>} : () -> (tensor<3xi32>)
    // CHECK: %[[CONST_1:.*]], %[[CTRL_3:.*]] = Const {{.*}} name("i3")
    %Identity_10, %ctl_11 = Identity(%Const_8) name("i3") {T = i32} : (tensor<3xi32>) -> (tensor<3xi32>)
    // CHECK: Identity(%[[VAR_1]]) [%[[CTRL_3]]] name("r3")
    %Reshape_12, %ctl_13 = Reshape(%VariableV2_6, %Identity_10) name("r3") {T = f32, Tshape = i32} : (tensor<5x5x5xf32>, tensor<3xi32>) -> (tensor<5x5x5xf32>)
    %Square_14, %ctl_15 = Square(%Reshape_12) name("s3") {T = f32} : (tensor<5x5x5xf32>) -> (tensor<5x5x5xf32>)
    // CHECK: %[[VAR_2:.*]], %[[CTRL_4:.*]] = VariableV2 name("v4")
    %VariableV2_16, %ctl_17 = VariableV2 name("v4") {container = "", dtype = f32, shape = #tf_type.shape<5x5x5>, shared_name = ""} : () -> (tensor<5x5x5xf32>)
    %Const_18, %ctl_19 = Const [%ctl_17] name("c4") {dtype = i32, value = dense<[5, -1, 5]> : tensor<3xi32>} : () -> (tensor<3xi32>)
    // CHECK: %[[CONST_2:.*]], %[[CTRL_5:.*]] = Const {{.*}} name("i4")
    %Identity_20, %ctl_21 = Identity(%Const_18) name("i4") {T = i32} : (tensor<3xi32>) -> (tensor<3xi32>)
    // CHECK: Identity(%[[VAR_2]]) [%[[CTRL_5]]] name("r4")
    %Reshape_22, %ctl_23 = Reshape(%VariableV2_16, %Identity_20) name("r4") {T = f32, Tshape = i32} : (tensor<5x5x5xf32>, tensor<3xi32>) -> (tensor<5x5x5xf32>)
    %Square_24, %ctl_25 = Square(%Reshape_22) name("s4") {T = f32} : (tensor<5x5x5xf32>) -> (tensor<5x5x5xf32>)
    %VariableV2_26, %ctl_27 = VariableV2 name("v2") {container = "", dtype = f32, shape = #tf_type.shape<17x1>, shared_name = ""} : () -> (tensor<17x1xf32>)
    %Const_28, %ctl_29 = Const [%ctl_27] name("c2") {dtype = i32, value = dense<17> : tensor<1xi32>} : () -> (tensor<1xi32>)
    // CHECK: Reshape{{.*}} name("r2")
    %Reshape_30, %ctl_31 = Reshape(%VariableV2_26, %Const_28) name("r2") {T = f32, Tshape = i32} : (tensor<17x1xf32>, tensor<1xi32>) -> (tensor<17xf32>)
    %Square_32, %ctl_33 = Square(%Reshape_30) name("s2") {T = f32} : (tensor<17xf32>) -> (tensor<17xf32>)
    return
  }
}

