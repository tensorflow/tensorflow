// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1143, min_consumer = 0> {
    // CHECK: %[[VAR:.*]], {{.*}} VariableV2 name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<3x5x2>, shared_name = ""} : () -> (tensor<3x5x2xf32>)
    // CHECK:, %[[CTRL0:.*]] = Const name("begin")
    %Const, %ctl_0 = Const name("begin") {dtype = i32, value = dense<0> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK:, %[[CTRL2:.*]] = Const name("end")
    %Const_1, %ctl_2 = Const name("end") {dtype = i32, value = dense<[3, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK:, %[[CTRL4:.*]] = Const name("strides")
    %Const_3, %ctl_4 = Const name("strides") {dtype = i32, value = dense<1> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: %[[VAR2:.*]], {{.*}} VariableV2 name("in2")
    %VariableV2_5, %ctl_6 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<4x6x2>, shared_name = ""} : () -> (tensor<4x6x2xf32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL0]], %[[CTRL2]], %[[CTRL4]]] name("s1") {{.*}} -> (tensor<3x5x2xf32>)
    %StridedSlice, %ctl_7 = StridedSlice(%VariableV2, %Const, %Const_1, %Const_3) name("s1") {Index = i32, T = f32, begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<3x5x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<3x5x2xf32>)
    // CHECK: StridedSlice(%[[VAR2]], {{.*}} name("s2")
    %StridedSlice_8, %ctl_9 = StridedSlice(%VariableV2_5, %Const, %Const_1, %Const_3) name("s2") {Index = i32, T = f32, begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4x6x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<3x5x2xf32>)
    %Add, %ctl_10 = Add(%StridedSlice, %StridedSlice_8) name("out") {T = f32} : (tensor<3x5x2xf32>, tensor<3x5x2xf32>) -> (tensor<3x5x2xf32>)
  }
}

