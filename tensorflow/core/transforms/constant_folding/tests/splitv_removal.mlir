// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: %[[VAR:.*]], {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = f32, shape = #tf_type.shape<2>, shared_name = ""} : () -> (tensor<2x!tf_type.f32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = f32, shape = #tf_type.shape<5>, shared_name = ""} : () -> (tensor<5x!tf_type.f32ref>)
    // CHECK: , %[[CTRL2:.*]] = Const name("split_dim")
    %Const, %ctl_2 = Const name("split_dim") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: , %[[CTRL4:.*]] = Const name("size_splits1")
    %Const_3, %ctl_4 = Const name("size_splits1") {dtype = i32, value = dense<2> : tensor<1xi32>} : () -> (tensor<1xi32>)
    %Const_5, %ctl_6 = Const name("size_splits2") {dtype = i32, value = dense<[2, 3]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    // CHECK: Identity(%[[VAR]]) [%[[CTRL4]], %[[CTRL2]]] name("s1/Identity")
    %SplitV, %ctl_7 = SplitV(%VariableV2, %Const_3, %Const) name("s1") {T = f32, Tlen = i32, num_split = 1 : i64} : (tensor<2x!tf_type.f32ref>, tensor<1xi32>, tensor<i32>) -> (tensor<*xf32>)
    %SplitV_8:2, %ctl_9 = SplitV(%VariableV2_0, %Const_5, %Const) name("s2") {T = f32, Tlen = i32, num_split = 2 : i64} : (tensor<5x!tf_type.f32ref>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
    %Add, %ctl_10 = Add(%SplitV, %SplitV_8#0) name("out") {T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  }
}
