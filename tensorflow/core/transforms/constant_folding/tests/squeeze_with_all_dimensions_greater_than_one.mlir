// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK: %[[VAR:.*]], {{.*}} name("in1")
    %VariableV2, %ctl = VariableV2 name("in1") {container = "", dtype = i32, shape = #tf_type.shape<2x3>, shared_name = ""} : () -> (tensor<2x3x!tf_type.int32ref>)
    %VariableV2_0, %ctl_1 = VariableV2 name("in2") {container = "", dtype = i32, shape = #tf_type.shape<1x2x3x1>, shared_name = ""} : () -> (tensor<1x2x3x1x!tf_type.int32ref>)
    // CHECK: Identity(%[[VAR]]) name("s1/Identity")
    %Squeeze, %ctl_2 = Squeeze(%VariableV2) name("s1") {T = i32, squeeze_dims = []} : (tensor<2x3x!tf_type.int32ref>) -> (tensor<*xi32>)
    %Squeeze_3, %ctl_4 = Squeeze(%VariableV2_0) name("s2") {T = i32, squeeze_dims = []} : (tensor<1x2x3x1x!tf_type.int32ref>) -> (tensor<*xi32>)
    %Add, %ctl_5 = Add(%Squeeze, %Squeeze_3) name("out") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  }
}
