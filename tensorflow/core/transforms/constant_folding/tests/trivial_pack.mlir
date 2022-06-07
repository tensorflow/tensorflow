// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1143, min_consumer = 0> {
    %Const, %ctl = Const name("Const/Const") {dtype = i32, value = dense<2> : tensor<2xi32>} : () -> (tensor<*xi32>)
    // CHECK: %[[RANDOM:.*]], %[[CTRL:.*]] = RandomStandardNormal{{.*}} name("x")
    %RandomStandardNormal, %ctl_0 = RandomStandardNormal(%Const) name("x") {T = i32, dtype = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<*xi32>) -> (tensor<*xf32>)
    // CHECK: , %[[CTRL_2:.*]] = Const name("y")
    %Const_1, %ctl_2 = Const name("y") {dtype = f32, value = dense<2.000000e+00> : tensor<f32>} : () -> (tensor<*xf32>)
    // CHECK: %[[CONST_3:.*]], {{.*}} = Const [%[[CTRL]]] name("stack/_const_axis")
    // CHECK: ExpandDims(%[[RANDOM]], %[[CONST_3]]) [%[[CTRL_2]]] name("stack") {T = f32, Tdim = i32}
    %Pack, %ctl_3 = Pack(%RandomStandardNormal) [%ctl_2] name("stack") {N = 1 : i64, T = f32, axis = 1 : i64} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[CONST_6:.*]], {{.*}} = Const [%[[CTRL]]] name("stack_no_axis/_const_axis")
    // CHECK: ExpandDims(%[[RANDOM]], %[[CONST_6]]) name("stack_no_axis") {T = f32, Tdim = i32}
    %Pack_4, %ctl_5 = Pack(%RandomStandardNormal) name("stack_no_axis") {N = 1 : i64, T = f32, axis = 0 : i64} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Pack{{.*}} name("pack_with_multiple_args")
    %Pack_5, %ctl_6 = Pack(%RandomStandardNormal, %RandomStandardNormal) name("pack_with_multiple_args") {N = 2 : i64, T = f32, axis = 0 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  }
}
