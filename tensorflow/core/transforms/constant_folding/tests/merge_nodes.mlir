// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("Const/Const") {dtype = i32, value = dense<[3, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %RandomStandardNormal, %ctl_0 = RandomStandardNormal(%Const) name("x") {T = i32, dtype = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> (tensor<*xf32>)
    %Const_1, %ctl_2 = Const name("Const_1/Const") {dtype = i32, value = dense<[3, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %RandomStandardNormal_3, %ctl_4 = RandomStandardNormal(%Const_1) name("y") {T = i32, dtype = f32, seed = 0 : i64, seed2 = 0 : i64} : (tensor<2xi32>) -> (tensor<*xf32>)
    %Const_5, %ctl_6 = Const [%ctl_0] name("const1") {dtype = f32, value = dense<2.700000e+00> : tensor<3x5xf32>} : () -> (tensor<3x5xf32>)
    %Const_7, %ctl_8 = Const name("const2") {dtype = f32, value = dense<3.140000e+00> : tensor<3x5xf32>} : () -> (tensor<3x5xf32>)
    %Const_9, %ctl_10 = Const [%ctl_0] name("const3") {dtype = f32, value = dense<3.140000e+00> : tensor<3x5xf32>} : () -> (tensor<3x5xf32>)
    // CHECK-DAG: {{.*}}, %[[CTRL:.*]] = {{.*}} name("m1")
    // CHECK-DAG: Const [%[[CTRL]]] name("m1/_const")
    // CHECK-DAG: Const [%[[CTRL]]] name("m1/_index")
    %Merge:2, %ctl_11 = Merge(%RandomStandardNormal, %Const_5, %Const_7) name("m1") {N = 3 : i64, T = f32} : (tensor<*xf32>, tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<*xf32>, tensor<*xi32>)
    %Merge_12:2, %ctl_13 = Merge(%Const_5, %Const_9) name("m2") {N = 2 : i64, T = f32} : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<*xf32>, tensor<*xi32>)
    %Merge_14:2, %ctl_15 = Merge(%RandomStandardNormal, %RandomStandardNormal_3) name("m3") {N = 2 : i64, T = f32} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>)
    %Merge_16:2, %ctl_17 = Merge(%RandomStandardNormal, %Const_5) name("m4") {N = 2 : i64, T = f32} : (tensor<*xf32>, tensor<3x5xf32>) -> (tensor<*xf32>, tensor<*xi32>)
    %Identity, %ctl_18 = Identity(%Merge#0) name("out1") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Identity_19, %ctl_20 = Identity(%Merge#1) name("idx1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Identity_21, %ctl_22 = Identity(%Merge_12#0) name("out2") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Identity_23, %ctl_24 = Identity(%Merge_12#1) name("idx2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Identity_25, %ctl_26 = Identity(%Merge_14#0) name("out3") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Identity_27, %ctl_28 = Identity(%Merge_14#1) name("idx3") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Identity_29, %ctl_30 = Identity(%Merge_16#0) name("out4") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    %Identity_31, %ctl_32 = Identity(%Merge_16#1) name("idx4") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
  }
}
