// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const name("cf_half") {dtype = f32, value = dense<5.000000e-01> : tensor<1xf32>} : () -> (tensor<1xf32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} = Placeholder name("xf")
    %Placeholder, %ctl_0 = Placeholder name("xf") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    %Placeholder_1, %ctl_2 = Placeholder name("xi") {dtype = i32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xi32>)
    %Const_3, %ctl_4 = Const name("ci") {dtype = i32, value = dense<2> : tensor<1xi32>} : () -> (tensor<1xi32>)
    // CHECK: , %[[CTRL:.*]] = Const name("cf")
    %Const_5, %ctl_6 = Const name("cf") {dtype = f32, value = dense<2.000000e+00> : tensor<1xf32>} : () -> (tensor<1xf32>)
    %Div, %ctl_7 = Div(%Placeholder_1, %Const_3) name("div_i") {T = i32} : (tensor<2x2xi32>, tensor<1xi32>) -> (tensor<*xi32>)
    // CHECK: %[[CONST_DIVF:.*]], {{.*}} = Const [%[[CTRL]]] name("div_f/cf/_recip")
    // CHECK: Mul(%[[PLACEHOLDER]], %[[CONST_DIVF]]) name("div_f")
    %Div_8, %ctl_9 = Div(%Placeholder, %Const_5) name("div_f") {T = f32} : (tensor<2x2xf32>, tensor<1xf32>) -> (tensor<*xf32>)
    // CHECK: %[[CONST_REAL:.*]], {{.*}} = Const [%[[CTRL]]] name("realdiv/cf/_recip")
    // CHECK: Mul(%[[PLACEHOLDER]], %[[CONST_REAL]]) name("realdiv")
    %RealDiv, %ctl_10 = RealDiv(%Placeholder, %Const_5) name("realdiv") {T = f32} : (tensor<2x2xf32>, tensor<1xf32>) -> (tensor<*xf32>)
  }
}
