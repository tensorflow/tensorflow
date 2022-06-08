// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %Placeholder, %ctl = Placeholder name("a") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    %Square, %ctl_0 = Square(%Placeholder) name("b") {T = f32} : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
    %Mul, %ctl_1 = Mul(%Placeholder, %Square) name("c") {T = f32} : (tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<*xf32>)
    %Shape, %ctl_2 = Shape(%Placeholder) name("d") {T = f32, out_type = i32} : (tensor<2x2xf32>) -> (tensor<*xi32>)
    %Shape_3, %ctl_4 = Shape(%Square) name("e") {T = f32, out_type = i32} : (tensor<2x2xf32>) -> (tensor<*xi32>)
    // CHECK-DAG: , %[[C0:.*]] = Const{{.*}} name("f/bcastargs_0")
    // CHECK-DAG: , %[[C1:.*]] = Const{{.*}} name("f/bcastargs_1")
    %BroadcastGradientArgs:2, %ctl_5 = BroadcastGradientArgs(%Shape, %Shape_3) name("f") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    // CHECK: Const [%[[C0]]] name("o1
    %Identity, %ctl_6 = Identity(%BroadcastGradientArgs#0) name("o1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[C1]]] name("o2
    %Identity_7, %ctl_8 = Identity(%BroadcastGradientArgs#1) name("o2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[C0]]] name("o3
    %Identity_1, %ctl_19 = Identity(%BroadcastGradientArgs#0) name("o3") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const [%[[C1]]] name("o4
    %Identity_8, %ctl_20 = Identity(%BroadcastGradientArgs#1) name("o4") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Placeholder_9, %ctl_10 = Placeholder name("g") {dtype = f32, shape = #tf_type.shape<1>} : () -> (tensor<1xf32>)
    %Placeholder_10, %ctl_11 = Placeholder name("z") {dtype = f32, shape = #tf_type.shape<?x?>} : () -> (tensor<?x?xf32>)
    %Shape_12, %ctl_21 = Shape(%Placeholder_10) name("x") {T = f32, out_type = i32} : (tensor<?x?xf32>) -> (tensor<*xi32>)
    %Shape_11, %ctl_12 = Shape(%Placeholder_9) name("h") {T = f32, out_type = i32} : (tensor<1xf32>) -> (tensor<*xi32>)
    // CHECK: BroadcastGradientArgs{{.*}} name("i")
    %BroadcastGradientArgs_13:2, %ctl_14 = BroadcastGradientArgs(%Shape_12, %Shape_11) name("i") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    // CHECK: Identity{{.*}} name("p1")
    %Identity_15, %ctl_16 = Identity(%BroadcastGradientArgs_13#0) name("p1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Identity{{.*}} name("p2")
    %Identity_17, %ctl_18 = Identity(%BroadcastGradientArgs_13#1) name("p2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Const, %ctl_22 = Const name("c2") {dtype = f32, value = dense<2> : tensor<2xi32>} : () -> (tensor<2xi32>)
    %BroadcastGradientArgs_14:2, %ctl_23 = BroadcastGradientArgs(%Shape, %Const) name("i2") {T = i32} : (tensor<*xi32>, tensor<2xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    // CHECK: Const{{.*}} name("p12")
    // CHECK-SAME: value = dense<> : tensor<0xi32>
    %Identity_16, %ctl_24 = Identity(%BroadcastGradientArgs_14#0) name("p12") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    // CHECK: Const{{.*}} name("p22")
    // CHECK-SAME: value = dense<> : tensor<0xi32>
    %Identity_18, %ctl_25 = Identity(%BroadcastGradientArgs_14#1) name("p22") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    return
  }
}
