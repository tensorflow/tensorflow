// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder name("a") {dtype = f32, shape = #tf_type.shape<?x?>} : () -> (tensor<?x?xf32>)
    %Square, %ctl_0 = Square(%Placeholder) name("b") {T = f32} : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
    %Mul, %ctl_1 = Mul(%Placeholder, %Square) name("c") {T = f32} : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<*xf32>)
    %Shape, %ctl_2 = Shape(%Placeholder) name("d") {T = f32, out_type = i32} : (tensor<?x?xf32>) -> (tensor<*xi32>)
    %Shape_3, %ctl_4 = Shape(%Square) name("e") {T = f32, out_type = i32} : (tensor<?x?xf32>) -> (tensor<*xi32>)
    // CHECK-DAG: Const{{.*}} name("f-bcastargs-0")
    // CHECK-DAG: Const{{.*}} name("f-bcastargs-1")
    %BroadcastGradientArgs:2, %ctl_5 = BroadcastGradientArgs(%Shape, %Shape_3) name("f") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    %Identity, %ctl_6 = Identity(%BroadcastGradientArgs#0) name("o1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Identity_7, %ctl_8 = Identity(%BroadcastGradientArgs#1) name("o2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Placeholder_9, %ctl_10 = Placeholder name("g") {dtype = f32, shape = #tf_type.shape<1>} : () -> (tensor<1xf32>)
    %Shape_11, %ctl_12 = Shape(%Placeholder_9) name("h") {T = f32, out_type = i32} : (tensor<1xf32>) -> (tensor<*xi32>)
    // CHECK-DAG: Const{{.*}} name("i-bcastargs-0")
    // CHECK-DAG: Const{{.*}} name("i-bcastargs-1")
    %BroadcastGradientArgs_13:2, %ctl_14 = BroadcastGradientArgs(%Shape, %Shape_11) name("i") {T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    %Identity_15, %ctl_16 = Identity(%BroadcastGradientArgs_13#0) name("p1") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Identity_17, %ctl_18 = Identity(%BroadcastGradientArgs_13#1) name("p2") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
  }
}
