// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK-DAG: %[[PLACEHOLDER:.*]], %[[CTRL:.*]] = {{.*}} name("x")
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<1x2x3x4>} : () -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: Const [%[[CTRL]]] name("ones_like")
    %OnesLike, %ctl_0 = OnesLike(%Placeholder) name("ones_like") {T = f32} : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: Const [%[[CTRL]]] name("zeros_like")
    %ZerosLike, %ctl_1 = ZerosLike(%Placeholder) name("zeros_like") {T = f32} : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: {{.*}}, %[[CTRL1:.*]] = {{.*}} name("Const/Const")
    %Const, %ctl_2 = Const name("Const/Const") {dtype = i32, value = dense<[4, 3, 2, 1]> : tensor<4xi32>} : () -> (tensor<4xi32>)
    // CHECK-DAG: {{.*}}, %[[CTRL2:.*]] = {{.*}} name("Const_1/Const")
    %Const_3, %ctl_4 = Const name("Const_1/Const") {dtype = i32, value = dense<42> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: Const [%[[CTRL1]], %[[CTRL2]]] name("tfg.FillConst_folded")
    %Fill, %ctl_5 = Fill(%Const, %Const_3) name("fill") {T = i32, index_type = i32} : (tensor<4xi32>, tensor<i32>) -> (tensor<4x3x2x1xi32>)
  }
}
