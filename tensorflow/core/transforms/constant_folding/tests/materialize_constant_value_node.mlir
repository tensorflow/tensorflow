// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    // CHECK-DAG: %[[PLACEHOLDER:.*]], %[[CTRL:.*]] = {{.*}} name("x")
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<1x2x3x4>} : () -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: Const [%[[CTRL]]] name("ones_like/const_folded") {{.*}} value = dense<1.000000e+00> {{.*}} -> (tensor<1x2x3x4xf32>)
    %OnesLike, %ctl_0 = OnesLike(%Placeholder) name("ones_like") {T = f32} : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: Const [%[[CTRL]]] name("zeros_like/const_folded") {{.*}} value = dense<0.000000e+00> {{.*}} -> (tensor<1x2x3x4xf32>)
    %ZerosLike, %ctl_1 = ZerosLike(%Placeholder) name("zeros_like") {T = f32} : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>)
    // CHECK-DAG: {{.*}}, %[[CTRL1:.*]] = {{.*}} name("Const/Const")
    %Const, %ctl_2 = Const name("Const/Const") {dtype = i32, value = dense<[4, 3, 2, 1]> : tensor<4xi32>} : () -> (tensor<4xi32>)
    // CHECK-DAG: {{.*}}, %[[CTRL2:.*]] = {{.*}} name("Const_1/Const")
    %Const_3, %ctl_4 = Const name("Const_1/Const") {dtype = i32, value = dense<42> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: Const [%[[CTRL1]], %[[CTRL2]]] name("fill/const_folded") {{.*}} value = dense<42> : tensor<4x3x2x1xi32>
    %Fill, %ctl_5 = Fill(%Const, %Const_3) name("fill") {T = i32, index_type = i32} : (tensor<4xi32>, tensor<i32>) -> (tensor<4x3x2x1xi32>)
    // CHECK: Fill(%[[PLACEHOLDER]], %{{.*}}) name("fill_1")
    %Fill_1, %ctl_6 = Fill(%Placeholder, %Const_3) name("fill_1") {T = i32, index_type = i32} : (tensor<1x2x3x4xf32>, tensor<i32>) -> (tensor<*xi32>)
    // CHECK: Fill(%{{.*}}, %[[PLACEHOLDER]]) name("fill_2")
    %Fill_2, %ctl_7 = Fill(%Const_3, %Placeholder) name("fill_2") {T = i32, index_type = i32} : (tensor<i32>, tensor<1x2x3x4xf32>) -> (tensor<*xi32>)
    // Note that this op is supposed to be folded by operation evaluation. Not by the MaterializeFillNode pattern
    // CHECK: Const{{.*}} name("fill_3/eval_0/const_folded")
    %Fill_3, %ctl_8 = Fill(%Const, %Const_3) name("fill_3") {T = i32, index_type = i32} : (tensor<4xi32>, tensor<i32>) -> (tensor<*xi32>)
  }
}
