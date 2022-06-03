// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @mlir_lifted_graph(%arg: tensor<2x2xf32> {tfg.name = "name"}) {
    // CHECK: %[[P0:.*]], {{.*}} = Placeholder name("x")
    %Placeholder, %ctl = Placeholder name("x") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[P1:.*]], {{.*}} = Placeholder name("y")
    %Placeholder_0, %ctl_1 = Placeholder name("y") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[P2:.*]], {{.*}} = Placeholder name("z")
    %Placeholder_2, %ctl_3 = Placeholder name("z") {dtype = f32, shape = #tf_type.shape<2x2>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[C0:.*]], {{.*}} = Const name("c1")
    %Const, %ctl_4 = Const name("c1") {dtype = f32, value = dense<1.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[C1:.*]], {{.*}} = Const name("c2")
    %Const_5, %ctl_6 = Const name("c2") {dtype = f32, value = dense<2.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: %[[C2:.*]], {{.*}} = Const name("c3")
    %Const_7, %ctl_8 = Const name("c3") {dtype = f32, value = dense<3.000000e+00> : tensor<2x2xf32>} : () -> (tensor<2x2xf32>)
    // CHECK: Const{{.*}} name("acc0/eval_0/const_folded")
    %AccumulateNV2, %ctl_9 = AccumulateNV2(%Const, %Const_5, %Const_7) name("acc0") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: AccumulateNV2(%[[P0]], %[[P1]], %[[P2]]) name("acc1")
    %AccumulateNV2_10, %ctl_11 = AccumulateNV2(%Placeholder, %Placeholder_0, %Placeholder_2) name("acc1") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: AccumulateNV2(%[[C0]], %[[P0]], %[[P1]]) name("acc2")
    %AccumulateNV2_12, %ctl_13 = AccumulateNV2(%Const, %Placeholder, %Placeholder_0) name("acc2") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: %[[ACC3C:.*]], {{.*}} = Const{{.*}} name("acc3/_partial_split_2/eval_0/const_folded")
    // CHECK: AccumulateNV2(%[[ACC3C]], %[[P2]]) name("acc3")
    // CHECK-SAME: N = 2
    %AccumulateNV2_14, %ctl_15 = AccumulateNV2(%Const, %Const_5, %Placeholder_2) name("acc3") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: %[[ACC4C:.*]], {{.*}} = Const{{.*}} name("acc4/_partial_split_2/eval_0/const_folded")
    // CHECK: AccumulateNV2(%[[ACC4C]], %[[P1]]) name("acc4")
    // CHECK-SAME: N = 2
    %AccumulateNV2_16, %ctl_17 = AccumulateNV2(%Const, %Placeholder_0, %Const_5) name("acc4") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: %[[ACC5C:.*]], {{.*}} = Const{{.*}} name("acc5/_partial_split_2/eval_0/const_folded")
    // CHECK: AccumulateNV2(%[[ACC5C]], %[[P0]]) name("acc5")
    // CHECK-SAME: N = 2
    %AccumulateNV2_18, %ctl_19 = AccumulateNV2(%Placeholder, %Const, %Const_5) name("acc5") {N = 3 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    // CHECK: %[[ACC6C:.*]], {{.*}} = Const{{.*}} name("acc6/_partial_split_2/eval_0/const_folded")
    // CHECK: AccumulateNV2(%[[ACC6C]], %[[P0]], %[[P1]]) name("acc6")
    // CHECK-SAME: N = 3
    %AccumulateNV2_20, %ctl_21 = AccumulateNV2(%Placeholder, %Const, %Placeholder_0, %Const_5) name("acc6") {N = 4 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    %Pack, %ctl_22 = Pack(%AccumulateNV2, %AccumulateNV2_10, %AccumulateNV2_12, %AccumulateNV2_14, %AccumulateNV2_16, %AccumulateNV2_18, %AccumulateNV2_20) name("stack") {N = 7 : i64, T = f32, axis = 0 : i64} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<7x2x2xf32>)
    // CHECK: AccumulateNV2{{.*}} name("acc7")
    %AccumulateNV2_21, %ctl_23 = AccumulateNV2(%arg, %Const, %Placeholder_0, %Const_5) name("acc7") {N = 4 : i64, T = f32, shape = #tf_type.shape<2x2>} : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
    return
  }
}
