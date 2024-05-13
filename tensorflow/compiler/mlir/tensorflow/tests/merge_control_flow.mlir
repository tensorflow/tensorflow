// RUN: tf-opt %s -tf-merge-control-flow | FileCheck %s

// Check that IfRegions with different predicates are not merged.

// CHECK-LABEL: func @different_predicate_no_merge
func.func @different_predicate_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    "tf.IfRegion"(%1) ({
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegions with same predicates but different block are not merged.

// CHECK-LABEL: func @different_block_no_merge
func.func @different_block_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %3 = "tf.A"() : () -> (tensor<?xf32>)
    %4 = "tf.B"() : () -> (tensor<i32>)
    "tf.WhileRegion"(%4, %3) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
      "tf.IfRegion"(%0) ({
        %2 = "tf.A"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
        }, {
        "tf.Yield"() : () -> ()
       }) {is_stateless = true} : (tensor<i1>) -> ()
       "tf.Yield"(%1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
      "tf.IfRegion"(%0) ({
        %2 = "tf.B"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
        }, {
        "tf.Yield"() : () -> ()
       }) {is_stateless = true} : (tensor<i1>) -> ()
      "tf.Yield"(%arg1, %arg2) : (tensor<i32>, tensor<?xf32>) -> ()
    }) {is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegions with same predicates and no returns are merged.

// CHECK-LABEL: func @same_predicate_no_returns_merged
func.func @same_predicate_no_returns_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:         _else_func_name = "elseFunc1"
  // CHECK-SAME:   _then_func_name = "thenFunc1"

  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true, _else_func_name = "elseFunc1", _then_func_name = "thenFunc1"} : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ({
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true, _else_func_name = "elseFunc2", _then_func_name = "thenFunc2"} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegions with same predicate intermediate data dependency are not merged.

// CHECK-LABEL: func @same_predicate_intermediate_dependency_no_merge
func.func @same_predicate_intermediate_dependency_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<f32>)
    %4 = "tf.E"(%3) : (tensor<f32>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.B"(%4) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegions with same predicate intermediate side effect dependency are not merged.

// CHECK-LABEL: func @same_predicate_side_effect_dependency_no_merge
func.func @same_predicate_side_effect_dependency_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    "tf.D"(%1) : (tensor<f32>) -> ()
    "tf.IfRegion"(%0) ({
      %4 = "tf.B"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that merged IfRegions correctly set is_stateless attribute.

// CHECK-LABEL: func @same_predicate_stateless_merge
func.func @same_predicate_stateless_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        is_stateless = false
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %4 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegions with same predicates and returns are merged.

// CHECK-LABEL: func @same_predicate_returns_merged
func.func @same_predicate_returns_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        %[[IF_OUTPUT:[0-9]*]]:2 = "tf.IfRegion"
  // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK-NEXT:     %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK-NEXT:     "tf.Yield"(%[[A_OUTPUT]], %[[B_OUTPUT]])
  // CHECK:          %[[C_OUTPUT:[0-9]*]] = "tf.C"
  // CHECK-NEXT:     %[[D_OUTPUT:[0-9]*]] = "tf.D"
  // CHECK-NEXT:     "tf.Yield"(%[[C_OUTPUT]], %[[D_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  // CHECK         "tf.E"(%[[IF_OUTPUT]]#0, %[[IF_OUTPUT]]#1)
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.B"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}
// Check that IfRegions with same predicates and unused returns.

// CHECK-LABEL: func @same_predicate_returns_unused
func.func @same_predicate_returns_unused() {
  // CHECK:      tf_device.cluster
  // CHECK:        %[[IF_OUTPUT:[0-9]*]] = "tf.IfRegion"
  // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK-NEXT:     %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK-NEXT:     "tf.Yield"(%[[B_OUTPUT]])
  // CHECK:          %[[C_OUTPUT:[0-9]*]] = "tf.C"
  // CHECK-NEXT:     %[[D_OUTPUT:[0-9]*]] = "tf.D"
  // CHECK-NEXT:     "tf.Yield"(%[[D_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  // CHECK         "tf.E"(%[[IF_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.B"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%2) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// CHECK-LABEL: func @same_predicate_dependency
func.func @same_predicate_dependency() {
  // CHECK:      tf_device.cluster
  // CHECK:        %[[IF_OUTPUT:[0-9]*]] = "tf.IfRegion"
  // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK-NEXT:     %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK-NEXT:     "tf.Yield"(%[[B_OUTPUT]])
  // CHECK:          %[[C_OUTPUT:[0-9]*]] = "tf.C"
  // CHECK-NEXT:     %[[D_OUTPUT:[0-9]*]] = "tf.D"
  // CHECK-NEXT:     "tf.Yield"(%[[D_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  // CHECK         "tf.E"(%[[IF_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.B"(%1) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%2) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Checks that results from first IfRegion are moved after merged IfRegion op as needed.

// CHECK-LABEL: func @same_predicate_results_moved
func.func @same_predicate_results_moved(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  // CHECK:      tf_device.cluster
  // CHECK:        %[[IF_OUTPUT:[0-9]*]]:2 = "tf.IfRegion"
  // CHECK:          %[[A_OUTPUT:[0-9]*]] = "tf.A"
  // CHECK-NEXT:     %[[B_OUTPUT:[0-9]*]] = "tf.B"
  // CHECK-NEXT:     "tf.Yield"(%[[A_OUTPUT]], %[[B_OUTPUT]])
  // CHECK:          %[[C_OUTPUT:[0-9]*]] = "tf.C"
  // CHECK-NEXT:     %[[D_OUTPUT:[0-9]*]] = "tf.D"
  // CHECK-NEXT:     "tf.Yield"(%[[C_OUTPUT]], %[[D_OUTPUT]])
  // CHECK-NOT:    "tf.IfRegion"
  // CHECK         "tf.AssignVariableOp(arg0, %[[IF_OUTPUT#0]])
  // CHECK         "tf.E"(%[[IF_OUTPUT#1]])
  // CHECK-NEXT    "tf.F"(%[[IF_OUTPUT#1]])
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
    %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %5 = "tf.IfRegion"(%0) ({
      %3 = "tf.B"(%4) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"(%4) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    %6 = "tf.E"(%5) : (tensor<i32>) -> (tensor<f32>)
    "tf.F"(%1, %6) : (tensor<f32>, tensor<f32>) -> ()
    tf_device.return %1 : tensor<f32>
  }) {cluster_attr = "cluster_attr"} : () -> (tensor<f32>)
  func.return
}

// Checks that side effect successor of op in first IfRegion are moved after merged IfRegion op as needed.

// CHECK-LABEL: func @same_predicate_side_effect_moved
func.func @same_predicate_side_effect_moved(%arg0: tensor<!tf_type.resource<tensor<f32>>>) {
  // CHECK:      tf_device.cluster
  // CHECK:        %[[IF_OUTPUT:[0-9]*]]:2 = "tf.IfRegion"
  // CHECK:         "tf.A"
  // CHECK-NEXT:    "tf.AssignVariableOp"
  // CHECK-NEXT:    "tf.B"
  // CHECK:         "tf.C"
  // CHECK-NEXT:    "tf.D"
  // CHECK-NOT:    "tf.IfRegion"
  // CHECK         "tf.ReadVariableOp(arg0)
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.AssignVariableOp"(%arg0, %3) : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>)
    %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %5 = "tf.IfRegion"(%0) ({
      %3 = "tf.B"(%4) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"(%4) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    %6 = "tf.E"(%5) : (tensor<i32>) -> (tensor<f32>)
    "tf.F"(%1, %6) : (tensor<f32>, tensor<f32>) -> ()
    tf_device.return %8 : tensor<f32>
  }) {cluster_attr = "cluster_attr"} : () -> (tensor<f32>)
  func.return
}

// Check that 3 IfRegions with same predicates and no intermediate dependencies are merged.

// CHECK-LABEL: func @same_predicate_3_ifregions
func.func @same_predicate_3_ifregions() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) { is_stateless = true } : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ({
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ({
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that 3 IfRegions with same predicates where 2nd and 3rd IfRegions
// can be merged but not 1st IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_only_merge2
func.func @same_predicate_3_ifregions_only_merge2() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.A"
  // CHECK:        "tf.D"
  // CHECK-NEXT    "tf.IfRegion"
  // CHECK:          "tf.E"
  // CHECK-NEXT:     "tf.G"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %4 = "tf.E"(%3) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.G"(%3) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}


// Check that 3 IfRegions  where 1st and 3rd IfRegions
// can be merged but not 2nd IfRegion and 2nd IfRegion should be moved after
// newly merged IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_reorder
func.func @same_predicate_3_ifregions_reorder() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.A"
  // CHECK:          "tf.G"
  // CHECK-NEXT    "tf.IfRegion"
  // CHECK:          "tf.E"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%8) ({
      %4 = "tf.E"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.G"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that 3 IfRegions where 1st and 3rd IfRegions
// can't be merged due to an intermediate dep in the 2nd IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_intermediate_dep
func.func @same_predicate_3_ifregions_intermediate_dep() {
  // CHECK-COUNT-3:        "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ({
      %4 = "tf.E"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.G"(%9) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that 3 IfRegions where 1st and 3rd IfRegions
// can't be merged due to an intermediate side effecting IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_intermediate_side_effect
func.func @same_predicate_3_ifregions_intermediate_side_effect() {
  // CHECK-COUNT-3:   "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ({
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.G"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check case for 4 IfRegions where 1st and 3rd IfRegions
// can be merged and ensure that side effect analysis is regenerated.

// CHECK-LABEL: func @side_effect_analysis_updated
func.func @side_effect_analysis_updated() {
  // CHECK-COUNT-3:   "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ({
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ({
      %5 = "tf.G"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = false} : (tensor<i1>) -> ()
    "tf.IfRegion"(%8) ({
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that 2 IfRegions can be merged when the first IfRegion contains multiple side effecting ops.

// CHECK-LABEL: func @same_predicate_2_ifregions_multiple_side_effect_ops
func.func @same_predicate_2_ifregions_multiple_side_effect_ops() {
  // CHECK:       "tf.IfRegion"
  // CHECK-NOT:   "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ({
      %2 = "tf.A"() : () -> (tensor<f32>)
      %3 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%0) ({
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that the algorithm that moves ops to after the merged IfRegion is
// reasonably efficient.

// CHECK-LABEL: func @moved_ops_with_many_dependencies
func.func @moved_ops_with_many_dependencies() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ( {
      %1 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%1) : (tensor<f32>) -> ()
      }, {
      %1 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"(%1) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.D"(%2) : (tensor<f32>) -> (tensor<f32>)
    %4 = "tf.D"(%2, %3) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %5 = "tf.D"(%2, %3, %4) : (tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %6 = "tf.D"(%2, %3, %4, %5) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %7 = "tf.D"(%2, %3, %4, %5, %6) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %8 = "tf.D"(%2, %3, %4, %5, %6, %7) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %9 = "tf.D"(%2, %3, %4, %5, %6, %7, %8) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %10 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %11 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %12 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %13 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %14 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %15 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %16 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %17 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %18 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %19 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %20 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %21 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %22 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %23 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %24 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %25 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %26 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %27 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %28 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %29 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %30 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %31 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %32 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %33 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %34 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %35 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %36 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %37 = "tf.D"(%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}


// Check that two IfRegion groups where each of them contains three IfRegions with
// the same predicate are merged. There are no dependencies between IfRegions.

// CHECK-LABEL: func @three_if_regions_with_same_predicate_merged
func.func @three_if_regions_with_same_predicate_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.A"
  // CHECK:        "tf.C"
  // CHECK:        "tf.E"
  // CHECK:        "tf.B"
  // CHECK:        "tf.D"
  // CHECK:        "tf.F"
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.G"
  // CHECK:        "tf.H"
  // CHECK:        "tf.I"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %5 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
     }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.IfRegion"(%0) ({
      %6 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%6) : (tensor<f32>) -> ()
      }, {
      %6 = "tf.D"() : () -> (tensor<f32>)
      "tf.Yield"(%6) : (tensor<f32>) -> ()
     }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %7 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %7 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
     }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%1) ({
      %8 = "tf.G"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    "tf.IfRegion"(%1) ({
      %9 = "tf.H"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    "tf.IfRegion"(%1) ({
      %10 = "tf.I"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that IfRegion groups with nested IfRegion with the same predicate are
// merged at the same block level. There are no dependencies between IfRegions.

// CHECK-LABEL: func @nested_IfRegions_with_same_predicate_same_block_level_merged
func.func @nested_IfRegions_with_same_predicate_same_block_level_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.A"
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.B"
  // CHECK:        "tf.D"
  // CHECK:        "tf.F"
  // CHECK:        "tf.C"
  // CHECK:        "tf.E"
  // CHECK:        "tf.G"
  // CHECK:        "tf.H"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %5 = "tf.A"() : () -> (tensor<i1>)
      "tf.IfRegion"(%5) ({
        %6 = "tf.B"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }, {
        %6 = "tf.C"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }) {is_stateless = false} : (tensor<i1>) -> ()
      "tf.IfRegion"(%5) ({
        %7 = "tf.D"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }, {
        %7 = "tf.E"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }) {is_stateless = false} : (tensor<i1>) -> ()
      "tf.IfRegion"(%5) ({
        %8 = "tf.F"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }, {
        %8 = "tf.G"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
      }) {is_stateless = false} : (tensor<i1>) -> ()
      "tf.Yield"(%5) : (tensor<i1>) -> ()
      }, {
      %5 = "tf.H"() : () -> (tensor<i1>)
      "tf.Yield"(%5) : (tensor<i1>) -> ()
      }) {is_stateless = false} : (tensor<i1>) -> (tensor<i1>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that when two different IfRegion groups are overlapped and there is no
// control dependency or data dependency, both of the groups can be merged

// CHECK-LABEL: func @two_overlapped_if_groups_with_no_dependency_merged
func.func @two_overlapped_if_groups_with_no_dependency_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<5.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<9.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<6.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<1.000000e+01> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.Const"() <{value = dense<3.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<7.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<1.100000e+01> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<4.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<8.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK           "tf.Const"() <{value = dense<1.200000e+01> : tensor<f32>}> : () -> tensor<f32>
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.IfRegion"(%1) ({
      %7 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %7 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %5 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.IfRegion"(%1) ({
      %9 = "tf.Const"() {value = dense<7.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }, {
      %9 = "tf.Const"() {value = dense<8.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %10 = "tf.IfRegion"(%0) ({
      %11 = "tf.Const"() {value = dense<9.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }, {
      %11 = "tf.Const"() {value = dense<10.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %12 = "tf.IfRegion"(%1) ({
      %13 = "tf.Const"() {value = dense<11.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }, {
      %13 = "tf.Const"() {value = dense<12.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that when two different IfRegion groups are overlapped and there is a
// control dependency or data dependency, then they cannot be merged.

// CHECK-LABEL: func @two_overlapped_if_groups_with_dependency_not_merged_for_first_if_region_group
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.A"
  // CHECK:          "tf.B"
  // CHECK:        "tf.AA"
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.C"
  // CHECK:          "tf.D"
  // CHECK:        "tf.AB"
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.E"
  // CHECK:          "tf.F"
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.Const"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<3.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<5.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK:          "tf.Const"() <{value = dense<4.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK;          "tf.Const"() <{value = dense<6.000000e+00> : tensor<f32>}> : () -> tensor<f32>
  // CHECK-NOT:    "tf.IfRegion"
func.func @two_overlapped_if_groups_with_dependency_not_merged_for_first_if_region_group() {
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.IfRegion"(%1) ({
      %7 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %7 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %14 = "tf.AA"(%2) : (tensor<f32>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %5 = "tf.C"(%14) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.D"(%14) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.IfRegion"(%1) ({
      %9 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }, {
      %9 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %15 = "tf.AB"(%4) : (tensor<f32>) -> (tensor<f32>)
    %10 = "tf.IfRegion"(%0) ({
      %11 = "tf.E"(%15) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }, {
      %11 = "tf.F"(%15) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }) {is_stateless = false} : (tensor<i1>) -> (tensor<f32>)
    %12 = "tf.IfRegion"(%1) ({
      %13 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }, {
      %13 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that when two different IfRegion groups are overlapped and there is no
// control dependency or data dependency. They can be merged.
// The second IfRegion moves after the first IfRegion.

// CHECK-LABEL: func @two_overlapped_if_groups_with_dependency_merged_v1
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK         "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<9.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CEHCK         "tf.Const"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK:        "tf.IfRegion"
  // CHECK         "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Add"(%0, %cst_1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<7.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<1.100000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Add"(%0, %cst_1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<1.200000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NOT:    "tf.IfRegion"
func.func @two_overlapped_if_groups_with_dependency_merged_v1() {
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.IfRegion"(%1) ({
      %50 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      %7 = "tf.Add"(%2, %50) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %50 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      %7 = "tf.Add"(%2, %50) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %5 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.IfRegion"(%1) ({
      %9 = "tf.Const"() {value = dense<7.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }, {
      %9 = "tf.Const"() {value = dense<8.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %10 = "tf.IfRegion"(%0) ({
      %11 = "tf.Const"() {value = dense<9.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }, {
      %11 = "tf.Const"() {value = dense<10.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %12 = "tf.IfRegion"(%1) ({
      %13 = "tf.Const"() {value = dense<11.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }, {
      %13 = "tf.Const"() {value = dense<12.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that when two different IfRegion groups are overlapped and there is no
// control dependency or data dependency. They can be merged.
// The first IfRegion moves after the second IfRegion.

// CHECK-LABEL: func @two_overlapped_if_groups_with_dependency_merged_v2
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK         "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<7.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<1.100000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<1.200000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK:        "tf.IfRegion"
  // CHECK         "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Add"(%0, %cst_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<9.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK         "tf.Const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
   // CHECK        "tf.Add"(%0, %cst_2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
   // CHECK        "tf.Const"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NOT:    "tf.IfRegion"
func.func @two_overlapped_if_groups_with_dependency_merged_v2() {
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.IfRegion"(%1) ({
      %7 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %7 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %50 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      %5 = "tf.Add"(%6, %50) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %50 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      %5 = "tf.Add"(%6, %50) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.IfRegion"(%1) ({
      %9 = "tf.Const"() {value = dense<7.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }, {
      %9 = "tf.Const"() {value = dense<8.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%9) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %10 = "tf.IfRegion"(%0) ({
      %11 = "tf.Const"() {value = dense<9.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }, {
      %11 = "tf.Const"() {value = dense<10.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%11) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %12 = "tf.IfRegion"(%1) ({
      %13 = "tf.Const"() {value = dense<11.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }, {
      %13 = "tf.Const"() {value = dense<12.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%13) : (tensor<f32>) -> ()
      }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that two IfRegion groups where each of them contains three IfRegions with
// the same predicate are merged. There are no dependencies between IfRegions.

// CHECK-LABEL: func @three_if_regions_with_same_predicate_and_correct_return_indices_merged_v1
func.func @three_if_regions_with_same_predicate_and_correct_return_indices_merged_v1() {
  //CHECK  "tf_device.cluster"
  //CHECK  "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  //CHECK  "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  //CHECK-SAME  "tf.IfRegion"(%cst)
  //CHECK  "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Add"(%cst_1, %cst_3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Add"(%cst_1, %cst_3) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %5 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.IfRegion"(%0) ({
      %6 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%6) : (tensor<f32>) -> ()
      }, {
      %6 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%6) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %7 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      %8 = "tf.Add"(%2, %7) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %7 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      %8 = "tf.Add"(%2, %7) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that merged IfRegion will not contain unused return variables

// CHECK-LABEL: func @three_if_regions_with_same_predicate_and_correct_return_indices_merged_v2
func.func @three_if_regions_with_same_predicate_and_correct_return_indices_merged_v2() {
  //CHECK  "tf_device.cluster"
  //CHECK  "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  //CHECK  "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  //CHECK-SAME  "tf.IfRegion"(%cst)
  //CHECK  "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Add"(%cst_1, %cst_3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
  //CHECK  "tf.Add"(%cst_1, %cst_3) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%0) ({
      %5 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      %7 = "tf.Add"(%2, %5) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      %7 = "tf.Add"(%2, %5) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%7) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.IfRegion"(%0) ({
      %9 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      %10 = "tf.Add"(%2, %9) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%10) : (tensor<f32>) -> ()
      }, {
      %9 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      %10 = "tf.Add"(%2, %9) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%10) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

// Check that merged IfRegion will not contain unused return variables

// CHECK-LABEL: func @one_use_between_two_IfRegions_groups
  //CHECK tf_device.cluster
  //CHECK "tf.IfRegion"
  //CHECK-NOT "tf.Add"
  //CHECK "tf.IfRegion"
  //CHECK "tf.Add"
  //CHECK-NOT "tf.IfRegion"
func.func @one_use_between_two_IfRegions_groups() {
  "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.IfRegion"(%0) ({
      %3 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %4 = "tf.IfRegion"(%1) ({
      %5 = "tf.Const"() {value = dense<3.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
      }, {
      %5 = "tf.Const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%5) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %6 = "tf.Add"(%2, %4) : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    %7 = "tf.IfRegion"(%1) ({
      %8 = "tf.Const"() {value = dense<5.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%8) : (tensor<f32>) -> ()
      }, {
      %8 = "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%8) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%0) ({
      %10 = "tf.Const"() {value = dense<7.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%10) : (tensor<f32>) -> ()
      }, {
      %10 = "tf.Const"() {value = dense<8.0> : tensor<f32>} : () -> tensor<f32>
      "tf.Yield"(%10) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  func.return
}

