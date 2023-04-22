// RUN: tf-opt %s -tf-merge-control-flow | FileCheck %s

// Check that IfRegions with different predicates are not merged.

// CHECK-LABEL: func @different_predicate_no_merge
func @different_predicate_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    "tf.IfRegion"(%1) ( {
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that IfRegions with same predicates but different block are not merged.

// CHECK-LABEL: func @different_block_no_merge
func @different_block_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %3 = "tf.A"() : () -> (tensor<?xf32>)
    %4 = "tf.B"() : () -> (tensor<i32>)
    "tf.WhileRegion"(%4, %3) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
      "tf.IfRegion"(%0) ( {
        %2 = "tf.A"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
        }, {
        "tf.Yield"() : () -> ()
       }) {is_stateless = true} : (tensor<i1>) -> ()
       "tf.Yield"(%1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<?xf32>):
      "tf.IfRegion"(%0) ( {
        %2 = "tf.B"() : () -> (tensor<f32>)
        "tf.Yield"() : () -> ()
        }, {
        "tf.Yield"() : () -> ()
       }) {is_stateless = true} : (tensor<i1>) -> ()
      "tf.Yield"(%arg1, %arg2) : (tensor<i32>, tensor<?xf32>) -> ()
    }) {is_stateless = false} : (tensor<i32>, tensor<?xf32>) -> (tensor<i32>, tensor<?xf32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that IfRegions with same predicates and no returns are merged.

// CHECK-LABEL: func @same_predicate_no_returns_merged
func @same_predicate_no_returns_merged() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:         _else_func_name = "elseFunc1"
  // CHECK-SAME:   _then_func_name = "thenFunc1"

  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true, _else_func_name = "elseFunc1", _then_func_name = "thenFunc1"} : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ( {
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true, _else_func_name = "elseFunc2", _then_func_name = "thenFunc2"} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that IfRegions with same predicate intermediate data dependency are not merged.

// CHECK-LABEL: func @same_predicate_intermediate_dependency_no_merge
func @same_predicate_intermediate_dependency_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<f32>)
    %4 = "tf.E"(%3) : (tensor<f32>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.B"(%4) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that IfRegions with same predicate intermediate side effect dependency are not merged.

// CHECK-LABEL: func @same_predicate_side_effect_dependency_no_merge
func @same_predicate_side_effect_dependency_no_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    "tf.D"(%1) : (tensor<f32>) -> ()
    "tf.IfRegion"(%0) ( {
      %4 = "tf.B"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that merged IfRegions correctly set is_stateless attribute.

// CHECK-LABEL: func @same_predicate_stateless_merge
func @same_predicate_stateless_merge() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        is_stateless = false
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
     }) {is_stateless = true} : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %4 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that IfRegions with same predicates and returns are merged.

// CHECK-LABEL: func @same_predicate_returns_merged
func @same_predicate_returns_merged() {
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
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ( {
      %3 = "tf.B"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}
// Check that IfRegions with same predicates and unused returns.

// CHECK-LABEL: func @same_predicate_returns_unused
func @same_predicate_returns_unused() {
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
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ( {
      %3 = "tf.B"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"() : () -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%2) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @same_predicate_dependency
func @same_predicate_dependency() {
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
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %2 = "tf.IfRegion"(%0) ( {
      %3 = "tf.B"(%1) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
      }, {
      %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<i32>)
      "tf.Yield"(%3) : (tensor<i32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<i32>)
    "tf.E"(%2) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Checks that results from first IfRegion are moved after merged IfRegion op as needed.

// CHECK-LABEL: func @same_predicate_results_moved
func @same_predicate_results_moved(%arg0: tensor<!tf.resource<tensor<f32>>>) {
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
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.AssignVariableOp"(%arg0, %1) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
    %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %5 = "tf.IfRegion"(%0) ( {
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
  return
}

// Checks that side effect successor of op in first IfRegion are moved after merged IfRegion op as needed.

// CHECK-LABEL: func @same_predicate_side_effect_moved
func @same_predicate_side_effect_moved(%arg0: tensor<!tf.resource<tensor<f32>>>) {
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
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %3 = "tf.A"() : () -> (tensor<f32>)
      "tf.AssignVariableOp"(%arg0, %3) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.Yield"(%3) : (tensor<f32>) -> ()
      }, {
      %3 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%3) : (tensor<f32>) -> ()
     }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %8 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> (tensor<f32>)
    %4 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %5 = "tf.IfRegion"(%0) ( {
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
  return
}

// Check that 3 IfRegions with same predicates and no intermediate dependencies are merged.

// CHECK-LABEL: func @same_predicate_3_ifregions
func @same_predicate_3_ifregions() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) { is_stateless = true } : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ( {
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ( {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that 3 IfRegions with same predicates where 2nd and 3rd IfRegions
// can be merged but not 1st IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_only_merge2
func @same_predicate_3_ifregions_only_merge2() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.A"
  // CHECK:        "tf.D"
  // CHECK-NEXT    "tf.IfRegion"
  // CHECK:          "tf.E"
  // CHECK-NEXT:     "tf.G"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %3 = "tf.D"(%1) : (tensor<f32>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %4 = "tf.E"(%3) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.G"(%3) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}


// Check that 3 IfRegions  where 1st and 3rd IfRegions
// can be merged but not 2nd IfRegion and 2nd IfRegion should be moved after
// newly merged IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_reorder
func @same_predicate_3_ifregions_reorder() {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.A"
  // CHECK:          "tf.G"
  // CHECK-NEXT    "tf.IfRegion"
  // CHECK:          "tf.E"
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%8) ( {
      %4 = "tf.E"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.G"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that 3 IfRegions where 1st and 3rd IfRegions
// can't be merged due to an intermediate dep in the 2nd IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_intermediate_dep
func @same_predicate_3_ifregions_intermediate_dep() {
  // CHECK-COUNT-3:        "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ( {
      %4 = "tf.E"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.G"(%9) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = true } : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that 3 IfRegions where 1st and 3rd IfRegions
// can't be merged due to an intermediate side effecting IfRegion.

// CHECK-LABEL: func @same_predicate_3_ifregions_intermediate_side_effect
func @same_predicate_3_ifregions_intermediate_side_effect() {
  // CHECK-COUNT-3:   "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ( {
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.G"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check case for 4 IfRegions where 1st and 3rd IfRegions
// can be merged and ensure that side effect analysis is regenerated.

// CHECK-LABEL: func @side_effect_analysis_updated
func @side_effect_analysis_updated() {
  // CHECK-COUNT-3:   "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %8 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%8) ( {
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %5 = "tf.G"(%1) : (tensor<f32>) -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = false} : (tensor<i1>) -> ()
    "tf.IfRegion"(%8) ( {
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// Check that 2 IfRegions can be merged when the first IfRegion contains multiple side effecting ops.

// CHECK-LABEL: func @same_predicate_2_ifregions_multiple_side_effect_ops
func @same_predicate_2_ifregions_multiple_side_effect_ops() {
  // CHECK:       "tf.IfRegion"
  // CHECK-NOT:   "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      %3 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
      }, {
      %2 = "tf.C"() : () -> (tensor<f32>)
      "tf.Yield"(%2) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    %9 = "tf.IfRegion"(%0) ( {
      %4 = "tf.E"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
      }, {
      %4 = "tf.F"() : () -> (tensor<f32>)
      "tf.Yield"(%4) : (tensor<f32>) -> ()
    }) { is_stateless = false } : (tensor<i1>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}
