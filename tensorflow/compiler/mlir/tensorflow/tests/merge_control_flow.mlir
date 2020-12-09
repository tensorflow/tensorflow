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
  // CHECK-NOT:    "tf.IfRegion"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ( {
      %2 = "tf.A"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
    "tf.IfRegion"(%0) ( {
      %2 = "tf.B"() : () -> (tensor<f32>)
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
     }) {is_stateless = true} : (tensor<i1>) -> ()
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
