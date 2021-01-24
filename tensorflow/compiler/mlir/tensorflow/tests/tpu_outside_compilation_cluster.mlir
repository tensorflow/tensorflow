// RUN: tf-opt %s -tf-tpu-outside-compilation-cluster | FileCheck %s

// CHECK-LABEL: func @one_cluster_no_dependencies
func @one_cluster_no_dependencies() {
  // CHECK: "tf.opA"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "{{[a-zA-Z_0-9]+}}"
  // CHECK: "tf.opC"
  "tf_device.cluster"() ( {
    "tf.opA"() : () -> ()
    "tf.opB"() {_xla_outside_compilation = "0"} : () -> ()
    "tf.opC"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @one_cluster_with_one_op
func @one_cluster_with_one_op() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "{{[a-zA-Z_0-9]+}}"
  // CHECK-NEXT: "tf.opC"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf.opC"(%b) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @one_cluster_with_two_ops
func @one_cluster_with_two_ops() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER2:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER2]]"
  // CHECK-NEXT: "tf.opD"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf.opD"(%c) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @one_cluster_with_three_ops
func @one_cluster_with_three_ops() {
  // CHECK: "tf.opA"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER3:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER3]]"
  // CHECK: "tf.opD"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER3]]"
  // CHECK: "tf.opE"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"(%b, %c) {_xla_outside_compilation = "0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.opE"(%d) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_no_dependencies
func @two_clusters_no_dependencies() {
  // CHECK: "tf.opA"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER4:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER5:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opD"
  "tf_device.cluster"() ( {
    "tf.opA"() : () -> ()
    "tf.opB"() {_xla_outside_compilation = "0"} : () -> ()
    "tf.opC"() {_xla_outside_compilation = "0"} : () -> ()
    "tf.opD"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_with_one_op_each
func @two_clusters_with_one_op_each() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER6:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opC"
  // CHECK-NEXT: "tf.opD"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER6]]"
  // CHECK-NEXT: "tf.opE"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"(%c) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf.opE"(%d) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_with_two_ops_each
func @two_clusters_with_two_ops_each() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER8:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER8]]"
  // CHECK-NEXT: "tf.opD"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER8]]"
  // CHECK-NEXT: "tf.opF"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER8]]"
  // CHECK-NEXT: "tf.opG"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"(%c) : (tensor<i32>) -> tensor<i32>
    %e = "tf.opE"(%d) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %f = "tf.opF"(%e) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf.opG"(%f) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_transitive_data_dependency
func @two_clusters_transitive_data_dependency() {
  // CHECK: "tf.opA"
  // CHECK: "tf.Const"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.Identity"
  // CHECK: "tf.AddV2"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    "tf.opA"() : () -> ()
    %1 = "tf.Const"() {_xla_outside_compilation = "0", value = dense<1.0> : tensor<f32>} : () -> (tensor<f32>)
    %2 = "tf.Identity"(%1) : (tensor<f32>) -> (tensor<f32>)
    "tf.AddV2"(%1, %2) {_xla_outside_compilation = "0"} : (tensor<f32>, tensor<f32>) -> (tensor<f32>)
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @resource_side_effect_cycle
func @resource_side_effect_cycle(%arg0: tensor<!tf.resource<tensor<f32>>>, %arg1: tensor<!tf.resource<tensor<f32>>>) {
  // CHECK: "tf.ReadVariableOp"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK-NEXT: "tf.AssignVariableOp"
  // CHECK-NOT:  {_xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    %read0 = "tf.ReadVariableOp"(%arg0) {_xla_outside_compilation = "0"} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %idet0 = "tf.Identity"(%read0) {_xla_outside_compilation = "0"} : (tensor<f32>) -> tensor<f32>
    "tf.AssignVariableOp"(%arg1, %idet0) : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
    %read1 = "tf.ReadVariableOp"(%arg1) {_xla_outside_compilation = "0"} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %idet1 = "tf.Identity"(%read1) {_xla_outside_compilation = "0"} : (tensor<f32>) -> tensor<f32>
    %add0 = "tf.AddV2"(%idet0, %idet1) {_xla_outside_compilation = "0"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tf.AssignVariableOp"(%arg0, %add0) {_xla_outside_compilation = "0"} : (tensor<!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_with_same_parent
func @two_clusters_with_same_parent() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER10:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opB"
  // CHECK-NEXT: "tf.opC"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER10]]"
  // CHECK-NEXT: "tf.opD"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER12:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NEXT: "tf.opF"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER12]]"
  // CHECK-NEXT: "tf.opG"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %b = "tf.opB"(%a) : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %e = "tf.opE"(%d) : (tensor<i32>) -> tensor<i32>
    %f = "tf.opF"(%e) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %g = "tf.opG"(%c, %f) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_with_same_outside_compiled_parent
func @two_clusters_with_same_outside_compiled_parent() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER12:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opB"
  // CHECK-NEXT: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER13:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opD"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER13]]"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: "tf.opF"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER13]]"
  // CHECK-NEXT: "tf.opG"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER13]]"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %b = "tf.opB"(%a) : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %e = "tf.Identity"(%d) : (tensor<i32>) -> tensor<i32>
    %f = "tf.opF"(%e) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %g = "tf.opG"(%c, %f) {_xla_outside_compilation = "0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @parent_with_a_non_outside_compiled_child
func @parent_with_a_non_outside_compiled_child() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER14:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER14]]"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %c = "tf.opC"(%a, %b) {_xla_outside_compilation = "0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @outside_compile_with_block
func @outside_compile_with_block() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER15:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER15]]"
  // CHECK: "tf.opC"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER15]]"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf_device.cluster" () ( {
      tf_device.return
    }) {cluster_attr = "cluster_attr"} : () -> ()
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @two_clusters_with_one_op_each_with_indirect_dependency
func @two_clusters_with_one_op_each_with_indirect_dependency() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER16:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opC"
  // CHECK-NEXT: "tf.opD"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER16]]"
  // CHECK-NEXT: "tf.opF"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"(%c) : (tensor<i32>) -> tensor<i32>
    %e = "tf.opE"(%d) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    "tf.opF"(%e) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @check_ops_with_data_dependency_added_as_host_cluster
func @check_ops_with_data_dependency_added_as_host_cluster() {
  // CHECK: "tf.opA"
  // CHECK-NEXT: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER16:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: "tf.Identity"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER16]]"
  // CHECK-NEXT: "tf.opF"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() : () -> tensor<i32>
    %b = "tf.opB"(%a) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %c = "tf.Identity"(%b) : (tensor<i32>) -> tensor<i32>
    %d = "tf.Identity"(%c) : (tensor<i32>) -> tensor<i32>
    %e = "tf.opE"(%d, %b, %c) {_xla_outside_compilation = "0"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "tf.opF"(%e) : (tensor<i32>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @check_op_inside_nested_region_clustered
func @check_op_inside_nested_region_clustered(%arg0 : tensor<*x!tf.resource>) {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-NEXT:     "tf.B"
  // CHECK-NEXT:     "tf.C"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK-NEXT:     "tf.WriteSummary"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17]]"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "tf.IfRegion"(%0) ( {
      %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %2 = "tf.B"() : () -> (tensor<i64>)
      %3 = "tf.C"() : () -> (tensor<f32>)
      %4 = "tf.Const"() {_xla_outside_compilation = "auto0", value = dense<"logits"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      %5 = "tf.Const"() {_xla_outside_compilation = "auto1", value = dense<"\0A\09\0A\07scalars"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      "tf.WriteSummary"(%arg0, %2, %3, %4, %5) {_xla_outside_compilation = "auto2", device = "/device:CPU:0"} : (tensor<*x!tf.resource>, tensor<i64>, tensor<f32>, tensor<!tf.string>, tensor<!tf.string>) -> ()
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }, {
      %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }) { is_stateless = true } : (tensor<i1>) -> tensor<i1>

    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @check_ops_inside_different_block_clustered
func @check_ops_inside_different_block_clustered(%arg0 : tensor<*x!tf.resource>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.B"
  // CHECK-SAME:   _xla_outside_compilation = "[[CLUSTER17:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT:   "tf.C"
  // CHECK-SAME:   _xla_outside_compilation = "[[CLUSTER18:[a-zA-Z_0-9]+]]"
  // CHECK:      "tf.IfRegion"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK-NEXT:     "tf.WriteSummary"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK:          "tf.Const"
  // CHECK-NEXT:     "tf.Const"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER18]]"
  // CHECK-NEXT:     "tf.D"
  // CHECK-SAME:     _xla_outside_compilation = "[[CLUSTER18]]"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.B"() {_xla_outside_compilation = "auto1"} : () -> (tensor<i64>)
    %3 = "tf.C"() {_xla_outside_compilation = "auto2"} : () -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %4 = "tf.Const"() {_xla_outside_compilation = "auto3", value = dense<"logits"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      %5 = "tf.Const"() {_xla_outside_compilation = "auto4", value = dense<"\0A\09\0A\07scalars"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      "tf.WriteSummary"(%arg0, %2, %3, %4, %5) {_xla_outside_compilation = "auto2", device = "/device:CPU:0"} : (tensor<*x!tf.resource>, tensor<i64>, tensor<f32>, tensor<!tf.string>, tensor<!tf.string>) -> ()
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }, {
      %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
      %4 = "tf.Const"() {_xla_outside_compilation = "auto5", value = dense<"a"> : tensor<!tf.string>} : () -> tensor<!tf.string>
      "tf.D"(%3, %4, %1) {_xla_outside_compilation = "auto6"} : (tensor<f32>, tensor<!tf.string>, tensor<i1>) -> ()
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }) { is_stateless = true } : (tensor<i1>) -> tensor<i1>

    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @check_clustering_ops_inside_nested_control_flow
func @check_clustering_ops_inside_nested_control_flow(%arg0 : tensor<*x!tf.resource>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.B"
  // CHECK-SAME:   _xla_outside_compilation = "[[CLUSTER17:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT:   "tf.C"
  // CHECK:        _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.IfRegion"
  // CHECK-NEXT:       "tf.Const"
  // CHECK-NEXT:       "tf.Const"
  // CHECK-SAME:       _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK-NEXT:       "tf.Const"
  // CHECK-SAME:       _xla_outside_compilation = "[[CLUSTER17]]"
  // CHECK-NEXT:       "tf.WriteSummary"
  // CHECK-SAME:       _xla_outside_compilation = "[[CLUSTER17]]"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.B"() {_xla_outside_compilation = "auto1"} : () -> (tensor<i64>)
    %3 = "tf.C"() {_xla_outside_compilation = "auto2"} : () -> (tensor<f32>)
    "tf.IfRegion"(%0) ( {
      %6 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      "tf.IfRegion"(%6) ( {
        %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
        %4 = "tf.Const"() {_xla_outside_compilation = "auto3", value = dense<"logits"> : tensor<!tf.string>} : () -> tensor<!tf.string>
        %5 = "tf.Const"() {_xla_outside_compilation = "auto4", value = dense<"\0A\09\0A\07scalars"> : tensor<!tf.string>} : () -> tensor<!tf.string>
        "tf.WriteSummary"(%arg0, %2, %3, %4, %5) {_xla_outside_compilation = "auto2", device = "/device:CPU:0"} : (tensor<*x!tf.resource>, tensor<i64>, tensor<f32>, tensor<!tf.string>, tensor<!tf.string>) -> ()
        "tf.Yield"(%1) : (tensor<i1>) -> ()
      }, {
        %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
        "tf.Yield"(%1) : (tensor<i1>) -> ()
      }) { is_stateless = true } : (tensor<i1>) -> tensor<i1>
      "tf.Yield"(%6) : (tensor<i1>) -> ()
    }, {
      %7 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
      "tf.Yield"(%7) : (tensor<i1>) -> ()
    }) { is_stateless = true } : (tensor<i1>) -> tensor<i1>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @single_variant_input
func @single_variant_input() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK: "tf.opC"
  "tf_device.cluster"() ( {
    %1= "tf.opA"() : () -> tensor<!tf.variant<tensor<f32>>>
    "tf.opB"(%1) {_xla_outside_compilation = "0"} : (tensor<!tf.variant<tensor<f32>>>) -> ()
    "tf.opC"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @chained_variant_input
func @chained_variant_input() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    %1 = "tf.opA"() : () -> tensor<!tf.variant<tensor<f32>>>
    %2 = "tf.opB"(%1) : (tensor<!tf.variant<tensor<f32>>>) -> (tensor<!tf.variant<tensor<f32>>>)
    "tf.opC"(%2) {_xla_outside_compilation = "0"} : (tensor<!tf.variant<tensor<f32>>>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @single_variant_output
func @single_variant_output() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK: "tf.opC"
  "tf_device.cluster"() ( {
    %1= "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<!tf.variant<tensor<f32>>>
    "tf.opB"(%1) : (tensor<!tf.variant<tensor<f32>>>) -> ()
    "tf.opC"() : () -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @chained_variant_output
func @chained_variant_output() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    %1 = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<!tf.variant<tensor<f32>>>
    %2 = "tf.opB"(%1) : (tensor<!tf.variant<tensor<f32>>>) -> (tensor<!tf.variant<tensor<f32>>>)
    "tf.opC"(%2) : (tensor<!tf.variant<tensor<f32>>>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @variant_input_output
func @variant_input_output() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK: "tf.opB"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  // CHECK: "tf.opC"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    %1 = "tf.opA"() : () -> tensor<!tf.variant<tensor<f32>>>
    %2 = "tf.opB"(%1) {_xla_outside_compilation = "0"} : (tensor<!tf.variant<tensor<f32>>>) -> (tensor<!tf.variant<tensor<f32>>>)
    "tf.opC"(%2) : (tensor<!tf.variant<tensor<f32>>>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @variant_input_nested
func @variant_input_nested(%arg0 : tensor<*x!tf.resource>) {
  // CHECK:      tf_device.cluster
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.C"
  // CHECK-SAME:   _xla_outside_compilation = "[[CLUSTER1:[a-zA-Z_0-9]+]]"
  // CHECK:        "tf.IfRegion"
  // CHECK:          "tf.opD"
  // CHECK:        _xla_outside_compilation = "[[CLUSTER1]]"
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %2 = "tf.C"() {_xla_outside_compilation = "auto0"} : () -> (tensor<!tf.variant<tensor<f32>>>)
    "tf.IfRegion"(%0) ( {
      %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      "tf.opD"(%2) : (tensor<!tf.variant<tensor<f32>>>) -> ()
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }, {
      %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
      "tf.Yield"(%1) : (tensor<i1>) -> ()
      }) { is_stateless = true, _xla_outside_compilation = "auto1" } : (tensor<i1>) -> tensor<i1>
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}

// CHECK-LABEL: func @variant_output_nested
func @variant_output_nested(%arg0 : tensor<*x!tf.resource>) {
  // CHECK:      tf_device.cluster
  // CHECK:        "tf.IfRegion"
  // CHECK:        "tf.C"
  // CHECK-NOT: _xla_outside_compilation
  // CHECK:        "tf.D"
  // CHECK-NOT: _xla_outside_compilation
  // CHECK:        "tf.Yield"
  // CHECK: _xla_outside_compilation
  "tf_device.cluster"() ( {
    %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %1 = "tf.IfRegion"(%0) ( {
      %2 = "tf.C"()  : () -> (tensor<!tf.variant<tensor<f32>>>)
      "tf.Yield"(%2) : (tensor<!tf.variant<tensor<f32>>>) -> ()
      }, {
      %2 = "tf.D"() : () -> (tensor<!tf.variant<tensor<f32>>>)
      "tf.Yield"(%2) : (tensor<!tf.variant<tensor<f32>>>) -> ()
      }) { is_stateless = true, _xla_outside_compilation = "auto1" } : (tensor<i1>) -> tensor<!tf.variant<tensor<f32>>>
    "tf.E"(%1) {_xla_outside_compilation = "auto0"} : (tensor<!tf.variant<tensor<f32>>>) -> ()
    tf_device.return
  }) {cluster_attr = "cluster_attr"} : () -> ()
  return
}
