// RUN: tf-opt %s -tf-tpu-outside-compilation-cluster | FileCheck %s --dump-input-on-failure

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
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER4]]"
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
  // CHECK-SAME: _xla_outside_compilation = "{{[a-zA-Z_0-9]+}}"
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
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER9:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opF"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER9]]"
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

// CHECK-LABEL: func @two_clusters_with_same_parent
func @two_clusters_with_same_parent() {
  // CHECK: "tf.opA"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER10:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opB"
  // CHECK-NEXT: "tf.opC"
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER10]]"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER11:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opD"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER10]]"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NEXT: "tf.opF"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER11]]"
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
  // CHECK-NOT: _xla_outside_compilation = "[[CLUSTER12]]"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER13:[a-zA-Z_0-9]+]]"
  // CHECK-NEXT: "tf.opD"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER12]]"
  // CHECK-NEXT: "tf.opE"
  // CHECK-NEXT: "tf.opF"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER13]]"
  // CHECK-NEXT: "tf.opG"
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER13]]"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %b = "tf.opB"(%a) : (tensor<i32>) -> tensor<i32>
    %c = "tf.opC"(%b) {_xla_outside_compilation = "0"} : (tensor<i32>) -> tensor<i32>
    %d = "tf.opD"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %e = "tf.opE"(%d) : (tensor<i32>) -> tensor<i32>
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
  // CHECK-SAME: _xla_outside_compilation = "[[CLUSTER15]]"
  "tf_device.cluster"() ( {
    %a = "tf.opA"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    %b = "tf.opB"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
    "tf_device.cluster" () ( {
      tf_device.return
    }) {cluster_attr = "cluster_attr"} : () -> ()
    %c = "tf.opC"() {_xla_outside_compilation = "0"} : () -> tensor<i32>
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
  // CHECK-SAME: _xla_outside_compilation = "{{[a-zA-Z_0-9]+}}"
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
