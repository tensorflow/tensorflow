// RUN: tf-opt -cluster-ops-by-policy="oplist=tf.Neg,tf.Add,tf.Neg policy-name=foo" %s -print-ir-after-all | FileCheck %s

// CHECK-LABEL: func @multiple_clusters

// Tests general cluster formation with and without device attribute.

func @multiple_clusters(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Add"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) {device="cpu"} : (tensor<i32>) -> tensor<i32>
  %4 = "tf.Add"(%3, %arg1) {device="cpu"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %5 = "tf.Neg"(%4) {device="cpu"} : (tensor<i32>) -> tensor<i32>
  // CHECK:      %[[RESULT_0:.*]] = "tf_device.cluster"() ( {
  // CHECK-NEXT:   "tf.Neg"
  // CHECK-NEXT:   "tf.Add"
  // CHECK-NEXT:   %[[RESULT_1:.*]] = "tf.Neg"
  // CHECK-NEXT:   tf_device.return %[[RESULT_1:.*]]
  // CHECK-NOT:  {device="cpu"}
  // CHECK:      %[[RESULT_2:.*]] = "tf_device.cluster"() ( {
  // CHECK-NEXT:   "tf.Neg"
  // CHECK-NEXT:   "tf.Add"
  // CHECK-NEXT:   %[[RESULT_3:.*]] = "tf.Neg"
  // CHECK-NEXT:   tf_device.return %[[RESULT_3:.*]]
  // CHECK-NEXT:   }) {device = "cpu", policy = "foo"}
  // CHECK:      return %[[RESULT_2:.*]]
  return %5 : tensor<i32>
}

// Tests cluster formation for non-consequitive operations.

// CHECK-LABEL: func @op_in_between
func @op_in_between(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.A"(%arg1) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK:      "tf.A"
  // CHECK-NEXT: tf_device.cluster
  return %3 : tensor<i32>
}

// Tests that we form a single cluster when matching sequences overlap.

// CHECK-LABEL: func @match_overlap
func @match_overlap(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Add"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
  %3 = "tf.Add"(%0, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.Neg"(%3) : (tensor<i32>) -> tensor<i32>
  // CHECK: tf_device.cluster
  // CHECK-NOT: tf_device.cluster
  return %4 : tensor<i32>
}

// Tests that cluster is not formed when a matching op has more than one use.

// CHECK-LABEL: func @multiple_uses
func @multiple_uses(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Add"(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.Sub"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK-NOT: tf_device.cluster
  return %3 : tensor<i32>
}

// Tests that we form a single cluster when maching sequences share a tail.

// CHECK-LABEL: func @double_head
func @double_head(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%arg1) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Neg"(%2) : (tensor<i32>) -> tensor<i32>
  // CHECK: tf_device.cluster
  // CHECK-NOT: tf_device.cluster
  return %3 : tensor<i32>
}


