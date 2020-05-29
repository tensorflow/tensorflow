// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-extract-head-tail-outside-compilation | FileCheck %s --dump-input-on-failure

// Tests extraction of a outside compiled ops at head of TPU computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @head_single_outside_compiled_op
  func @head_single_outside_compiled_op(%arg0: tensor<i32>) {
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   "tf.A"
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ( {
      "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
      "tf.B"() : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    return
  }

  // CHECK-LABEL: func @head_single_outside_compiled_op_no_operands
  func @head_single_outside_compiled_op_no_operands() {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.B"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ( {
      %a = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
      "tf.C"(%b) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    return
  }

  // CHECK-LABEL: func @head_operand_op_outside_cluster
  func @head_operand_op_outside_cluster() {
    // CHECK:      %[[A_OUT:.*]] = "tf.A"
    %a = "tf.A"() : () -> tensor<i32>
    // CHECK-NEXT: %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.C"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   "tf.D"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ( {
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b) : (tensor<i32>) -> tensor<i32>
      "tf.D"(%c) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    return
  }

  // CHECK-LABEL: func @head_aliased_output
  func @head_aliased_output() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      %[[CLUSTER_OUT:.*]]:2 = "tf_device.cluster"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"
    // CHECK-NEXT:   tf_device.return %[[C_OUT]], %[[B_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  padding_map = []
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    %cluster:3 = "tf_device.cluster"() ( {
      %a = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b) : (tensor<i32>) -> tensor<i32>
      tf_device.return %a, %c, %b : tensor<i32>, tensor<i32>, tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> (tensor<i32>, tensor<i32>, tensor<i32>)
    // CHECK:      return %[[LAUNCH_OUT]], %[[CLUSTER_OUT]]#0, %[[CLUSTER_OUT]]#1
    return %cluster#0, %cluster#1, %cluster#2 : tensor<i32>, tensor<i32>, tensor<i32>
  }

  // CHECK-LABEL: func @head_all_cluster_op
  func @head_all_cluster_op(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"(%[[B_OUT]], %arg0)
    // CHECK-NEXT:   tf_device.return %[[C_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   tf_device.return
    %cluster = "tf_device.cluster"() ( {
      %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b, %arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %c : tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> tensor<i32>
    // CHECK:      return %[[LAUNCH_OUT]]
    return %cluster : tensor<i32>
  }

  // CHECK-LABEL: func @head_multiple_outside_compiled_ops
  func @head_multiple_outside_compiled_ops(%arg0: tensor<i32>) {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.D"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ( {
      %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      "tf.C"(%b, %arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.D"(%b) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    return
  }

  // CHECK-LABEL: func @head_replicated_outside_compilation
  func @head_replicated_outside_compilation(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // CHECK:      tf_device.replicate([%arg0, %arg1] as %[[RI:.*]]: tensor<i32>)
    //
    // CHECK-NEXT:   %[[LAUNCH_OUT:.*]] = "tf_device.launch"()
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"(%[[RI]])
    // CHECK-NEXT:     tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   device = "TPU_REPLICATED_HOST"
    //
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     "tf.B"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:     tf_device.return
    tf_device.replicate([%arg0, %arg1] as %ri : tensor<i32>) {n = 2 : i32} {
      "tf_device.cluster"() ( {
        %a = "tf.A"(%ri) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
        "tf.B"(%a) : (tensor<i32>) -> ()
        tf_device.return
      }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
      tf_device.return
    }
    return
  }
}
