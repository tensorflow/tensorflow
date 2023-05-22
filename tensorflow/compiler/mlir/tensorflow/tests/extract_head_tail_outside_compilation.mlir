// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-extract-head-tail-outside-compilation | FileCheck %s

// Tests extraction of a outside compiled ops at head of TPU computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @head_single_outside_compiled_op
  func.func @head_single_outside_compiled_op(%arg0: tensor<i32>) {
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
      "tf.B"() : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @head_single_outside_compiled_op_no_operands
  func.func @head_single_outside_compiled_op_no_operands() {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.B"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      %a = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
      "tf.C"(%b) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @head_operand_op_outside_cluster
  func.func @head_operand_op_outside_cluster() {
    // CHECK:      %[[A_OUT:.*]] = "tf.A"
    %a = "tf.A"() : () -> tensor<i32>
    // CHECK-NEXT: %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.C"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   "tf.D"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b) : (tensor<i32>) -> tensor<i32>
      "tf.D"(%c) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @head_aliased_output
  func.func @head_aliased_output() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
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
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    %cluster:3 = "tf_device.cluster"() ({
      %a = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b) : (tensor<i32>) -> tensor<i32>
      tf_device.return %a, %c, %b : tensor<i32>, tensor<i32>, tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> (tensor<i32>, tensor<i32>, tensor<i32>)
    // CHECK:      return %[[LAUNCH_OUT]], %[[CLUSTER_OUT]]#0, %[[CLUSTER_OUT]]#1
    func.return %cluster#0, %cluster#1, %cluster#2 : tensor<i32>, tensor<i32>, tensor<i32>
  }

  // CHECK-LABEL: func @head_all_cluster_op
  func.func @head_all_cluster_op(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"(%[[B_OUT]], %arg0)
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[C_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   tf_device.return
    %cluster = "tf_device.cluster"() ({
      %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b, %arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %c : tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> tensor<i32>
    // CHECK:      return %[[LAUNCH_OUT]]
    func.return %cluster : tensor<i32>
  }

  // CHECK-LABEL: func @head_multiple_outside_compiled_ops
  func.func @head_multiple_outside_compiled_ops(%arg0: tensor<i32>) {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   "tf.C"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.D"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      "tf.C"(%b, %arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> ()
      "tf.D"(%b) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @head_replicated_outside_compilation
  func.func @head_replicated_outside_compilation(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // CHECK:      tf_device.replicate([%arg0, %arg1] as %[[RI:.*]]: tensor<i32>)
    //
    // CHECK-NEXT:   %[[LAUNCH_OUT:.*]] = "tf_device.launch"()
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"(%[[RI]])
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK-NEXT:     tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   device = "TPU_REPLICATED_HOST_0"
    //
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:     "tf.B"(%[[LAUNCH_OUT]])
    // CHECK-NEXT:     tf_device.return
    tf_device.replicate([%arg0, %arg1] as %ri : tensor<i32>) {n = 2 : i32} {
      "tf_device.cluster"() ({
        %a = "tf.A"(%ri) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
        "tf.B"(%a) : (tensor<i32>) -> ()
        tf_device.return
      }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
      tf_device.return
    }
    func.return
  }

  // CHECK-LABEL: func @head_while_op
  func.func @head_while_op(%arg0: tensor<i32>) {
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[A_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK:        %[[C_OUT:.*]] = "tf.C"(%[[WHILE_OUT]]#1)
    // CHECK:        tf_device.return %[[C_OUT]]
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.D"(%[[LAUNCH_OUT]])
    "tf_device.cluster"() ({
      %a = "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<f32>
      %w1, %w2 = "tf.WhileRegion"(%a, %b) ({
      ^bb0(%arg1: tensor<i32>, %arg2: tensor<f32>):
        %7 = "tf.H"(%arg1) :  (tensor<i32>) -> tensor<i1>
        "tf.Yield"(%7) : (tensor<i1>) -> ()
      }, {
      ^bb0(%arg1: tensor<i32>, %arg2: tensor<f32>):
        %8 = "tf.C"(%arg1) : (tensor<i32>) -> tensor<i32>
        %9 = "tf.D"(%arg1, %arg2) : (tensor<i32>, tensor<f32>) -> tensor<f32>
        "tf.Yield"(%8, %9) : (tensor<i32>, tensor<f32>) -> ()
      }) { is_stateless = false, _xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
      %c = "tf.C"(%w2) {_xla_outside_compilation = "cluster1"} : (tensor<f32>) -> (tensor<f32>)
      "tf.D"(%c) : (tensor<f32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @tail_single_outside_compiled_op
  func.func @tail_single_outside_compiled_op() {
    // CHECK:      %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   "tf.NoOp"
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   "tf.B"(%[[CLUSTER_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<i32>
      "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
      "tf.NoOp"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @tail_single_outside_compiled_op_user
  func.func @tail_single_outside_compiled_op_user() -> tensor<i32> {
    // CHECK:      %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   "tf.NoOp"
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[CLUSTER_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    %cluster = "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      "tf.NoOp"() : () -> ()
      tf_device.return %b : tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> tensor<i32>
    // CHECK:      return %[[LAUNCH_OUT]]
    func.return %cluster : tensor<i32>
  }

  // CHECK-LABEL: func @tail_multiple_outside_compiled_ops
  func.func @tail_multiple_outside_compiled_ops(%arg0: tensor<i32>) {
    // CHECK:      %[[CLUSTER_OUT:.*]]:2 = "tf_device.cluster"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:   tf_device.return %[[B_OUT]], %[[A_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"(%arg0, %[[CLUSTER_OUT]]#1)
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   "tf.D"(%[[C_OUT]], %arg0, %[[CLUSTER_OUT]]#0)
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<i32>
      %b = "tf.B"(%arg0) : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%arg0, %a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.D"(%c, %arg0, %b) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @tail_multiple_nested_outside_compiled_ops
  func.func @tail_multiple_nested_outside_compiled_ops(%arg0: tensor<i32>) {
    // CHECK:      %[[CLUSTER_OUT:.*]]:3 = "tf_device.cluster"
    // CHECK-NEXT:   %[[CONST_OUT:.*]] = "tf.Const"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:   tf_device.return %[[B_OUT]], %[[CONST_OUT]], %[[A_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  padding_map = []
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"(%arg0, %[[CLUSTER_OUT]]#2)
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK         "tf.IfRegion"
    // CHECK:          "tf.D"(%[[C_OUT]], %arg0, %[[CLUSTER_OUT]]#0)
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK:        tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      %0 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %a = "tf.A"() : () -> tensor<i32>
      %b = "tf.B"(%arg0) : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%arg0, %a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.IfRegion"(%0) ({
        "tf.D"(%c, %arg0, %b) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
      "tf.Yield"() : () -> ()
      }, {
      "tf.Yield"() : () -> ()
      }) {is_stateless = true, _xla_outside_compilation = "cluster1"} : (tensor<i1>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", padding_map = [], topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @tail_aliased_output
  func.func @tail_aliased_output() -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
    // CHECK-NEXT: %[[A_OUT:.*]] = "tf.A"
    %a = "tf.A"() : () -> tensor<i32>
    // CHECK-NEXT: %[[B_OUT:.*]] = "tf.B"
    %b = "tf.B"() : () -> tensor<i32>
    // CHECK:      %[[CLUSTER_OUT:.*]]:2 = "tf_device.cluster"
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"
    // CHECK-NEXT:   %[[E_OUT:.*]] = "tf.Const"
    // CHECK-NEXT:   tf_device.return %[[C_OUT]], %[[E_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      %[[LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[D_OUT:.*]] = "tf.D"(%[[CLUSTER_OUT]]#0, %[[A_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    %cluster:5 = "tf_device.cluster"() ({
      %c = "tf.C"()  : () -> tensor<i32>
      %d = "tf.D"(%c, %a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %e = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      tf_device.return %a, %b, %c, %d, %e : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)
    // CHECK:      return %[[A_OUT]], %[[B_OUT]], %[[CLUSTER_OUT]]#0, %[[LAUNCH_OUT]], %[[CLUSTER_OUT]]#1
    func.return %cluster#0, %cluster#1, %cluster#2, %cluster#3, %cluster#4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }

  // CHECK-LABEL: func @tail_replicated_outside_compilation
  func.func @tail_replicated_outside_compilation(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // CHECK:      tf_device.replicate([%arg0, %arg1] as %[[RI:.*]]: tensor<i32>)
    //
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"(%[[RI]])
    // CHECK-NEXT:     tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   {
    // CHECK-DAG:    num_cores_per_replica = 1
    // CHECK-DAG:    step_marker_location = ""
    // CHECK-DAG:    topology = ""
    // CHECK-DAG:    device_assignment = []
    //
    // CHECK-NEXT:   "tf_device.launch"()
    // CHECK-NEXT:     %[[B_OUT:.*]] = "tf.B"(%[[CLUSTER_OUT]], %[[RI]])
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   device = "TPU_REPLICATED_HOST_0"
    tf_device.replicate([%arg0, %arg1] as %ri : tensor<i32>) {n = 2 : i32} {
      "tf_device.cluster"() ({
        %a = "tf.A"(%ri) : (tensor<i32>) -> tensor<i32>
        %b = "tf.B"(%a, %ri) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        tf_device.return
      }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
      tf_device.return
    }
    func.return
  }

  // CHECK-LABEL: func @head_tail_no_extraction_middle_outside_compiled_ops
  func.func @head_tail_no_extraction_middle_outside_compiled_ops(%arg0: tensor<i32>) {
    // CHECK-NOT:  "tf_device.launch"
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.Identity"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   "tf.Identity"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      %a = "tf.Identity"(%arg0) : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %c = "tf.Identity"(%b) : (tensor<i32>) -> tensor<i32>
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @head_tail_simple_extraction
  func.func @head_tail_simple_extraction(%arg0: tensor<i32>) -> tensor<i32> {
    // CHECK:      %[[HEAD_LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"(%arg0)
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:   %[[B_OUT:.*]] = "tf.B"(%[[HEAD_LAUNCH_OUT]])
    // CHECK-NEXT:   tf_device.return %[[B_OUT]]
    // CHECK-NEXT: {
    // CHECK-DAG:  num_cores_per_replica = 1
    // CHECK-DAG:  step_marker_location = ""
    // CHECK-DAG:  topology = ""
    // CHECK-DAG:  device_assignment = []
    //
    // CHECK:      %[[TAIL_LAUNCH_OUT:.*]] = "tf_device.launch"
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"(%[[CLUSTER_OUT]])
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return %[[C_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    %cluster = "tf_device.cluster"() ({
      %a = "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      %b = "tf.B"(%a) : (tensor<i32>) -> tensor<i32>
      %c = "tf.C"(%b) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
      tf_device.return %c : tensor<i32>
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> tensor<i32>
    // CHECK:      return %[[TAIL_LAUNCH_OUT]]
    func.return %cluster : tensor<i32>
  }

  // CHECK-LABEL: func @head_tail_replicated_outside_compilation
  func.func @head_tail_replicated_outside_compilation(%arg0: tensor<i32>, %arg1: tensor<i32>) {
    // CHECK:      tf_device.replicate([%arg0, %arg1] as %[[RI:.*]]: tensor<i32>)
    //
    // CHECK-NEXT:   %[[HEAD_LAUNCH_OUT:.*]] = "tf_device.launch"()
    // CHECK-NEXT:     %[[A_OUT:.*]] = "tf.A"(%[[RI]])
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK-NEXT:     tf_device.return %[[A_OUT]]
    // CHECK-NEXT:   device = "TPU_REPLICATED_HOST_0"
    //
    // CHECK:        %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:     %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:     %[[C_OUT:.*]] = "tf.C"(%[[RI]], %[[B_OUT]])
    // CHECK-NEXT:     "tf.IdentityN"(%[[C_OUT]], %[[HEAD_LAUNCH_OUT]])
    // CHECK-NEXT:     tf_device.return %[[C_OUT]]
    // CHECK-NEXT:   {
    // CHECK-DAG:    num_cores_per_replica = 1
    // CHECK-DAG:    step_marker_location = ""
    // CHECK-DAG:    topology = ""
    // CHECK-DAG:    device_assignment = []
    //
    // CHECK-NEXT:   "tf_device.launch"()
    // CHECK-NEXT:     "tf.D"(%[[HEAD_LAUNCH_OUT]], %[[CLUSTER_OUT]], %[[RI]])
    // CHECK-NOT:      _xla_outside_compilation
    // CHECK-NEXT:     tf_device.return
    // CHECK-NEXT:   device = "TPU_REPLICATED_HOST_0"
    tf_device.replicate([%arg0, %arg1] as %ri : tensor<i32>) {n = 2 : i32} {
      "tf_device.cluster"() ({
        %a = "tf.A"(%ri) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> tensor<i32>
        %b = "tf.B"() : () -> tensor<i32>
        %c = "tf.C"(%ri, %b) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
        %d = "tf.D"(%a, %c, %ri) {_xla_outside_compilation = "cluster1"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
        %e:2 = "tf.IdentityN"(%c, %a) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
        tf_device.return
      }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
      tf_device.return
    }
    func.return
  }

  // CHECK-LABEL: func @side_effect_middle
  func.func @side_effect_middle() {
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.A"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      "tf.A"() : () -> ()
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @side_effect_head_no_operand
  func.func @side_effect_head_no_operand() {
    // CHECK:      %[[HEAD_LAUNCH_OUT:.*]] = "tf_device.launch"()
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   %[[C_OUT:.*]] = "tf.C"
    // CHECK-NEXT:   tf_device.return %[[C_OUT]]
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.Const"
    // CHECK-NEXT:   "tf.D"(%[[HEAD_LAUNCH_OUT]])
    // CHECK-NEXT:   tf_device.return

    "tf_device.cluster"() ({
      %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      "tf.B"() {_xla_outside_compilation = "cluster1"} : () -> ()
      %c = "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> tensor<i32>
      "tf.D"(%c) : (tensor<i32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // CHECK-LABEL: func @side_effect_tail_no_operand
  func.func @side_effect_tail_no_operand() {
    // CHECK:      %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
    // CHECK-NEXT:   %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:   "tf.Const"
    // CHECK-NEXT:   tf_device.return %[[A_OUT]]

    // CHECK:      "tf_device.launch"()
    // CHECK-NEXT:   "tf.B"(%[[CLUSTER_OUT]])
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      %a = "tf.A"() : () -> tensor<i32>
      "tf.B"(%a) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
      "tf.C"() {_xla_outside_compilation = "cluster1"} : () -> ()
      %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // Test embedding ops can be head extracted and side effect analysis
  // predecessors are ignored.

  // CHECK-LABEL: func @embedding_head_extraction
  func.func @embedding_head_extraction(%arg0: tensor<!tf_type.string>) {
    // CHECK:      "tf_device.launch"()
    // CHECK-NEXT:   "tf.EnqueueTPUEmbeddingRaggedTensorBatch"
    // CHECK-NEXT:   "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.UnknownOp"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      "tf.UnknownOp"() : () -> ()
      "tf.EnqueueTPUEmbeddingRaggedTensorBatch"(%arg0) {_xla_outside_compilation = "cluster1", table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
      "tf.EnqueueTPUEmbeddingArbitraryTensorBatch"(%arg0) {_xla_outside_compilation = "cluster1", table_ids = [1, 2]} : (tensor<!tf_type.string>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // Test side effecting op after embedding op can be head extracted.

  // CHECK-LABEL: func @op_after_embedding_head_extraction
  func.func @op_after_embedding_head_extraction() {
    // CHECK:      "tf_device.launch"()
    // CHECK-NEXT:   "tf.A"
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"

    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.RecvTPUEmbeddingActivations"
    // CHECK-NEXT:   "tf.SendTPUEmbeddingGradients"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      %0 = "tf.RecvTPUEmbeddingActivations"() {config = "test_config_recv_embedding"} : () -> tensor<512x256xf32>
      "tf.SendTPUEmbeddingGradients"(%0) {N = 1 : i64, NN = 0 : i64, config = "test_config_send_embedding", operand_segment_sizes = array<i32: 1, 0>} : (tensor<512x256xf32>) -> ()
      "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }

  // Test side effecting op before embedding op can be tail extracted.

  // CHECK-LABEL: func @op_before_embedding_tail_extraction
  func.func @op_before_embedding_tail_extraction() {
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.UnknownOp"
    // CHECK-NEXT:   "tf.RecvTPUEmbeddingActivations"
    // CHECK-NEXT:   "tf.SendTPUEmbeddingGradients"
    // CHECK-NEXT:   tf_device.return

    // CHECK:      "tf_device.launch"()
    // CHECK-NEXT:   "tf.A"
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:worker/replica:0/task:0/device:CPU:0"
    "tf_device.cluster"() ({
      "tf.UnknownOp"() : () -> ()
      "tf.A"() {_xla_outside_compilation = "cluster1"} : () -> ()
      %0 = "tf.RecvTPUEmbeddingActivations"() {config = "test_config_recv_embedding"} : () -> tensor<512x256xf32>
      "tf.SendTPUEmbeddingGradients"(%0) {N = 1 : i64, NN = 0 : i64, config = "test_config_send_embedding", operand_segment_sizes = array<i32: 1, 0>} : (tensor<512x256xf32>) -> ()
      tf_device.return
    }) {num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = []} : () -> ()
    func.return
  }
}

// -----
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0"]} {
  // CHECK-LABEL: func @head_single_outside_compiled_op_in_generic_pipeline
  func.func @head_single_outside_compiled_op_in_generic_pipeline(%arg0: tensor<i32>) {
    // CHECK:      "tf_device.launch"
    // CHECK-NEXT:   "tf.A"
    // CHECK-NOT:    _xla_outside_compilation
    // CHECK-NEXT:   tf_device.return
    // CHECK-NEXT: device = "/job:localhost/replica:0/task:0/device:CPU:0"
    //
    // CHECK:      "tf_device.cluster"
    // CHECK-NEXT:   "tf.B"
    // CHECK-NEXT:   "tf.C"
    // CHECK-NEXT:   tf_device.return
    "tf_device.cluster"() ({
      "tf.A"(%arg0) {_xla_outside_compilation = "cluster1"} : (tensor<i32>) -> ()
      "tf.B"() : () -> ()
      "tf.C"() : () -> ()
      tf_device.return
    }) : () -> ()
    func.return
  }
}
