// RUN: tf-opt %s -split-input-file -tf-tpu-space-to-depth-pass | FileCheck %s

// Tests for space to depth host and device transform.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:1" = {}, "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 390 : i32}} {
  func @main(%arg0: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg1: tensor<!tf.variant> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg2: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg3: tensor<!tf.variant> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg4: tensor<!tf.resource<tensor<7x7x3x64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg5: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg6: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg7: tensor<!tf.resource<tensor<i64>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) attributes {tf.entry_function = {control_outputs = "while", inputs = "iterator,iterator_1,iterator_2,iterator_3,while_input_6,while_input_7,while_input_8,while_input_9", outputs = ""}} {
    %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %3:10 = "tf.While"(%2, %1, %2, %0, %1, %arg2, %arg4, %arg5, %arg6, %arg7) {_lower_using_switch_merge = true, _num_original_outputs = 10 : i64, _read_only_resource_inputs = [], body = @while_body_2710, cond = @while_cond_2700, device = "", is_stateless = false, output_shapes = [#tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>, #tf.shape<>], parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>)
    return
  }
  // CHECK-LABEL: func @while_body_2710
  func @while_body_2710(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg6: tensor<!tf.resource<tensor<7x7x3x64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg7: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg8: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg9: tensor<!tf.resource<tensor<i64>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>) attributes {tf.signature.is_stateful} {
    %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // CHECK: %[[INPUT:.*]] = "tf.IteratorGetNext"
    %1 = "tf.IteratorGetNext"(%arg5) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf.resource>) -> tensor<2x224x224x3xf32>
    // CHECK-DAG: %[[SPACETODEPTH0:.*]] = "tf.SpaceToDepth"([[INPUT:.*]]) {block_size = 2 : i64, data_format = "NHWC"} : (tensor<2x224x224x3xf32>) -> tensor<2x112x112x12xf32>
    %2 = "tf.AddV2"(%arg2, %arg3) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tf.ReadVariableOp"(%arg6) : (tensor<!tf.resource<tensor<7x7x3x64xf32>>>) -> tensor<7x7x3x64xf32>
    %4 = "tf.ReadVariableOp"(%arg8) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %5 = "tf.ReadVariableOp"(%arg7) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
    %6 = "tf.ReadVariableOp"(%arg9) : (tensor<!tf.resource<tensor<i64>>>) -> tensor<i64>
    %7:2 = "tf_device.cluster_func"(%1, %3, %5, %6) {_tpu_replicate = "while/cluster_while_body_271", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [0, 0, 0, 0], func = @_func, host_compute_core = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", use_tpu = true} : (tensor<2x224x224x3xf32>, tensor<7x7x3x64xf32>, tensor<f32>, tensor<i64>) -> (tensor<7x7x3x64xf32>, tensor<i64>)
    "tf.AssignVariableOp"(%arg6, %7#0) : (tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<7x7x3x64xf32>) -> ()
    "tf.AssignVariableOp"(%arg9, %7#1) : (tensor<!tf.resource<tensor<i64>>>, tensor<i64>) -> ()
    %8 = "tf.Identity"(%arg1) {device = ""} : (tensor<i32>) -> tensor<i32>
    %9 = "tf.Identity"(%2) {device = ""} : (tensor<i32>) -> tensor<i32>
    %10 = "tf.AddV2"(%arg0, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %11 = "tf.Identity"(%10) {device = ""} : (tensor<i32>) -> tensor<i32>
    return %11, %8, %9, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>
  }
  func @while_cond_2700(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg6: tensor<!tf.resource<tensor<7x7x3x64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg7: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg8: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg9: tensor<!tf.resource<tensor<i64>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) -> tensor<i1> {
    %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.GreaterEqual"(%arg3, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = "tf.Less"(%arg3, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tf.Greater"(%arg2, %arg4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = "tf.LogicalAnd"(%2, %3) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = "tf.Less"(%arg2, %arg4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = "tf.LogicalAnd"(%1, %5) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = "tf.LogicalOr"(%6, %4) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = "tf.Less"(%arg0, %arg1) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %9 = "tf.LogicalAnd"(%8, %7) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = "tf.Identity"(%9) {device = ""} : (tensor<i1>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  // CHECK-LABEL: func @_func
  // CHECK-SAME: [[FUNCINPUT0:.*]]: tensor<2x112x112x12xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[FUNCINPUT1:%.*]]: tensor<7x7x3x64xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[FUNCINPUT2:%.*]]: tensor<f32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[VAL_59:%.*]]: tensor<i64> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<7x7x3x64xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<i64> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}) attributes {sym_visibility = "private"} {
  func @_func(%arg0: tensor<2x224x224x3xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<7x7x3x64xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<f32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<i64> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<7x7x3x64xf32> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<i64> {xla_hlo.sharding = "\08\01\1A\01\01\22\01\00"}) attributes {sym_visibility = "private"} {
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf.Const"() {value = dense<0> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
    %2 = "tf.Const"() {value = dense<[7, 7, 3, 64]> : tensor<4xi32>} : () -> tensor<4xi32>
    %3 = "tf.Const"() {value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %5 = "tf.Pad"(%arg0, %3) : (tensor<2x224x224x3xf32>, tensor<4x2xi32>) -> tensor<2x230x230x3xf32>
    // CHECK: "tf.Conv2D"
    // CHECK-SAME: strides = [1, 1, 1, 1]
    // CHECK-SAME: (tensor<2x115x115x12xf32>, tensor<4x4x12x64xf32>) -> tensor<2x112x112x64xf32>
    %6 = "tf.Conv2D"(%5, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<2x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<2x112x112x64xf32>
    // CHECK: %[[BACKPROP:.*]] = "tf.Conv2DBackpropFilter"
    // CHECK-SAME: strides = [1, 1, 1, 1]
    // CHECK-SAME: (tensor<2x115x115x12xf32>, tensor<4xi32>, tensor<2x112x112x64xf32>) -> tensor<4x4x12x64xf32>
    %7 = "tf.Conv2DBackpropFilter"(%5, %2, %6) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<2x230x230x3xf32>, tensor<4xi32>, tensor<2x112x112x64xf32>) -> tensor<7x7x3x64xf32>
    // CHECK: %[[CONST0:.*]] = "tf.Const"() {value = dense<
    // CHECK-SAME: [4, 4, 2, 2, 3, 64]
    // CHECK: %[[RESHAPE0:.*]] = "tf.Reshape"(%[[BACKPROP:.*]], %[[CONST0:.*]]) : (tensor<4x4x12x64xf32>, tensor<6xi64>) -> tensor<4x4x2x2x3x64xf32>
    // CHECK: %[[CONST1:.*]] = "tf.Const"() {value = dense<
    // CHECK-SAME: [0, 2, 1, 3, 4, 5]
    // CHECK: %[[TRANSPOSE0:.*]] = "tf.Transpose"(%[[RESHAPE0:.*]], %[[CONST1:.*]]) : (tensor<4x4x2x2x3x64xf32>, tensor<6xi32>) -> tensor<4x2x4x2x3x64xf32>
    // CHECK: %[[CONST2:.*]] = "tf.Const"() {value = dense<
    // CHECK-SAME: [8, 8, 3, 64]
    // CHECK: %[[RESHAPE1:.*]] = "tf.Reshape"(%[[TRANSPOSE1:.*]], %[[CONST2:.*]]) : (tensor<4x2x4x2x3x64xf32>, tensor<4xi64>) -> tensor<8x8x3x64xf32>
    // CHECK: %[[CONST3:.*]] = "tf.Const"() {value = dense<
    // CHECK-SAME: [7, 7, 3, 64]
    // CHECK: %[[CONST4:.*]] = "tf.Const"() {value = dense<
    // CHECK-SAME: 0
    // CHECK: %[[SLICE0:.*]] = "tf.Slice"(%[[RESHAPE1:.*]], %[[CONST4:.*]], %[[CONST3:.*]]) : (tensor<8x8x3x64xf32>, tensor<4xi64>, tensor<4xi32>) -> tensor<7x7x3x64xf32>
    %8 = "tf.CrossReplicaSum"(%7, %1) : (tensor<7x7x3x64xf32>, tensor<1x1xi32>) -> tensor<7x7x3x64xf32>
    %9 = "tf.Mul"(%arg2, %8) : (tensor<f32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %10 = "tf.Sub"(%arg1, %9) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %11 = "tf.AddV2"(%arg3, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %10, %11 : tensor<7x7x3x64xf32>, tensor<i64>
  }
}

// ----

