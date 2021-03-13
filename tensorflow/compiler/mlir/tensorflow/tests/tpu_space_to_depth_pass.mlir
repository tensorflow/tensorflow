// RUN: tf-opt %s -split-input-file -tf-tpu-space-to-depth-pass | FileCheck %s

// Tests for space to depth host and device transform.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:1" = {}, "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 390 : i32}} {
  func @main(%arg0: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg1: tensor<!tf.variant> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg2: tensor<!tf.resource> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg3: tensor<!tf.variant> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg4: tensor<!tf.resource<tensor<7x7x3x64xf32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg5: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg6: tensor<!tf.resource<tensor<f32>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}, %arg7: tensor<!tf.resource<tensor<i64>>> {tf.device = "/job:localhost/replica:0/task:0/device:TPU:0"}) attributes {tf.entry_function = {control_outputs = "while", inputs = "iterator,iterator_1,iterator_2,iterator_3,while_input_6,while_input_7,while_input_8,while_input_9", outputs = ""}} {
    %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %3:10 = "tf.While"(%2, %1, %2, %0, %1, %arg2, %arg4, %arg5, %arg6, %arg7) {_lower_using_switch_merge = true, _num_original_outputs = 10 : i64, _read_only_resource_inputs = [], body = @while_body_2710, cond = @while_cond_2700, device = "", is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<!tf.resource<tensor<7x7x3x64xf32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<f32>>>, tensor<!tf.resource<tensor<i64>>>)
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
  // CHECK-LABEL: func private @_func
  // CHECK-SAME: [[FUNCINPUT0:.*]]: tensor<2x112x112x12xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[FUNCINPUT1:%.*]]: tensor<7x7x3x64xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[FUNCINPUT2:%.*]]: tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, [[VAL_59:%.*]]: tensor<i64> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<7x7x3x64xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<i64> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
  func private @_func(%arg0: tensor<2x224x224x3xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<7x7x3x64xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<i64> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<7x7x3x64xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<i64> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
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

// -----

// Tests for space to depth host and device transform with replicate inputs.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:COMPOSITE:0" = {}, "/job:localhost/replica:0/task:0/device:CPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:1" = {}, "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 458 : i32}} {
  func @main(%arg0: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg1: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg2: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg3: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg4: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg5: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg6: tensor<*x!tf.resource<tensor<7x7x3x64xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg7: tensor<*x!tf.resource<tensor<64x1001xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg8: tensor<*x!tf.resource<tensor<1001xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg9: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg10: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg11: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg12: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}) attributes {tf.entry_function = {control_outputs = "IteratorGetNext,IteratorGetNext_1,CrossReplicaSum,AssignAddVariableOp,CrossReplicaSum_1,AssignAddVariableOp_1,CrossReplicaSum_2,AssignAddVariableOp_2,CrossReplicaSum_3,AssignAddVariableOp_3", inputs = "iterator,iterator_1,iterator_2,iterator_3,iterator_4,iterator_5,resnet50_conv1_conv2d_conv1_kernel_140365606309224_handle_inputs_0,resnet50_fc1000_matmul_fc1000_kernel_140365944145960_handle_inputs_0,resnet50_fc1000_biasadd_fc1000_bias_140365944146240_handle_inputs_0,total_140366323758976_handle_inputs_0,count_140366323759312_handle_inputs_0,total_140366323760264_handle_inputs_0,count_140366323760600_handle_inputs_0", outputs = ""}} {
    // CHECK: %[[INPUT00:.*]] = "tf.IteratorGetNext"
    // CHECK-DAG: %[[SPACETODEPTH00:.*]] = "tf.SpaceToDepth"([[INPUT00:.*]]#0) {block_size = 2 : i64, data_format = "NHWC"} : (tensor<2x224x224x3xf32>) -> tensor<2x112x112x12xf32>
    %0:2 = "tf.IteratorGetNext"(%arg2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*x!tf.resource>) -> (tensor<2x224x224x3xf32>, tensor<2x1xf32>)
    // CHECK: %[[INPUT01:.*]] = "tf.IteratorGetNext"
    // CHECK-DAG: %[[SPACETODEPTH01:.*]] = "tf.SpaceToDepth"([[INPUT01:.*]]#0) {block_size = 2 : i64, data_format = "NHWC"} : (tensor<2x224x224x3xf32>) -> tensor<2x112x112x12xf32>
    %1:2 = "tf.IteratorGetNext"(%arg4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*x!tf.resource>) -> (tensor<2x224x224x3xf32>, tensor<2x1xf32>)
    tf_device.replicate([%0#0, %1#0] as %arg13: tensor<2x224x224x3xf32>, [%0#1, %1#1] as %arg14: tensor<2x1xf32>, %arg6 as %arg15: tensor<*x!tf.resource<tensor<7x7x3x64xf32>>>, %arg8 as %arg16: tensor<*x!tf.resource<tensor<1001xf32>>>, %arg7 as %arg17: tensor<*x!tf.resource<tensor<64x1001xf32>>>, %arg9 as %arg18: tensor<*x!tf.resource<tensor<f32>>>, %arg10 as %arg19: tensor<*x!tf.resource<tensor<f32>>>, %arg11 as %arg20: tensor<*x!tf.resource<tensor<f32>>>, %arg12 as %arg21: tensor<*x!tf.resource<tensor<f32>>>) {_mirrored_variable_indices = [2, 3, 4, 5, 6, 7, 8], _replicated_input_indices = [1, 2, -1, -1, -1, -1, -1, -1, -1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
      %2 = "tf.ReadVariableOp"(%arg15) : (tensor<*x!tf.resource<tensor<7x7x3x64xf32>>>) -> tensor<7x7x3x64xf32>
      %3 = "tf.ReadVariableOp"(%arg16) : (tensor<*x!tf.resource<tensor<1001xf32>>>) -> tensor<1001xf32>
      %4 = "tf.ReadVariableOp"(%arg17) : (tensor<*x!tf.resource<tensor<64x1001xf32>>>) -> tensor<64x1001xf32>
      %5 = "tf.ReadVariableOp"(%arg18) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %6 = "tf.ReadVariableOp"(%arg19) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %7 = "tf.ReadVariableOp"(%arg20) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %8 = "tf.ReadVariableOp"(%arg21) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %9:4 = "tf_device.cluster_func"(%arg13, %arg14, %2, %4, %3, %5, %6, %7, %8) {_tpu_replicate = "cluster_eval_step", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], func = @_func, host_compute_core = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : (tensor<2x224x224x3xf32>, tensor<2x1xf32>, tensor<7x7x3x64xf32>, tensor<64x1001xf32>, tensor<1001xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
      "tf.AssignVariableOp"(%arg18, %9#0) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg19, %9#1) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg20, %9#2) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg21, %9#3) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_device.return
    }
    return
  }
  // CHECK-LABEL: func private @_func
  // CHECK-SAME: [[FUNCINPUT00:.*]]: tensor<2x112x112x12xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<2x1xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<64x1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg4: tensor<1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg5: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg6: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg7: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg8: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
  func private @_func(%arg0: tensor<2x224x224x3xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<2x1xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<64x1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg4: tensor<1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg5: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg6: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg7: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg8: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
    %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<[[0, 1]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
    %4 = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
    %5 = "tf.Const"() {value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32>
    %6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %7 = "tf.Const"() {value = dense<[-1, 1001]> : tensor<2xi32>} : () -> tensor<2xi32>
    %8 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %9 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %10 = "tf.Const"() {value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %11 = "tf.Pad"(%arg0, %10) : (tensor<2x224x224x3xf32>, tensor<4x2xi32>) -> tensor<2x230x230x3xf32>
    %12 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2x1xf32>) -> tensor<2x1xi64>
    %13 = "tf.Reshape"(%12, %9) : (tensor<2x1xi64>, tensor<1xi32>) -> tensor<2xi64>
    %14 = "tf.Squeeze"(%arg1) {squeeze_dims = [-1]} : (tensor<2x1xf32>) -> tensor<2xf32>
    // CHECK: "tf.Conv2D"
    // CHECK-SAME: strides = [1, 1, 1, 1]
    // CHECK-SAME: (tensor<2x115x115x12xf32>, tensor<4x4x12x64xf32>) -> tensor<2x112x112x64xf32>
    %15 = "tf.Conv2D"(%11, %arg2) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<2x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<2x112x112x64xf32>
    %16 = "tf.Mean"(%15, %8) {keep_dims = false} : (tensor<2x112x112x64xf32>, tensor<2xi32>) -> tensor<2x64xf32>
    %17 = "tf.MatMul"(%16, %arg3) {transpose_a = false, transpose_b = false} : (tensor<2x64xf32>, tensor<64x1001xf32>) -> tensor<2x1001xf32>
    %18 = "tf.BiasAdd"(%17, %arg4) {data_format = "NHWC"} : (tensor<2x1001xf32>, tensor<1001xf32>) -> tensor<2x1001xf32>
    %19 = "tf.Reshape"(%18, %7) : (tensor<2x1001xf32>, tensor<2xi32>) -> tensor<2x1001xf32>
    %loss, %backprop = "tf.SparseSoftmaxCrossEntropyWithLogits"(%19, %13) : (tensor<2x1001xf32>, tensor<2xi64>) -> (tensor<2xf32>, tensor<2x1001xf32>)
    %20 = "tf.Sum"(%loss, %6) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
    %21 = "tf.Mul"(%20, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = "tf.Sum"(%21, %4) {keep_dims = false} : (tensor<f32>, tensor<0xi32>) -> tensor<f32>
    %23 = "tf.CrossReplicaSum"(%22, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %24 = "tf.Softmax"(%18) : (tensor<2x1001xf32>) -> tensor<2x1001xf32>
    %25 = "tf.ArgMax"(%24, %2) : (tensor<2x1001xf32>, tensor<i32>) -> tensor<2xi64>
    %26 = "tf.Cast"(%25) {Truncate = false} : (tensor<2xi64>) -> tensor<2xf32>
    %27 = "tf.Equal"(%14, %26) {incompatible_shape_error = true} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %28 = "tf.Cast"(%27) {Truncate = false} : (tensor<2xi1>) -> tensor<2xf32>
    %29 = "tf.Sum"(%28, %6) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
    %30 = "tf.CrossReplicaSum"(%29, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %31 = "tf.AddV2"(%arg5, %23) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %32 = "tf.CrossReplicaSum"(%1, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %33 = "tf.AddV2"(%arg6, %32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %34 = "tf.AddV2"(%arg7, %30) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %35 = "tf.CrossReplicaSum"(%0, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %36 = "tf.AddV2"(%arg8, %35) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %31, %33, %34, %36 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

// -----

// Tests for space to depth host and device transform with replicate packed inputs.

module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:COMPOSITE:0" = {}, "/job:localhost/replica:0/task:0/device:CPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:0" = {}, "/job:localhost/replica:0/task:0/device:TPU:1" = {}, "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0" = {}}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 458 : i32}} {
  func @main(%arg0: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg1: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg2: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg3: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg4: tensor<*x!tf.resource> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg5: tensor<!tf.variant> {tf._user_specified_name = "iterator", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}, %arg6: tensor<*x!tf.resource<tensor<7x7x3x64xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg7: tensor<*x!tf.resource<tensor<64x1001xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg8: tensor<*x!tf.resource<tensor<1001xf32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg9: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg10: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg11: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}, %arg12: tensor<*x!tf.resource<tensor<f32>>> {tf._composite_device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0", tf.device = "/job:localhost/replica:0/task:0/device:COMPOSITE:0"}) attributes {tf.entry_function = {control_outputs = "IteratorGetNext,IteratorGetNext_1,CrossReplicaSum,AssignAddVariableOp,CrossReplicaSum_1,AssignAddVariableOp_1,CrossReplicaSum_2,AssignAddVariableOp_2,CrossReplicaSum_3,AssignAddVariableOp_3", inputs = "iterator,iterator_1,iterator_2,iterator_3,iterator_4,iterator_5,resnet50_conv1_conv2d_conv1_kernel_140365606309224_handle_inputs_0,resnet50_fc1000_matmul_fc1000_kernel_140365944145960_handle_inputs_0,resnet50_fc1000_biasadd_fc1000_bias_140365944146240_handle_inputs_0,total_140366323758976_handle_inputs_0,count_140366323759312_handle_inputs_0,total_140366323760264_handle_inputs_0,count_140366323760600_handle_inputs_0", outputs = ""}} {
    // CHECK: %[[INPUT00:.*]] = "tf.IteratorGetNext"
    // CHECK-DAG: %[[SPACETODEPTH00:.*]] = "tf.SpaceToDepth"([[INPUT00:.*]]#0) {block_size = 2 : i64, data_format = "NHWC"} : (tensor<2x224x224x3xf32>) -> tensor<2x112x112x12xf32>
    %0:2 = "tf.IteratorGetNext"(%arg2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*x!tf.resource>) -> (tensor<2x224x224x3xf32>, tensor<2x1xf32>)
    tf_device.replicate(%0#0 as %arg13: tensor<2x224x224x3xf32>, %0#1 as %arg14: tensor<2x1xf32>, %arg6 as %arg15: tensor<*x!tf.resource<tensor<7x7x3x64xf32>>>, %arg8 as %arg16: tensor<*x!tf.resource<tensor<1001xf32>>>, %arg7 as %arg17: tensor<*x!tf.resource<tensor<64x1001xf32>>>, %arg9 as %arg18: tensor<*x!tf.resource<tensor<f32>>>, %arg10 as %arg19: tensor<*x!tf.resource<tensor<f32>>>, %arg11 as %arg20: tensor<*x!tf.resource<tensor<f32>>>, %arg12 as %arg21: tensor<*x!tf.resource<tensor<f32>>>) {_mirrored_variable_indices = [2, 3, 4, 5, 6, 7, 8], _replicated_input_indices = [1, 2, -1, -1, -1, -1, -1, -1, -1], devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}, n = 2 : i32} {
      %2 = "tf.ReadVariableOp"(%arg15) : (tensor<*x!tf.resource<tensor<7x7x3x64xf32>>>) -> tensor<7x7x3x64xf32>
      %3 = "tf.ReadVariableOp"(%arg16) : (tensor<*x!tf.resource<tensor<1001xf32>>>) -> tensor<1001xf32>
      %4 = "tf.ReadVariableOp"(%arg17) : (tensor<*x!tf.resource<tensor<64x1001xf32>>>) -> tensor<64x1001xf32>
      %5 = "tf.ReadVariableOp"(%arg18) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %6 = "tf.ReadVariableOp"(%arg19) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %7 = "tf.ReadVariableOp"(%arg20) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %8 = "tf.ReadVariableOp"(%arg21) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
      %9:4 = "tf_device.cluster_func"(%arg13, %arg14, %2, %4, %3, %5, %6, %7, %8) {_tpu_replicate = "cluster_eval_step", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], func = @_func, host_compute_core = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : (tensor<2x224x224x3xf32>, tensor<2x1xf32>, tensor<7x7x3x64xf32>, tensor<64x1001xf32>, tensor<1001xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
      "tf.AssignVariableOp"(%arg18, %9#0) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg19, %9#1) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg20, %9#2) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      "tf.AssignVariableOp"(%arg21, %9#3) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
      tf_device.return
    }
    return
  }
  // CHECK-LABEL: func private @_func
  // CHECK-SAME: [[FUNCINPUT00:.*]]: tensor<2x112x112x12xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<2x1xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<64x1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg4: tensor<1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg5: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg6: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg7: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg8: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
  func private @_func(%arg0: tensor<2x224x224x3xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg1: tensor<2x1xf32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg2: tensor<7x7x3x64xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg3: tensor<64x1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg4: tensor<1001xf32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg5: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg6: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg7: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, %arg8: tensor<f32> {mhlo.is_same_data_across_replicas, mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}, tensor<f32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
    %0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tf.Const"() {value = dense<[[0, 1]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
    %4 = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
    %5 = "tf.Const"() {value = dense<2.500000e-01> : tensor<f32>} : () -> tensor<f32>
    %6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %7 = "tf.Const"() {value = dense<[-1, 1001]> : tensor<2xi32>} : () -> tensor<2xi32>
    %8 = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %9 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %10 = "tf.Const"() {value = dense<[[0, 0], [3, 3], [3, 3], [0, 0]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
    %11 = "tf.Pad"(%arg0, %10) : (tensor<2x224x224x3xf32>, tensor<4x2xi32>) -> tensor<2x230x230x3xf32>
    %12 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<2x1xf32>) -> tensor<2x1xi64>
    %13 = "tf.Reshape"(%12, %9) : (tensor<2x1xi64>, tensor<1xi32>) -> tensor<2xi64>
    %14 = "tf.Squeeze"(%arg1) {squeeze_dims = [-1]} : (tensor<2x1xf32>) -> tensor<2xf32>
    // CHECK: "tf.Conv2D"
    // CHECK-SAME: strides = [1, 1, 1, 1]
    // CHECK-SAME: (tensor<2x115x115x12xf32>, tensor<4x4x12x64xf32>) -> tensor<2x112x112x64xf32>
    %15 = "tf.Conv2D"(%11, %arg2) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<2x230x230x3xf32>, tensor<7x7x3x64xf32>) -> tensor<2x112x112x64xf32>
    %16 = "tf.Mean"(%15, %8) {keep_dims = false} : (tensor<2x112x112x64xf32>, tensor<2xi32>) -> tensor<2x64xf32>
    %17 = "tf.MatMul"(%16, %arg3) {transpose_a = false, transpose_b = false} : (tensor<2x64xf32>, tensor<64x1001xf32>) -> tensor<2x1001xf32>
    %18 = "tf.BiasAdd"(%17, %arg4) {data_format = "NHWC"} : (tensor<2x1001xf32>, tensor<1001xf32>) -> tensor<2x1001xf32>
    %19 = "tf.Reshape"(%18, %7) : (tensor<2x1001xf32>, tensor<2xi32>) -> tensor<2x1001xf32>
    %loss, %backprop = "tf.SparseSoftmaxCrossEntropyWithLogits"(%19, %13) : (tensor<2x1001xf32>, tensor<2xi64>) -> (tensor<2xf32>, tensor<2x1001xf32>)
    %20 = "tf.Sum"(%loss, %6) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
    %21 = "tf.Mul"(%20, %5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %22 = "tf.Sum"(%21, %4) {keep_dims = false} : (tensor<f32>, tensor<0xi32>) -> tensor<f32>
    %23 = "tf.CrossReplicaSum"(%22, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %24 = "tf.Softmax"(%18) : (tensor<2x1001xf32>) -> tensor<2x1001xf32>
    %25 = "tf.ArgMax"(%24, %2) : (tensor<2x1001xf32>, tensor<i32>) -> tensor<2xi64>
    %26 = "tf.Cast"(%25) {Truncate = false} : (tensor<2xi64>) -> tensor<2xf32>
    %27 = "tf.Equal"(%14, %26) {incompatible_shape_error = true} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
    %28 = "tf.Cast"(%27) {Truncate = false} : (tensor<2xi1>) -> tensor<2xf32>
    %29 = "tf.Sum"(%28, %6) {keep_dims = false} : (tensor<2xf32>, tensor<1xi32>) -> tensor<f32>
    %30 = "tf.CrossReplicaSum"(%29, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %31 = "tf.AddV2"(%arg5, %23) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %32 = "tf.CrossReplicaSum"(%1, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %33 = "tf.AddV2"(%arg6, %32) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %34 = "tf.AddV2"(%arg7, %30) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %35 = "tf.CrossReplicaSum"(%0, %3) : (tensor<f32>, tensor<1x2xi32>) -> tensor<f32>
    %36 = "tf.AddV2"(%arg8, %35) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %31, %33, %34, %36 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}

