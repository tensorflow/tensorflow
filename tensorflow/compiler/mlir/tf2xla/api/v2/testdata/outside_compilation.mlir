module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1654 : i32}} {
  func.func @main(%arg0: tensor<*x!tf_type.resource> {tf._user_specified_name = "input_1", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) attributes {allow_soft_placement = true, tf.entry_function = {control_outputs = "while,image_sample/write_summary/summary_cond", inputs = "image_sample_write_summary_summary_cond_input_1", outputs = ""}} {
    tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.TPUReplicatedInput"(%outputs, %outputs_0) {device = "", index = -1 : i64, is_mirrored_variable = false, is_packed = false} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Const"() {_post_device_rewrite = true, device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Const"() {_post_device_rewrite = true, device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
      %outputs_10, %control_11 = tf_executor.island wraps "tf.Pack"(%outputs_6, %outputs_8) {axis = 0 : i64, device = ""} : (tensor<0xi32>, tensor<0xi32>) -> tensor<*xi32>
      %outputs_12, %control_13 = tf_executor.island wraps "tf.Max"(%outputs_10, %outputs_4) {device = "", keep_dims = false} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %control_14 = tf_executor.island wraps "tf.NoOp"() {_pivot_for_cluster = "cluster_sample_sequence", device = ""} : () -> ()
      %control_15 = tf_executor.island(%control_14) wraps "tf.NoOp"() {_has_manual_control_dependencies = true, _tpu_replicate = "cluster_sample_sequence", device = ""} : () -> ()
      %control_16 = tf_executor.island(%control_15) wraps "tf.NoOp"() {device = ""} : () -> ()
      %control_17 = tf_executor.island(%control_15) wraps "tf.NoOp"() {device = ""} : () -> ()
      %control_18 = tf_executor.island(%control_14) wraps "tf.TPUReplicateMetadata"() {_has_manual_control_dependencies = true, _tpu_replicate = "cluster_sample_sequence", allow_soft_placement = true, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 1 : i64, num_replicas = 2 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> ()
      %outputs_19, %control_20 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_21, %control_22 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<5> : tensor<i64>} : () -> tensor<i64>
      %outputs_23, %control_24 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<[3, 32, 32, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
      %outputs_25, %control_26 = tf_executor.island(%control_18) wraps "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster_sample_sequence", device = ""} : () -> tensor<!tf_type.string>
      %outputs_27, %control_28 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_29, %control_30 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<[1, 1, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
      %outputs_31, %control_32 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_33, %control_34 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %outputs_35, %control_36 = tf_executor.island(%control_18) wraps "tf.Identity"(%outputs_2) {_tpu_input_identity = true, _tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xi32>) -> tensor<*xi32>
      %outputs_37, %control_38 = tf_executor.island wraps "tf.Equal"(%outputs_35, %outputs_31) {_tpu_replicate = "cluster_sample_sequence", device = "", incompatible_shape_error = true} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
      %outputs_39, %control_40 = tf_executor.island wraps "tf.LogicalAnd"(%outputs_37, %outputs_33) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xi1>, tensor<i1>) -> tensor<*xi1>
      %outputs_41, %control_42 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_43, %control_44 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<1024> : tensor<i32>} : () -> tensor<i32>
      %outputs_45, %control_46 = tf_executor.island wraps "tf.TensorListReserve"(%outputs_41, %outputs_43) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<1xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
      %outputs_47, %control_48 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
      %outputs_49, %control_50 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_51, %control_52 = tf_executor.island(%control_18) wraps "tf.Const"() {_tpu_replicate = "cluster_sample_sequence", device = "", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
      %outputs_53:4, %control_54 = tf_executor.island wraps "tf.While"(%outputs_49, %outputs_51, %outputs_19, %outputs_45) {_num_original_outputs = 4 : i64, _read_only_resource_inputs = [], _tpu_replicate = "cluster_sample_sequence", _xla_propagate_compile_time_consts = true, body = @while_body_260, cond = @while_cond_250, device = "", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>)
      %outputs_55, %control_56 = tf_executor.island wraps "tf.TensorListStack"(%outputs_53#3, %outputs_27) {_tpu_replicate = "cluster_sample_sequence", device = "", num_elements = 1024 : i64} : (tensor<!tf_type.variant>, tensor<1xi32>) -> tensor<*xf32>
      %outputs_57, %control_58 = tf_executor.island wraps "tf.Transpose"(%outputs_55, %outputs_47) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
      %outputs_59, %control_60 = tf_executor.island wraps "tf.Reshape"(%outputs_57, %outputs_23) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xf32>, tensor<4xi32>) -> tensor<*xf32>
      %outputs_61, %control_62 = tf_executor.island wraps "tf.Tile"(%outputs_59, %outputs_29) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xf32>, tensor<4xi32>) -> tensor<*xf32>
      %outputs_63, %control_64 = tf_executor.island wraps "tf.If"(%outputs_39, %outputs_61, %arg0, %outputs_21) {_read_only_resource_inputs = [], _tpu_replicate = "cluster_sample_sequence", _xla_propagate_compile_time_consts = true, device = "", else_branch = @image_sample_write_summary_summary_cond_false_710, is_stateless = false, then_branch = @image_sample_write_summary_summary_cond_true_700} : (tensor<*xi1>, tensor<*xf32>, tensor<*x!tf_type.resource>, tensor<i64>) -> tensor<*xi1>
      %outputs_65, %control_66 = tf_executor.island wraps "tf.Identity"(%outputs_63) {_tpu_replicate = "cluster_sample_sequence", device = ""} : (tensor<*xi1>) -> tensor<*xi1>
      tf_executor.fetch %control_54, %control_64 : !tf_executor.control, !tf_executor.control
    }
    return
  }
  func.func private @while_body_260(%arg0: tensor<i32> {tf._user_specified_name = "while/loop_counter"}, %arg1: tensor<i32> {tf._user_specified_name = "while/maximum_iterations"}, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*x!tf_type.variant>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    %0:4 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.RandomUniform"(%outputs_2) {device = "", seed = 87654321 : i64, seed2 = 0 : i64} : (tensor<1xi32>) -> tensor<*xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.AddV2"(%arg2, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Identity"(%outputs_6) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
      %outputs_10, %control_11 = tf_executor.island wraps "tf.TensorListSetItem"(%arg3, %arg2, %outputs_4) {device = "", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant>, tensor<i32>, tensor<*xf32>) -> tensor<*x!tf_type.variant>
      %outputs_12, %control_13 = tf_executor.island wraps "tf.Identity"(%outputs_10) {device = ""} : (tensor<*x!tf_type.variant>) -> tensor<*x!tf_type.variant>
      %outputs_14, %control_15 = tf_executor.island wraps "tf.AddV2"(%arg0, %outputs_0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
      %outputs_16, %control_17 = tf_executor.island wraps "tf.Identity"(%outputs_14) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
      %outputs_18, %control_19 = tf_executor.island wraps "tf.Identity"(%arg1) {device = ""} : (tensor<i32>) -> tensor<*xi32>
      tf_executor.fetch %outputs_16, %outputs_18, %outputs_8, %outputs_12 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*x!tf_type.variant>
    }
    return %0#0, %0#1, %0#2, %0#3 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*x!tf_type.variant>
  }
  func.func private @while_cond_250(%arg0: tensor<i32> {tf._user_specified_name = "while/loop_counter"}, %arg1: tensor<i32> {tf._user_specified_name = "while/maximum_iterations"}, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>) -> tensor<*xi1> attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<1024> : tensor<i32>} : () -> tensor<i32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Less"(%arg2, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Identity"(%outputs_0) {device = ""} : (tensor<*xi1>) -> tensor<*xi1>
      tf_executor.fetch %outputs_2 : tensor<*xi1>
    }
    return %0 : tensor<*xi1>
  }
  func.func private @image_sample_write_summary_summary_cond_false_710(%arg0: tensor<3x32x32x3xf32>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<i64>) -> tensor<*xi1> attributes {tf._construction_context = "kEagerRuntime"} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<false> : tensor<i1>} : () -> tensor<i1>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Identity"(%outputs) {device = ""} : (tensor<i1>) -> tensor<*xi1>
      tf_executor.fetch %outputs_0 : tensor<*xi1>
    }
    return %0 : tensor<*xi1>
  }
  func.func private @image_sample_write_summary_summary_cond_true_700(%arg0: tensor<3x32x32x3xf32> {tf._user_specified_name = "Tile"}, %arg1: tensor<*x!tf_type.resource> {tf._user_specified_name = "writer"}, %arg2: tensor<i64> {tf._user_specified_name = "Const_3"}) -> tensor<*xi1> attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<[3, 32, 32, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"x (image_sample/write_summary/summary_cond/assert_non_negative/x:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_10, %control_11 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_12, %control_13 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"x (image_sample/write_summary/summary_cond/assert_non_negative/x:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_14, %control_15 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_16, %control_17 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %outputs_18, %control_19 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_20, %control_21 = tf_executor.island wraps "tf.Range"(%outputs_18, %outputs_14, %outputs_16) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
      %outputs_22, %control_23 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<3> : tensor<i32>} : () -> tensor<i32>
      %outputs_24, %control_25 = tf_executor.island wraps "tf.LessEqual"(%outputs_0, %outputs_22) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
      %outputs_26, %control_27 = tf_executor.island wraps "tf.All"(%outputs_24, %outputs_20) {device = "/device:CPU:0", keep_dims = false} : (tensor<*xi1>, tensor<*xi32>) -> tensor<*xi1>
      %control_28 = tf_executor.island wraps "tf.Assert"(%outputs_26, %outputs_2, %outputs_4, %outputs_6, %outputs_22) {device = "/device:CPU:0", summarize = 3 : i64} : (tensor<*xi1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i32>) -> ()
      %outputs_29, %control_30 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<[3, 32, 32, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
      %control_31 = tf_executor.island wraps "tf.NoOp"() {device = "/device:CPU:0"} : () -> ()
      %outputs_32, %control_33 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<4> : tensor<i32>} : () -> tensor<i32>
      %control_34 = tf_executor.island wraps "tf.NoOp"() {device = "/device:CPU:0"} : () -> ()
      %outputs_35, %control_36 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %outputs_37, %control_38 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
      %outputs_39, %control_40 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<2.550000e+02> : tensor<f32>} : () -> tensor<f32>
      %outputs_41, %control_42 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<2.555000e+02> : tensor<f32>} : () -> tensor<f32>
      %outputs_43, %control_44 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_45, %control_46 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_47, %control_48 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_49, %control_50 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_51, %control_52 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_53, %control_54 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_55, %control_56 = tf_executor.island wraps "tf.StridedSlice"(%outputs, %outputs_49, %outputs_51, %outputs_53) {begin_mask = 0 : i64, device = "/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
      %outputs_57, %control_58 = tf_executor.island wraps "tf.AsString"(%outputs_55) {device = "/device:CPU:0", fill = "", precision = -1 : i64, scientific = false, shortest = false, width = -1 : i64} : (tensor<*xi32>) -> tensor<*x!tf_type.string>
      %outputs_59, %control_60 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_61, %control_62 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_63, %control_64 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
      %outputs_65, %control_66 = tf_executor.island wraps "tf.StridedSlice"(%outputs, %outputs_59, %outputs_61, %outputs_63) {begin_mask = 0 : i64, device = "/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
      %outputs_67, %control_68 = tf_executor.island wraps "tf.AsString"(%outputs_65) {device = "/device:CPU:0", fill = "", precision = -1 : i64, scientific = false, shortest = false, width = -1 : i64} : (tensor<*xi32>) -> tensor<*x!tf_type.string>
      %outputs_69, %control_70 = tf_executor.island wraps "tf.Pack"(%outputs_57, %outputs_67) {axis = 0 : i64, device = "/device:CPU:0"} : (tensor<*x!tf_type.string>, tensor<*x!tf_type.string>) -> tensor<*x!tf_type.string>
      %outputs_71, %control_72 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"\0A\08\0A\06images"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_73, %control_74 = tf_executor.island wraps "tf.Const"() {device = "/device:CPU:0", value = dense<"image_sample"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_75, %control_76 = tf_executor.island wraps "tf.Mul"(%arg0, %outputs_41) {device = "/device:CPU:0"} : (tensor<3x32x32x3xf32>, tensor<f32>) -> tensor<*xf32>
      %outputs_77, %control_78 = tf_executor.island wraps "tf.ClipByValue"(%outputs_75, %outputs_37, %outputs_39) {device = "/device:CPU:0"} : (tensor<*xf32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
      %outputs_79, %control_80 = tf_executor.island wraps "tf.Cast"(%outputs_77) {Truncate = false, device = "/device:CPU:0"} : (tensor<*xf32>) -> tensor<*xui8>
      %outputs_81, %control_82 = tf_executor.island wraps "tf.StridedSlice"(%outputs_79, %outputs_43, %outputs_45, %outputs_47) {begin_mask = 1 : i64, device = "/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<*xui8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xui8>
      %outputs_83, %control_84 = tf_executor.island wraps "tf.EncodePng"(%outputs_81) {compression = -1 : i64, device = "/device:CPU:0"} : (tensor<*xui8>) -> tensor<*x!tf_type.string>
      %outputs_85, %control_86 = tf_executor.island wraps "tf.ConcatV2"(%outputs_69, %outputs_83, %outputs_35) {device = "/device:CPU:0"} : (tensor<*x!tf_type.string>, tensor<*x!tf_type.string>, tensor<i32>) -> tensor<*x!tf_type.string>
      %control_87 = tf_executor.island wraps "tf.WriteSummary"(%arg1, %arg2, %outputs_85, %outputs_73, %outputs_71) {_has_manual_control_dependencies = true, device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>, tensor<i64>, tensor<*x!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> ()
      %outputs_88, %control_89 = tf_executor.island(%control_87) wraps "tf.Const"() {device = "/device:CPU:0", value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %control_90 = tf_executor.island(%control_28, %control_87) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_91, %control_92 = tf_executor.island(%control_90) wraps "tf.Identity"(%outputs_88) {device = ""} : (tensor<i1>) -> tensor<*xi1>
      tf_executor.fetch %outputs_91, %control_28, %control_87 : tensor<*xi1>, !tf_executor.control, !tf_executor.control
    }
    return %0 : tensor<*xi1>
  }
}