// RUN: tf-tfrt-opt -split-input-file -tf-mlrt-while-to-map-fn %s | FileCheck %s

// Test a while to map_fn conversion in which the max iteration is hard coded inside the predicate body.

// CHECK-LABEL: map/while_cond
func.func private @"map/while_cond"(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> tensor<i1> {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<3> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = "tf.Less"(%arg1, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = "tf.LogicalAnd"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: map/while_body
func.func private @"map/while_body"(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]> : tensor<9xf32>} : () -> tensor<9xf32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<3> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_3 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<9xf32>} : () -> tensor<9xf32>
  %cst_4 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst_4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.Mul"(%arg3, %cst_3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<9xf32>) -> tensor<9xf32>
  %2 = "tf.Reshape"(%1, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<9xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  %3 = "tf.AddV2"(%arg1, %cst_4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.GatherV2"(%cst_1, %arg1, %cst_0) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<3xi32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %5 = "tf.Cast"(%4) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  %6 = "tf.Mul"(%5, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<f32>, tensor<9xf32>) -> tensor<9xf32>
  %7 = "tf.Reshape"(%6, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<9xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  %8 = "tf.MatMul"(%2, %7) {device = "/job:localhost/replica:0/task:0/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
  %9 = "tf.MatrixDeterminant"(%8) {T = f32, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<3x3xf32>) -> tensor<f32>
  %10 = "tf.TensorListSetItem"(%arg2, %arg1, %9) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  return %0, %3, %10, %arg3 : tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>
}

// CHECK-LABEL: map/while_body/MapFnBody
// CHECK-SAME: (%arg0: !mlrt.future, %arg1: !mlrt.promise, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<?xf32>)
// CHECK: [[det:%.*]] = "tf.MatrixDeterminant"
// CHECK-NEXT: [[ta_0:%.*]] = "tf_mlrt.tf_await"(%arg0) : (!mlrt.future) -> tensor<!tf_type.variant<tensor<*xf32>>>
// CHECK-NEXT: [[ta_1:%.*]] = "tf.TensorListSetItem"([[ta_0]], %arg3, [[det]]) <{
// CHECK-NEXT:  "tf_mlrt.tf_promise"(%arg1, [[ta_1]]) : (!mlrt.promise, tensor<!tf_type.variant<tensor<*xf32>>>) -> ()
// CHECK-NEXT: return

//CHECK-LABEL: @serving_default
func.func @serving_default(%arg0: tensor<?xf32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> tensor<3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_input:0", outputs = "PartitionedCall:0"}} {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[tensor_list:%.*]] = "tf.TensorListReserve"([[shape:%.*]], [[reserve_size:%.*]]) {
  %0 = "tf.TensorListReserve"(%cst_1, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK: [[map_fn_result:%.*]] = tf_mlrt.tf_map_fn([[reserve_size]], [[tensor_list]], %arg0)
  // CHECK-SAME: {body_fn = @"map/while_body/MapFnBody", num_tensor_list_or_flow_in = 1 : i32}
  // CHECK-NOT: tf.While
  %1:4 = "tf.While"(%cst, %cst, %0, %arg0) {_lower_using_switch_merge = true, _num_original_outputs = 6 : i64, _read_only_resource_inputs = [], _xla_propagate_compile_time_consts = true, body = @"map/while_body", cond = @"map/while_cond", device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = true, parallel_iterations = 4 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>)
  // CHECK-NEXT: "tf.TensorListStack"([[map_fn_result]], %cst_0) <{
  %2 = "tf.TensorListStack"(%1#2, %cst_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = 3 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<0xi32>) -> tensor<3xf32>
  return %2 : tensor<3xf32>
}

// -----

// Test a while to map_fn conversion in which max_iterations are passed
// into the predicate function.

// CHECK-LABEL: @"map/while_cond"
func.func private @"map/while_cond"(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<i32>, %arg4: tensor<!tf_type.resource<tensor<3x1xf32>>>, %arg5: tensor<?x3xf32>, %arg6: tensor<?x4xf32>) -> tensor<i1> {
  %outputs =  "tf.Less"(%arg0, %arg3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %outputs_0  =  "tf.Less"(%arg1, %arg3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %outputs_2  =  "tf.LogicalAnd"(%outputs_0, %outputs) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i1>, tensor<i1>) -> tensor<i1>

  return %outputs_2 : tensor<i1>
}

// CHECK-LABEL: @"map/while_body"
func.func private @"map/while_body"(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<i32>, %arg4: tensor<!tf_type.resource<tensor<3x1xf32>>>, %arg5: tensor<?x3xf32>, %arg6: tensor<?x4xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<?x3xf32>, tensor<?x4xf32>) {
  %outputs =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_0  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_2  =  "tf.AddV2"(%arg0, %outputs_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %outputs_4  =  "tf.ReadVariableOp"(%arg4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
  %outputs_6  =  "tf.Identity"(%outputs_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> tensor<i32>
  %outputs_8  =  "tf.MatMul"(%arg5, %outputs_4) {device = "/job:localhost/replica:0/task:0/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<?x3xf32>, tensor<3x1xf32>) -> tensor<?x1xf32>
  %outputs_10  =  "tf.AddV2"(%arg1, %outputs_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %outputs_12  =  "tf.Identity"(%outputs_10) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> tensor<i32>
  %outputs_14  =  "tf.GatherV2"(%arg6, %arg1, %outputs) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4xf32>
  %outputs_16  =  "tf.AddV2"(%outputs_8, %outputs_14) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x1xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  %outputs_18  =  "tf.TensorListSetItem"(%arg2, %arg1, %outputs_16) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<?x4xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  return  %outputs_6, %outputs_12, %outputs_18, %arg3, %arg4, %arg5, %arg6 : tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<?x3xf32>, tensor<?x4xf32>
}

// CHECK-LABEL: @"map/while_body/MapFnBody"
// CHECK-SAME (%arg0: !mlrt.Future, %arg1: !mlrt.Promise, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<!tf_type.resource<tensor<3x1xf32>>>, %arg6: tensor<?x3xf32>, %arg7: tensor<?x4xf32>)
// CHECK-NEXT: [[cst_0:%.*]] = "tf.Const"
// CHECK-NEXT: [[cst_1:%.*]] = "tf.Const"
// CHECK-NEXT: [[loop_counter:%.*]] = "tf.AddV2"(%arg2, [[cst_1]])
// CHECK-NEXT: [[weight:%.*]] = "tf.ReadVariableOp"(%arg5)
// CHECK-NEXT: [[mpy:%.*]] = "tf.MatMul"(%arg6, [[weight]])
// CHECK-NEXT: [[element_index:%.*]] = "tf.AddV2"(%arg3, [[cst_1]])
// CHECK-NEXT: [[bias:%.*]] = "tf.GatherV2"(%arg7, %arg3, [[cst_0]])
// CHECK-NEXT: [[res:%.*]] = "tf.AddV2"([[mpy]], [[bias]])
// CHECK-NEXT: [[ta_0:%.*]] = "tf_mlrt.tf_await"(%arg0)
// CHECK-NEXT: [[ta_1:%.*]] = "tf.TensorListSetItem"([[ta_0]], %arg3, [[res]])
// CHECK-NEXT: "tf_mlrt.tf_promise"(%arg1, [[ta_1]])
// CHECK-NEXT: return

// CHECK-LABEL: func @main_while
func.func @main_while(%arg0: tensor<?x3xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x?x4xf32> {
  %outputs =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[-1, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_0  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %outputs_2  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_4  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_6  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_8  =  "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[elems:%.*]] = "tf.VarHandleOp"
  %outputs_10  =  "tf.VarHandleOp"() {_xla_inferred_shapes = [#tf_type.shape<>], allowed_devices = [], container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", shared_name = "w"} : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  %outputs_12  =  "tf.Shape"(%arg1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x4xf32>) -> tensor<2xi32>
  // CHECK: [[max_iter:%.*]] = "tf.StridedSlice"
  %outputs_14  =  "tf.StridedSlice"(%outputs_12, %outputs_2, %outputs_4, %outputs_4) {begin_mask = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  // CHECK: [[tensor_list:%.*]] = "tf.TensorListReserve"
  %outputs_16  =  "tf.TensorListReserve"(%outputs_0, %outputs_14) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK: tf_mlrt.tf_map_fn
  // CHECK-SAME: ([[max_iter]], [[tensor_list]], [[max_iter]], [[elems]], %arg0, %arg1)
  // CHECK-SAME: {body_fn = @"map/while_body/MapFnBody", num_tensor_list_or_flow_in = 1 : i32}
  // CHECK-NOT: tf.while
  %outputs_18:7  =  "tf.While"(%outputs_6, %outputs_6, %outputs_16, %outputs_14, %outputs_10, %arg0, %arg1) {_lower_using_switch_merge = true, _num_original_outputs = 8 : i64, _read_only_resource_inputs = [6], _xla_propagate_compile_time_consts = true, body = @"map/while_body", cond = @"map/while_cond", device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<?x3xf32>, tensor<?x4xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<?x3xf32>, tensor<?x4xf32>)
  %outputs_20  =  "tf.TensorListStack"(%outputs_18#2, %outputs) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = -1 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<2xi32>) -> tensor<?x?x4xf32>
  return %outputs_20 : tensor<?x?x4xf32>
}

// -----

// Test a while to map_fn conversion in which the passed in max_iterations 
// is not in typical location of %arg3 and there are identify chains in function bodies.

// CHECK-LABEL: @map_while_cond_170
func.func private @map_while_cond_170(%arg0: tensor<i32> {tf._user_specified_name = "map/while/loop_counter"}, %arg1: tensor<i32> {tf._user_specified_name = "map/while/maximum_iterations"}, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>, %arg4: tensor<*x!tf_type.variant>, %arg5: tensor<*xf32>) -> tensor<*xi1> attributes {tf._construction_context = "kEagerRuntime", tf._original_func_name = "map_while_cond_17"} {
  %outputs =  "tf.Const"() {device = "", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %outputs_0 =  "tf.Less"(%arg0, %arg1) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
  %outputs_2 =  "tf.Less"(%arg2, %outputs) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
  %outputs_4 =  "tf.LogicalAnd"(%outputs_0, %outputs_2) {device = ""} : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  %outputs_6 =  "tf.Identity"(%outputs_4) {device = ""} : (tensor<*xi1>) -> tensor<*xi1>
  return %outputs_6 : tensor<*xi1>
}

// Original input argument list (loop_counter, max_iterations, element_index, tensor_list, read_only_tensor_list, scale)
// CHECK-LABEL: @map_while_body_180
func.func private @map_while_body_180(%arg0: tensor<i32> {tf._user_specified_name = "map/while/loop_counter"}, %arg1: tensor<i32> {tf._user_specified_name = "map/while/maximum_iterations"}, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>, %arg4: tensor<!tf_type.variant> {tf._user_specified_name = "map/TensorArrayUnstack/TensorListFromTensor"}, %arg5: tensor<?xf32> {tf._user_specified_name = "input"}) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*x!tf_type.variant>, tensor<!tf_type.variant>, tensor<?xf32>) attributes {tf._construction_context = "kEagerRuntime", tf._original_func_name = "map_while_body_18"} {
  %outputs =  "tf.Const"() {device = "", value = dense<16> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_0 =  "tf.Const"() {device = "", value = dense<16> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_2 =  "tf.Const"() {device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %outputs_4 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_6 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_8 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_10 =  "tf.Const"() {device = "", value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %outputs_12 =  "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_14 =  "tf.Range"(%outputs_12, %outputs_10, %outputs_8) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
  %outputs_16 =  "tf.Cast"(%outputs_14) {Truncate = false, device = ""} : (tensor<*xi32>) -> tensor<*xf32>
  %outputs_18 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_20 =  "tf.Const"() {device = "", value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %outputs_22 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_24 =  "tf.Range"(%outputs_22, %outputs_20, %outputs_18) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
  %outputs_26 =  "tf.Cast"(%outputs_24) {Truncate = false, device = ""} : (tensor<*xi32>) -> tensor<*xf32>
  %outputs_28 =  "tf.Const"() {device = "", value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_30 =  "tf.Transpose"(%outputs_26, %outputs_28) {device = ""} : (tensor<*xf32>, tensor<1xi32>) -> tensor<*xf32>
  %outputs_32 =  "tf.AddV2"(%arg0, %outputs_6) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
  %outputs_34 =  "tf.Identity"(%outputs_32) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_36 =  "tf.Identity"(%arg1) {device = ""} : (tensor<i32>) -> tensor<*xi32>
  %outputs_38 =  "tf.Mul"(%outputs_16, %arg5) {device = ""} : (tensor<*xf32>, tensor<?xf32>) -> tensor<*xf32>
  %outputs_40 =  "tf.Reshape"(%outputs_38, %outputs) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
  %outputs_42 =  "tf.AddV2"(%arg2, %outputs_4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
  %outputs_44 =  "tf.Identity"(%outputs_42) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_46 =  "tf.TensorListGetItem"(%arg4, %arg2, %outputs_2) {device = ""} : (tensor<!tf_type.variant>, tensor<i32>, tensor<0xi32>) -> tensor<*xi32>
  %outputs_48 =  "tf.Cast"(%outputs_46) {Truncate = false, device = ""} : (tensor<*xi32>) -> tensor<*xf32>
  %outputs_50 =  "tf.Mul"(%outputs_30, %outputs_48) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %outputs_52 =  "tf.Reshape"(%outputs_50, %outputs_0) {device = ""} : (tensor<*xf32>, tensor<2xi32>) -> tensor<*xf32>
  %outputs_54 =  "tf.MatMul"(%outputs_40, %outputs_52) {device = "", transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %outputs_56 =  "tf.MatrixDeterminant"(%outputs_54) {T = f32, device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  %outputs_58 =  "tf.TensorListSetItem"(%arg3, %arg2, %outputs_56) {device = "", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant>, tensor<i32>, tensor<*xf32>) -> tensor<*x!tf_type.variant>
  %outputs_60 =  "tf.Identity"(%outputs_58) {device = ""} : (tensor<*x!tf_type.variant>) -> tensor<*x!tf_type.variant>
  return %outputs_34, %outputs_36, %outputs_44, %outputs_60, %arg4, %arg5 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*x!tf_type.variant>, tensor<!tf_type.variant>, tensor<?xf32>
}

// Converted input argument list (loop_counter, element_index, max_iterations, tensor_list, read_only_tensor_list, scale)
// CHECK-LABEL: @"map_while_body_180/MapFnBody"
// CHECK-SAME: (%arg0: !mlrt.future, %arg1: !mlrt.promise, %arg2: tensor<i32> {tf._user_specified_name = "map/while/loop_counter"}, %arg3: tensor<i32>, %arg4: tensor<i32> {tf._user_specified_name = "map/while/maximum_iterations"}, %arg5: tensor<!tf_type.variant> {tf._user_specified_name = "map/TensorArrayUnstack/TensorListFromTensor"}, %arg6: tensor<?xf32> {tf._user_specified_name = "input"})
// CHECK: [[res:%.*]] = "tf.MatrixDeterminant"
// CHECK-NEXT: [[ta_0:%.*]] = "tf_mlrt.tf_await"(%arg0)
// CHECK-NEXT: [[ta_1:%.*]] = "tf.TensorListSetItem"([[ta_0]], %arg3, [[res]])
// CHECK-NEXT: "tf_mlrt.tf_promise"(%arg1, [[ta_1]])
// CHECK-NEXT: return


// CHECK-LABEL: __inference_while_from_map_fn_810
// CHECK-SAME: ([[scale:%.*]]: tensor<?xf32>
func.func private @__inference_while_from_map_fn_810(%arg0: tensor<?xf32> {tf._user_specified_name = "input"}) -> tensor<*xf32> attributes {tf._construction_context = "kEagerRuntime", tf._original_func_name = "__inference_while_from_map_fn_81"} {
  // CHECK: [[element_index:%.*]] = "tf.Const"
  %outputs =  "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_0 =  "tf.Const"() {device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %outputs_2=  "tf.Const"() {device = "", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %outputs_4 =  "tf.Const"() {device = "", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.TensorListReserve
  %outputs_6 =  "tf.TensorListReserve"(%outputs_2, %outputs_4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xi32>>>
  %outputs_8 =  "tf.Const"() {device = "", value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %outputs_10 =  "tf.Const"() {device = "", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %outputs_12 =  "tf.Const"() {device = "", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[tensor_list:%.*]] = "tf.TensorListReserve"([[shape:%.*]], [[reserve_size:%.*]]) {
  %outputs_14 =  "tf.TensorListReserve"(%outputs_10, %outputs_12) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK-NEXT: [[loop_counter:%.*]] = "tf.Const"
  %outputs_16 =  "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: [[max_iterations:%.*]] = "tf.Const"
  %outputs_18 =  "tf.Const"() {device = "", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %outputs_20 =  "tf.Const"() {device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_22 =  "tf.Const"() {device = "", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %outputs_24 =  "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_26 =  "tf.Range"(%outputs_24, %outputs_22, %outputs_20) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<*xi32>
  // CHECK: [[read_only_tensor_list:%.*]] = "tf.TensorListFromTensor"
  %outputs_28 =  "tf.TensorListFromTensor"(%outputs_26, %outputs_0) {device = ""} : (tensor<*xi32>, tensor<0xi32>) -> tensor<*x!tf_type.variant>
// CHECK: [[map_fn_out:%.*]] = tf_mlrt.tf_map_fn
  // CHECK-SAME: ([[reserve_size]], [[tensor_list]], [[max_iterations]], [[read_only_tensor_list]], [[scale]])
  // CHECK-SAME: {body_fn = @"map_while_body_180/MapFnBody", num_tensor_list_or_flow_in = 1 : i32}
  // CHECK-NOT: tf.While
  %outputs_30:6 =  "tf.While"(%outputs_16, %outputs_18, %outputs, %outputs_14, %outputs_28, %arg0) {T = [i32, i32, i32, !tf_type.variant, !tf_type.variant, f32], _lower_using_switch_merge = true, _num_original_outputs = 6 : i64, _read_only_resource_inputs = [], body = @map_while_body_180, cond = @map_while_cond_170, device = "", is_stateless = true, output_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], parallel_iterations = 4 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<*x!tf_type.variant>, tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>, tensor<!tf_type.variant>, tensor<?xf32>)
  // CHECK-NEXT: "tf.TensorListStack"
  // CHECK-SAME: ([[map_fn_out]],
  %outputs_32 =  "tf.TensorListStack"(%outputs_30#3, %outputs_8) {device = "", num_elements = 16 : i64} : (tensor<!tf_type.variant>, tensor<0xi32>) -> tensor<*xf32>
  %outputs_34 =  "tf.Identity"(%outputs_32) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
  return %outputs_34 : tensor<*xf32>
}

// -----

// Test a while to map_fn conversion in which tensor array is used instead of
// tensor list.

// CHECK-LABEL: map/while/LoopCond_cond
func.func private @"map/while/LoopCond_cond"(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, %arg5: tensor<f32>, %arg6: tensor<2x!tf_type.resource<tensor<*xui8>>>) -> tensor<i1> {
  %outputs = "tf.Less"(%arg0, %arg3) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %outputs_0 = "tf.Less"(%arg1, %arg3) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %outputs_2 = "tf.LogicalAnd"(%outputs, %outputs_0) {device = ""} : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  %outputs_4 = "tf.ToBool"(%outputs_2) : (tensor<*xi1>) -> tensor<i1>
  return %outputs_4 : tensor<i1>
}

// CHECK-LABEL: map/while/LoopCond_body
func.func private @"map/while/LoopCond_body"(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, %arg5: tensor<f32>, %arg6: tensor<2x!tf_type.resource<tensor<*xui8>>>) -> (tensor<*xi32>, tensor<*xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>) {
  %outputs = "tf.Const"() {value = dense<224> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_4 = "tf.Identity"(%arg0) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_6 = "tf.AddV2"(%outputs_4, %outputs_2) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %outputs_8 = "tf.Identity"(%arg1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_10 = "tf.AddV2"(%outputs_8, %outputs_2) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %outputs_12 = "tf.Identity"(%arg2) {device = ""} : (tensor<f32>) -> tensor<f32>
  %outputs_14 = "tf.TensorArrayReadV3"(%arg4, %outputs_8, %arg5) {device = ""} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<*xi32>, tensor<f32>) -> tensor<*x!tf_type.string>
  %outputs_16 = "tf.DecodeJpeg"(%outputs_14) {acceptable_fraction = 1.000000e+00 : f32, channels = 3 : i64, dct_method = "INTEGER_FAST", device = "", fancy_upscaling = true, ratio = 1 : i64, try_recover_truncated = false} : (tensor<*x!tf_type.string>) -> tensor<?x?x3xui8>
  %outputs_18 = "tf.ExpandDims"(%outputs_16, %outputs_0) {device = ""} : (tensor<?x?x3xui8>, tensor<i32>) -> tensor<1x?x?x3xui8>
  %outputs_20 = "tf.ResizeBilinear"(%outputs_18, %outputs) {align_corners = false, device = "", half_pixel_centers = false} : (tensor<1x?x?x3xui8>, tensor<2xi32>) -> tensor<1x224x224x3xf32>
  %outputs_22 = "tf.Squeeze"(%outputs_20) {device = "", squeeze_dims = [0]} : (tensor<1x224x224x3xf32>) -> tensor<224x224x3xf32>
  %outputs_24 = "tf.Cast"(%outputs_22) {Truncate = false, device = ""} : (tensor<224x224x3xf32>) -> tensor<224x224x3xui8>
  %outputs_26 = "tf.TensorArrayWriteV3"(%arg6, %outputs_8, %outputs_24, %outputs_12) {device = ""} : (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<*xi32>, tensor<224x224x3xui8>, tensor<f32>) -> tensor<f32>
  return %outputs_6, %outputs_10, %outputs_26, %arg3, %arg4, %arg5, %arg6: tensor<*xi32>, tensor<*xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>
}

// CHECK-LABEL: @"map/while/LoopCond_body/MapFnBody"
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.TensorArrayReadV3
// CHECK-NEXT: tf.DecodeJpeg
// CHECK-NEXT: tf.ExpandDims
// CHECK-NEXT: tf.ResizeBilinear
// CHECK-NEXT: tf.Squeeze
// CHECK-NEXT: tf.Cast
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf.TensorArrayWriteV3
// CHECK-NEXT: tf_mlrt.tf_promise
// CHECK-NEXT: return

//CHECK-LABEL: map_while_test
func.func @map_while_test(%arg0: tensor<?x!tf_type.string>) -> tensor<?x224x224x3xui8> {
  %outputs = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<1xi32>
  %outputs_0 = "tf.Const"() {value = dense<224> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_4 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_8 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_10 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
  // CHECK: [[max_iter:%.*]] = "tf.StridedSlice"
  %outputs_12 = "tf.StridedSlice"(%outputs_10, %outputs_6, %outputs_4, %outputs_4) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: tf.Range
  %outputs_14 = "tf.Range"(%outputs_2, %outputs_12, %outputs_8) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-NEXT: [[handle_1:%.*]], [[flow_in_1:%.*]] = "tf.TensorArrayV3"
  %outputs_16:2 = "tf.TensorArrayV3"(%outputs_12) {clear_after_read = true, device = "", dtype = !tf_type.string, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>)
  // CHECK-NEXT: [[handle_2:%.*]] = "tf.TensorArrayScatterV3"
  %outputs_18 = "tf.TensorArrayScatterV3"(%outputs_16#0, %outputs_14, %arg0, %outputs_16#1) {device = ""} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<?xi32>, tensor<?x!tf_type.string>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: tf.Range
  %outputs_20 = "tf.Range"(%outputs_2, %outputs_12, %outputs_8) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-NEXT: [[tensor_array:%.*]], [[flow_in:%.*]] = "tf.TensorArrayV3"
  %outputs_22:2 = "tf.TensorArrayV3"(%outputs_12) {clear_after_read = true, device = "", dtype = ui8, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<f32>)
  // CHECK-NEXT: tf_mlrt.tf_map_fn
  // CHECK-SAME: ([[max_iter]], [[flow_in]], [[max_iter]], [[handle_1]], [[handle_2]], [[tensor_array]])
  // CHECK-SAME: {body_fn = @"map/while/LoopCond_body/MapFnBody", num_tensor_list_or_flow_in = 1 : i32} 
  // CHECK-NOT: tf.While
  %outputs_24:7 = "tf.While"(%outputs, %outputs, %outputs_22#1, %outputs_12, %outputs_16#0, %outputs_18, %outputs_22#0) {_xla_propagate_compile_time_consts = true, body = @"map/while/LoopCond_body", cond = @"map/while/LoopCond_cond", device = "", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<1xi32>, tensor<1xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>) -> (tensor<1xi32>, tensor<1xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>)
  // CHECK-NEXT: tf.TensorArrayGatherV3
  %outputs_26 = "tf.TensorArrayGatherV3"(%outputs_22#0, %outputs_20, %outputs_24#2) {device = "", element_shape = #tf_type.shape<224x224x3>} : (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<?xi32>, tensor<f32>) -> tensor<?x224x224x3xui8>
  return %outputs_26 : tensor<?x224x224x3xui8>
}

// -----
// Test non-applicable while is NOT converted to map_fn.

// CHECK-LABEL: func @while_cond_lt9
func.func @while_cond_lt9(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// CHECK-LABEL: func @while_body_add2
func.func @while_body_add2(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @while_test()
func.func @while_test() -> (tensor<i32>) {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.While
  %1 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  func.return %1 : tensor<i32>
}

// -----

// Test a case that the while body has multiple tensor lists.

// CHECK-LABEL: tf.MultiListWhileRegion_body
func.func private @tf.MultiListWhileRegion_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<!tf_type.variant<tensor<*xf32>>>, %arg4: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00], [8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]]> : tensor<2x8xf32>} : () -> tensor<2x8xf32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[[1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01], [2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01]]> : tensor<2x8xf32>} : () -> tensor<2x8xf32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.GatherV2"(%arg4, %cst_2, %cst_2) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>, tensor<i32>) -> tensor<f32>
  %1 = "tf.AddV2"(%arg0, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%arg1, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.GatherV2"(%cst_0, %arg1, %cst_2) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8xf32>
  %4 = "tf.Mul"(%0, %3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  %5 = "tf.TensorListSetItem"(%arg2, %arg1, %4) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<8xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %6 = "tf.GatherV2"(%cst, %arg1, %cst_2) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8xf32>
  %7 = "tf.Mul"(%0, %6) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<f32>, tensor<8xf32>) -> tensor<8xf32>
  %8 = "tf.TensorListSetItem"(%arg3, %arg1, %7) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<8xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  return %1, %2, %5, %8, %arg4 : tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>
}

// CHECK-LABEL: tf.MultiListWhileRegion_body/MapFnBody
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.GatherV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.GatherV2
// CHECK-NEXT: tf.Mul
// CHECK-NEXT: tf.GatherV2
// CHECK-NEXT: tf.Mul
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf.TensorListSetItem
// CHECK-NEXT: tf.TensorListSetItem
// CHECK-NEXT: tf_mlrt.tf_promise
// CHECK-NEXT: tf_mlrt.tf_promise
// CHECK-NEXT: return

// CHECK-LABEL: tf.MultiListWhileRegion_cond
func.func private @tf.MultiListWhileRegion_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<!tf_type.variant<tensor<*xf32>>>, %arg4: tensor<?xf32>) -> tensor<i1> {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = "tf.Less"(%arg1, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = "tf.LogicalAnd"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %2 : tensor<i1>
}

// CHECK-LABEL: multilist_serving
func.func private @multilist_serving(%arg0: tensor<?xf32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<2x8xf32>, tensor<2x8xf32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<8> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // CHECK: TensorListReserve
  %0 = "tf.TensorListReserve"(%cst_1, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK-NEXT: tf_mlrt.tf_map_fn
  %1:5 = "tf.While"(%cst, %cst, %0, %0, %arg0) {_lower_using_switch_merge = true, _num_original_outputs = 8 : i64, _read_only_resource_inputs = [], _xla_propagate_compile_time_consts = true, body = @tf.MultiListWhileRegion_body, cond = @tf.MultiListWhileRegion_cond, device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = true, parallel_iterations = 4 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>)
  // CHECK-NEXT: TensorListStack
  %2 = "tf.TensorListStack"(%1#2, %cst_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = 2 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<2x8xf32>
  %3 = "tf.TensorListStack"(%1#3, %cst_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = 2 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<2x8xf32>
  return %3, %2 : tensor<2x8xf32>, tensor<2x8xf32>
}


// -----

// Convert a while with multiple tensor array to map_fn

// CHECK-LABEL: tf.WhileRegion1_body(
func.func private @tf.WhileRegion1_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg6: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg7: tensor<*xi32>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<*xi32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.AddV2"(%arg1, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.RaggedTensorToVariant"(%arg7) {RAGGED_RANK = 0 : i64, Tsplits = i64, Tvalues = i32, batched_input = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*xi32>) -> tensor<!tf_type.variant>
  %4 = "tf.TensorArrayWriteV3"(%arg5, %arg1, %3, %arg2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<i32>, tensor<!tf_type.variant>, tensor<f32>) -> tensor<f32>
  %5 = "tf.RaggedTensorToVariant"(%arg7) {RAGGED_RANK = 0 : i64, Tsplits = i64, Tvalues = f32, batched_input = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*xi32>) -> tensor<!tf_type.variant>
  %6 = "tf.TensorArrayWriteV3"(%arg6, %arg1, %5, %arg3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<i32>, tensor<!tf_type.variant>, tensor<f32>) -> tensor<f32>
  return %0, %1, %4, %6, %arg4, %arg5, %arg6, %arg7 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<*xi32>
}

// CHECK-LABEL: func.func private @"tf.WhileRegion1_body/MapFnBody"(%arg0: !mlrt.future, %arg1: !mlrt.promise, %arg2: !mlrt.future, %arg3: !mlrt.promise, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg8: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg9: tensor<*xi32>) attributes {tfrt.cost_threshold = 4294967295 : i64} 
// CHECK: [[result_0:%.*]] = "tf.RaggedTensorToVariant"
// CHECK: [[result_1:%.*]] = "tf.RaggedTensorToVariant"
// CHECK-NEXT: [[flow_in_0:%.*]] = "tf_mlrt.tf_await"(%arg0) : (!mlrt.future) -> tensor<f32>
// CHECK-NEXT: [[flow_in_1:%.*]] = "tf_mlrt.tf_await"(%arg2) : (!mlrt.future) -> tensor<f32>
// CHECK-NEXT: [[flow_out_0:%.*]] = "tf.TensorArrayWriteV3"(%arg7, %arg5, [[result_0]], [[flow_in_0]])
// CHECK-NEXT: [[flow_out_1:%.*]] = "tf.TensorArrayWriteV3"(%arg8, %arg5, [[result_1]], [[flow_in_1]])
// CHECK-NEXT: "tf_mlrt.tf_promise"(%arg1, [[flow_out_0]]) : (!mlrt.promise, tensor<f32>) -> ()
// CHECK-NEXT: "tf_mlrt.tf_promise"(%arg3, [[flow_out_1]]) : (!mlrt.promise, tensor<f32>) -> ()
// CHECK-NEXT: return

// CHECK-LABEL: tf.WhileRegion1_cond
func.func private @tf.WhileRegion1_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg6: tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, %arg7: tensor<*xi32>) -> (tensor<i1>) {
  %0 = "tf.Less"(%arg0, %arg4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
  %1 = "tf.Less"(%arg1, %arg4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<*xi1>
  %2 = "tf.LogicalAnd"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  %3 = "tf.ToBool"(%2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*xi1>) -> tensor<i1>
  return %3 : tensor<i1>
}

// CHECK-LABEL: func.func private @tf.WhileRegion2_body(
func.func private @tf.WhileRegion2_body(%arg0: tensor<*xi32>) -> (tensor<?x!tf_type.variant>, tensor<?x!tf_type.variant>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[-1, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %max_iter = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<4> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.TensorArrayV3"
  %handle_12, %flow_13 = "tf.TensorArrayV3"(%max_iter) {device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = !tf_type.variant, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<f32>)
  // CHECK: "tf.TensorArrayV3"
  %handle_14, %flow_15 = "tf.TensorArrayV3"(%max_iter) {device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = !tf_type.variant, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<f32>)
  // CHECK: tf_mlrt.tf_map_fn
  // CHECK-SAME: {body_fn = @"tf.WhileRegion1_body/MapFnBody", num_tensor_list_or_flow_in = 2 : i32}
  %4:8 = "tf.While"(%cst_0, %cst_0, %flow_13, %flow_15, %max_iter, %handle_12, %handle_14, %arg0) {body = @tf.WhileRegion1_body, cond = @tf.WhileRegion1_cond, device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<*xi32>)
  // CHECK: TensorArrayGatherV3
  %5 = "tf.TensorArrayGatherV3"(%handle_12, %1, %4#2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<i32>, tensor<f32>) -> tensor<?x!tf_type.variant>
  // CHECK: TensorArrayGatherV3
  %6 = "tf.TensorArrayGatherV3"(%handle_14, %2, %4#3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.variant>>>, tensor<i32>, tensor<f32>) -> tensor<?x!tf_type.variant>
  return %5, %6 : tensor<?x!tf_type.variant>, tensor<?x!tf_type.variant>
}

// -----

// Test a while to map_fn conversion in which tensor array is used instead of
// tensor list and the tensor array size and the number of iterations are bounded
// by separate constants of the same value.

// CHECK-LABEL: map2/while/LoopCond_body
func.func private @"map2/while/LoopCond_body"(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, %arg5: tensor<f32>, %arg6: tensor<2x!tf_type.resource<tensor<*xui8>>>) -> (tensor<*xi32>, tensor<*xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>) {
  %outputs = "tf.Const"() {value = dense<224> : tensor<2xi32>} : () -> tensor<2xi32>
  %outputs_0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_4 = "tf.Identity"(%arg0) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_6 = "tf.AddV2"(%outputs_4, %outputs_2) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %outputs_8 = "tf.Identity"(%arg1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  %outputs_10 = "tf.AddV2"(%outputs_8, %outputs_2) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %outputs_12 = "tf.Identity"(%arg2) {device = ""} : (tensor<f32>) -> tensor<f32>
  %outputs_14 = "tf.TensorArrayReadV3"(%arg4, %outputs_8, %arg5) {device = ""} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<*xi32>, tensor<f32>) -> tensor<*x!tf_type.string>
  %outputs_16 = "tf.DecodeJpeg"(%outputs_14) {acceptable_fraction = 1.000000e+00 : f32, channels = 3 : i64, dct_method = "INTEGER_FAST", device = "", fancy_upscaling = true, ratio = 1 : i64, try_recover_truncated = false} : (tensor<*x!tf_type.string>) -> tensor<?x?x3xui8>
  %outputs_18 = "tf.ExpandDims"(%outputs_16, %outputs_0) {device = ""} : (tensor<?x?x3xui8>, tensor<i32>) -> tensor<1x?x?x3xui8>
  %outputs_20 = "tf.ResizeBilinear"(%outputs_18, %outputs) {align_corners = false, device = "", half_pixel_centers = false} : (tensor<1x?x?x3xui8>, tensor<2xi32>) -> tensor<1x224x224x3xf32>
  %outputs_22 = "tf.Squeeze"(%outputs_20) {device = "", squeeze_dims = [0]} : (tensor<1x224x224x3xf32>) -> tensor<224x224x3xf32>
  %outputs_24 = "tf.Cast"(%outputs_22) {Truncate = false, device = ""} : (tensor<224x224x3xf32>) -> tensor<224x224x3xui8>
  %outputs_26 = "tf.TensorArrayWriteV3"(%arg6, %outputs_8, %outputs_24, %outputs_12) {device = ""} : (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<*xi32>, tensor<224x224x3xui8>, tensor<f32>) -> tensor<f32>
  return %outputs_6, %outputs_10, %outputs_26, %arg3, %arg4, %arg5, %arg6: tensor<*xi32>, tensor<*xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>
}

// CHECK-LABEL: @"map2/while/LoopCond_body/MapFnBody"
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.Const
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.TensorArrayReadV3
// CHECK-NEXT: tf.DecodeJpeg
// CHECK-NEXT: tf.ExpandDims
// CHECK-NEXT: tf.ResizeBilinear
// CHECK-NEXT: tf.Squeeze
// CHECK-NEXT: tf.Cast
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf.TensorArrayWriteV3
// CHECK-NEXT: tf_mlrt.tf_promise
// CHECK-NEXT: return

// CHECK-LABEL: map2/while/LoopCond_cond
func.func private @"map2/while/LoopCond_cond"(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, %arg5: tensor<f32>, %arg6: tensor<2x!tf_type.resource<tensor<*xui8>>>) -> tensor<i1> {
  %cst = "tf.Const"() {value = dense<224> : tensor<i32>} : () -> tensor<i32>
  %outputs = "tf.Less"(%arg0, %cst) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %outputs_0 = "tf.Less"(%arg1, %cst) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi1>
  %outputs_2 = "tf.LogicalAnd"(%outputs, %outputs_0) {device = ""} : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
  %outputs_4 = "tf.ToBool"(%outputs_2) : (tensor<*xi1>) -> tensor<i1>
  return %outputs_4 : tensor<i1>
}

//CHECK-LABEL: map2_while_test
func.func private @map2_while_test(%arg0: tensor<?x!tf_type.string>) -> tensor<?x224x224x3xui8> {
  // CHECK-NEXT: tf.Const
  %outputs = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<1xi32>
  // CHECK-NEXT: [[max_iter:%.*]] = "tf.Const"
  %cst_0 = "tf.Const"() {value = dense<224> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %outputs_2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %outputs_4 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_6 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %outputs_8 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %outputs_10 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<1xi32>
  // CHECK: tf.Range
  %outputs_14 = "tf.Range"(%outputs_2, %cst_0, %outputs_8) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-NEXT: tf.TensorArrayV3
  %outputs_16:2 = "tf.TensorArrayV3"(%cst_0) {clear_after_read = true, device = "", dtype = !tf_type.string, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>)
  // CHECK-NEXT: tf.TensorArrayScatterV3
  %outputs_18 = "tf.TensorArrayScatterV3"(%outputs_16#0, %outputs_14, %arg0, %outputs_16#1) {device = ""} : (tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<?xi32>, tensor<?x!tf_type.string>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: tf.Range
  %outputs_20 = "tf.Range"(%outputs_2, %cst_0, %outputs_8) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-NEXT: [[tensor_array:%.*]], [[flow_in:%.*]] = "tf.TensorArrayV3"
  %outputs_22:2 = "tf.TensorArrayV3"(%cst_0) {clear_after_read = true, device = "", dtype = ui8, dynamic_size = false, element_shape = #tf_type.shape<*>, identical_element_shapes = true, tensor_array_name = ""} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<f32>)
  // CHECK-NEXT: tf_mlrt.tf_map_fn
  // CHECK-SAME: ([[max_iter]], [[flow_in]], %cst_1
  // CHECK-SAME: {body_fn = @"map2/while/LoopCond_body/MapFnBody", num_tensor_list_or_flow_in = 1 : i32}
  // CHECK-NOT: tf.While
  %outputs_24:7 = "tf.While"(%outputs, %outputs, %outputs_22#1, %cst_1, %outputs_16#0, %outputs_18, %outputs_22#0) {_xla_propagate_compile_time_consts = true, body = @"map2/while/LoopCond_body", cond = @"map2/while/LoopCond_cond", device = "", is_stateless = false, parallel_iterations = 10 : i64, shape_invariant} : (tensor<1xi32>, tensor<1xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>) -> (tensor<1xi32>, tensor<1xi32>, tensor<f32>, tensor<i32>, tensor<2x!tf_type.resource<tensor<*x!tf_type.string>>>, tensor<f32>, tensor<2x!tf_type.resource<tensor<*xui8>>>)
  // CHECK-NEXT: tf.TensorArrayGatherV3
  %outputs_26 = "tf.TensorArrayGatherV3"(%outputs_22#0, %outputs_20, %outputs_24#2) {device = "", element_shape = #tf_type.shape<224x224x3>} : (tensor<2x!tf_type.resource<tensor<*xui8>>>, tensor<?xi32>, tensor<f32>) -> tensor<?x224x224x3xui8>
  return %outputs_26 : tensor<?x224x224x3xui8>
}

// -----
// Test a nest while in which the while body is after the usage.

// CHECK-LABEL: nested_while
func.func @nested_while(%arg0: tensor<?xf32> {tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<16x16x?xf32>)  {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[16, -1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.TensorListReserve
  %0 = "tf.TensorListReserve"(%cst_1, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  // CHECK-NEXT: tf_mlrt.tf_map_fn
  %1:4 = "tf.While"(%cst, %cst, %0, %arg0) {_lower_using_switch_merge = true, _num_original_outputs = 6 : i64, _read_only_resource_inputs = [], _xla_propagate_compile_time_consts = true, body = @tf.NestedWhileRegion1_body, cond = @tf.NestedWhileRegion1_cond, device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = true, parallel_iterations = 4 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>)
  %2 = "tf.TensorListStack"(%1#2, %cst_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = 16 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<2xi32>) -> tensor<16x16x?xf32>
  return %2 : tensor<16x16x?xf32>
}
// CHECK-LABEL: tf.NestedWhileRegion1_body
func.func private @tf.NestedWhileRegion1_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>} : () -> tensor<16xi32>
  %cst_2 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %cst_3 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %cst_4 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.TensorListReserve"(%cst_4, %cst_3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %1 = "tf.AddV2"(%arg0, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.AddV2"(%arg1, %cst_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.GatherV2"(%cst_1, %arg1, %cst_0) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<16xi32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %4 = "tf.Cast"(%3) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  %5 = "tf.Mul"(%arg3, %4) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %6:4 = "tf.While"(%cst_0, %cst_0, %0, %5) {_lower_using_switch_merge = true, _num_original_outputs = 6 : i64, _read_only_resource_inputs = [], _xla_propagate_compile_time_consts = true, body = @tf.NestedWhileRegion_body, cond = @tf.NestedWhileRegion_cond, device = "/job:localhost/replica:0/task:0/device:CPU:0", is_stateless = true, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>)
  %7 = "tf.TensorListStack"(%6#2, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0", num_elements = 16 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<16x?xf32>
  %8 = "tf.TensorListSetItem"(%arg2, %arg1, %7) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<16x?xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  return %1, %2, %8, %arg3 : tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>
}

//CHECK-LABEL: @"tf.NestedWhileRegion1_body/MapFnBody"(%arg0: !mlrt.future, %arg1: !mlrt.promise, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<?xf32>) 
// CHECK: tf.TensorListReserve
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.GatherV2
// CHECK-NEXT: tf.Cast
// CHECK-NEXT: tf.Mul
// CHECK-NEXT: tf_mlrt.tf_map_fn
// CHECK-NEXT: tf.TensorListStack
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf.TensorListSetItem
// CHECK-NEXT: tf_mlrt.tf_promise
// CHECK-NEXT: return

// CHECK-LABEL: tf.NestedWhileRegion1_cond
func.func private @tf.NestedWhileRegion1_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> tensor<i1> {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = "tf.Less"(%arg1, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = "tf.LogicalAnd"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %2 : tensor<i1>
}
// CHECK-LABEL: tf.NestedWhileRegion_body
func.func private @tf.NestedWhileRegion_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>) {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %cst_0 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi32>} : () -> tensor<16xi32>
  %cst_1 = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.AddV2"(%arg1, %cst_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.GatherV2"(%cst_0, %arg1, %cst) {batch_dims = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<16xi32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  %3 = "tf.Cast"(%2) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  %4 = "tf.Mul"(%arg3, %3) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %5 = "tf.TensorListSetItem"(%arg2, %arg1, %4) {device = "/job:localhost/replica:0/task:0/device:CPU:0", resize_if_index_out_of_bounds = false} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<?xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  return %0, %1, %5, %arg3 : tensor<i32>, tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<?xf32>
}

// CHECK-LABEL: tf.NestedWhileRegion_body/MapFnBody
// CHECK: tf.AddV2
// CHECK-NEXT: tf.AddV2
// CHECK-NEXT: tf.GatherV2
// CHECK-NEXT: tf.Cast
// CHECK-NEXT: tf.Mul
// CHECK-NEXT: tf_mlrt.tf_await
// CHECK-NEXT: tf.TensorListSetItem
// CHECK-NEXT: "tf_mlrt.tf_promise
// CHECK-NEXT: return

// CHECK-LABEL: tf.NestedWhileRegion_cond
func.func private @tf.NestedWhileRegion_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<!tf_type.variant<tensor<*xf32>>>, %arg3: tensor<?xf32>) -> tensor<i1> {
  %cst = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<16> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = "tf.Less"(%arg1, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %2 = "tf.LogicalAnd"(%0, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %2 : tensor<i1>
}

