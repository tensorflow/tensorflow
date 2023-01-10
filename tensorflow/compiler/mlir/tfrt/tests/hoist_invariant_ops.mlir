// RUN: tf-tfrt-opt -split-input-file -tfrt-lower-tf-savedmodel=hoist-invariant-ops=true %s | FileCheck %s --dump-input=fail --dump-input-filter=all

module attributes {tf_saved_model.semantics} {

// Test hoisting varhandle op.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[handle:%.*]] = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
// CHECK: [[x:%.*]] = "tf.ReadVariableOp"([[handle]]) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
// CHECK: "tf._TfrtSetResource"([[x]]) {device = "/CPU:0", index = 0 : i64} : (tensor<i32>) -> ()

// CHECK-LABEL: func @test_hoist_varhandleop
func.func @hoist_varhandleop(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_varhandleop"]} {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.ReadVariableOp
  // CHECK: [[v:%.*]] = "tf._TfrtGetResource"() {container = [""], device = "/CPU:0", indices = [0], shared_name = [""]} : () -> tensor<i32>
  // CHECK: [[r:%.*]] = "tf.AddV2"({{.*}}, [[v]]) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: return [[r]]
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %x = "tf.ReadVariableOp"(%handle) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test hoisting hash table op.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[handle:%.*]] = "tf.HashTableV2"()
// CHECK-SAME: shared_name = "x"
// CHECK: "tf._TfrtSetResource"([[handle]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0", index = [[handle_idx:.*]] : i64}
// CHECK: [[x:%.*]] = "tf.LookupTableSizeV2"([[handle]])
// CHECK: "tf._TfrtSetResource"([[x]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0", index = [[size_idx:.*]] : i64} : (tensor<i64>) -> ()

// CHECK: func @test_hoist_hash_table
func.func @hoist_hash_table(%arg: tensor<?x!tf_type.string> {tf_saved_model.index_path = ["input"]}, %default: tensor<i64> {tf_saved_model.index_path = ["default"]}) -> (tensor<i64> {tf_saved_model.index_path = ["r"]}, tensor<*xi64> {tf_saved_model.index_path = ["r1"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_hash_table"]} {
  // CHECK-NOT: tf.HashTableV2
  // CHECK-NOT: tf.LookupTableSizeV2
  // CHECK: [[v:%.*]]:2 = "tf._TfrtGetResource"() {container = ["", ""], device = "/job:localhost/replica:0/task:0/device:CPU:0", indices = [0, 1], shared_name = [{{.*}}, {{.*}}]}
  // CHECK: [[r:%.*]] = "tf.LookupTableFindV2"([[v]]#[[handle_idx]]
  // CHECK: return [[v]]#[[size_idx]], [[r]]
  %0 = "tf.HashTableV2"() {container = "", device = "", key_dtype = !tf_type.string, shared_name = "x", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
  %1 = "tf.LookupTableSizeV2"(%0) {device = ""} : (tensor<!tf_type.resource>) -> tensor<i64>
  %2 = "tf.LookupTableFindV2"(%0, %arg, %default) {device = "/CPU:0"} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
  func.return %1, %2 : tensor<i64>, tensor<*xi64>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test hoisting const op.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[const:%.*]] = "tf.Const"() {device = "/CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK: [[x:%.*]] = "tf.AddV2"([[const]], [[const]]) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK: "tf._TfrtSetResource"([[x]]) {device = "/CPU:0", index = 0 : i64} : (tensor<i32>) -> ()
// CHECK: [[const_1:%.*]] = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK: "tf._TfrtSetResource"([[const_1]]) {device = "/CPU:0", index = 1 : i64} : (tensor<i32>) -> ()

// CHECK-LABEL: func @test_hoist_const
func.func @hoist_const(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_const"]} {
  // CHECK-NOT: tf.Const
  // CHECK: [[v:%.*]] = "tf._TfrtGetResource"() {container = [""], device = "/CPU:0", indices = [0], shared_name = [""]} : () -> tensor<i32>
  // CHECK-NEXT: "tf.AddV2"({{.*}}, [[v]]) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: return
  %const = "tf.Const"() {device = "/CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %x = "tf.AddV2"(%const, %const) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

// CHECK-LABEL: func @test_hoist_const_return
func.func @hoist_const_return(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_const_return"]} {
  // CHECK-NOT: tf.Const
  // CHECK: [[v:%.*]] = "tf._TfrtGetResource"() {container = [""], device = "/CPU:0", indices = [1], shared_name = [""]} : () -> tensor<i32>
  // CHECK-NEXT: return [[v]]
  %const = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  func.return %const : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test hoisting write side-effect ops.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[const:%.*]] = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK: "tf._TfrtSetResource"([[const]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0", index = [[const_idx:.*]] : i64} : (tensor<i32>) -> ()
// CHECK: [[handle:%.*]] = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
// CHECK: "tf._TfrtSetResource"([[handle]]) {device = "/job:localhost/replica:0/task:0/device:CPU:0", index = [[handle_idx:.*]] : i64} : (tensor<!tf_type.resource<tensor<i32>>>) -> ()

// CHECK: func @test_hoist_var_read_write
func.func @hoist_var_read_write() -> (tensor<i32> {tf_saved_model.index_path = ["x"]}, tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_hoist_var_read_write"]} {
  // CHECK-NOT: tf.Const
  // CHECK-NOT: tf.VarHandleOp
  // CHECK: [[v:%.*]]:2 = "tf._TfrtGetResource"() {container = ["", ""], device = "/job:localhost/replica:0/task:0/device:CPU:0", indices = [0, 1], shared_name = [{{.*}}, {{.*}}]} : () -> ({{.*}})
  // CHECK: [[x:%.*]] = "tf.ReadVariableOp"([[v]]#[[handle_idx]]) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK-NEXT: "tf.AssignVariable"([[v]]#[[handle_idx]], [[v]]#[[const_idx]]) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  // CHECK-NEXT: [[r:%.*]] = "tf.ReadVariableOp"([[v]]#[[handle_idx]]) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  // CHECK-NEXT: return [[x]], [[r]]
  %const = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %x = "tf.ReadVariableOp"(%handle) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  "tf.AssignVariable"(%handle, %const) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  %r = "tf.ReadVariableOp"(%handle) {device = "/CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  func.return %x, %r : tensor<i32>, tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test not hoisting read variable op that used by control flow ops if var handle op and read variable op are separated, but still hoists const ops and var handle ops.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK: [[handle:%.*]] = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
// CHECK: "tf._TfrtSetResource"([[handle]])
// CHECK-SAME: index = [[handle_index:.*]]
// CHECK: [[handle1:%.*]] = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
// CHECK: "tf._TfrtSetResource"([[handle1]])
// CHECK-SAME: index = [[handle1_index:.*]]
// CHECK: [[const:%.*]] = "tf.Const"() {device = "/CPU:0", value = dense<true> : tensor<i1>} : () -> tensor<i1>
// CHECK: "tf._TfrtSetResource"([[const]])
// CHECK-SAME: index = [[const_index:.*]]
func.func private @some_func(
    %arg: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg) {device = "cpu"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @test_not_hoist_stateful_call
func.func @not_hoist_stateful_call(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_not_hoist_stateful_call"]} {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK:  "tf._TfrtGetResource"()
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK: tf.StatefulPartitionedCall
  %x = "tf.StatefulPartitionedCall"(%handle) {device = "/CPU:0", config = "", config_proto = "", executor_type = "", f = @some_func} : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<i32>)
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

// CHECK-LABEL: func @test_not_hoist_if
func.func @not_hoist_if(%arg: tensor<i32> {tf_saved_model.index_path = ["input"]}) -> (tensor<i32> {tf_saved_model.index_path = ["r"]})
  attributes {tf_saved_model.exported_names = ["test_not_hoist_if"]} {
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK-NOT: tf.Const
  // CHECK:  "tf._TfrtGetResource"() 
  %cond = "tf.Const"() {device = "/CPU:0", value = dense<true> : tensor<i1>} : () -> tensor<i1>
  // CHECK: tf.If
  %x = "tf.If"(%cond, %handle) {then_branch = @some_func, else_branch = @some_func, is_stateless = false} : (tensor<i1>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %r = "tf.AddV2"(%arg, %x) {device = "/CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test hoist var handle op and read variable op in the batch function.

// CHECK-LABEL: func private @batched_function
func.func private @batched_function(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32>
  attributes {tf._input_shapes = [#tf_type.shape<1x3>, #tf_type.shape<*>], tf.signature.is_stateful} {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK-NOT: tf.ReadVariableOp
  // CHECK:  "tf._TfrtGetResource"() 
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %2 = "tf.AddV2"(%arg0, %1) {device = "/device:CPU:0"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %3 = "tf.Identity"(%2) {device = "/device:CPU:0"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %3 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32> {tf_saved_model.index_path = ["input"]}) -> (tensor<*xf32> {tf_saved_model.index_path = ["r"]}) 
  attributes {tf_saved_model.exported_names = ["main"]} {
  // CHECK-NOT: tf.VarHandleOp
  // CHECK:  "tf._TfrtGetResource"() 
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  // CHECK: "tf.BatchFunction"(%arg0, %0)
  // CHECK: operand_segment_sizes = array<i32: 1, 1>
  %1 = "tf.BatchFunction"(%arg0, %0) {allowed_batch_sizes = [6], batch_timeout_micros = 100000 : i64, batching_queue = "", container = "", device = "/device:CPU:0", enable_large_batch_splitting = false, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>, shared_name = "batch/"} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test not hoisting callees in init functions.

"tf_saved_model.session_initializer"() {initializers = [@init]} : () -> ()

func.func @init() attributes {tf_saved_model.exported_names = ["__tf_saved_model_session_initializer_init"]} {
  %var0 = "tf.VarHandleOp"() {container = "", shared_name = "var0"} : () -> tensor<!tf_type.resource<tensor<i1>>>
  %cond = "tf.ReadVariableOp"(%var0) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i1>>>) -> tensor<i1>
  %x = "tf.StatefulPartitionedCall"(%cond) {device = "/CPU:0", config = "", config_proto = "", executor_type = "", f = @some_func} : (tensor<i1>) -> (tensor<i32>)
  %var1 = "tf.VarHandleOp"() {container = "", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  "tf.AssignVariable"(%var1, %x) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}


// CHECK-LABEL: func @_tfrt_resource_init
// CHECK-NEXT: return

// CHECK-LABEL: func private @some_func
func.func private @some_func(%arg: tensor<i1>) -> tensor<i32> {
  // CHECK-NOT: tf._TfrtGetResource
  %const = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0 = "tf.ReadVariableOp"(%handle) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %r = "tf.SelectV2"(%arg, %const, %0) {device = "/CPU:0"} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test not hoisting in TPU functions.

// CHECK-LABEL: func @_tfrt_resource_init
// CHECK-NEXT: return

// CHECK-LABEL: func private @func2
func.func private @func2(%arg: tensor<i1>) -> tensor<i32> {
  // CHECK-NOT: tf._TfrtGetResource
  "tf.TPUReplicateMetadata"() {_tpu_replicate = "0",  allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], host_compute_core = [], num_cores_per_replica = 4 : i64, num_replicas = 1 : i64, padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : () -> ()
  %const = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  %0 = "tf.ReadVariableOp"(%handle) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %r = "tf.SelectV2"(%arg, %const, %0) {device = "/CPU:0"} : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

}
