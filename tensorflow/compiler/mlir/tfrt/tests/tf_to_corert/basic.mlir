// RUN: tf-tfrt-opt -pass-pipeline='builtin.func(tf-tensor-device-copy),tfrt-lower-tf-savedmodel{hoist-invariant-ops=true},tf-to-tfrt{tfrt-cost-threshold=1024 tfrt-upper-cost-threshold=65536 tfrt-merge-inter-dependent-streams=true}' %s | FileCheck %s --dump-input-filter=all

// CHECK-NOT: tf_saved_model.semantics
// CHECK: tfrt.cost_threshold = 1024
// CHECK-SAME: tfrt.merge_inter_dependent_streams = true
// CHECK-SAME: tfrt.upper_cost_threshold = 65536
module attributes {tf_saved_model.semantics} {

// CHECK-NOT: "tf_saved_model.global_tensor"
"tf_saved_model.global_tensor"() {is_mutable, sym_name = "y", type = tensor<1x3xf32>, value = dense<[[1.67482901, -0.529208779, -0.803792417]]> : tensor<1x3xf32>} : () -> ()

// CHECK-NOT: "tf_saved_model.session_initializer"
"tf_saved_model.session_initializer"() { initializers = [@func_init] } : () -> ()

// CHECK-LABEL: _tfrt_resource_init
// CHECK: tf.VarHandleOp
// CHECK: tf.ReadVariableOp
// CHECK: tfrt_fallback_async.set_resource
// CHECK-SAME: {device = "/device:CPU:0", index = 0 : i64}


// CHECK-LABEL: func @init
// CHECK-SAME: {tfrt.cost_threshold = 1 : i64}
func @func_init() attributes {tf_saved_model.exported_names = ["init"]} {
  return
}

// CHECK-LABEL: func @basic
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain
// CHECK-SAME: [[arg0_th:%.*]]: !corert.tensorhandle,
// CHECK-SAME: [[arg1_th:%.*]]: !corert.tensorhandle {tf.resource_name = "y"})
// CHECK-SAME: -> (!tfrt.chain, !corert.tensorhandle)
func @func_basic(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>> {tf_saved_model.bound_input = @y})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["basic"]} {
  // CHECK-NEXT: [[cpu_device:%.*]] = corert.get_op_handler
  // CHECK-SAME: "/device:CPU:0"

  // CHECK-NOT: tf.VarHandleOp
  %handle = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<3xf32>>>
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%handle) {_output_shapes = ["tfshape$dim { size: 3 }"], device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>
  // CHECK-NOT: tf.ReadVariableOp
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>

  // CHECK-NEXT: [[ch:%.*]], [[result:%.*]] = tfrt_fallback_async.get_resource [[in_chain]] {device = "/device:CPU:0", indices = [0]} : (!tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  // CHECK-NEXT: [[r0_th:%.*]] = corert.executeop([[cpu_device]]) "tf.MatMul"([[arg0_th]], [[arg1_th]])
  %2 = "tf.MatMul"(%arg0, %1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: [[result_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[result]] {device = "/device:CPU:0"}
  // CHECK-NEXT: [[r1_th:%.*]] = corert.executeop([[cpu_device]]) "tf.BiasAdd"([[r0_th]], [[result_th]])
  %3 = "tf.BiasAdd"(%2, %0) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], data_format = "NHWC", device = "/device:CPU:0"} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: [[r2_th:%.*]] = corert.executeop([[cpu_device]]) "tf.Tanh"([[r1_th]]) {T = f32, device = "/device:CPU:0"}
  %4 = "tf.Tanh"(%3) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NOT: tf.Identity
  %5 = "tf.Identity"(%4) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NOT: tf.IdentityN
  %6:2 = "tf.IdentityN"(%5, %4) {T = [f32, f32], _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }", "tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>, tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  // CHECK-NEXT: [[out_ch:%.*]] = tfrt.merge.chains [[ch]], [[in_chain]] : !tfrt.chain, !tfrt.chain
  // CHECK-NEXT: tfrt.return [[out_ch]], [[r2_th]] : !tfrt.chain, !corert.tensorhandle
  return %6#0 : tensor<3x3xf32>
}

}
