// RUN: tf-tfrt-opt -pass-pipeline='builtin.module(func.func(tf-tensor-device-copy),tfrt-lower-tf-savedmodel{hoist-invariant-ops=true},tf-to-tfrt{tfrt-cost-threshold=1024 tfrt-merge-inter-dependent-streams=true})' %s | FileCheck %s --dump-input-filter=all

// CHECK-NOT: tf_saved_model.semantics
// CHECK: tfrt.cost_threshold = 1024
// CHECK-SAME: tfrt.merge_inter_dependent_streams = true
module attributes {tf_saved_model.semantics} {

// CHECK-NOT: "tf_saved_model.session_initializer"
"tf_saved_model.session_initializer"() { initializers = [@func_init] } : () -> ()

// CHECK-LABEL: _tfrt_resource_init
// CHECK: tf.VarHandleOp
// CHECK: tf.ReadVariableOp
// CHECK: tfrt_fallback_async.set_resource
// CHECK-SAME: {device = "/device:CPU:0", index = 0 : i64}


// CHECK-LABEL: func @init
// CHECK-SAME: {tfrt.cost_threshold = 1 : i64}
func.func @func_init() attributes {tf_saved_model.exported_names = ["init"]} {
  func.return
}

// CHECK-LABEL: func @basic
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain
// CHECK-SAME: [[arg0:%.*]]: !tfrt_fallback.tf_tensor,
// CHECK-SAME: [[arg1:%.*]]: !tfrt_fallback.tf_tensor)
// CHECK-SAME: -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @func_basic(
    %arg0: tensor<3x1xf32> {tf_saved_model.index_path = [0]},
    %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>> {tf_saved_model.index_path = [1]})
      -> (tensor<3x3xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["basic"]} {
  %handle = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<3xf32>>>
  %0 = "tf.ReadVariableOp"(%handle) {_output_shapes = ["tfshape$dim { size: 3 }"], device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>

  // CHECK-NEXT: [[ready_ch:%.*]] = tfrt.new.chain
  // CHECK-NEXT: [[ch:%.*]], [[result:%.*]] = tfrt_fallback_async.get_resource [[ready_ch]] {device = "/device:CPU:0", indices = [0]} : (!tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  // CHECK-NEXT: [[ch1:%.*]], [[var:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) {{.*}} "tf.ReadVariableOp"([[arg1]])
  // CHECK-NEXT: [[r0:%.*]] = tfrt_fallback_async.executeop {{.*}} "tf.MatMul"([[arg0]], [[var]])
  %2 = "tf.MatMul"(%arg0, %1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: [[r1:%.*]] = tfrt_fallback_async.executeop {{.*}} "tf.BiasAdd"([[r0]], [[result]])
  %3 = "tf.BiasAdd"(%2, %0) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], data_format = "NHWC", device = "/device:CPU:0"} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: [[r2:%.*]] = tfrt_fallback_async.executeop {{.*}} "tf.Tanh"([[r1]]) {T = f32}
  %4 = "tf.Tanh"(%3) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NOT: tf.Identity
  %5 = "tf.Identity"(%4) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NOT: tf.IdentityN
  %6:2 = "tf.IdentityN"(%5, %4) {T = [f32, f32], _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }", "tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0"} : (tensor<3x3xf32>, tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3x3xf32>)
  // CHECK-NEXT: [[out_ch:%.*]] = tfrt.merge.chains [[ch]], [[ch1]] : !tfrt.chain, !tfrt.chain
  // CHECK-NEXT: tfrt.return [[out_ch]], [[r2]] : !tfrt.chain, !tfrt_fallback.tf_tensor
  func.return %6#0 : tensor<3x3xf32>
}

}
