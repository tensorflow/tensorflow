// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @assign_variable
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain) -> !tfrt.chain
func.func @assign_variable() {
  // CHECK: [[ch1:%.*]], %results = tfrt_fallback_async.executeop.seq([[in_chain]]) key(0) cost({{.*}}) device("/device:CPU:0") "tf.VarHandleOp"
  // CHECK-NEXT: [[ch2:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(1) cost({{.*}}) device("/device:CPU:0") "tf.AssignVariableOp"
  // CHECK-NEXT: [[out_ch:%.*]] = tfrt.merge.chains [[ch1]], [[ch2]]
  // CHECK-NEXT: tfrt.return [[out_ch]]

  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<4.200000e+01> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  "tf.AssignVariableOp"(%1, %0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<f32>>>, tensor<f32>) -> ()
  func.return
}
