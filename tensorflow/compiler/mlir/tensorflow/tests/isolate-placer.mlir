// RUN: tf-opt %s --run-tf-graph-optimization --graph-passes=IsolatePlacerInspectionRequiredOpsPass  | FileCheck %s

func @main() {
  %0:2 = "_tf.VarHandleOp"() {dtype = "tfdtype$DT_FLOAT", shape = "tfshape$"} : () -> (tensor<!tf.resource>, !_tf.control)
  %1:2 = "_tf.StatefulPartitionedCall"(%0#0) {Tin = ["tfdtype$DT_RESOURCE"], Tout = ["tfdtype$DT_RESOURCE"], config = "", config_proto = "", executor_type = "", f = @foo} : (tensor<!tf.resource>) -> (tensor<!tf.resource>, !_tf.control) loc("call_foo")
  return
}

func @foo(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> {
  return %arg0 : tensor<!tf.resource>
}

// The IsolatePlacerInspectionRequiredOpsPass adds Identities for each input/output of function-calling ops.

// Capture the result of input to function call.
// CHECK:      [[VARIABLE_REG:%[0-9]*]]:2 = tf_executor.island wraps "tf.VarHandleOp"()

// Test for the presence of Identity op between input and function call.
// CHECK: [[IDENTITY_REG:%[0-9]*]]:2 = tf_executor.island wraps "tf.Identity"([[VARIABLE_REG]]#0)

// CHECK: [[CALL_RESULT_REG:%[0-9]*]]:2 = tf_executor.island wraps "tf.StatefulPartitionedCall"([[IDENTITY_REG]]#0)
// CHECK-SAME: f = @[[FUNCTION:[a-zA-Z0-9_]*]]

// Match the inserted Identity op for call output.
// CHECK: "tf.Identity"([[CALL_RESULT_REG]]#0)

// Match the function name
// CHECK: func @[[FUNCTION]]
