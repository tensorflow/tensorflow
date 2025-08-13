// RUN: tf-opt %s -allow-unregistered-dialect --tf-order-for-program-key --split-input-file | FileCheck %s

// CHECK-LABEL: func @test_reorder_launches
func.func @test_reorder_launches() {
  // CHECK: TPUCompileMlir
  // CHECK: tf.OpA
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  return
}

// -----

// CHECK-LABEL: func @test_reorder_ops
func.func @test_reorder_ops() {
  // CHECK: TPUCompileMlir
  // CHECK: tf.OpA
  // CHECK: tf.OpB
  "tf.OpA"() : () -> ()
  %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.OpB"() : () -> ()
  return
}

// -----

// CHECK-LABEL: func @test_compile_with_dependencies
func.func @test_compile_with_dependencies() {
  // CHECK: tf.OpA
  // CHECK: TPUCompileMlir
  // CHECK: tf.OpB
  %a = "tf.OpA"() : () -> tensor<i64>
  %b = "tf.OpB"() : () -> tensor<i64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%a) { metadata = "...", mlir_module = "..." } : (tensor<i64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  return
}

// -----

// CHECK-LABEL: func @test_side_effects
func.func @test_side_effects(%arg0: tensor<!tf_type.resource<tensor<i64>>>) {
  // CHECK: ReadVariable
  // CHECK: TPUCompileMlir
  // CHECK: AssignVariable
  %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i64>>>) -> tensor<i64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%1) { metadata = "...", mlir_module = "..." } : (tensor<i64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  %2 = "tf.Cast"(%compilation_status) : (tensor<!tf_type.string>) -> tensor<i64>
  %3 = builtin.unrealized_conversion_cast to tensor<i64>
  "tf.AssignVariableOp"(%arg0, %2) : (tensor<!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()
  "tf.AssignVariableOp"(%arg0, %3) : (tensor<!tf_type.resource<tensor<i64>>>, tensor<i64>) -> ()
  return
}
