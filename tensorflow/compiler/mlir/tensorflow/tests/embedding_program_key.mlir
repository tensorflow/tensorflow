// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-embedding-program-key | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: func @single_op_program_key
func.func @single_op_program_key() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  return
}

// CHECK-LABEL: func @multiple_ops_program_key
func.func @multiple_ops_program_key() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpB"(%[[COMPILE_LAUNCH]]#1
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
    "tf.OpB"(%cst_0) { mini_batch_in_csr = ""} : (tensor<1x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  return
}

// CHECK-LABEL: func @reorder_single_op_program_key
func.func @reorder_single_op_program_key() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
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

// CHECK-LABEL: func @reorder_multiple_ops_program_key
func.func @reorder_multiple_ops_program_key() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpB"(%[[COMPILE_LAUNCH]]#1
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1= "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> (tensor<2xi32>)
    "tf.OpB"(%cst_0, %1) { mini_batch_in_csr = ""} : (tensor<1x!tf_type.string>, tensor<2xi32>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  return
}

// CHECK-LABEL: func @reorder_multiple_ops_with_successors_program_key
func.func @reorder_multiple_ops_with_successors_program_key() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpC"
  // CHECK: "tf.OpB"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpD"
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1= "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> (tensor<2xi32>)
    %2 = "tf.OpC"(%1) {} : (tensor<2xi32>) -> (tensor<2xi32>)
    %3 = "tf.OpB"(%cst_0, %2) { mini_batch_in_csr = ""} : (tensor<1x!tf_type.string>, tensor<2xi32>) -> (tensor<2xi32>)
    %4 = "tf.OpD"(%3) {} : (tensor<2xi32>) -> (tensor<2xi32>)
    tf_device.return %4 : tensor<2xi32>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<2xi32>)
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  return
}

// CHECK-LABEL: func @launch_intermediate_usage
func.func @launch_intermediate_usage() {
  // CHECK: %[[ORIG_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: "tf.OpG"(%[[ORIG_LAUNCH]]#1
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: "tf.OpA"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpC"
  // CHECK: "tf.OpB"(%[[COMPILE_LAUNCH]]#1
  // CHECK: "tf.OpD"
  %0:2 = "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    %1 = "tf.OpF"() {} : () -> (tensor<2xi32>)
    %2= "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> (tensor<2xi32>)
    %3 = "tf.OpC"(%2) {} : (tensor<2xi32>) -> (tensor<2xi32>)
    %4 = "tf.OpB"(%cst_0, %3) { mini_batch_in_csr = ""} : (tensor<1x!tf_type.string>, tensor<2xi32>) -> (tensor<2xi32>)
    %5 = "tf.OpD"(%4) {} : (tensor<2xi32>) -> (tensor<2xi32>)
    tf_device.return %4, %1 : tensor<2xi32>, tensor<2xi32>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<2xi32>, tensor<2xi32>)
  "tf.OpG"(%0#1) : (tensor<2xi32>) -> ()
  %6:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  return
}

// CHECK-LABEL: func @preprocess_not_in_launch
func.func @preprocess_not_in_launch() {
  // CHECK: %[[COMPILE_LAUNCH:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: TPUCompileMlir
  // CHECK: %[[CONSTANT:[a-z0-9]*]] = "tf.Const"
  // CHECK: "tf.OpA"(%[[CONSTANT]]
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
  return
}

// CHECK-LABEL: func @compile_not_in_launch
func.func @compile_not_in_launch() {
  // CHECK: TPUCompileMlir
  // CHECK: %[[CONSTANT:[a-z0-9]*]] = "tf.Const"
  // CHECK: "tf.OpA"(%[[CONSTANT]]
   %compilation_status, %program = "tf._TPUCompileMlir"() { metadata = "...", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf_device.launch"() ({
    %cst_0 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
    "tf.OpA"(%cst_0) { mini_batch_splits = ""} : (tensor<1x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  return
}

