// RUN: tf-tfrt-opt -verify-diagnostics -split-input-file -tfrt-fuse-tpu-compile-and-execute-ops -canonicalize %s | FileCheck %s --dump-input=fail --dump-input-filter=all

module attributes {tf_saved_model.semantics} {

// Test fusing _TPUCompileMlirOp and TPUExecuteOp into TPUCompileMlirAndExecuteOp.

// CHECK-LABEL: func private @test_fuse_tpu_ops
func.func private @test_fuse_tpu_ops(%arg0: tensor<*xi32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<*xi32> {
  // CHECK-NOT: tf._TPUCompileMlirOp
  // CHECK-NOT: tf.TPUCompileSucceededAssert
  // CHECK-NOT: tf.TPUExecuteOp

  // CHECK-NEXT: %0 = "tf.ReadVariableOp"(%arg1)
  // CHECK:      [[key:%.*]], [[exec_result:%.*]] = "tf.TPUCompileMlirAndExecute"(%arg0, %0) <{metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 2, 0>, operands_with_static_shape = [], producer_name = "default"}> : (tensor<*xi32>, tensor<*xi32>) -> (tensor<3x!tf_type.string>, tensor<*xi32>)
  // CHECK-NEXT: return [[exec_result]] : tensor<*xi32>

  %0 = "tf.ReadVariableOp"(%arg1) {device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %1 = "tf.Shape"(%arg0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %3 = "tf.TPUExecute"(%arg0, %0, %program) {device = "/TPU:0"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  func.return %3 : tensor<*xi32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test models using Outside Compilation

// CHECK-LABEL: func private @test_outside_compilation
func.func private @test_outside_compilation(%arg0: tensor<*xi32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<*xi32> {
  // CHECK-NOT: tf._TPUCompileMlirOp
  // CHECK-NOT: tf.TPUCompileSucceededAssert
  // CHECK-NOT: tf.TPUExecuteOp

  // CHECK-NEXT: %0 = "tf.ReadVariableOp"(%arg1)
  // CHECK:      [[key:%.*]], [[exec_result:%.*]] = "tf.TPUCompileMlirAndExecute"(%arg0, %0) <{metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 2, 0>, operands_with_static_shape = [], producer_name = "default"}> : (tensor<*xi32>, tensor<*xi32>) -> (tensor<3x!tf_type.string>, tensor<*xi32>)
  // CHECK-NEXT: "tf._XlaSendFromHost"(%arg0, %0, [[key]]) <{device_ordinal = 0 : i64, key = "host_compute_channel_0_retvals"}> {_xla_has_host_transfer = true, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> ()
  // CHECK-NEXT: return [[exec_result]] : tensor<*xi32>
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %1 = "tf.Shape"(%arg0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  "tf._XlaSendFromHost"(%arg0, %0, %program) {_xla_has_host_transfer = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", device_ordinal = 0 : i64, key = "host_compute_channel_0_retvals"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> ()
  %3 = "tf.TPUExecute"(%arg0, %0, %program) {device = "/TPU:0"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  func.return %3 : tensor<*xi32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test models with dynamic bounds ops.

// CHECK-LABEL: func private @test_fuse_dynamic_dimension_ops
func.func private @test_fuse_dynamic_dimension_ops(%arg0: tensor<?x?xi32>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<2xi64>, %arg3: tensor<?xi64>, %arg4: tensor<?xi64>) -> tensor<*xi32> {
  // CHECK-NOT: tf._TPUCompileMlirOp
  // CHECK-NOT: tf.TPUCompileSucceededAssert
  // CHECK-NOT: tf.TPUExecuteOp
  // CHECK-NOT: tf.SetStaticDimensionBounds

  // CHECK: [[read_result:%.*]] = "tf.ReadVariableOp"(%arg1)
  // CHECK: [[shape_result_1:%.*]] = "tf.Shape"(%arg0) {device = "/CPU:0"} : (tensor<?x?xi32>) -> tensor<?xi64>
  // CHECK: [[shape_result_2:%.*]] = "tf.Shape"([[read_result]]) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  // CHECK: [[key:%.*]], [[exec_result:%.*]] = "tf.TPUCompileMlirAndExecute"(%arg0, [[shape_result_2]], %0, %0, %arg2, %arg4, %arg3) <{metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 4, 3>, operands_with_static_shape = [0 : i32, 1 : i32, 3 : i32], producer_name = "default"}> : (tensor<?x?xi32>, tensor<?xi64>, tensor<*xi32>, tensor<*xi32>, tensor<2xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<3x!tf_type.string>, tensor<*xi32>)
  // CHECK: [[key_1:%.*]], [[exec_result_1:%.*]] = "tf.TPUCompileMlirAndExecute"(%arg0, %2, %0, %1) <{metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 4, 0>, operands_with_static_shape = [], producer_name = "default"}> : (tensor<?x?xi32>, tensor<?xi64>, tensor<*xi32>, tensor<?xi64>) -> (tensor<3x!tf_type.string>, tensor<*xi32>)
  // CHECK-NEXT: return [[exec_result]] : tensor<*xi32>
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %dyn_arg0 = "tf.SetStaticDimensionBounds" (%arg0, %arg2) :(tensor<?x?xi32>, tensor<2xi64>) -> tensor<?x?xi32>
  %dyn_0 = "tf.SetStaticDimensionBounds" (%0, %arg3) :(tensor<*xi32>, tensor<?xi64>) -> tensor<?xi64>
  %1 = "tf.Shape"(%dyn_arg0) {device = "/CPU:0"} : (tensor<?x?xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %dyn_2 = "tf.SetStaticDimensionBounds" (%2, %arg4) :(tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %3 = "tf.TPUExecute"(%dyn_arg0, %dyn_2, %0, %dyn_0, %program) {device = "/TPU:0"} : (tensor<?x?xi32>, tensor<?xi64>, tensor<*xi32>, tensor<?xi64>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  %compilation_status_2, %program_2 = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %4 = "tf.TPUExecute"(%arg0, %2, %0, %1, %program_2) {device = "/TPU:0"} : (tensor<?x?xi32>, tensor<?xi64>, tensor<*xi32>, tensor<?xi64>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  func.return %3 : tensor<*xi32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// CHECK-LABEL: func private @reorder_execute_arg_defining_ops
// CHECK: tf.VarHandleOp
// CHECK-NEXT: tf.ReadVariableOp
// CHECK-NEXT: tf.TPUCompileMlirAndExecute
func.func private @reorder_execute_arg_defining_ops(%arg0: tensor<1x3xf32> {tf.device = "/CPU:0"}) -> (tensor<1x1xf32> {tf.device = "/TPU:0"}) {
  %compilation_status, %program = "tf._TPUCompileMlir"() {device = "/CPU:0", metadata = "metadata", mlir_module = "propgram"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %0 = "tf.VarHandleOp"() {_xla_inferred_shapes = [#tf_type.shape<>], allowed_devices = [], container = "", device = "/CPU:0", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<3x1xf32>>>) -> tensor<3x1xf32>
  %2 = "tf.TPUExecute"(%arg0, %1, %program) {_producer_name = "UNKNOWN", device = "/TPU:0"} : (tensor<1x3xf32>, tensor<3x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  return %2 : tensor<1x1xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {
// CHECK-LABEL: func private @spmd_fuse_mulitple_execute_ops
// CHECK-NEXT: %0 = "tf.VarHandleOp"()
// CHECK-NEXT: %1 = "tf.ReadVariableOp"(%0)
// CHECK-NEXT: %rendezvous_key_base, %results = "tf.TPUCompileMlirAndExecute"(%arg0, %1)
func.func private @spmd_fuse_mulitple_execute_ops(%arg0: tensor<1x4xf32> {tf.device = "/CPU:0"}) -> (tensor<1x1xf32> {tf.device = "/TPU:0"}) {
  %cst = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %compilation_status, %program:2 = "tf._TPUCompileMlir"() {device = "/CPU:0", metadata = "metadata", mlir_module = "propgram"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %0 = "tf.VarHandleOp"() {_xla_inferred_shapes = [#tf_type.shape<>], allowed_devices = [], container = "", device = "/CPU:0", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<2x1xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<2x1xf32>>>) -> tensor<2x1xf32>
  %2:2 = "tf.Split"(%cst, %arg0) {device = "/CPU:0"} : (tensor<i32>,  tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  %3 = "tf.TPUExecute"(%2#0, %1, %program#0) {_producer_name = "UNKNOWN", device = "/TPU:0"} : (tensor<1x2xf32>, tensor<2x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  %4 = "tf.TPUExecute"(%2#1, %1, %program#1) {_producer_name = "UNKNOWN", device = "/TPU:1"} : (tensor<1x2xf32>, tensor<2x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  return %3 : tensor<1x1xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {
// CHECK-LABEL: func private @spmd_fuse_mulitple_execute_ops_2
// CHECK-NEXT: %0 = "tf.VarHandleOp"()
// CHECK-NEXT: %1 = "tf.ReadVariableOp"(%0)
// CHECK-NEXT: %rendezvous_key_base, %results = "tf.TPUCompileMlirAndExecute"(%arg0, %1)
func.func private @spmd_fuse_mulitple_execute_ops_2(%arg0: tensor<1x1xf32> {tf.device = "/CPU:0"}) -> (tensor<1x1xf32> {tf.device = "/TPU:0"}) {
  %cst = "tf.Const"() {device = "/CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %compilation_status, %program:2 = "tf._TPUCompileMlir"() {device = "/CPU:0", metadata = "metadata", mlir_module = "propgram"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %0 = "tf.VarHandleOp"() {_xla_inferred_shapes = [#tf_type.shape<>], allowed_devices = [], container = "", device = "/CPU:0", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<2x1xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<2x1xf32>>>) -> tensor<2x1xf32>
  %2:2 = "tf.Split"(%cst, %1) {device = "/CPU:0"} : (tensor<i32>,  tensor<2x1xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
  %3 = "tf.TPUExecute"(%arg0, %2#0, %program#0) {_producer_name = "UNKNOWN", device = "/TPU:0"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  %4 = "tf.TPUExecute"(%arg0, %2#1, %program#1) {_producer_name = "UNKNOWN", device = "/TPU:1"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  return %3 : tensor<1x1xf32>
}

}

// -----

module attributes {tf_saved_model.semantics} {
// CHECK-LABEL: func private @spmd_fuse_split_nd_ops
// CHECK-NEXT: %0 = "tf.VarHandleOp"()
// CHECK-NEXT: %1 = "tf.ReadVariableOp"(%0)
// CHECK-NEXT: %rendezvous_key_base, %results = "tf.TPUCompileMlirAndExecute"(%arg0, %1)
func.func private @spmd_fuse_split_nd_ops(%arg0: tensor<1x4xf32> {tf.device = "/CPU:0"}) -> (tensor<1x1xf32> {tf.device = "/TPU:0"}) {
  %cst = "tf.Const"() {device = "/CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %compilation_status, %program:4 = "tf._TPUCompileMlir"() {device = "/CPU:0", metadata = "metadata", mlir_module = "propgram"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %0 = "tf.VarHandleOp"() {_xla_inferred_shapes = [#tf_type.shape<>], allowed_devices = [], container = "", device = "/CPU:0", shared_name = "y"} : () -> tensor<!tf_type.resource<tensor<1x1xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/CPU:0"} : (tensor<!tf_type.resource<tensor<1x1xf32>>>) -> tensor<1x1xf32>
  %2:2 = "tf.Split"(%cst, %arg0) {device = "/CPU:0"} : (tensor<i32>,  tensor<1x4xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>)
  %3:2 = "tf.Split"(%cst, %2#0) {device = "/CPU:0"} : (tensor<i32>,  tensor<1x2xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
  %4:2 = "tf.Split"(%cst, %2#1) {device = "/CPU:0"} : (tensor<i32>,  tensor<1x2xf32>) -> (tensor<1x1xf32>, tensor<1x1xf32>)
  %5 = "tf.TPUExecute"(%3#0, %1, %program#0) {_producer_name = "UNKNOWN", device = "/TPU:0"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  %6 = "tf.TPUExecute"(%3#1, %1, %program#1) {_producer_name = "UNKNOWN", device = "/TPU:1"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  %7 = "tf.TPUExecute"(%4#0, %1, %program#2) {_producer_name = "UNKNOWN", device = "/TPU:2"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  %8 = "tf.TPUExecute"(%4#1, %1, %program#3) {_producer_name = "UNKNOWN", device = "/TPU:3"} : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<3x!tf_type.string>) -> tensor<1x1xf32>
  return %5 : tensor<1x1xf32>
}

}
