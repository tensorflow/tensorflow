// RUN: tf-opt --tf-tpu-compile-replication %s | FileCheck %s

// CHECK: func @test(%[[ARG_0:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %[[ARG_1:.*]]: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"})
func @test(%arg0: tensor<i32> {tf.device = "/job:worker/replica:0/task:0/device:CPU:0"}, %arg1: tensor<i32> {tf.device = "/job:worker/replica:0/task:1/device:CPU:0"}) -> (tensor<i32>, tensor<i32>) {
  // CHECK-NEXT: %[[STATUS_0:.*]], %[[PROGRAM_0:.*]] = "tf._TPUCompileMlir"() {device = "/job:worker/replica:0/task:1/device:CPU:0", metadata = "metadata", mlir_module = "mlir_module"}
  // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[STATUS_0]]) {device = "/job:worker/replica:0/task:1/device:CPU:0"}
  // CHECK-NEXT: %[[STATUS_1:.*]], %[[PROGRAM_1:.*]] = "tf._TPUCompileMlir"() {device = "/job:worker/replica:0/task:0/device:CPU:0", metadata = "metadata", mlir_module = "mlir_module"}
  %compilation_status, %program = "tf._TPUCompileMlir"() {device = "/job:worker/replica:0/task:0/device:CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : () -> (tensor<!tf.string>, tensor<2x!tf.string>)
  // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[STATUS_1]]) {device = "/job:worker/replica:0/task:0/device:CPU:0"}
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/job:worker/replica:0/task:0/device:CPU:0"} : (tensor<!tf.string>) -> ()
  // CHECK-NEXT: %[[ADD_0:.*]] = "tf.AddV2"(%[[ARG_0]], %[[ARG_0]]) {device = "/job:worker/replica:0/task:0/device:TPU:0"}
  %0 = "tf.AddV2"(%arg0, %arg0) {device = "/job:worker/replica:0/task:0/device:TPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[EXECUTE_0:.*]] = "tf.TPUExecute"(%[[ADD_0]], %[[PROGRAM_1]]) {device = "/job:worker/replica:0/task:0/device:TPU:0"}
  %1 = "tf.TPUExecute"(%0, %program) {device = "/job:worker/replica:0/task:0/device:TPU:0"} : (tensor<i32>, tensor<2x!tf.string>) -> tensor<i32>
  // CHECK-NEXT: %[[ADD_1:.*]] = "tf.AddV2"(%[[ARG_1]], %[[ARG_1]]) {device = "/job:worker/replica:0/task:1/device:TPU:0"}
  %2 = "tf.AddV2"(%arg1, %arg1) {device = "/job:worker/replica:0/task:1/device:TPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[EXECUTE_1:.*]] = "tf.TPUExecute"(%[[ADD_1]], %[[PROGRAM_0]]) {device = "/job:worker/replica:0/task:1/device:TPU:0"}
  %3 = "tf.TPUExecute"(%2, %program) {device = "/job:worker/replica:0/task:1/device:TPU:0"} : (tensor<i32>, tensor<2x!tf.string>) -> tensor<i32>
  // CHECK-NEXT: return %[[EXECUTE_0]], %[[EXECUTE_1]]
  return %1, %3 : tensor<i32>, tensor<i32>
}

