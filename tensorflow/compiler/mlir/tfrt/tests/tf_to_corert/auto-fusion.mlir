// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline="auto-fusion-oplist=tf.Rsqrt,tf.Tanh auto-fusion-min-cluster-size=1" -split-input-file %s \
// RUN: | FileCheck %s --dump-input=always

// CHECK-LABEL: func @single_op_cluster
// CHECK: %[[ARG0:.*]]: !tfrt.chain
// CHECK: %[[ARG1:.*]]: !corert.tensorhandle
func.func @single_op_cluster(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[ARG:.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor
  // CHECK-SAME:          %[[ARG1]]
  // CHECK-SAME:          device = "/job:localhost/replica:0/task:0/device:CPU:0"
  // CHECK: %[[RES:.*]] = tf_jitrt.fallback.execute @kernel::@compute(%[[ARG]])
  // CHECK: %[[OUT:.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle
  // CHECK-SAME:          %[[RES]]
  // CHECK-SAME:          device = "/job:localhost/replica:0/task:0/device:CPU:0"
  // CHECK: tfrt.return %[[ARG0]], %[[OUT]] : !tfrt.chain, !corert.tensorhandle
  %0 = "tf.Rsqrt"(%arg0) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK-LABEL: func @compute
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xf32>
// CHECK: %[[RES:.*]] = "tf.Rsqrt"(%[[ARG0]])
// CHECK: return %[[RES]]

// -----

// CHECK-LABEL: func @one_compiled_cluster
func.func @one_compiled_cluster(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[RES:.*]] = tf_jitrt.fallback.execute @kernel::@compute
  // CHECK-NOT: Rsqrt
  // CHECK-NOT: Tanh
  %0 = "tf.Rsqrt"(%arg0) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Tanh"(%0) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK-LABEL: func @compute
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xf32>
// CHECK: %[[RES0:.*]] = "tf.Rsqrt"(%[[ARG0]])
// CHECK: %[[RES1:.*]] = "tf.Tanh"(%[[RES0]])
// CHECK: return %[[RES1]]

// -----

// CHECK-LABEL: func @two_compiled_clusters
func.func @two_compiled_clusters(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: tf_jitrt.fallback.execute @kernel::@compute
  %0 = "tf.Rsqrt"(%arg0) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tfrt_fallback_async.executeop {{.*}} "tf.Sqrt"
  %1 = "tf.Sqrt"(%0) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: tf_jitrt.fallback.execute @kernel_0::@compute
  %2 = "tf.Tanh"(%1) {T = f32, device="/device:CPU:0"} : (tensor<?xf32>) -> tensor<?xf32>
  func.return %2 : tensor<?xf32>
}

// CHECK: module @kernel
// CHECK: tf.Rsqrt
// CHECK: module @kernel_0
// CHECK: tf.Tanh
