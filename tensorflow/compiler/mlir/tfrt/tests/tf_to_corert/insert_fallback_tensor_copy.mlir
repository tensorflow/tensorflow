// RUN: tf-tfrt-opt %s -tfrt-insert-fallback-tensor-copy | FileCheck %s

// CHECK-LABEL: func @test_insert_copy
// CHECK-SAME: ([[arg:%.*]]: !tfrt_fallback.tf_tensor
func @test_insert_copy(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) attributes {tfrt.cost_threshold = 1024} {
  // CHECK: [[value:%.*]] = tfrt_fallback_async.executeop key({{.*}}) {{.*}} "tf.AddV2"([[arg]], [[arg]])
  %0 = tfrt_fallback_async.executeop key(0) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  // CHECK: [[copy:%.*]] = tfrt_fallback_async.copy_if_small [[value]]
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) {{.*}} "tf.AddV2"([[copy]], [[copy]])
  %1 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %2 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %3 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %4 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %5 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %6 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %7 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %8 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %9 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  %10 = tfrt_fallback_async.executeop key(1) cost(512) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%0, %0) {T = f32} : 1
  tfrt.return %1, %2, %3 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_insert_copy_for_arg
// CHECK-SAME: ([[arg:%.*]]: !tfrt_fallback.tf_tensor
func @test_insert_copy_for_arg(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) attributes {tfrt.cost_threshold = 1024} {
  // CHECK: [[copy:%.*]] = tfrt_fallback_async.copy_if_small [[arg]]
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) {{.*}} "tf.AddV2"([[copy]], [[copy]])
  %0 = tfrt_fallback_async.executeop key(0) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %1 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %2 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %3 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %4 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %5 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %6 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %7 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %8 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %9 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  %10 = tfrt_fallback_async.executeop key(1) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  tfrt.return %0, %1, %2 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_no_copy_for_return
func @test_no_copy_for_return(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) attributes {tfrt.cost_threshold = 1024} {
  // CHECK-NOT: tfrt_fallback_async.copy_if_small
  %0 = tfrt_fallback_async.executeop key(0) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AddV2"(%arg, %arg) {T = f32} : 1
  tfrt.return %arg, %0 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_no_copy_for_few_uses
func @test_no_copy_for_few_uses(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) attributes {tfrt.cost_threshold = 1024} {
  // CHECK-NOT: tfrt_fallback_async.copy_if_small
  %0 = tfrt_fallback_async.executeop key(0) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.Relu"(%arg) {T = f32} : 1
  %1 = tfrt_fallback_async.executeop key(0) cost(1024) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.Relu"(%arg) {T = f32} : 1
  tfrt.return %0, %1 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}
