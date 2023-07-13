// RUN: tf-tfrt-opt -split-input-file -tf-mlrt-fuse %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK-SAME: ([[f0:%.*]]: !mlrt.future, [[f1:%.*]]: !mlrt.future, [[f2:%.*]]: !mlrt.future)
func.func @main(%f0: !mlrt.future, %f1: !mlrt.future, %f2: !mlrt.future) -> (!tf_mlrt.tensor, !tf_mlrt.tensor, !tf_mlrt.tensor) {
  // CHECK-NEXT: [[t:%.*]]:3 = tf_mlrt.await_all [[f0]], [[f1]], [[f2]]
  // CHECK-NOT: tf_mlrt.await
  // CHECK-NEXT: return [[t]]#0, [[t]]#1, [[t]]#2
  %t0 = tf_mlrt.await %f0
  %t1 = tf_mlrt.await %f1
  %t2 = tf_mlrt.await %f2
  func.return %t0, %t1, %t2 : !tf_mlrt.tensor, !tf_mlrt.tensor, !tf_mlrt.tensor
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: ([[f0:%.*]]: !mlrt.future, [[f1:%.*]]: !mlrt.future, [[f2:%.*]]: !mlrt.future)
func.func @main(%f0: !mlrt.future, %f1: !mlrt.future, %f2: !mlrt.future) -> (!tf_mlrt.tensor, !tf_mlrt.tensor) {
  // CHECK-NEXT: [[t:%.*]]:2 = tf_mlrt.await_all [[f0]], [[f1]]
  // CHECK-NOT: tf_mlrt.await
  // CHECK-NEXT: [[t2:%.*]] = tf_mlrt.executeop([[t]]#0, [[t]]#1)
  // CHECK-NEXT: [[t3:%.*]] = tf_mlrt.await [[f2]]
  // CHECK-NEXT: return [[t2]], [[t3]]
  %t0 = tf_mlrt.await %f0
  %t1 = tf_mlrt.await %f1
  %t2 = tf_mlrt.executeop(%t0, %t1) {node_def = "AddV2", op_key = 0 : i32} : (!tf_mlrt.tensor, !tf_mlrt.tensor) -> (!tf_mlrt.tensor)
  %t3 = tf_mlrt.await %f2
  func.return %t2, %t3 : !tf_mlrt.tensor, !tf_mlrt.tensor
}

// -----

// CHECK-LABEL: @main
// CHECK-SAME: ([[f0:%.*]]: !mlrt.async_handle, [[f1:%.*]]: !mlrt.async_handle, [[f2:%.*]]: !mlrt.async_handle)
func.func @main(%f0: !mlrt.async_handle, %f1: !mlrt.async_handle, %f2: !mlrt.async_handle) -> () {
  // CHECK-NEXT: mlrt.await_all_handle [[f0]], [[f1]], [[f2]]
  // CHECK-NOT: mlrt.await_handle
  // CHECK-NEXT: return
  mlrt.await_handle %f0
  mlrt.await_handle %f1
  mlrt.await_handle %f2
  func.return
}

// -----

// CHECK-LABEL: @main
func.func @main() -> (!tf_mlrt.tensor, !tf_mlrt.tensor) {
  // CHECK-NEXT: [[r:%.*]]:3 = tf_mlrt.get_resource {indices = [2, 0, 1]}
  // CHECK-NEXT: [[v:%.*]] = tf_mlrt.executeop([[r]]#0, [[r]]#1)
  // CHECK-NEXT: return [[v]], [[r]]#2
  %0 = tf_mlrt.get_resource {indices = [2]} : !tf_mlrt.tensor
  %1 = tf_mlrt.get_resource {indices = [0]} : !tf_mlrt.tensor
  %r = tf_mlrt.executeop(%0, %1) {node_def = "AddV2", op_key = 0 : i32} : (!tf_mlrt.tensor, !tf_mlrt.tensor) -> (!tf_mlrt.tensor)
  %2 = tf_mlrt.get_resource {indices = [1]} : !tf_mlrt.tensor
  func.return %r, %2 : !tf_mlrt.tensor, !tf_mlrt.tensor
}

// -----

// CHECK-LABEL: @fuse_promise_return
// CHECK-SAME: ([[p:%.*]]: !mlrt.promise, [[v:%.*]]: !tf_mlrt.tensor)
func.func @fuse_promise_return(%p: !mlrt.promise, %v: !tf_mlrt.tensor) -> () {
  // CHECK: tf_mlrt.promise_return [[p]], [[v]]
  tf_mlrt.promise %p, %v
  func.return
}

// -----

// CHECK-LABEL: @not_fuse_promise_return
// CHECK-SAME: ([[p:%.*]]: !mlrt.promise, [[v:%.*]]: !tf_mlrt.tensor)
func.func @not_fuse_promise_return(%p: !mlrt.promise, %v: !tf_mlrt.tensor) -> (!tf_mlrt.tensor) {
  // CHECK-NOT: tf_mlrt.promise_return
  tf_mlrt.promise %p, %v
  func.return %v : !tf_mlrt.tensor
}
