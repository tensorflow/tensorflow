// RUN: xla-runtime-opt %s --xla-rt-move-allocas-to-entry-block | FileCheck %s

func.func @compute(
  %arg0: !rt.execution_context,
  %arg1: !async.value<memref<f32>>
) -> !async.token attributes {passthrough = ["presplitcoroutine"]} {
  // CHECK:   %alloca = memref.alloca() {alignment = 64 : i64} : memref<f32>
  // CHECK:   %0 = async.runtime.create : !async.token
  // CHECK:   %1 = async.coro.id
  // CHECK:   %2 = async.coro.begin %1
  // CHECK:   %3 = async.coro.save %2
  // CHECK:   async.runtime.resume %2
  // CHECK:   async.coro.suspend %3, ^bb9, ^bb1, ^bb8
  // CHECK: ^bb1:  // pred: ^bb0
  // CHECK:   %status = rt.call %arg0["test.producer"] (%alloca)
  // CHECK:     : (memref<f32>) -> ()
  %0 = async.runtime.create : !async.token
  %1 = async.coro.id
  %2 = async.coro.begin %1
  %3 = async.coro.save %2
  async.runtime.resume %2
  async.coro.suspend %3, ^bb9, ^bb1, ^bb8
^bb1:  // pred: ^bb0
  %alloca = memref.alloca() {alignment = 64 : i64} : memref<f32>
  %status = rt.call %arg0["test.producer"] (%alloca) : (memref<f32>) -> ()
  %4 = rt.is_ok %status
  cf.cond_br %4, ^bb2, ^bb6
^bb2:  // pred: ^bb1
  %5 = async.coro.save %2
  async.runtime.await_and_resume %arg1, %2 : !async.value<memref<f32>>
  async.coro.suspend %5, ^bb9, ^bb3, ^bb8
^bb3:  // pred: ^bb2
  %6 = async.runtime.is_error %arg1 : !async.value<memref<f32>>
  cf.cond_br %6, ^bb6, ^bb4
^bb4:  // pred: ^bb3
  %7 = async.runtime.load %arg1 : <memref<f32>>
  %status_0 = rt.call %arg0["test.consumer"] (%alloca) : (memref<f32>) -> ()
  %8 = rt.is_ok %status_0
  cf.cond_br %8, ^bb5, ^bb6
^bb5:  // pred: ^bb4
  async.runtime.set_available %0 : !async.token
  cf.br ^bb7
^bb6:  // 3 preds: ^bb1, ^bb3, ^bb4
  async.runtime.set_error %0 : !async.token
  cf.br ^bb7
^bb7:  // 2 preds: ^bb5, ^bb6
  async.coro.free %1, %2
  cf.br ^bb9
^bb8:  // 2 preds: ^bb0, ^bb2
  async.coro.free %1, %2
  cf.br ^bb9
^bb9:  // 4 preds: ^bb0, ^bb2, ^bb7, ^bb8
  async.coro.end %2
  return %0 : !async.token
}
