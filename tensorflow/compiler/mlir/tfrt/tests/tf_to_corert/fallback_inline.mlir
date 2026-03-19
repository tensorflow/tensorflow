// RUN: tf-tfrt-opt %s -inline | FileCheck %s

func.func @_tfrt_fallback_init(%arg0: !tfrt.chain) -> !tfrt.chain {
  %0 = tfrt_fallback_async.createop(%arg0) key(0) device("/device:CPU:0") "tf.Less"() {T = i32} num_args(2)
  tfrt.return %0 : !tfrt.chain
}

func.func @callee(%ch: !tfrt.chain, %arg: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  %const = tfrt_fallback_async.const_dense_tensor dense<9> : tensor<i32> {_tfrt_cost = 1 : i64}
  %result = tfrt_fallback_async.executeop key(0) cost(3) device("/device:CPU:0") "tf.Less"(%arg, %const) {T = i32} : 1
  tfrt.return %ch, %result : !tfrt.chain, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_inline_fallback_ops
// CHECK-SAME: ([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !tfrt_fallback.tf_tensor
func.func @test_inline_fallback_ops(%ch: !tfrt.chain, %arg: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  // CHECK-NOT: tfrt.call
  // CHECK: [[const:%.*]] = tfrt_fallback_async.const_dense_tensor dense<9> : tensor<i32>
  // CHECK-NEXT: [[result:%.*]] = tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.Less"([[arg]], [[const]]) {T = i32} : 1
  // CHECK-NEXT: tfrt.return [[ch]], [[result]] : !tfrt.chain, !tfrt_fallback.tf_tensor
  %out_ch, %result = tfrt.call @callee(%ch, %arg) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  tfrt.return %out_ch, %result : !tfrt.chain, !tfrt_fallback.tf_tensor
}
