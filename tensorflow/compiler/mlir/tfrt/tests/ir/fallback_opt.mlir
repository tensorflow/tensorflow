// RUN: tf-tfrt-opt %s | tf-tfrt-opt | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @const_tensor_proto
func.func @const_tensor_proto() -> !tfrt_fallback.tf_tensor {
  // CHECK: tfrt_fallback_async.const_tensor_proto "fake serialized proto"
  %0 = tfrt_fallback_async.const_tensor_proto "fake serialized proto"
  tfrt.return %0 : !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @const_dense_tensor
func.func @const_dense_tensor() -> !tfrt_fallback.tf_tensor {
  // CHECK: tfrt_fallback_async.const_dense_tensor
  %0 = tfrt_fallback_async.const_dense_tensor dense<0.0> : tensor<f32> {_tfrt_cost = 1 : i64}
  tfrt.return %0 : !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @const_string_tensor
func.func @const_string_tensor() -> !tfrt_fallback.tf_tensor {
  // CHECK: tfrt_fallback_async.const_string_tensor
  %0 = tfrt_fallback_async.const_string_tensor {shape = [1, 2], value = ["const", "string"], _tfrt_cost = 1 : i64}
  tfrt.return %0 : !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @convert
func.func @convert() -> !corert.tensorhandle {
  %0 = corert.const_dense_tensor dense<0.0> : tensor<f32>
  // CHECK: tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor
  %1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %0 {_tfrt_cost = 1 : i64, device = "cpu"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  // CHECK: tfrt_fallback_async.executeop key(0) cost(100) device("cpu") "tf.Relu"(%{{.*}}) {T = f32} : 1
  %2 = tfrt_fallback_async.executeop key(0) cost(100) device("cpu") "tf.Relu"(%1) {T = f32} : 1
  // CHECK: tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle
  %3 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %2 {_tfrt_cost = 1 : i64, device="cpu"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)
  tfrt.return %3 : !corert.tensorhandle
}

// CHECK-LABEL: func @predicate
func.func @predicate() -> i1 {
  %0 = tfrt_fallback_async.const_dense_tensor dense<0.0> : tensor<f32> {_tfrt_cost = 1 : i64}
  // CHECK: tfrt_fallback_async.predicate
  %1 = tfrt_fallback_async.predicate %0 {_tfrt_cost = 1 : i64, device = "cpu"}
  tfrt.return %1 : i1
}

// CHECK-LABEL: func @createop
func.func @createop(%in_ch: !tfrt.chain) -> !tfrt.chain {

  // CHECK: [[ch:%.*]] = tfrt_fallback_async.createop(%{{.*}}) key(100) device("cpu") "tf.AddV2"() {T = i32} num_args(2)
  %out_ch = tfrt_fallback_async.createop(%in_ch) key(100) device("cpu") "tf.AddV2"() {T = i32} num_args(2)

  // CHECK: tfrt.return [[ch]]
  tfrt.return %out_ch: !tfrt.chain
}

// CHECK-LABEL: func @fallback_resource
func.func @fallback_resource(%ch0: !tfrt.chain) -> !tfrt.chain {

  %ra = tfrt_fallback_async.const_dense_tensor dense<0.0> : tensor<f32> {_tfrt_cost = 1 : i64}
  %rb = tfrt_fallback_async.const_dense_tensor dense<0.5> : tensor<f32> {_tfrt_cost = 1 : i64}

  // CHECK: tfrt_fallback_async.set_resource {{%.*}}, {{%.*}} {device = "cpu", index = 0 : i64}
  // CHECK: tfrt_fallback_async.set_resource {{%.*}}, {{%.*}} {device = "cpu", index = 1 : i64}
  // CHECK: tfrt_fallback_async.get_resource {{%.*}} {_tfrt_cost = 1 : i64, device = "cpu", indices = [0, 1]}
  %ch1 = tfrt_fallback_async.set_resource %ch0, %ra {device = "cpu", index = 0 : i64}
  %ch2 = tfrt_fallback_async.set_resource %ch1, %rb {device = "cpu", index = 1 : i64}
  %ch3, %a, %b = tfrt_fallback_async.get_resource %ch2 {_tfrt_cost = 1 : i64, device = "cpu", indices = [0 : i64, 1 : i64]} : (!tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  tfrt.return %ch3: !tfrt.chain
}

// CHECK-LABEL: func @copy_if_small
func.func @copy_if_small(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) {
  // CHECK: tfrt_fallback_async.copy_if_small {{%.*}} {_tfrt_cost = 1 : i64} : (!tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)
  %0:2 = tfrt_fallback_async.copy_if_small %arg {_tfrt_cost = 1 : i64} : (!tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)
  tfrt.return %0#0, %0#1 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @custom_allocator
func.func @custom_allocator(%ch: !tfrt.chain, %arg: !tfrt_fallback.tf_tensor, %allocator: !tfrt_fallback.tf_allocator) -> (!tfrt.chain, !tfrt_fallback.tf_tensor) {
  // CHECK: tfrt_fallback_async.executeop.allocator(%{{.*}}) key(200) cost(100) device("cpu") "tf.Cast"(%{{.*}}) {Truncate = false, T = f32} : 1
  %0 = tfrt_fallback_async.executeop.allocator(%allocator) key(200) cost(100) device("cpu") "tf.Cast"(%arg) {Truncate = false, T = f32} : 1
  // CHECK: tfrt_fallback_async.executeop.seq.allocator(%{{.*}}, %{{.*}}) key(201) cost(100) device("cpu") "tf.Cast"(%{{.*}}) {Truncate = false, T = i32} : 1
  %out_ch, %1 = tfrt_fallback_async.executeop.seq.allocator(%ch, %allocator) key(201) cost(100) device("cpu") "tf.Cast"(%0) {Truncate = false, T = i32} : 1
  tfrt.return %out_ch, %1 : !tfrt.chain, !tfrt_fallback.tf_tensor
}
