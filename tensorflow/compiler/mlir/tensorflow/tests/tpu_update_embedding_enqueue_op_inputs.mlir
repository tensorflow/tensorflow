// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-update-embedding-enqueue-op-inputs | FileCheck %s

// CHECK-LABEL: func @check_enqueue_ops_update_for_eval
// CHECK-SAME: %[[ARG_0:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_1:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_2:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_3:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_4:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_5:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_6:[a-z0-9]*]]: tensor<!tf_type.string>
// CHECK-SAME: %[[ARG_7:[a-z0-9]*]]: tensor<!tf_type.string>
func.func @check_enqueue_ops_update_for_eval(%arg0: tensor<?x2xi32>, %arg1: tensor<?x2xi32>,
  %arg2 :tensor<?x2xi32>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<?xi32>,
  %arg6: tensor<!tf_type.string>, %arg7: tensor<!tf_type.string>) -> () {
  // CHECK: %[[CONST_0:.*]] = "tf.Const"()
  %0 = "tf.Const"() {value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>

  // CHECK: %[[CONST_MODE:.*]] = "tf.Const"() <{value = dense<"inference"> : tensor<!tf_type.string>}> {_xla_outside_compilation = "0"} : () -> tensor<!tf_type.string>
  // CHECK: "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]], %[[ARG_4]], %[[ARG_5]], %[[CONST_0]], %[[CONST_0]], %[[CONST_0]], %[[CONST_MODE]])
  "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %0, %0, %0, %arg7) {_tpu_embedding_layer = "call1", _xla_outside_compilation = "0", combiners = ["mean", "sum"], device_ordinal = -1 : i64, max_sequence_lengths = [0, 0, 0], table_ids = [1, 1, 0]} : (tensor<?x2xi32>, tensor<?x2xi32>, tensor<?x2xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<!tf_type.string>) -> ()
  %2:2 = "tf.RecvTPUEmbeddingActivations"() {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D"} : () -> (tensor<2x2xf32>, tensor<4x4xf32>)
  func.return
}

// -----

// CHECK-LABEL: func @check_enqueue_ops_update_for_training
// CHECK-SAME: %[[ARG_0:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_1:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_2:[a-z0-9]*]]: tensor<?x2xi32>
// CHECK-SAME: %[[ARG_3:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_4:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_5:[a-z0-9]*]]: tensor<?xi32>
// CHECK-SAME: %[[ARG_6:[a-z0-9]*]]: tensor<!tf_type.string>
// CHECK-SAME: %[[ARG_7:[a-z0-9]*]]: tensor<!tf_type.string>
func.func @check_enqueue_ops_update_for_training(%arg0: tensor<?x2xi32>, %arg1: tensor<?x2xi32>,
  %arg2 :tensor<?x2xi32>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<?xi32>,
  %arg6: tensor<!tf_type.string>, %arg7: tensor<!tf_type.string>) -> () {
  // CHECK: %[[CONST_0:.*]] = "tf.Const"()
  %0 = "tf.Const"() {value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>

  %2 = "tf.Const"() {value = dense<0.0> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
  %3 = "tf.Const"() {value = dense<0.0> : tensor<4x4xf32>} : () -> tensor<4x4xf32>
  "tf.SendTPUEmbeddingGradients"(%2, %3) {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D", operandSegmentSizes = array<i32: 2, 0>} : (tensor<2x2xf32>, tensor<4x4xf32>) -> ()

  // CHECK: %[[CONST_MODE:.*]] = "tf.Const"() <{value = dense<"train"> : tensor<!tf_type.string>}> {_xla_outside_compilation = "0"} : () -> tensor<!tf_type.string>
  // CHECK: "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%[[ARG_0]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]], %[[ARG_4]], %[[ARG_5]], %[[CONST_0]], %[[CONST_0]], %[[CONST_0]], %[[CONST_MODE]])
  "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %0, %0, %0, %arg7) {_tpu_embedding_layer = "call1", _xla_outside_compilation = "0", combiners = ["mean", "sum"], device_ordinal = -1 : i64, max_sequence_lengths = [0, 0, 0], table_ids = [1, 1, 0]} : (tensor<?x2xi32>, tensor<?x2xi32>, tensor<?x2xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<!tf_type.string>) -> ()
  %4:2 = "tf.RecvTPUEmbeddingActivations"() {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D"} : () -> (tensor<2x2xf32>, tensor<4x4xf32>)
  func.return
}

// -----

func.func @check_enqueue_ops_with_different_attr_disallowed(%arg0: tensor<?x2xi32>, %arg1: tensor<?x2xi32>,
  %arg2 :tensor<?x2xi32>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<?xi32>,
  %arg6: tensor<!tf_type.string>, %arg7: tensor<!tf_type.string>, %arg8: tensor<i1>) -> () {
  %0 = "tf.Const"() {value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
  %1 = "tf.SelectV2"(%arg8, %arg6, %arg7) : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<!tf_type.string>
  // expected-error @+1 {{'tf.EnqueueTPUEmbeddingSparseTensorBatch' op must have a corresponding 'tf.RecvTPUEmbeddingActivations' op}}
  "tf.EnqueueTPUEmbeddingSparseTensorBatch"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %0, %0, %0, %1) {_tpu_embedding_layer = "call_123", _xla_outside_compilation = "0", combiners = ["mean", "sum"], device_ordinal = -1 : i64, max_sequence_lengths = [0, 0, 0], table_ids = [1, 1, 0]} : (tensor<?x2xi32>, tensor<?x2xi32>, tensor<?x2xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<0xf32>, tensor<0xf32>, tensor<0xf32>, tensor<!tf_type.string>) -> ()
  %2:2 = "tf.RecvTPUEmbeddingActivations"() {_tpu_embedding_layer = "call1", config = "\0A\0B\0C\0D"} : () -> (tensor<2x2xf32>, tensor<4x4xf32>)
  func.return
}

