// RUN: tf-tfrt-opt %s                                                         \
// RUN:   -split-input-file                                                    \
// RUN:   -tf-executor-to-tfrt-pipeline="                                      \
// RUN:       enable-native-ops=false                                          \
// RUN:       enable-optimizer=true                                            \
// RUN:       tfrt-cost-threshold=1024                                         \
// RUN:       auto-fusion-oplist=tf.Relu,tf.Transpose,tf.Const                 \
// RUN:       auto-fusion-min-cluster-size=1"                                  \
// RUN: | FileCheck %s --dump-input=always

// Check TF->JitRT JIT compiled operations clustering and outlining starting
// from the Tensorflow executor dialect.

// -----
// Simple cluster consisting of a single operation.

module attributes {tf.versions = {producer = 462 : i32}} {
  // CHECK: func @_tfrt_fallback_init(%[[ARG:.*]]: !tfrt.chain)
  // CHECK:   %[[COMPILED:.*]] = tf_jitrt.fallback.compile @kernel::@compute
  // CHECK:   %[[CHAIN:.*]] = tfrt.merge.chains %[[ARG]], %[[COMPILED]]
  // CHECK:   tfrt.return %[[CHAIN]]

  // CHECK: func @call
  func.func @call(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>)
      attributes { tf.entry_function = {control_outputs = "",
                                        inputs = "input_0",
                                        outputs = "output_0"}} {
    // CHECK: tf_jitrt.fallback.execute @kernel::@compute
    %0 = tf_executor.graph {
      %outs, %control = tf_executor.island wraps "tf.Relu"(%arg0)
                          {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outs: tensor<?x?xf32>
    }
    func.return %0 : tensor<?x?xf32>
  }
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK:      func @compute(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<?x?xf32>
// CHECK-SAME: ) -> tensor<?x?xf32> {
// CHECK:        %[[RELU:.*]] = "tf.Relu"(%[[ARG0]])
// CHECK:        return %[[RELU]]
// CHECK:      }

// -----
// Two identical clusters (except the _class attribute) consisting of a single
// `Relu` operation. Check that outlined clusters are deduplicated and we
// compile only once.

module attributes {tf.versions = {producer = 462 : i32}} {
  // CHECK:     func @_tfrt_fallback_init
  // CHECK:       tf_jitrt.fallback.compile @kernel::@compute
  // CHECK-NOT:   tf_jitrt.fallback.compile

  // CHECK: func @call
  func.func @call(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>)
      attributes { tf.entry_function = {control_outputs = "",
                                        inputs = "input_0",
                                        outputs = "output_0"}} {
    // CHECK: tf_jitrt.fallback.execute @kernel::@compute
    // CHECK: tfrt_fallback_async.executeop {{.*}} "tf.Sqrt"
    // CHECK: tf_jitrt.fallback.execute @kernel::@compute
    %0 = tf_executor.graph {
      %outs0, %control0 = tf_executor.island wraps "tf.Relu"(%arg0)
                            {device = "", _class = ["loc:@Relu_0"]}
                            : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %outs1, %control1 = tf_executor.island wraps "tf.Sqrt"(%outs0)
                            {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      %outs2, %control2 = tf_executor.island wraps "tf.Relu"(%outs1)
                            {device = "", _class = ["loc:@Relu_1"]}
                            : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outs2: tensor<?x?xf32>
    }
    func.return %0 : tensor<?x?xf32>
  }
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK:      func @compute(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<?x?xf32>
// CHECK-SAME: ) -> tensor<?x?xf32> {
// CHECK:        %[[RELU:.*]] = "tf.Relu"(%[[ARG0]])
// CHECK:        return %[[RELU]]
// CHECK:      }

// -----
// Constants sunk into the outlined compiled functions.

module attributes {tf.versions = {producer = 462 : i32}} {
  // CHECK: func @_tfrt_fallback_init
  // CHECK:   tf_jitrt.fallback.compile @kernel::@compute

  // CHECK: func @call
  func.func @call(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>)
      attributes { tf.entry_function = {control_outputs = "",
                                        inputs = "input_0",
                                        outputs = "output_0"}} {
    // CHECK: tf_jitrt.fallback.execute @kernel::@compute
    %0 = tf_executor.graph {
      %perm, %perm_ctl = tf_executor.island wraps "tf.Const"()
                         {device = "", value = dense<[1, 0]> : tensor<2xi32>}
                         : () -> tensor<2xi32>
      %out, %out_ctl = tf_executor.island wraps "tf.Transpose"(%arg0, %perm)
                       {device = ""}
                       : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
      tf_executor.fetch %out: tensor<?x?xf32>
    }
    func.return %0 : tensor<?x?xf32>
  }
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK:      func @compute(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<?x?xf32>
// CHECK-SAME: ) -> tensor<?x?xf32> {
// CHECK:        %[[PERM:.*]] = "tf.Const"() {{.*}} dense<[1, 0]>
// CHECK:        %[[RET:.*]] = "tf.Transpose"(%[[ARG0]], %[[PERM]])
// CHECK:        return %[[RET]]
// CHECK:      }

// -----
// tf.Transpose: a non-const permutation parameter cannot be sunk into the
// compiled function. Such a transpose should, however, support clustering,
// and its permutation parameter should compile to be value-constrained.

module attributes {tf.versions = {producer = 462 : i32}} {
  // CHECK: func @_tfrt_fallback_init
  // CHECK:   tf_jitrt.fallback.compile @kernel::@compute

  // CHECK: func @call
  func.func @call(%arg0: tensor<?x?xf32>, %arg1: tensor<?xi32>) -> (tensor<?x?xf32>)
      attributes { tf.entry_function = {control_outputs = "",
                                        inputs = "input_0,input_1",
                                        outputs = "output_0"}} {
    // CHECK: tf_jitrt.fallback.execute @kernel::@compute
    %0 = tf_executor.graph {
      %out, %out_ctl = tf_executor.island wraps "tf.Transpose"(%arg0, %arg1)
                       {device = ""}
                       : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<?x?xf32>
      tf_executor.fetch %out: tensor<?x?xf32>
    }
    func.return %0 : tensor<?x?xf32>
  }
}

// CHECK:      module @kernel attributes {
// CHECK-SAME:   tfrt.compiled
// CHECK-SAME:   "tfrt.max-arg-size" = 1 : i64
// CHECK-SAME: }
// CHECK:      func @compute(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:.*]]: tensor<?xi32> {jitrt.constraint = "value"}
// CHECK-SAME: ) -> tensor<?x?xf32> {
// CHECK-NEXT:   %[[RET:.*]] = "tf.Transpose"(%[[ARG0]], %[[ARG1]])
// CHECK:        return %[[RET]]
// CHECK:      }

// -----
// Operations with unsupported data type operands/results are not clustered.

module attributes {tf.versions = {producer = 462 : i32}} {
  func.func @call(%arg0: tensor<?x?x!tf_type.string>) -> (tensor<?x?x!tf_type.string>)
      attributes { tf.entry_function = {control_outputs = "",
                                        inputs = "input_0",
                                        outputs = "output_0"}} {
    // CHECK-NOT: tf_jitrt.fallback.compile
    // CHECK-NOT: tf_jitrt.fallback.execute
    // CHECK-NOT: module @kernel
    %0 = tf_executor.graph {
      %perm, %perm_ctl =
        tf_executor.island wraps "tf.Const"()
        {device = "", value = dense<[1, 0]> : tensor<2xi32>}
        : () -> tensor<2xi32>
      %out, %out_ctl =
        tf_executor.island wraps "tf.Transpose"(%arg0, %perm) {device = ""}
        : (tensor<?x?x!tf_type.string>, tensor<2xi32>) -> tensor<?x?x!tf_type.string>
      tf_executor.fetch %out: tensor<?x?x!tf_type.string>
    }
    func.return %0 : tensor<?x?x!tf_type.string>
  }
}
