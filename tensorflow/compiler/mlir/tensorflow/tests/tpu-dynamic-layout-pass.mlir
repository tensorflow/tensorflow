// RUN: tf-opt %s -split-input-file -tf-tpu-dynamic-layout-pass | FileCheck %s --dump-input=fail

// Tests that the pass can transform non-replicated execution.

// CHECK: func @non_replicated(%[[ARG0:.*]]: tensor<*x!tf.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @non_replicated(%arg0: tensor<*x!tf.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}
  // CHECK: %[[ITER:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK: "tf_device.launch"
  // CHECK-NEXT: "tf.TPUCompileSucceededAssert"
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK: "tf_device.launch"
  // CHECK-NEXT: "tf.TPUExecute"(%[[COPY0]], %[[COPY1]], %[[COMPILE]]#1)
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not transform two execute ops sharing the same
// compile op.

// CHECK-LABEL: func @multiple_compile_uses
func @multiple_compile_uses(%arg0: tensor<*x!tf.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute0 = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  %4:2 = "tf._UnKnownOp_"() : () -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  %execute1 = "tf_device.launch"() ( {
    %5 = "tf.TPUExecute"(%4#0, %4#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
    tf_device.return %5 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute1 : tensor<i32>
}

// -----

// Tests that the pass does not transform when tf.IteratorGetNext is on TPU.

// CHECK-LABEL: func @on_tpu_iter
func @on_tpu_iter(%arg0: tensor<*x!tf.resource> {tf.device = "/device:TPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:TPU:0"}
    : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not change unsupported input ops.

// CHECK-LABEL: func @unsupported_ops
func @unsupported_ops(%arg0: tensor<3x3x1x32xf32>) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2 = "tf._Unknown_"() : () -> tensor<3x3x1x32xf32>
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%arg0, %2, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass can transform replicated execution.

// CHECK: func @replicated(%[[ARG0:.*]]: tensor<*x!tf.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @replicated(%arg0: tensor<*x!tf.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[ITER0:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK: %[[ITER1:.*]]:2 = "tf.IteratorGetNext"
  %3:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK-DAG: %[[COPY2:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#0, %[[LAYOUT0]]) {device = "/device:TPU:1"}
  // CHECK-DAG: %[[COPY3:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#1, %[[LAYOUT1]]) {device = "/device:TPU:1"}
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK: tf_device.replicate([%[[COPY0]], %[[COPY2]]] as %[[R0:.*]]: tensor<3x3x1x32xf32>, [%[[COPY1]], %[[COPY3]]] as %[[R1:.*]]: tensor<3x3x1x32xf32>)
  %5:2 = tf_device.replicate([%2#0, %3#0] as %r0: tensor<3x3x1x32xf32>, [%2#1, %3#1] as %r1: tensor<3x3x1x32xf32>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    // CHECK: "tf.TPUExecute"(%[[R0]], %[[R1]], %[[COMPILE]]#1)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%r0, %r1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}

// -----

// Tests that the pass does not change inputs inside replicate.

// CHECK-LABEL: func @inside_replicated
func @inside_replicated(%arg0: tensor<*x!tf.resource>, %arg1: tensor<*x!tf.resource>) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf.string>, tensor<!tf.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf.string>, tensor<!tf.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %5:2 = tf_device.replicate([%arg0, %arg1] as %r0: tensor<*x!tf.resource>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    %2:2 = "tf.IteratorGetNext"(%r0)
      : (tensor<*x!tf.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%2#0, %2#1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<!tf.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}
