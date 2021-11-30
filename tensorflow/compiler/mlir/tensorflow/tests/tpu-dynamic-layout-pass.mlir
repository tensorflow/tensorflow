// RUN: tf-opt %s -split-input-file -tf-tpu-dynamic-layout-pass | FileCheck %s

// Tests that the pass can transform non-replicated execution.

// CHECK: func @non_replicated(%[[ARG0:.*]]: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @non_replicated(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}
  // CHECK: %[[ITER:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK: "tf_device.launch"
  // CHECK-NEXT: "tf.TPUCompileSucceededAssert"
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK: "tf_device.launch"
  // CHECK-NEXT: "tf.TPUExecute"(%[[COPY0]], %[[COPY1]], %[[COMPILE]]#1)
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not transform two execute ops sharing the same
// compile op.

// CHECK-LABEL: func @multiple_compile_uses
func @multiple_compile_uses(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute0 = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  %4:2 = "tf._UnKnownOp_"() : () -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  %execute1 = "tf_device.launch"() ( {
    %5 = "tf.TPUExecute"(%4#0, %4#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %5 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute1 : tensor<i32>
}

// -----

// Tests that the pass does not transform when tf.IteratorGetNext is on TPU.

// CHECK-LABEL: func @on_tpu_iter
func @on_tpu_iter(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:TPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:TPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not transform when tf.IteratorGetNext is on CPU
// but generator is on TPU.

// CHECK-LABEL: func @arg_on_tpu_iter_on_cpu
func @arg_on_tpu_iter_on_cpu(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:TPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not transform when tf.IteratorGetNext is on CPU but
// generator is on TPU. All intermediate nodes like tf.Identity between
// generator and IteratorGetNext are on CPU too.

// CHECK-LABEL: func @arg_on_tpu_intermediate_ops_on_cpu
func @arg_on_tpu_intermediate_ops_on_cpu(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:TPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  %id1 = "tf.Identity"(%arg0) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> (tensor<*x!tf_type.resource>)
  %id2 = "tf.Identity"(%id1) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> (tensor<*x!tf_type.resource>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%id2) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not transform when tf.IteratorGetNext is on CPU but
// generator is on TPU.

// CHECK-LABEL: func @var_handle_on_tpu_iter_on_cpu
func @var_handle_on_tpu_iter_on_cpu() -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  %var = "tf.VarHandleOp"() {container = "c", shared_name = "v", device = "/device:TPU:0"} : () -> tensor<!tf_type.resource<tensor<3x3x1x32xf32>>>
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2:2 = "tf.IteratorGetNext"(%var) {device = "/device:CPU:0"}
    : (tensor<!tf_type.resource<tensor<3x3x1x32xf32>>>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%2#0, %2#1, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass does not change unsupported input ops.

// CHECK-LABEL: func @unsupported_ops
func @unsupported_ops(%arg0: tensor<3x3x1x32xf32> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  %2 = "tf._Unknown_"() : () -> tensor<3x3x1x32xf32>
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %execute = "tf_device.launch"() ( {
    %3 = "tf.TPUExecute"(%arg0, %2, %compile#1)
      : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
    tf_device.return %3 : tensor<i32>
  }) {device = "/device:TPU:0"} : () -> tensor<i32>
  return %execute : tensor<i32>
}

// -----

// Tests that the pass can transform replicated execution.

// CHECK: func @replicated(%[[ARG0:.*]]: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @replicated(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[ITER0:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}
  // CHECK: %[[ITER1:.*]]:2 = "tf.IteratorGetNext"
  %3:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY2:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#0, %[[LAYOUT0]]) {device = "/device:TPU:1"}
  // CHECK-DAG: %[[COPY3:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#1, %[[LAYOUT1]]) {device = "/device:TPU:1"}
  // CHECK: tf_device.replicate([%[[COPY0]], %[[COPY2]]] as %[[R0:.*]]: tensor<3x3x1x32xf32>, [%[[COPY1]], %[[COPY3]]] as %[[R1:.*]]: tensor<3x3x1x32xf32>)
  %5:2 = tf_device.replicate([%2#0, %3#0] as %r0: tensor<3x3x1x32xf32>, [%2#1, %3#1] as %r1: tensor<3x3x1x32xf32>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    // CHECK: "tf.TPUExecute"(%[[R0]], %[[R1]], %[[COMPILE]]#1)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%r0, %r1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}

// -----

// Tests that the pass can transform replicated execution with packed inputs.

// CHECK: func @replicated_packed(%[[ARG0:.*]]: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @replicated_packed(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[ITER0:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}

  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK: tf_device.replicate(%[[COPY0]] as %[[R0:.*]]: tensor<3x3x1x32xf32>, %[[COPY1]] as %[[R1:.*]]: tensor<3x3x1x32xf32>)
  %5:2 = tf_device.replicate(%2#0 as %r0: tensor<3x3x1x32xf32>, %2#1 as %r1: tensor<3x3x1x32xf32>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    // CHECK: "tf.TPUExecute"(%[[R0]], %[[R1]], %[[COMPILE]]#1)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%r0, %r1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}

// -----

// Tests that the pass can transform replicated execution with both replicated
// and packed operands.

// CHECK: func @replicated(%[[ARG0:.*]]: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32>
func @replicated(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}, %arg1: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  // CHECK: %[[ITER0:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
  // CHECK: %[[COMPILE:.*]]:2 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 1 : i64, is_output = false}
  // CHECK: %[[ITER1:.*]] = "tf.IteratorGetNext"
  %3 = "tf.IteratorGetNext"(%arg1) {device = "/device:CPU:0"}
    : (tensor<*x!tf_type.resource>) -> tensor<3x3x1x32xf32>
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#1, %[[LAYOUT1]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY2:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]], %[[LAYOUT0]]) {device = "/device:TPU:1"}
  // CHECK: tf_device.replicate([%[[COPY0]], %[[COPY2]]] as %[[R0:.*]]: tensor<3x3x1x32xf32>, %[[COPY1]] as %[[R1:.*]]: tensor<3x3x1x32xf32>)
  %5:2 = tf_device.replicate([%2#0, %3] as %r0: tensor<3x3x1x32xf32>, %2#1 as %r1: tensor<3x3x1x32xf32>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    // CHECK: "tf.TPUExecute"(%[[R0]], %[[R1]], %[[COMPILE]]#1)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%r0, %r1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}

// -----

// Tests that the pass does not change inputs inside replicate.

// CHECK-LABEL: func @inside_replicated
func @inside_replicated(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}, %arg1: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) -> tensor<i32> {
  %compile:2 = "tf_device.launch"() ( {
    %1:2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      // The metadata encodes 2 parameter and two return values.
      metadata = "\0A\0E\08\01\18\01\22\08\08\01\1A\01\01\22\01\00\0A \08\01\12\10\12\02\08\03\12\02\08\03\12\02\08\01\12\02\08 \18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\02 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-NOT: "tf.TPUGetLayoutOp"
  // CHECK-NOT: "tf.TPUCopyWithLayout"
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  %5:2 = tf_device.replicate([%arg0, %arg1] as %r0: tensor<*x!tf_type.resource>)
      {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"]}} {
    %2:2 = "tf.IteratorGetNext"(%r0)
      : (tensor<*x!tf_type.resource>) -> (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>)
    %execute = "tf_device.launch"() ( {
      %4 = "tf.TPUExecute"(%2#0, %2#1, %compile#1) : (tensor<3x3x1x32xf32>, tensor<3x3x1x32xf32>, tensor<2x!tf_type.string>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {device = "TPU_REPLICATED_CORE_0"} : () -> tensor<i32>
    tf_device.return %execute : tensor<i32>
  }
  return %5#0 : tensor<i32>
}

// -----

// Tests that the pass can transform execution with model parallelism and no
// replication.
//
// The following TPUCompileMetadataProto is used:
// args {
//   dtype: DT_FLOAT
//   shape {
//     dim {
//       size: 128
//     }
//   }
// }
// num_replicas: 1
// num_cores_per_replica: 2

// CHECK-LABEL: func @parallel_execute
func @parallel_execute(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) {
  // CHECK: %[[COMPILE:.*]]:3 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:3 = "tf_device.launch"() ( {
    %1:3 = "tf._TPUCompileMlir"() {NumDynamicShapes = 0 : i64, metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\01 \02", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1, %1#2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#2) {index = 0 : i64, is_output = false}
  // CHECK: %[[ITER:.*]]:2 = "tf.IteratorGetNext"
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> (tensor<128xf32>, tensor<128xf32>)
  // CHECK: "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK: "tf_device.parallel_execute"
  "tf_device.parallel_execute"() ({
    // CHECK-NEXT: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#0, %[[LAYOUT0]])
    // CHECK-SAME: device = "/device:TPU:0"
    // CHECK-NEXT: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[COPY0]], %[[COMPILE]]#1)
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: device = "/device:TPU:0"
    "tf_device.launch"() ( {
      "tf.TPUExecute"(%2#0, %compile#1) : (tensor<128xf32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/device:TPU:0"} : () -> ()
    tf_device.return
  },
  {
    // CHECK: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER]]#1, %[[LAYOUT1]])
    // CHECK-SAME: device = "/device:TPU:1"
    // CHECK-NEXT: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[COPY1]], %[[COMPILE]]#2)
    // CHECK-NEXT: tf_device.return
    // CHECK-NEXT: device = "/device:TPU:1"
    "tf_device.launch"() ( {
      "tf.TPUExecute"(%2#1, %compile#2) : (tensor<128xf32>, tensor<2x!tf_type.string>) -> ()
      tf_device.return
    }) {device = "/device:TPU:1"} : () -> ()
    tf_device.return
  }) {} : () -> ()
  return
}

// -----

// Tests that the pass can transform execution with model parallelism and
// replication.
//
// The following TPUCompileMetadataProto is used:
// args {
//   dtype: DT_FLOAT
//   shape {
//     dim {
//       size: 128
//     }
//   }
// }
// num_replicas: 2
// num_cores_per_replica: 2

// CHECK-LABEL: func @replicated_parallel_execute
// CHECK-SAME: %[[ARG0:[a-z0-9]+]]: tensor<*x!tf_type.resource>
// CHECK-SAME: %[[ARG1:[a-z0-9]+]]: tensor<*x!tf_type.resource>
func @replicated_parallel_execute(%arg0: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}, %arg1: tensor<*x!tf_type.resource> {tf.device = "/device:CPU:0"}) {
  // CHECK: %[[COMPILE:.*]]:3 = "tf_device.launch"
  // CHECK-NEXT: "tf._TPUCompileMlir"()
  %compile:3 = "tf_device.launch"() ( {
    %1:3 = "tf._TPUCompileMlir"() {NumDynamicShapes = 0 : i64, metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\02 \02", mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1#0, %1#1, %1#2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>
  }) {device = "/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>)
  // CHECK-DAG: %[[LAYOUT0:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#1) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[LAYOUT1:.*]] = "tf.TPUGetLayoutOp"(%[[COMPILE]]#2) {index = 0 : i64, is_output = false}
  // CHECK-DAG: %[[ITER0:.*]]:2 = "tf.IteratorGetNext"(%[[ARG0]])
  // CHECK-DAG: %[[ITER1:.*]]:2 = "tf.IteratorGetNext"(%[[ARG1]])
  %2:2 = "tf.IteratorGetNext"(%arg0) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> (tensor<128xf32>, tensor<128xf32>)
  %3:2 = "tf.IteratorGetNext"(%arg1) {device = "/device:CPU:0"} : (tensor<*x!tf_type.resource>) -> (tensor<128xf32>, tensor<128xf32>)
  // CHECK: "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
  "tf_device.launch"() ( {
    "tf.TPUCompileSucceededAssert"(%compile#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/device:CPU:0"} : () -> ()
  // CHECK-DAG: %[[COPY0:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#0, %[[LAYOUT0]]) {device = "/device:TPU:0"}
  // CHECK-DAG: %[[COPY1:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#0, %[[LAYOUT0]]) {device = "/device:TPU:1"}
  // CHECK-DAG: %[[COPY2:.*]] = "tf.TPUCopyWithLayout"(%[[ITER0]]#1, %[[LAYOUT1]]) {device = "/device:TPU:2"}
  // CHECK-DAG: %[[COPY3:.*]] = "tf.TPUCopyWithLayout"(%[[ITER1]]#1, %[[LAYOUT1]]) {device = "/device:TPU:3"}
  // CHECK-NEXT: tf_device.replicate
  // CHECK-SAME: ([%[[COPY0]], %[[COPY1]]] as %[[R0:[a-z0-9]+]]: tensor<128xf32>, [%[[COPY2]], %[[COPY3]]] as %[[R1:[a-z0-9]+]]: tensor<128xf32>)
  tf_device.replicate([%2#0, %3#0] as %r0: tensor<128xf32>, [%2#1, %3#1] as %r1: tensor<128xf32>) {n = 2 : i32, devices = {TPU_REPLICATED_CORE_0 = ["/device:TPU:0", "/device:TPU:1"], TPU_REPLICATED_CORE_1 = ["/device:TPU:2", "/device:TPU:3"]}} {
    // CHECK: "tf_device.parallel_execute"
    "tf_device.parallel_execute"() ({
      // CHECK: "tf.TPUExecute"(%[[R0]], %[[COMPILE]]#1)
      // CHECK-NEXT: tf_device.return
      // CHECK-NEXT: device = "TPU_REPLICATED_CORE_0"
      "tf_device.launch"() ( {
        "tf.TPUExecute"(%r0, %compile#1) : (tensor<128xf32>, tensor<2x!tf_type.string>) -> ()
        tf_device.return
      }) {device = "TPU_REPLICATED_CORE_0"} : () -> ()
      tf_device.return
    },
    {
      // CHECK: "tf.TPUExecute"(%[[R1]], %[[COMPILE]]#2)
      // CHECK-NEXT: tf_device.return
      // CHECK-NEXT: device = "TPU_REPLICATED_CORE_1"
      "tf_device.launch"() ( {
        "tf.TPUExecute"(%r1, %compile#2) : (tensor<128xf32>, tensor<2x!tf_type.string>) -> ()
        tf_device.return
      }) {device = "TPU_REPLICATED_CORE_1"} : () -> ()
      tf_device.return
    }) {} : () -> ()
    tf_device.return
  }
  return
}
