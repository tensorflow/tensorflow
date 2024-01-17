// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail --dump-input-filter=all
// RUN: tf-tfrt-opt -pass-pipeline='builtin.module(tf-to-tfrt{target-tpurt=true tpu-use-core-selector=false})' %s | FileCheck %s --dump-input=fail --dump-input-filter=all

// CHECK-LABEL: func @_tfrt_fallback_init
// CHECK-SAME: {{.*}} !tfrt.chain
// CHECK: tfrt_fallback_async.createop(%arg0) key(0) device("/device:CPU:0") "tf.ParseExampleV2"()
// CHECK-SAME: Tdense = [f32, f32], dense_shapes = [#corert.shape<>, #corert.shape<>]
// CHECK-SAME: num_sparse = 2 : i64, ragged_split_types = [], ragged_value_types = []
// CHECK-SAME: sparse_types = [!corert.string, i64]}
// CHECK-SAME: num_args(7)

// CHECK: tfrt_fallback_async.createop(%0) key(1) device("/device:CPU:0") "tf.ReadVariableOp"() {dtype = f32} num_args(1)

// CHECK: tfrt_fallback_async.createop(%1) key(2) device("/device:CPU:0") "tf.MatMul"() {T = f32, transpose_a = false, transpose_b = false} num_args(2)

// CHECK-LABEL: func @main
// CHECK-SAME: {{.*}} !tfrt.chain
// CHECK-SAME: [[serialized:%.*]]: !corert.tensorhandle
func.func @main(%serialized: tensor<32x!tf_type.string>) -> (tensor<?x2xi64>) attributes {tf.entry_function = {inputs = "input0", outputs = "ParseExample/ParseExampleV2"}} {
  %dense_default_0 = "tf.Const"() {device = "/device:CPU:0", dtype = f32, value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
  %dense_default_1 = "tf.Const"() {device = "/device:CPU:0", dtype = f32, value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
  %dense_keys = "tf.Const"() {device = "/device:CPU:0", dtype = !tf_type.string, value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %names = "tf.Const"() {device = "/device:CPU:0", dtype = !tf_type.string, value = dense<""> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  %ragged_keys = "tf.Const"() {device = "/device:CPU:0", dtype = !tf_type.string, value = dense<""> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  %sparse_keys = "tf.Const"() {device = "/device:CPU:0", dtype = !tf_type.string, value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>

  // CHECK: [[fallback_serialized:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[serialized]]
  // CHECK-SAME: device = "/job:localhost/replica:0/task:0/device:CPU:0"
  // CHECK: [[outputs:%.*]]:8 = tfrt_fallback_async.executeop key(0) cost({{.*}}) device("/device:CPU:0") "tf.ParseExampleV2"
  // CHECK-SAME: ([[fallback_serialized]]
  // CHECK-NOT: device
  // CHECK-SAME: Tdense = [f32, f32]
  // CHECK-SAME: dense_shapes = [#corert.shape<>, #corert.shape<>]
  // CHECK-SAME: num_sparse = 2 : i64
  // CHECK-SAME: ragged_split_types = []
  // CHECK-SAME: ragged_value_types = []
  // CHECK-SAME: sparse_types = [!corert.string, i64]
  %outputs:8 = "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %dense_keys, %ragged_keys, %dense_default_0, %dense_default_1)
    {
      Tdense = [f32, f32], dense_shapes = [#tf_type.shape<>, #tf_type.shape<>],
      device = "/device:CPU:0", num_sparse = 2 : i64, ragged_split_types = [], ragged_value_types = [],
      resultSegmentSizes = array<i32: 2, 2, 2, 2, 0, 0>,
      sparse_types = [!tf_type.string, !tf_type.string]
    } : (tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0xf32>, tensor<0xf32>)
    -> (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>, tensor<2xi64>, tensor<32xf32>, tensor<32xf32>)

  // CHECK: [[result:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[outputs]]#0
  // CHECK-SAME: device = "/device:CPU:0"
  // CHECK: tfrt.return {{.*}}, [[result]]
  func.return %outputs#0 : tensor<?x2xi64>
}

// CHECK-LABEL: func @no_native
func.func @no_native(%arg0: tensor<3x1xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<3x3xf32> {
  // CHECK-NOT: corert.executeop
  // CHECK: tfrt_fallback_async.executeop.seq({{.*}}) key(1) cost({{.*}}) device("/device:CPU:0") "tf.ReadVariableOp"
  // CHECK: tfrt_fallback_async.executeop key(2) cost({{.*}}) device("/device:CPU:0") "tf.MatMul"
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %1 = "tf.MatMul"(%arg0, %0) {T = f32, device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  func.return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: func @gpu_device
func.func @gpu_device(%arg0: tensor<3x1xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<3x3xf32> {
  // CHECK: {{%.*}} = corert.get_op_handler %arg0 "/device:GPU:0"
  // CHECK: {{.*}} = corert.executeop.seq({{.*}}) "tf.ReadVariableOp"({{.*}}) {dtype = f32} : 1
  // CHECK: {{.*}} = corert.executeop({{.*}}) "tf.MatMul"({{.*}}) {T = f32, transpose_a = false, transpose_b = false} : 1
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:GPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %1 = "tf.MatMul"(%arg0, %0) {T = f32, device = "/device:GPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  func.return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: func @tpu_device
func.func @tpu_device(%arg0: tensor<3x1xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<3x3xf32> {
  // CHECK-NOT: corert.executeop
  // CHECK: tfrt_fallback_async.executeop.seq({{.*}}) key({{.*}}) cost({{.*}}) device("/device:TPU:0") "tf.ReadVariableOp"
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.MatMul"
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:TPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  %1 = "tf.MatMul"(%arg0, %0) {T = f32, device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  func.return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: func @tfrt_set_resource
// CHECK-SAME: ([[in_ch:%.*]]: !tfrt.chain,
func.func @tfrt_set_resource(%arg0: tensor<3x1xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) {
  // CHECK: [[ch0:%.*]] = tfrt_fallback_async.set_resource [[in_ch]], {{.*}} {device = "/device:CPU:0", index = 0 : i64}
  // CHECK: [[ch1:%.*]], [[result:%.*]] = tfrt_fallback_async.executeop.seq([[ch0]]) key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.ReadVariableOp"
  // CHECK: [[ch2:%.*]] = tfrt_fallback_async.set_resource [[ch1]], [[result]] {device = "/device:CPU:0", index = 1 : i64}

  "tf._TfrtSetResource"(%arg0) {device = "/device:CPU:0", index = 0} : (tensor<3x1xf32>) -> ()
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  "tf._TfrtSetResource"(%0) {device = "/device:CPU:0", index = 1} : (tensor<1x3xf32>) -> ()
  func.return
}

// CHECK-LABEL: func @tfrt_get_resource
func.func @tfrt_get_resource() -> tensor<3x3xf32> {
  // CHECK: [[ready_ch:%.*]] = tfrt.new.chain
  // CHECK: [[ch3:%.*]], [[results:%.*]]:2 = tfrt_fallback_async.get_resource [[ready_ch]] {device = "/device:CPU:0", indices = [0, 1]}
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.MatMul"([[results]]#0, [[results]]#1)
  %a, %b = "tf._TfrtGetResource"() {device = "/device:CPU:0", indices = [0, 1], shared_name = ["", ""], container = ["", ""]} : () -> (tensor<3x1xf32>, tensor<1x3xf32>)
  %1 = "tf.MatMul"(%a, %b) {T = f32, device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  func.return %1 : tensor<3x3xf32>
}

// CHECK-LABEL: func @tensor_array
func.func @tensor_array() -> (tensor<1x1x512xf32>) {
  %value = "tf.Const"() {device = "/device:CPU:0", value = dense<0.1> : tensor<1x512xf32>} : () -> (tensor<1x512xf32>)
  %index = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  %size = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
  %indices = "tf.Const"() {device = "/device:CPU:0", value = dense<[0]> : tensor<1xi32>} : () -> (tensor<1xi32>)
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.TensorArrayV3"
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.TensorArrayWriteV3"
  // CHECK: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.TensorArrayGatherV3"
  %handle, %flow = "tf.TensorArrayV3"(%size) {clear_after_read = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<?x512>, identical_element_shapes = true, tensor_array_name = "output"} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<1x512xf32>>>, tensor<f32>)
  %flow_1 = "tf.TensorArrayWriteV3"(%handle, %index, %value, %flow) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<1x512xf32>>>, tensor<i32>, tensor<1x512xf32>, tensor<f32>) -> tensor<f32>
  %result = "tf.TensorArrayGatherV3"(%handle, %indices, %flow_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", element_shape = #tf_type.shape<1x512>} : (tensor<2x!tf_type.resource<tensor<1x512xf32>>>, tensor<1xi32>, tensor<f32>) -> tensor<1x1x512xf32>
  func.return %result : tensor<1x1x512xf32>
}
