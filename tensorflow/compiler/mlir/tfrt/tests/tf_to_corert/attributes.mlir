// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// _output_shapes and f.* attributes are removed during tf-to-tfrt lowering.
// CHECK-LABEL: func @remove_unused_attr
func @remove_unused_attr() {
  // CHECK: %out_op_chain = tfrt_fallback_async.executeop.seq(%arg0) key(0) cost({{.*}}) device("/device:CPU:0") "tf.SomeOp2"()
  "tf.SomeOp2"() {device = "/device:CPU:0", _output_shapes = ["tfshape$"], f.Tin = [f32], f._read_only_resource_inputs = []} : () -> ()
  return
}

// CHECK-LABEL: func @basic
func @basic(
    %arg0: tensor<3x1xf32>,
    %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> (tensor<3x3xf32>) {
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = f32} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>

  // CHECK: {{%.*}} = corert.executeop({{%.*}}) "tf.MatMul"
  // CHECK-SAME: {T = f32, device = "/device:CPU:0", transpose_a = false, transpose_b = false}
  %2 = "tf.MatMul"(%arg0, %1) {T = f32, _output_shapes = ["tfshape$dim { size: 3 } dim { size: 3 }"], device = "/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<3x1xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}

// CHECK-LABEL: func @string_type
func @string_type(%arg: tensor<1x2x!tf_type.string>) -> tensor<?x6x!tf_type.string> {
  %multiples = "tf.Const"() { device = "/device:CPU:0", value = dense<[7,3]> : tensor<2xi32> } : () -> tensor<2xi32>
  // CHECK: T = !corert.string
  %output = "tf.Tile"(%arg, %multiples) { device = "/device:CPU:0" } : (tensor<1x2x!tf_type.string>, tensor<2xi32>) -> tensor<?x6x!tf_type.string>
  return %output : tensor<?x6x!tf_type.string>
}

// CHECK-LABEL: func @shape
func @shape() {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf.TensorArrayV3
  // CHECK-SAME: element_shape = #corert.shape<*>
  %ta:2 = "tf.TensorArrayV3"(%size) {device = "/device:CPU:0", dtype = f32, element_shape = #tf_type.shape<*>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf_type.resource>, tensor<f32>)
  return
}

// CHECK-LABEL: func @resource
func @resource() {
  // CHECK: tf.SomeOp
  // CHECK-SAME: dtype = !corert.resource
  "tf.SomeOp"() {device = "/device:CPU:0", dtype = !tf_type.resource} : () -> ()
  return
}

// CHECK-LABEL: func @variant
func @variant(%arg: tensor<!tf_type.variant>) {
  // CHECK: tf.ZerosLike
  // CHECK-SAME: T = !corert.variant
  %0 = "tf.ZerosLike"(%arg) {device = "/device:CPU:0", T = !tf_type.variant} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  return
}

// Checks that TF quantized attrs are lowered to the corert types
// CHECK-LABEL: func @quantized_types
func @quantized_types(%arg0: tensor<!tf_type.resource<tensor<1x3x!tf_type.quint8>>>,
                      %arg1: tensor<!tf_type.resource<tensor<1x3x!tf_type.quint16>>>,
                      %arg2: tensor<!tf_type.resource<tensor<1x3x!tf_type.qint8>>>,
                      %arg3: tensor<!tf_type.resource<tensor<1x3x!tf_type.qint16>>>,
                      %arg4: tensor<!tf_type.resource<tensor<1x3x!tf_type.qint32>>>) {
  // CHECK: tf.ReadVariableOp
  // CHECK-SAME: dtype = !corert.quint8
  %0 = "tf.ReadVariableOp"(%arg0) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = !tf_type.quint8} : (tensor<!tf_type.resource<tensor<1x3x!tf_type.quint8>>>) -> tensor<1x3x!tf_type.quint8>

  // CHECK: tf.ReadVariableOp
  // CHECK-SAME: dtype = !corert.quint16
  %1 = "tf.ReadVariableOp"(%arg1) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = !tf_type.quint16} : (tensor<!tf_type.resource<tensor<1x3x!tf_type.quint16>>>) -> tensor<1x3x!tf_type.quint16>

  // CHECK: tf.ReadVariableOp
  // CHECK-SAME: dtype = !corert.qint8
  %2 = "tf.ReadVariableOp"(%arg2) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = !tf_type.qint8} : (tensor<!tf_type.resource<tensor<1x3x!tf_type.qint8>>>) -> tensor<1x3x!tf_type.qint8>

  // CHECK: tf.ReadVariableOp
  // CHECK-SAME: dtype = !corert.qint16
  %3 = "tf.ReadVariableOp"(%arg3) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = !tf_type.qint16} : (tensor<!tf_type.resource<tensor<1x3x!tf_type.qint16>>>) -> tensor<1x3x!tf_type.qint16>

  // CHECK: tf.ReadVariableOp
  // CHECK-SAME: dtype = !corert.qint32
  %4 = "tf.ReadVariableOp"(%arg4) {_output_shapes = ["tfshape$dim { size: 1 } dim { size: 3 }"], device = "/device:CPU:0", dtype = !tf_type.qint32} : (tensor<!tf_type.resource<tensor<1x3x!tf_type.qint32>>>) -> tensor<1x3x!tf_type.qint32>
  return
}
