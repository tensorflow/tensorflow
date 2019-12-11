// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tpu-rewrite -tpu_compile_metadata_debug | FileCheck %s --dump-input=fail

// Tests module with missing `tf.versions` attribute.

// expected-error@+1 {{requires attribute 'tf.versions'}}
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @missing_tf_versions() {
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests collecting compilation and execution devices results in an error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_devices() {
    // expected-error@+1 {{error in fetching TPU compilation/execution devices: no TPU_SYSTEM devices found}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with missing `num_cores_per_replicas`
// attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @missing_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'num_cores_per_replica'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with bad `num_cores_per_replicas` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'num_cores_per_replica'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = "", step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with missing `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'step_marker_location'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with bad `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_step_marker_location() {
    // expected-error@+1 {{requires attribute 'step_marker_location'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = 1, padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with unparsable `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @unparsable_step_marker_location() {
    // expected-error@+1 {{bad 'step_marker_location' attribute with value 'test'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "test", padding_map = []} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with missing `padding_map` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @missing_padding_map() {
    // expected-error@+1 {{requires attribute 'padding_map'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP"} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with bad `padding_map` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_padding_map() {
    // expected-error@+1 {{requires attribute 'padding_map'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ""} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with bad element in `padding_map` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @bad_element_padding_map() {
    // expected-error@+1 {{bad 'padding_map' attribute at index 0, not a string}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = [1]} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with unparsable element in `padding_map` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @unparsable_element_padding_map() {
    // expected-error@+1 {{bad 'padding_map' attribute at index 0 with value 'test'}}
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["test"]} : () -> ()
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests `tf_device.launch_func` with unsupported operand type.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func @unsupported_operand_type(%arg0: tensor<?xi2>) {
    // expected-error@+1 {{failed to determine operand type at index 0: Converting i2 to DataType}}
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_ENTRY", padding_map = []} : (tensor<?xi2>) -> tensor<?xi2>
    return
  }
  func @empty_func(%arg0: tensor<?xi2>) -> tensor<?xi2> {
    return %arg0 : tensor<?xi2>
  }
}

// -----

// Tests `tf_device.launch_func` with empty `step_marker_location` attribute
// defaults to `STEP_MARK_AT_ENTRY`.
//
// The expected TPUCompileMetadataProto is:
//   num_replicas: 1
//   num_cores_per_replica: 1

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @default_step_marker_location
  func @default_step_marker_location() {
    "tf_device.launch_func"() {_tpu_replicate = "cluster0", device = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : () -> ()
    // CHECK:      metadata
    // CHECK-SAME: num_replicas: 1
    // CHECK-SAME: num_cores_per_replica: 1
    return
  }
  func @empty_func() {
    return
  }
}

// -----

// Tests argument with unranked shape. Empty shape should be populated in the
// metadata for associated argument.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @unranked_shape_arg
  func @unranked_shape_arg(%arg0: tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<*xi32>) -> tensor<*xi32>
    // CHECK:      metadata
    // CHECK-SAME: shape {\0A unknown_rank: true

    return %0: tensor<*xi32>
  }
  func @_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
    return %arg0 : tensor<*xi32>
  }
}

// -----

// Tests argument with partial shape.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @partial_shape_arg
  func @partial_shape_arg(%arg0: tensor<?x?x3xi32>) -> tensor<?x?x3xi32> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<?x?x3xi32>) -> tensor<?x?x3xi32>
    // CHECK:      metadata
    // CHECK-SAME: args
    // CHECK-SAME: shape {\0A dim {\0A size: -1\0A }\0A dim {\0A size: -1\0A }\0A dim {\0A size: 3\0A }\0A }
    return %0: tensor<?x?x3xi32>
  }
  func @_func(%arg0: tensor<?x?x3xi32>) -> tensor<?x?x3xi32> {
    return %arg0 : tensor<?x?x3xi32>
  }
}

// -----

// Tests argument with static shape.

// The expected TensorShapeProto is:
//   shape {
//     dim {
//       size: 1
//     }
//     dim {
//       size: 2
//     }
//     dim {
//       size: 3
//     }
//   }

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @static_shape_arg
  func @static_shape_arg(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
    // CHECK:      metadata
    // CHECK-SAME: args
    // CHECK-SAME: shape
    // CHECK-SAME: dim
    // CHECK-SAME: size: 1
    // CHECK-SAME: dim
    // CHECK-SAME: size: 2
    // CHECK-SAME: dim
    // CHECK-SAME: size: 3

    return %0: tensor<1x2x3xi32>
  }
  func @_func(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
    return %arg0 : tensor<1x2x3xi32>
  }
}

// -----

// Tests argument that is a resource variable.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @resource_arg
  func @resource_arg(%arg0: tensor<*x!tf.resource>) -> tensor<*x!tf.resource> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<*x!tf.resource>) -> tensor<*x!tf.resource>
    // CHECK:      metadata
    // CHECK:      dtype: DT_RESOURCE
    // CHECK-SAME: kind: VARIABLE

    return %0: tensor<*x!tf.resource>
  }
  func @_func(%arg0: tensor<*x!tf.resource>) -> tensor<*x!tf.resource> {
    return %arg0 : tensor<*x!tf.resource>
  }
}

// -----

// Tests argument that is a parameter.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @parameter_arg
  func @parameter_arg(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:      metadata
    // CHECK:      dtype: DT_FLOAT
    // CHECK-SAME: kind: PARAMETER

    return %0: tensor<*xf32>
  }
  func @_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    return %arg0 : tensor<*xf32>
  }
}

// -----

// The following padding map is used in subsequent test cases:
// Proto debug string:
//   arg_index: 1
//   shape_index: 2
//   padding_arg_index: 3
// Serialized string:
//   "\08\01\10\02\18\03"

// -----

// Tests metadata is populated correctly based on launch_func op and attributes.
//
// The expected TPUCompileMetadataProto is:
//   args {
//     dtype: DT_INT32
//     shape {
//       dim {
//         size: 8
//       }
// .   }
//     kind: PARAMETER
//     sharding {
//       type: MAXIMAL
//       tile_assignment_dimensions: 1
//       tile_assignment_devices: 0
//     }
//   }
//   retvals {
//     sharding {
//       type: MAXIMAL
//       tile_assignment_dimensions: 1
//       tile_assignment_devices: 0
//     }
//   }
//   num_replicas: 1
//   num_cores_per_replica: 1
//   padding_maps {
//     arg_index: 1
//     shape_index: 2
//     padding_arg_index: 3
//   }
//   step_marker_location: STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @metadata
  func @metadata(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %0 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<8xi32>) -> tensor<8xi32>
    // CHECK:      metadata
    // CHECK-SAME: args
    // CHECK-SAME: dtype: DT_INT32
    // CHECK-SAME: shape
    // CHECK-SAME: dim
    // CHECK-SAME: size: 8
    // CHECK-SAME: kind: PARAMETER
    // CHECK-SAME: sharding
    // CHECK-SAME: type: MAXIMAL
    // CHECK-SAME: tile_assignment_dimensions: 1
    // CHECK-SAME: tile_assignment_devices: 0
    // CHECK-SAME: retvals
    // CHECK-SAME: sharding
    // CHECK-SAME: type: MAXIMAL
    // CHECK-SAME: tile_assignment_dimensions: 1
    // CHECK-SAME: tile_assignment_devices: 0
    // CHECK-SAME: num_replicas: 1
    // CHECK-SAME: num_cores_per_replica: 1
    // CHECK-SAME: padding_maps
    // CHECK-SAME: arg_index: 1
    // CHECK-SAME: shape_index: 2
    // CHECK-SAME: padding_arg_index: 3
    // CHECK-SAME: step_marker_location: STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP

    return %0: tensor<8xi32>
  }
  func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    return %arg0 : tensor<8xi32>
  }
}

// -----

// Tests shape ops are only generated for operands with non static shapes.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @static_and_dynamic_shapes
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_1:[a-z0-9]*]]: tensor<8xi32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<8xi32>)
  func @static_and_dynamic_shapes(%arg0: tensor<*xi32>, %arg1: tensor<8xi32>, %arg2: tensor<*xi32>, %arg3: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK-NOT:  "tf.Shape"(%[[ARG_1]])
    // CHECK-NOT:  "tf.Shape"(%[[ARG_3]])
    // CHECK:      %[[ARG_0_SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_0]])
    // CHECK:      %[[ARG_2_SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_2]])
    %0 = "tf_device.launch_func"(%arg0, %arg1, %arg2, %arg3) {_tpu_replicate = "cluster0", device = "", func = @_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<*xi32>, tensor<8xi32>, tensor<*xi32>, tensor<8xi32>) -> tensor<8xi32>
    // CHECK:      "tf._TPUCompileMlir"(%[[ARG_0_SHAPE]], %[[ARG_2_SHAPE]])
    // CHECK-SAME: NumDynamicShapes = 2

    return %0: tensor<8xi32>
  }
  func @_func(%arg0: tensor<*xi32>, %arg1: tensor<8xi32>, %arg2: tensor<*xi32>, %arg3: tensor<8xi32>) -> tensor<8xi32> {
    return %arg1 : tensor<8xi32>
  }
}

// -----

// Tests simple case of `tf_device.launch_func` on TPU with single input and
// single output.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @single_tpu_launch_func
  func @single_tpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:CPU:0"
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)
    // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:TPU:0"

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests simple case of `tf_device.launch_func` on TPU with replication.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  // CHECK-LABEL: func @replicated_tpu_launch_func
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func @replicated_tpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK-SAME: ([%[[A_OUTPUT]], %[[ARG_0]]] as %[[RI_0:[a-z0-9]*]]: tensor<?xi32>)
    // CHECK-SAME: devices = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]
    // CHECK-SAME: n = 2
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[RI_0]])
      // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
      // CHECK-SAME: NumDynamicShapes = 1
      // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:CPU:0"
      // CHECK-SAME: metadata
      // CHECK-SAME: mlir_module
      // CHECK-SAME: func @main
      // CHECK-SAME: tf.B
      // CHECK-NOT: func = @tpu0_func
      // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[RI_0]], %[[COMPILE_OUTPUT]]#1)
      %2 = "tf_device.launch_func"(%ri_0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: tf_device.return %[[EXECUTE_OUTPUT]]
      tf_device.return %2 : tensor<?xi32>
    }

    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[REPLICATE]]#1)
    %2 = "tf.C"(%1#1) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: return %[[C_OUTPUT]]
    return %2 : tensor<?xi32>
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests that launch_func without _tpu_replicate attribute is ignored.

module attributes {tf.versions = {producer = 888 : i32}} {
  // CHECK-LABEL: func @single_gpu_launch_func
  func @single_gpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    %1 = "tf_device.launch_func"(%0) {device = "gpu0", func = @gpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: tf_device.launch_func
    // CHECK-SAME: device = "gpu0"
    // CHECK-SAME: func = @gpu0_func
    // CHECK-SAME: num_cores_per_replica = 1
    // CHECK-SAME: padding_map = ["\08\01\10\02\18\03"]
    // CHECK-SAME: step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP"
    // CHECK-NOT: metadata

    return %1 : tensor<?xi32>
  }

  func @gpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.launch_func` on TPU with nested function calls.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_nested_func
  func @with_nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @nested_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @nested_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.launch_func` on TPU with referenced function that's not
// via a standard call op.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_referenced_func
  func @with_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.launch_func` on TPU with a chain of referenced
// functions.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_referenced_func_chain
  func @with_referenced_func_chain(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: @referenced_func1
    // CHECK-SAME: tf.D
    // CHECK-SAME: @referenced_func2
    // CHECK-SAME: tf.E
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @referenced_func2(%0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.E"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.launch_func` on TPU with multiple calls to same
// function.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_multiple_call_same_referenced_func
  func @with_multiple_call_same_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-COUNT-2: call @referenced_func
    // CHECK-COUNT-1: func @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    %1 = call @referenced_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    %2 = call @referenced_func(%1) : (tensor<?xi32>) -> tensor<?xi32>
    return %2 : tensor<?xi32>
  }

  func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.launch_func` on TPU with different computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @multiple_launch_different_func
  func @multiple_launch_different_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func0, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func0
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)

    %2 = "tf_device.launch_func"(%1) {_tpu_replicate = "cluster1", device = "", func = @tpu0_func1, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func1
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @tpu0_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.launch_func` on TPU with same computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @multiple_launch_same_func
  func @multiple_launch_same_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)

    %2 = "tf_device.launch_func"(%1) {_tpu_replicate = "cluster1", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

// -----

// Tests Functions referenced by TPU function via SymbolRefAttr nested in
// ArrayAttr and DictionaryAttr.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @single_tpu_launch_func
  func @single_tpu_launch_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", padding_map = ["\08\01\10\02\18\03"]} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: NumDynamicShapes = 1
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func @referenced_func3
    // CHECK-SAME: tf.I
    // CHECK-SAME: func @referenced_func2
    // CHECK-SAME: tf.H
    // CHECK-SAME: func @referenced_func1
    // CHECK-SAME: tf.G
    // CHECK-SAME: func @referenced_func0
    // CHECK-SAME: tf.F
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.D"(%0) {array_attr_funcs = [@referenced_func0, @referenced_func1]} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.E"(%1) {dictionary_attr_funcs = {fn1 = @referenced_func2, fn2 = @referenced_func3}} : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func @referenced_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.F"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.G"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.H"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }

  func @referenced_func3(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.I"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}

// -----

// Tests that TPUCompilationResult operations are properly rewritten

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @tpu_compilation_result
  func @tpu_compilation_result(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<!tf.string>, tensor<!tf.string>) {

    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf._TPUCompileMlir"
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"
    %1 = "tf_device.launch_func"(%arg0) {_tpu_replicate = "cluster0", device = "", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "", padding_map = []} : (tensor<?xi32>) -> tensor<?xi32>

    %compile_result = "tf.TPUCompilationResult"() {_tpu_replicate = "cluster0"} : () -> tensor<!tf.string>
    %compile_result2 = "tf.TPUCompilationResult"() {_tpu_replicate = "cluster0"} : () -> tensor<!tf.string>

    // CHECK-NOT: "tf.TPUCompilationResult"

    // CHECK: return %[[EXECUTE_OUTPUT]], %[[COMPILE_OUTPUT]]#0, %[[COMPILE_OUTPUT]]#0
    return %1, %compile_result, %compile_result2 : tensor<?xi32>, tensor<!tf.string>, tensor<!tf.string>
  }

  func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}
