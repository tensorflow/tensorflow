// RUN: tf-opt %s -split-input-file -verify-diagnostics --tf-tpu-rewrite=tpu-compile-metadata-debug | FILECHECK_OPTS="" FileCheck %s

// Tests module with missing `tf.versions` attribute.

// expected-error@+1 {{requires attribute 'tf.versions'}}
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_tf_versions() {
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests collecting compilation and execution devices results in an error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_devices() {
    // expected-error@+1 {{error in fetching TPU compilation/execution devices: no TPU_SYSTEM devices found}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `num_cores_per_replicas`
// attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'num_cores_per_replica'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with bad `num_cores_per_replicas` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'num_cores_per_replica'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = "", step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_num_cores_per_replica() {
    // expected-error@+1 {{requires attribute 'step_marker_location'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with bad `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_step_marker_location() {
    // expected-error@+1 {{requires attribute 'step_marker_location'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = 1, topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with unparsable `step_marker_location` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @unparsable_step_marker_location() {
    // expected-error@+1 {{bad 'step_marker_location' attribute with value 'test'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "test", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `topology` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_topology() {
    // expected-error@+1 {{requires attribute 'topology'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with bad `topology` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_topology() {
    // expected-error@+1 {{requires attribute 'topology'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = 1 : i32, device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with `topology` attribute resulting in device assignment error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @invalid_topology() {
    // expected-error@+1 {{error in fetching TPU compilation/execution devices}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "test", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `device_assignment` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_device_assignment() {
    // expected-error@+1 {{requires attribute 'device_assignment'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with bad `device_assignment` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_device_assignment() {
    // expected-error@+1 {{requires attribute 'device_assignment'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = "", input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with bad element in `device_assignment` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_element_device_assignment() {
    // expected-error@+1 {{bad 'device_assignment' attribute at index 0, not an int}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [""], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// The following topology is used in subsequent test cases:
// Proto debug string:
//   mesh_shape: 1
//   mesh_shape: 1
//   mesh_shape: 1
//   mesh_shape: 2
//   num_tasks: 1
//   num_tpu_devices_per_task: 2
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 1
// Serialized string:
//   "\0A\04\01\01\01\02\10\01\18\02\22\06\00\00\00\00\00\00\00\01"

// -----

// Tests `tf_device.cluster_func` with `device_assignment` attribute resulting in device assignment error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @invalid_device_assignment() {
    // expected-error@+1 {{error in fetching TPU compilation/execution devices}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\03\01\01\02\10\01\18\02\22\06\00\00\00\00\00\01", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `input_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_input_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{requires attribute 'input_sharding_configuration'}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", device_assignment = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// The following op sharding is used in subsequent test cases:
// Proto debug string:
//   type: MAXIMAL
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0
// Serialized string:
//   "\08\01\1A\01\01\22\01\00"

// -----

// Tests `tf_device.cluster_func` with bad `input_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_input_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{requires attribute 'input_sharding_configuration'}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = "", output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with mismatched `input_sharding_configuration` attribute size.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @mismatched_size_input_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'input_sharding_configuration' attribute, expected array attribute of size 1, got size 0}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with unsupported operand type.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @unsupported_operand_type(%arg0: tensor<?xi2>) {
    // expected-error@+1 {{failed to determine operand type at index 0: Converting i2 to DataType}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi2>) -> tensor<?xi2>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi2>) -> tensor<?xi2> {
    func.return %arg0 : tensor<?xi2>
  }
}

// -----

// Tests `tf_device.cluster_func` with bad element in `input_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_element_input_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'input_sharding_configuration' attribute at index 0, not a string}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = [1], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with unparsable element in `input_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @unparsable_element_input_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'input_sharding_configuration' attribute at index 0 with value 'test': failed to parse to xla::OpSharding}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["test"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with missing `output_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_output_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{requires attribute 'output_sharding_configuration'}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with bad `output_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_output_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{requires attribute 'output_sharding_configuration'}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = "", use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with mismatched `output_sharding_configuration` attribute size.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @mismatched_size_output_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'output_sharding_configuration' attribute, expected array attribute of size 1, got size 0}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----


// Tests `tf_device.cluster_func` with bad element in `output_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @bad_element_output_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'output_sharding_configuration' attribute at index 0, not a string}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = [1], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with unparsable element in `output_sharding_configuration` attribute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @unparsable_element_output_sharding_configuration(%arg0: tensor<?xi32>) {
    // expected-error@+1 {{bad 'output_sharding_configuration' attribute at index 0 with value 'test': failed to parse to xla::OpSharding}}
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["test"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    func.return
  }
  func.func @empty_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    func.return %arg0 : tensor<?xi32>
  }
}

// -----

// Tests `tf_device.cluster_func` with empty `step_marker_location` attribute
// defaults to `STEP_MARK_AT_ENTRY`.
//
// The expected TPUCompileMetadataProto is:
//   num_replicas: 1
//   num_cores_per_replica: 1

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @default_step_marker_location
  func.func @default_step_marker_location() {
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    // CHECK:      metadata
    // CHECK-SAME: num_replicas: 1
    // CHECK-SAME: num_cores_per_replica: 1
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests argument with unranked shape. Empty shape should be populated in the
// metadata for associated argument.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @unranked_shape_arg
  func.func @unranked_shape_arg(%arg0: tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<*xi32>) -> tensor<*xi32>
    // CHECK:      metadata
    // CHECK-SAME: shape {\0A unknown_rank: true

    func.return %0: tensor<*xi32>
  }
  func.func @_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
    func.return %arg0 : tensor<*xi32>
  }
}

// -----

// Tests argument with partial shape.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @partial_shape_arg
  func.func @partial_shape_arg(%arg0: tensor<?x?x3xi32>) -> tensor<?x?x3xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?x?x3xi32>) -> tensor<?x?x3xi32>
    // CHECK:      metadata
    // CHECK-SAME: args
    // CHECK-SAME: shape {\0A dim {\0A size: -1\0A }\0A dim {\0A size: -1\0A }\0A dim {\0A size: 3\0A }\0A }
    func.return %0: tensor<?x?x3xi32>
  }
  func.func @_func(%arg0: tensor<?x?x3xi32>) -> tensor<?x?x3xi32> {
    func.return %arg0 : tensor<?x?x3xi32>
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
  func.func @static_shape_arg(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
    // CHECK:      metadata
    // CHECK-SAME: args
    // CHECK-SAME: shape
    // CHECK-SAME: dim
    // CHECK-SAME: size: 1
    // CHECK-SAME: dim
    // CHECK-SAME: size: 2
    // CHECK-SAME: dim
    // CHECK-SAME: size: 3

    func.return %0: tensor<1x2x3xi32>
  }
  func.func @_func(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
    func.return %arg0 : tensor<1x2x3xi32>
  }
}

// -----

// Tests argument that is a resource variable.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @resource_arg
  func.func @resource_arg(%arg0: tensor<*x!tf_type.resource>) -> tensor<*x!tf_type.resource> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<*x!tf_type.resource>) -> tensor<*x!tf_type.resource>
    // CHECK:      metadata
    // CHECK:      dtype: DT_RESOURCE
    // CHECK-SAME: kind: VARIABLE

    func.return %0: tensor<*x!tf_type.resource>
  }
  func.func @_func(%arg0: tensor<*x!tf_type.resource>) -> tensor<*x!tf_type.resource> {
    func.return %arg0 : tensor<*x!tf_type.resource>
  }
}

// -----

// Tests argument that is a parameter.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @parameter_arg
  func.func @parameter_arg(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<*xf32>) -> tensor<*xf32>
    // CHECK:      metadata
    // CHECK:      dtype: DT_FLOAT
    // CHECK-SAME: kind: PARAMETER

    func.return %0: tensor<*xf32>
  }
  func.func @_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    func.return %arg0 : tensor<*xf32>
  }
}

// -----

// Tests metadata is populated correctly based on cluster_func op and attributes.
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
//   step_marker_location: STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @metadata
  func.func @metadata(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> tensor<8xi32>
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
    // CHECK-SAME: step_marker_location: STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP

    func.return %0: tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}

// -----

// Tests metadata is populated correctly for use_spmd_for_xla_partitioning ==
// true.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @metadata_use_spmd
  func.func @metadata_use_spmd(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = true} : (tensor<8xi32>) -> tensor<8xi32>
    // CHECK:      metadata
    // CHECK-SAME: use_spmd_for_xla_partitioning: true
    func.return %0: tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}

// -----

// Tests metadata is populated correctly for is_same_data_across_replicas.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @metadata_same_data_across_replicas
  func.func @metadata_same_data_across_replicas(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> tensor<8xi32>
    // CHECK:      metadata
    // CHECK-SAME: is_same_data_across_replicas: true
    // CHECK-SAME: mhlo.is_same_data_across_replicas = true
    func.return %0: tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32> {mhlo.is_same_data_across_replicas = true}) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}

// -----

// Tests shape ops are only generated for operands with non static shapes.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @static_and_dynamic_shapes
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_1:[a-z0-9]*]]: tensor<8xi32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<8xi32>)
  func.func @static_and_dynamic_shapes(%arg0: tensor<*xi32>, %arg1: tensor<8xi32>, %arg2: tensor<*xi32>, %arg3: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK-NOT:  "tf.Shape"(%[[ARG_1]])
    // CHECK-NOT:  "tf.Shape"(%[[ARG_3]])
    // CHECK:      %[[ARG_0_SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_0]])
    // CHECK:      %[[ARG_2_SHAPE:[0-9]*]] = "tf.Shape"(%[[ARG_2]])
    %0 = "tf_device.cluster_func"(%arg0, %arg1, %arg2, %arg3) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<*xi32>, tensor<8xi32>, tensor<*xi32>, tensor<8xi32>) -> tensor<8xi32>
    // CHECK:      "tf._TPUCompileMlir"(%[[ARG_0_SHAPE]], %[[ARG_2_SHAPE]])

    func.return %0: tensor<8xi32>
  }
  func.func @_func(%arg0: tensor<*xi32>, %arg1: tensor<8xi32>, %arg2: tensor<*xi32>, %arg3: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg1 : tensor<8xi32>
  }
}

// -----

// Tests simple case of `tf_device.cluster_func` on TPU with single input and
// single output.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @single_tpu_cluster_func
  func.func @single_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-SAME: device = "/job:worker/replica:0/task:0/device:TPU:0"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests simple case of `tf_device.cluster_func` on TPU with replication. Under
// data parallelism replicated host devices are also added to the
// tf_device.replicate

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  // CHECK-LABEL: func @replicated_tpu_cluster_func
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<?xi32>)
  func.func @replicated_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    // CHECK-SAME: ([%[[A_OUTPUT]], %[[ARG_0]]] as %[[RI_0:[a-z0-9]*]]: tensor<?xi32>)
    // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"], TPU_REPLICATED_HOST_0 = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:CPU:0"]}
    // CHECK-SAME: n = 2
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[RI_0]])
      // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
      // CHECK-SAME: metadata
      // CHECK-SAME: mlir_module
      // CHECK-SAME: func @main
      // CHECK-SAME: tf.B
      // CHECK-NOT: func = @tpu0_func
      // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
      // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT: "tf.TPUExecute"(%[[RI_0]], %[[COMPILE_OUTPUT]]#1)
      %2 = "tf_device.cluster_func"(%ri_0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>

      // CHECK: tf_device.return %[[EXECUTE_OUTPUT]]
      tf_device.return %2 : tensor<?xi32>
    }

    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[REPLICATE]]#1)
    %2 = "tf.C"(%1#1) : (tensor<?xi32>) -> tensor<?xi32>

    // CHECK: return %[[C_OUTPUT]]
    func.return %2 : tensor<?xi32>
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests that cluster_func without _xla_compile_device_type = "TPU", _replication_info attribute is ignored.

module attributes {tf.versions = {producer = 888 : i32}} {
  // CHECK-LABEL: func @single_gpu_cluster_func
  func.func @single_gpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>

    %1 = "tf_device.cluster_func"(%0) {device = "gpu0", func = @gpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: tf_device.cluster_func
    // CHECK-SAME: func = @gpu0_func
    // CHECK-SAME: device = "gpu0"
    // CHECK-SAME: num_cores_per_replica = 1
    // CHECK-SAME: step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP"
    // CHECK-NOT: metadata

    func.return %1 : tensor<?xi32>
  }

  func.func @gpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.cluster_func` on TPU with nested function calls.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_nested_func
  func.func @with_nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func private @nested_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:TPU:0"}>
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = func.call @nested_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }

  func.func @nested_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests of `tf_device.cluster_func` on TPU with referenced function that's not
// via a standard call op.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_referenced_func
  func.func @with_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func private @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func} : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }

  func.func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.cluster_func` on TPU with a chain of referenced
// functions.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_referenced_func_chain
  func.func @with_referenced_func_chain(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: @referenced_func1
    // CHECK-SAME: tf.D
    // CHECK-SAME: @referenced_func2
    // CHECK-SAME: tf.E
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }

  func.func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = func.call @referenced_func2(%0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }

  func.func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.E"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests rewriting `tf_device.cluster_func` on TPU with multiple calls to same
// function.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @with_multiple_call_same_referenced_func
  func.func @with_multiple_call_same_referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-COUNT-2: call @referenced_func
    // CHECK-COUNT-1: func private @referenced_func
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) {body = @referenced_func1} : (tensor<?xi32>) -> tensor<?xi32>
    %1 = func.call @referenced_func(%0) : (tensor<?xi32>) -> tensor<?xi32>
    %2 = func.call @referenced_func(%1) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %2 : tensor<?xi32>
  }

  func.func @referenced_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.cluster_func` on TPU with different computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @multiple_cluster_different_func
  func.func @multiple_cluster_different_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func0, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func0
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE0_OUTPUT]]#0)
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)

    %2 = "tf_device.cluster_func"(%1) {_xla_compile_device_type = "TPU", _replication_info = "cluster1", func = @tpu0_func1, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.D
    // CHECK-NOT: func = @tpu0_func1
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE1_OUTPUT]]#0)
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    func.return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }

  func.func @tpu0_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.D"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests multiple `tf_device.cluster_func` on TPU with same computation.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @multiple_cluster_same_func
  func.func @multiple_cluster_same_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE0_OUTPUT]]#0)
    // CHECK: %[[EXECUTE0_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE0_OUTPUT]]#1)

    %2 = "tf_device.cluster_func"(%1) {_xla_compile_device_type = "TPU", _replication_info = "cluster1", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[EXECUTE0_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[EXECUTE0_OUTPUT]])
    // CHECK: %[[COMPILE1_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[EXECUTE0_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-NOT: func = @tpu0_func
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE1_OUTPUT]]#0)
    // CHECK: %[[EXECUTE1_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[EXECUTE0_OUTPUT]], %[[COMPILE1_OUTPUT]]#1)

    %3 = "tf.C"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE1_OUTPUT]])

    func.return %3 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests Functions referenced by TPU function via SymbolRefAttr nested in
// ArrayAttr and DictionaryAttr.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @single_tpu_cluster_func
  func.func @single_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"

    %1 = "tf_device.cluster_func"(%0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[A_SHAPE_OUTPUT:[0-9]*]] = "tf.Shape"(%[[A_OUTPUT]])
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"(%[[A_SHAPE_OUTPUT]])
    // CHECK-SAME: metadata
    // CHECK-SAME: mlir_module
    // CHECK-SAME: func @main
    // CHECK-SAME: tf.B
    // CHECK-SAME: func private @referenced_func3
    // CHECK-SAME: tf.I
    // CHECK-SAME: func private @referenced_func2
    // CHECK-SAME: tf.H
    // CHECK-SAME: func private @referenced_func1
    // CHECK-SAME: tf.G
    // CHECK-SAME: func private @referenced_func0
    // CHECK-SAME: tf.F
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"(%[[A_OUTPUT]], %[[COMPILE_OUTPUT]]#1)

    %2 = "tf.C"(%1) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[C_OUTPUT:[0-9]*]] = "tf.C"(%[[EXECUTE_OUTPUT]])

    func.return %2 : tensor<?xi32>
    // CHECK: return %[[C_OUTPUT]]
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.D"(%0) {array_attr_funcs = [@referenced_func0, @referenced_func1]} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.E"(%1) {dictionary_attr_funcs = {fn1 = @referenced_func2, fn2 = @referenced_func3}} : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }

  func.func @referenced_func0(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.F"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }

  func.func @referenced_func1(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.G"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }

  func.func @referenced_func2(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.H"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }

  func.func @referenced_func3(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %1 = "tf.I"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %1 : tensor<?xi32>
  }
}

// -----

// Test `tf_device.cluster_func` on TPU with pre-split replicate sharded
// input/output using `tf.TPUPartitionedInputV2` and `tf.TPUPartitionedOutputV2`.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    // CHECK: %[[READ_VAR_0:[0-9]*]] = "tf.ReadVariableOp"(%arg0)
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    // CHECK: %[[READ_VAR_1:[0-9]*]] = "tf.ReadVariableOp"(%arg1)
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    // CHECK-NOT: tf.TPUPartitionedInputV2
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {N = 2 : i64, partition_dims = []} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:3 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"()
    // CHECK: "tf_device.launch"
    // CHECK: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: [[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:TPU:0"}>
    // CHECK-NEXT: "tf.TPUExecute"(%[[READ_VAR_0]], %[[COMPILE_OUTPUT]]#1)
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:TPU:1"}>
    // CHECK-NEXT: "tf.TPUExecute"(%[[READ_VAR_1]], %[[COMPILE_OUTPUT]]#2)
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    // CHECK-NOT: tf.TPUPartitionedOutputV2
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    // CHECK: "tf.AssignVariableOp"(%arg0, %[[PARALLEL_EXECUTE_OUTPUT]]#0)
    // CHECK: "tf.AssignVariableOp"(%arg1, %[[PARALLEL_EXECUTE_OUTPUT]]#1)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}

// -----

// Test `tf_device.cluster_func` on TPU with pre-split tile sharded input/
// output using `tf.TPUPartitionedInputV2` and `tf.TPUPartitionedOutputV2`.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<3x2xf32>>>, %arg1: tensor<!tf_type.resource<tensor<3x2xf32>>>) {
    // CHECK: %[[READ_VAR_0:[0-9]*]] = "tf.ReadVariableOp"(%arg0)
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<3x2xf32>>>) -> tensor<3x2xf32>
    // CHECK: %[[READ_VAR_1:[0-9]*]] = "tf.ReadVariableOp"(%arg1)
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<3x2xf32>>>) -> tensor<3x2xf32>
    // CHECK-NOT: tf.TPUPartitionedInputV2
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {_XlaSharding = "\08\03\1A\02\01\02\22\02\00\01", partition_dims = [1, 2]} : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x4xf32>
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:3 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"()
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: [[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:TPU:0"}>
    // CHECK-NEXT: "tf.TPUExecute"(%[[READ_VAR_0]], %[[COMPILE_OUTPUT]]#1)
    // CHECK: "tf_device.launch"() <{device = "/job:worker/replica:0/task:0/device:TPU:1"}>
    // CHECK-NEXT: "tf.TPUExecute"(%[[READ_VAR_1]], %[[COMPILE_OUTPUT]]#2)
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01"], output_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01"], use_spmd_for_xla_partitioning = true} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    // CHECK-NOT: tf.TPUPartitionedOutputV2
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {_XlaSharding = "\08\03\1A\02\01\02\22\02\00\01", partition_dims = [1, 2]} : (tensor<3x4xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>)
    // CHECK: "tf.AssignVariableOp"(%arg0, %[[PARALLEL_EXECUTE_OUTPUT]]#0)
    // CHECK: "tf.AssignVariableOp"(%arg1, %[[PARALLEL_EXECUTE_OUTPUT]]#1)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<3x2xf32>>>, tensor<3x2xf32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<3x2xf32>>>, tensor<3x2xf32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
    func.return %arg0: tensor<3x4xf32>
  }
}

// -----

// Test that unsupported input sharding type of TPUPartitionedInputV2Op inputs of
// ClusterFuncOp result in error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {N = 2 : i64, partition_dims = []} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error@+1 {{unsupported input sharding type MAXIMAL for 0-th input}}
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1],
      input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}

// -----

// Test that unsupported output sharding type of TPUPartitionedOutputV2Op outputs
// of ClusterFuncOp result in error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {N = 2 : i64, partition_dims = []} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error@+1 {{unsupported output sharding type MAXIMAL for 0-th output}}
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1],
      input_sharding_configuration = [""], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}

// -----

// Test that multiple uses of ClusterFuncOp output along with
// TPUPartitionedOutputV2Op results in error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {N = 2 : i64, partition_dims = []} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    // expected-error@+1 {{'tf.TPUPartitionedOutputV2' op must be a unique user of TPU Cluster}}
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf._SomeOp"(%computation) : (tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}
// -----

// Test that ParallelExecute that is not a direct parent of ClusterFunc results
// in an error.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0 : tensor<i1>) {
    "tf_device.parallel_execute"() ({
      "tf.IfRegion"(%arg0) ({
        // expected-error@+1 {{The ParallelExecute ancestor of a ClusterFunc must be its direct parent.}}
        "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
        "tf.Yield"() : () -> ()
      }, {
        "tf.Yield"() : () -> ()
      }) {is_stateless = false, device = "/device:CPU:0"} : (tensor<i1>) -> ()
      tf_device.return
    }) : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Tests that TPUCompilationResult operations are properly rewritten.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  // CHECK-LABEL: func @tpu_compilation_result
  func.func @tpu_compilation_result(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<!tf_type.string>, tensor<!tf_type.string>) {

    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
    // CHECK-NEXT: "tf._TPUCompileMlir"
    // CHECK: %[[COMPILE_RESULT_0:.*]] = "tf.Identity"(%[[COMPILE_OUTPUT]]#0)
    // CHECK: %[[COMPILE_RESULT_1:.*]] = "tf.Identity"(%[[COMPILE_RESULT_0]])
    // CHECK: "tf_device.launch"
    // CHECK-NEXT: "tf.TPUCompileSucceededAssert"
    // CHECK: %[[EXECUTE_OUTPUT:[0-9]*]] = "tf_device.launch"
    // CHECK-NEXT: "tf.TPUExecute"
    %1 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>

    %compile_result = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster0"} : () -> tensor<!tf_type.string>
    %compile_result2 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster0"} : () -> tensor<!tf_type.string>

    // CHECK-NOT: "tf.TPUCompilationResult"

    // CHECK: return %[[EXECUTE_OUTPUT]], %[[COMPILE_OUTPUT]]#0, %[[COMPILE_OUTPUT]]#0
    func.return %1, %compile_result, %compile_result2 : tensor<?xi32>, tensor<!tf_type.string>, tensor<!tf_type.string>
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests simple case of `tf_device.cluster_func` on TPU with replication and
// parallel_execute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  // CHECK-LABEL: func @replicated_parallel_tpu_cluster_func
  func.func @replicated_parallel_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[REPLICATE:[0-9]*]]:2 = tf_device.replicate
    %1:2 = tf_device.replicate([%0, %arg0] as %ri_0: tensor<?xi32>) {n = 2 : i32} {
      // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK: "tf._TPUCompileMlir"
      // CHECK: "tf.TPUCompileSucceededAssert"
      // CHECK: "tf_device.parallel_execute"
      // CHECK-NOT:"tf._XlaCompileMlirPlaceholderProgramKey"
      // CHECK:    "tf.D"(%[[COMPILE_OUTPUT]]#1
      // CHECK:    "tf.TPUExecute"
      // CHECK-NOT:"tf._XlaCompileMlirPlaceholderProgramKey"
      // CHECK:    "tf.E"(%[[COMPILE_OUTPUT]]#1
      %3 = "tf_device.parallel_execute"() ({
         %program = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        "tf.D"(%program) : (tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }, {
        %4 = "tf_device.cluster_func"(%ri_0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "", device_assignment = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }, {
        %program = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        "tf.E"(%program) : (tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }) : () -> (tensor<?xi32>)
      tf_device.return %3 : tensor<?xi32>
    }
    %2 = "tf.C"(%1#1) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %2 : tensor<?xi32>
  }

  func.func @tpu0_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests case of `tf_device.cluster_func` on TPU with replication, model
// parallelism and parallel_execute.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2", "/job:worker/replica:0/task:0/device:TPU:3", "/job:worker/replica:0/task:0/device:TPU:4", "/job:worker/replica:0/task:0/device:TPU:5", "/job:worker/replica:0/task:0/device:TPU:6", "/job:worker/replica:0/task:0/device:TPU:7"]} {
  // CHECK-LABEL: func @replicated_model_parallel_execute_tpu_cluster_func
  func.func @replicated_model_parallel_execute_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
    %1:4 = tf_device.replicate([%0, %arg0, %arg0, %arg0] as %ri_0: tensor<?xi32>) {n = 4 : i32} {
      // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:3 = "tf_device.launch"
      // CHECK: "tf._TPUCompileMlir"
      // CHECK: "tf.TPUCompileSucceededAssert"
      // CHECK: "tf_device.parallel_execute"
      // CHECK-NOT:"tf._XlaCompileMlirPlaceholderProgramKey"
      // CHECK:    "tf.D"(%[[COMPILE_OUTPUT]]#1
      // CHECK:    "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK:    "tf.TPUExecute"
      // CHECK:     "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK:    "tf.TPUExecute"
      // CHECK-NOT:    "tf.TPUExecute"
      %3 = "tf_device.parallel_execute"() ({
         %program = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        "tf.D"(%program) : (tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }, {
        %4 = "tf_device.cluster_func"(%ri_0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01*\02\08\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }) : () -> (tensor<?xi32>)
      tf_device.return %3 : tensor<?xi32>
    }
    %2 = "tf.C"(%1#1) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %2 : tensor<?xi32>
  }

  func.func @tpu0_func(%arg0: tensor<?xi32> {mhlo.sharding = ""}) -> (tensor<?xi32> {mhlo.sharding = ""}) {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests case of `tf_device.cluster_func` on TPU with replication, model
// parallelism and parallel_execute with 2 parallel_execute children after the
// cluster.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2", "/job:worker/replica:0/task:0/device:TPU:3", "/job:worker/replica:0/task:0/device:TPU:4", "/job:worker/replica:0/task:0/device:TPU:5", "/job:worker/replica:0/task:0/device:TPU:6", "/job:worker/replica:0/task:0/device:TPU:7"]} {
  // CHECK-LABEL: func @replicated_model_parallel_execute_2proc_tpu_cluster_func
  func.func @replicated_model_parallel_execute_2proc_tpu_cluster_func(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    // CHECK: %[[A_OUTPUT:[0-9]*]] = "tf.A"
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    // CHECK: %[[REPLICATE:[0-9]*]]:4 = tf_device.replicate
    %1:4 = tf_device.replicate([%0, %arg0, %arg0, %arg0] as %ri_0: tensor<?xi32>) {n = 4 : i32} {
      // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:3 = "tf_device.launch"
      // CHECK: "tf._TPUCompileMlir"
      // CHECK: "tf.TPUCompileSucceededAssert"
      // CHECK: "tf_device.parallel_execute"
      // CHECK:    "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK:    "tf.TPUExecute"
      // CHECK:    "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK:    "tf.TPUExecute"
      // CHECK-NOT:    "tf.TPUExecute"
      // CHECK-NOT:"tf._XlaCompileMlirPlaceholderProgramKey"
      // CHECK:    "tf.D"(%[[COMPILE_OUTPUT]]#1
      // CHECK:    "tf.E"(%[[COMPILE_OUTPUT]]#1
      %3 = "tf_device.parallel_execute"() ({
        %4 = "tf_device.cluster_func"(%ri_0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01*\02\08\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<?xi32>) -> tensor<?xi32>
        tf_device.return %4 : tensor<?xi32>
      }, {
        %program = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        "tf.D"(%program) : (tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }, {
        %program = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        "tf.E"(%program) : (tensor<3x!tf_type.string>) -> ()
        tf_device.return
      }) : () -> (tensor<?xi32>)
      tf_device.return %3 : tensor<?xi32>
    }
    %2 = "tf.C"(%1#1) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %2 : tensor<?xi32>
  }

  func.func @tpu0_func(%arg0: tensor<?xi32> {mhlo.sharding = ""}) -> (tensor<?xi32> {mhlo.sharding = ""}) {
    %0 = "tf.B"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }
}

// -----

// Tests devices are set properly for non replicated model parallelism.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @non_replicated_parallel_execute
  func.func @non_replicated_parallel_execute(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT:   "tf._TPUCompileMlir"()
    // CHECK-NEXT:   tf_device.return
    // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
    // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
    // CHECK-NEXT:   tf_device.return
    // CHECK:      "tf_device.parallel_execute"
    // CHECK-NEXT:   "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
    // CHECK-NEXT:     "tf.TPUExecute"
    // CHECK-NEXT:     tf_device.return
    // CHECK:        "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:1"}>
    // CHECK-NEXT:     "tf.TPUExecute"
    // CHECK-NEXT:     tf_device.return
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> tensor<8xi32>
    func.return %0 : tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}

// -----

// The following topology is used in subsequent test cases:
// Proto debug string:
//   mesh_shape: 1
//   mesh_shape: 2
//   mesh_shape: 1
//   mesh_shape: 2
//   num_tasks: 2
//   num_tpu_devices_per_task: 2
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 1
//   device_coordinates: 0
//   device_coordinates: 1
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 0
//   device_coordinates: 1
//   device_coordinates: 0
//   device_coordinates: 1
// Serialized string:
//   "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01"
// -----

// Tests devices are set properly for replicated model parallelism.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @replicated_parallel_execute
  func.func @replicated_parallel_execute(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: devices = {TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"], TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"], TPU_REPLICATED_HOST_0 = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:1/device:CPU:0"], TPU_REPLICATED_HOST_1 = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:1/device:CPU:0"]}
    %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<8xi32>) {n = 2 : i32} {
      // CHECK-NEXT: %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"()
      // CHECK-NEXT:   tf_device.return
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK-NEXT:   tf_device.return
      // CHECK:      "tf_device.parallel_execute"
      // CHECK-NEXT:   "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return
      // CHECK:        "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return
      %1 = "tf_device.cluster_func"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> tensor<8xi32>
      tf_device.return %1 : tensor<8xi32>
    }
    func.return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}


// -----

// Tests that inputs are inputs with maximal and replicate sharding are set
// properly for replicated model parallelism.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_input_with_sharding_configurations
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<8xi32>, %[[ARG_1:[a-z0-9]*]]: tensor<8xi32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi1>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi1>, %[[ARG_4:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_5:[a-z0-9]*]]: tensor<*xi32>)
  func.func @parallel_execute_with_input_with_sharding_configurations(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>, %arg2: tensor<*xi1>, %arg3: tensor<*xi1>, %arg4: tensor<*xi32>, %arg5: tensor<*xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<8xi32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi1>
    // CHECK-SAME: [%[[ARG_4]], %[[ARG_5]]] as %[[RI_2:[a-z0-9]*]]: tensor<*xi32>
    %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<8xi32>, [%arg2, %arg3] as %ri2: tensor<*xi1>, [%arg4, %arg5] as %ri3: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"
      // CHECK:      "tf._TPUCompileMlir"
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[RI_0]], %[[RI_1]], %[[RI_2]], %[[COMPILE]]#1)
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_OUTPUT]]
      // CHECK:        "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     "tf.TPUExecute"(%[[RI_1]], %[[RI_2]], %[[COMPILE]]#2)
      %1 = "tf_device.cluster_func"(%ri, %ri2, %ri3) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "", ""], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>, tensor<*xi1>, tensor<*xi32>) -> tensor<8xi32>
      tf_device.return %1 : tensor<8xi32>
    }
    func.return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>, %arg1: tensor<*xi1>, %arg2: tensor<*xi32>) -> tensor<8xi32> {
    %1 = "tf.A"(%arg0, %arg1, %arg2) : (tensor<8xi32>, tensor<*xi1>, tensor<*xi32>) -> (tensor<8xi32>)
    func.return %1 : tensor<8xi32>
  }
}

// -----

// Tests devices are set properly for replicated model parallelism with outputs
// to TPU computation placed on logical device 0.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_different_outputs
  func.func @parallel_execute_with_different_outputs(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<8xi32>) {n = 2 : i32} {
      // CHECK-NEXT: %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"()
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]] = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_OUTPUT]]
      // CHECK:        "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     "tf.TPUExecute"
      %1 = "tf_device.cluster_func"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> tensor<8xi32>
      tf_device.return %1 : tensor<8xi32>
    }
    func.return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    func.return %arg0 : tensor<8xi32>
  }
}

// -----

// Tests devices are set properly for replicated model parallelism with
// TPU computation with maximal and replicated outputs.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_replicated_output
  func.func @parallel_execute_with_replicated_output(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri: tensor<8xi32>) {n = 2 : i32} {
      // CHECK-NEXT: %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"()
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<8xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<8xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<8xi32>) -> (tensor<*xi32>, tensor<*xi1>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %1, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests inputs are correctly split and fed into TPU computation for tiled input
// sharding.

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_tiled_input
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @parallel_execute_with_tiled_input(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      //
      // CHECK:      %[[CONST_SPLIT_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_DIM]], %[[RI_0]])
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      //
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_OUT]]#0, %[[COMPILE]]#1)
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_OUT]]#1, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests that outputs are correctly merged and fed from TPU computation for
// tiled output sharding.

// The following OpSharding is used for TPU computation outputs in below test:
// Proto debug string:
//  output 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"
//
// output 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_tiled_output
  func.func @parallel_execute_with_tiled_output(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      //
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      //
      // CHECK:     %[[CONST_CONCAT_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#0, %[[PARALLEL_EXECUTE_OUTPUT]]#2

      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests that outputs are correctly merged and fed from TPU computation for
// tiled output sharding with MAXIMAL sharding for one of the output and OTHER
// for latter output.

// The following OpSharding is used for TPU computation outputs in below test:
// Proto debug string:
//  output 0
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"
//
// output 1
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_tiled_output
  func.func @parallel_execute_with_tiled_output(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      //
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      //
      // CHECK:     %[[CONST_CONCAT_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#1, %[[PARALLEL_EXECUTE_OUTPUT]]#2

      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {
        _xla_compile_device_type = "TPU", _replication_info = "cluster0",
        func = @tpu0_func, num_cores_per_replica = 2,
        step_marker_location = "",
        topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01",
        device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\00"],
        output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\03\1A\02\01\02\22\02\00\01"],
        use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 4
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//  Serialized string:
//   "\08\03\12\12\10\0b\1a\02\01\04\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\01\04\22\04\00\01\02\03"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"
//
// -----


// Tests tile sharding of inputs with number of splits that does not evenly divide
// the input results in an error, when shapes are not fully known.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  func.func @uneven_input_sharding_disallowed(%arg0: tensor<?x10xf32>, %arg1: tensor<?x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<?x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
    // expected-error@+1 {{incorrect input sharding configuration received. 1-th dimension of the input must be evenly divisible by 4}}
    %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\01\04\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\01\04\22\04\00\01\02\03", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<?x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<?x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<?x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// The following OpSharding is used for TPU computation outputs in below test:
// Proto debug string:
//  output 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 4
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\01\04\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\01\04\22\04\00\01\02\03"
//
// output 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"
//
// -----

// Tests tile sharding of outputs with number of splits that exeed number
// of logical devices is not allowed.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  func.func @uneven_output_sharding_disallowed(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
    // expected-error@+1 {{incorrect sharding format for outputs. Number of tiled outputs(4) must match the number of logical devices(2)}}
    %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["", ""], output_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\01\04\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\01\04\22\04\00\01\02\03", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi1>, tensor<*xi32>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %3, %4 : tensor<*xi1>, tensor<*xi32>
  }
}

// -----

// The following topology is used in subsequent test cases:
// Proto debug string:
//  mesh_shape: 2
//  mesh_shape: 1
//  mesh_shape: 2
//  num_tasks: 2
//  num_tpu_devices_per_task: 2
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 1

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_shape {
//    element_type: F32
//    dimensions: 2
//    dimensions: 2
//    layout {
//     minor_to_major: 1
//     minor_to_major: 0
//     format: DENSE
//    }
//    is_dynamic_dimension: false
//    is_dynamic_dimension: false
//   }
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// Tests inputs to TPUComputation that are tiled in multiple dimensions.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_multi_dimension_tiled_input
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @parallel_execute_with_multi_dimension_tiled_input(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[CONST_SPLIT_0_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_0_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_0_DIM]], %[[RI_0]])
      // CHECK:      %[[CONST_SPLIT_1_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_1_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_1_DIM]], %[[SPLIT_0_OUT]]#0)
      // CHECK:      %[[CONST_SPLIT_2_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_2_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_2_DIM]], %[[SPLIT_0_OUT]]#1)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#0, %[[COMPILE]]#1)
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#1, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#0, %[[COMPILE]]#3)
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#1, %[[COMPILE]]#4)
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}


// -----

// The following topology is used in subsequent test cases:
// Proto debug string:
//  mesh_shape: 2
//  mesh_shape: 1
//  mesh_shape: 2
//  num_tasks: 2
//  num_tpu_devices_per_task: 2
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 0
//  device_coordinates: 1
//  device_coordinates: 0
//  device_coordinates: 1

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_shape {
//    element_type: F32
//    dimensions: 2
//    dimensions: 2
//    layout {
//     minor_to_major: 1
//     minor_to_major: 0
//     format: DENSE
//    }
//    is_dynamic_dimension: false
//    is_dynamic_dimension: false
//   }
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// Tests inputs to TPUComputation that are tiled in multiple dimensions.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_multi_dimension_tiled_input
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @parallel_execute_with_multi_dimension_tiled_input(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[CONST_SPLIT_0_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_0_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_0_DIM]], %[[RI_0]])
      // CHECK:      %[[CONST_SPLIT_1_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_1_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_1_DIM]], %[[SPLIT_0_OUT]]#0)
      // CHECK:      %[[CONST_SPLIT_2_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_2_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_2_DIM]], %[[SPLIT_0_OUT]]#1)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#0, %[[COMPILE]]#1)
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#1, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#0, %[[COMPILE]]#3)
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#1, %[[COMPILE]]#4)
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests inputs to TPUComputation that are tiled in multiple dimensions with
// replicate_on_last_tile_dim set.

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
//   replicate_on_last_tile_dim: true
// Serialized string:
//  "\08\03\1A\03\02\01\02\22\04\00\01\02\030\01"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @multi_dimension_tiled_input_replicate_last_dim
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @multi_dimension_tiled_input_replicate_last_dim(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[CONST_SPLIT_0_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_0_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_0_DIM]], %[[RI_0]])
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#0, %[[COMPILE]]#1)
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#0, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#1, %[[COMPILE]]#3)
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#1, %[[COMPILE]]#4)
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\03\1A\03\02\01\02\22\04\00\01\02\030\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests that tiled output with multiple dimension sharding works properly.

// The following OpSharding is used for TPU computation outputs in below test:
// output 0
//  Proto debug string:
//   type: OTHER
//   tile_shape {
//    element_type: F32
//    dimensions: 2
//    dimensions: 2
//    layout {
//     minor_to_major: 1
//     minor_to_major: 0
//     format: DENSE
//    }
//    is_dynamic_dimension: false
//    is_dynamic_dimension: false
//   }
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03"
//
// output 1
//  Proto debug string:
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @multiple_dimension_output_sharding
  func.func @multiple_dimension_output_sharding(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      // CHECK:     %[[CONST_CONCAT_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#0, %[[PARALLEL_EXECUTE_OUTPUT]]#2
      // CHECK:     %[[CONST_CONCAT2_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT2_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT2_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#3, %[[PARALLEL_EXECUTE_OUTPUT]]#4
      // CHECK:     %[[CONST_CONCAT3_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT3_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT3_DIM]], %[[CONCAT_OUTPUT]], %[[CONCAT2_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\00\01\02\03", "\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests that tiled output with multiple dimension sharding works properly with
// replicate_on_last_tile_dim set.

// The following OpSharding is used for TPU computation outputs in below test:
// output 0
//  Proto debug string:
//   type: OTHER
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
//   replicate_on_last_tile_dim: true
// Serialized string:
//  "\08\03\1A\03\02\01\02\22\04\00\01\02\030\01"
//
// output 1
//  Proto debug string:
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\00"

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @multi_dimension_tiled_output_replicate_last_dim
  func.func @multi_dimension_tiled_output_replicate_last_dim(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      // CHECK:     %[[CONST_CONCAT_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#0, %[[PARALLEL_EXECUTE_OUTPUT]]#3
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\03\1A\03\02\01\02\22\04\00\01\02\030\01", "\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests inputs device assignment order is well preserved for tiled input sharding.

// The following OpSharding is used for TPU computation inputs in below test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_shape {
//    element_type: F32
//    dimensions: 2
//    dimensions: 2
//    layout {
//     minor_to_major: 1
//     minor_to_major: 0
//     format: DENSE
//    }
//    is_dynamic_dimension: false
//    is_dynamic_dimension: false
//   }
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 3
//   tile_assignment_devices: 2
//   tile_assignment_devices: 1
//   tile_assignment_devices: 0
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\03\02\01\00"
//
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"
//
// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @tiled_input_sharding_with_device_assignment_order
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @tiled_input_sharding_with_device_assignment_order(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[CONST_SPLIT_0_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_0_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_0_DIM]], %[[RI_0]])
      // CHECK:      %[[CONST_SPLIT_1_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_1_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_1_DIM]], %[[SPLIT_0_OUT]]#0)
      // CHECK:      %[[CONST_SPLIT_2_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_2_OUT:[a-z0-9]+]]:2 = "tf.Split"(%[[CONST_SPLIT_2_DIM]], %[[SPLIT_0_OUT]]#1)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#1, %[[COMPILE]]#1)
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_2_OUT]]#0, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#1, %[[COMPILE]]#3)
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_1_OUT]]#0, %[[COMPILE]]#4)
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\03\02\01\00", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests device assignment is well preserved for tile sharded outputs.

// The following OpSharding is used for TPU computation outputs in below test:
// output 0
//  Proto debug string:
//   type: OTHER
//   tile_shape {
//    element_type: F32
//    dimensions: 2
//    dimensions: 2
//    layout {
//     minor_to_major: 1
//     minor_to_major: 0
//     format: DENSE
//    }
//    is_dynamic_dimension: false
//    is_dynamic_dimension: false
//   }
//   tile_assignment_dimensions: 2
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 3
//   tile_assignment_devices: 2
//   tile_assignment_devices: 1
//   tile_assignment_devices: 0
// Serialized string:
//  "\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\03\02\01\00"
//
// output 1
//  Proto debug string:
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @device_order_preserved_for_tiled_output
  func.func @device_order_preserved_for_tiled_output(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      // CHECK:     %[[CONST_CONCAT_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#4, %[[PARALLEL_EXECUTE_OUTPUT]]#3
      // CHECK:     %[[CONST_CONCAT2_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT2_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT2_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#2, %[[PARALLEL_EXECUTE_OUTPUT]]#0
      // CHECK:     %[[CONST_CONCAT3_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT3_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT3_DIM]], %[[CONCAT_OUTPUT]], %[[CONCAT2_OUTPUT]]
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\02\02\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\02\02\22\04\03\02\01\00", "\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}


// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @missing_compilation_and_replication_attributes() {
    // expected-error@+1 {{'tf_device.cluster_func' op is expected to have either both or none of '_replication_info' and '_xla_compile_device_type' attributes}}
    "tf_device.cluster_func"() {_replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @empty_replication_attribute() {
    // expected-error@+1 {{'tf_device.cluster_func' op has an empty '_replication_info' attribute}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @empty_compilation_attribute() {
    // expected-error@+1 {{'tf_device.cluster_func' op has invalid '_xla_compile_device_type' value 'XPU'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "XPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0"]} {
  func.func @invalid_compilation_attribute() {
    // expected-error@+1 {{'tf_device.cluster_func' op has invalid '_xla_compile_device_type' value 'XPU'}}
    "tf_device.cluster_func"() {_xla_compile_device_type = "XPU", _replication_info = "cluster0", func = @empty_func, num_cores_per_replica = 1, step_marker_location = "", topology = "", device_assignment = [], input_sharding_configuration = [], output_sharding_configuration = [], use_spmd_for_xla_partitioning = false} : () -> ()
    func.return
  }
  func.func @empty_func() {
    func.return
  }
}

// -----

// Test `tf.TPUPartitionedInputV2` has outputs not in `tf_device.cluster_func`

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    // expected-error@+1 {{Output of TPUPartitionedInputV2 must be in tpu computation.}}
    %partitioned_input = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {N = 2 : i64, partition_dims = []} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<!tf_type.resource<tensor<i32>>>) -> tensor<!tf_type.resource<tensor<i32>>>

    %read = "tf.ReadVariableOp"(%partitioned_input) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>

    %computation = "tf_device.cluster_func"(%read) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%computation) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}

// -----

// Test `tf.TPUPartitionedOutputV2` has inputs not in `tf_device.cluster_func`

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<!tf_type.resource<tensor<i32>>>) {
    %read0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
    %partitioned_input = "tf.TPUPartitionedInputV2"(%read0, %read1) {N = 2 : i64, partition_dims = []} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %computation = "tf_device.cluster_func"(%partitioned_input) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @computation, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = [""], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = true} : (tensor<i32>) -> tensor<i32>
    %add_result = "tf.Add"(%computation, %computation) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // expected-error@+1 {{Input of TPUPartitionedOutputV2 must be in tpu computation.}}
    %partitioned_output:2 = "tf.TPUPartitionedOutputV2"(%add_result) {N = 2 : i64, partition_dims = []} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
    "tf.AssignVariableOp"(%arg0, %partitioned_output#0) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    "tf.AssignVariableOp"(%arg1, %partitioned_output#1) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
    func.return
  }
  func.func @computation(%arg0: tensor<i32>) -> tensor<i32> {
    func.return %arg0: tensor<i32>
  }
}

// -----

// The following xla.OpSharding is used:
// Proto debug string:
//   type : TUPLE
//   tuple_shardings: {
//     type: OTHER
//     tile_assignment_dimensions: [ 2, 1 ]
//     tile_assignment_devices   : [ 0, 1 ]
//   }
//   tuple_shardings: {
//     type: OTHER
//     tile_assignment_dimensions: [ 2, 1 ]
//     tile_assignment_devices   : [ 0, 1 ]
//   }
// Serialized string:
//   "\08\02*\0A\08\03\1A\02\02\01\22\02\00\01*\0A\08\03\1A\02\02\01\22\02\00\01"

// Test that an attempt to map an invalid cluster output index to core program
// index is caught. The output has sharding type TUPLE, which causes the
// cluster output index to be invalid for core 0.

module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster_to_core_index_check() -> tensor<128xf32> {
    // expected-error@+1 {{Attempted to map cluster_func output index 0 to program assigned to core 0}}
    %0 = "tf_device.cluster_func"() {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\01\01\02\10\01\18\02\22\08\00\00\00\00\00\00\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1], input_sharding_configuration = [], output_sharding_configuration = ["\08\02*\0A\08\03\1A\02\02\01\22\02\00\01*\0A\08\03\1A\02\02\01\22\02\00\01"], use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> tensor<128xf32>
    func.return %0 : tensor<128xf32>
  }
  func.func @_func() -> tensor<128xf32> {
    %0 = "tf.Const"() {value = dense<0.0> : tensor<128xf32>} : () -> tensor<128xf32>
    func.return %0 : tensor<128xf32>
  }
}

// -----
// CHECK-LABEL: func @return_from_host_and_tpu
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1199 : i32}} {
  func.func @return_from_host_and_tpu() -> (tensor<?xi32>, tensor<?x!tf_type.string>) attributes {tf._construction_context = "kEagerRuntime", tf.signature.is_stateful} {
      // CHECK:     %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:2 = "tf_device.parallel_execute"
      // CHECK:       %[[LAUNCH_0_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK:         %[[B_OUTPUT:[0-9]*]] = "tf.B"
      // CHECK:         tf_device.return %[[B_OUTPUT:[0-9]*]]
      // CHECK:       %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:TPU:0"}>
      // CHECK-NEXT:    %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK:         tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:    return %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#1, %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]#0
    %0:2 = "tf_device.parallel_execute"() ({
      %1 = "tf_device.launch"() ({
        %2 = "tf._XlaCompileMlirPlaceholderProgramKey"() : () -> tensor<3x!tf_type.string>
        %3 = "tf._XlaRecvAtHost"(%2) {_xla_has_host_transfer = true, device_ordinal = 0 : i64, key = "host_compute_channel_0_args"} : (tensor<3x!tf_type.string>) -> tensor<?xi32>
        %4 = "tf.B"(%3) : (tensor<?xi32>) -> tensor<?x!tf_type.string>
        tf_device.return %4 : tensor<?x!tf_type.string>
      }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<?x!tf_type.string>
      tf_device.return %1 : tensor<?x!tf_type.string>
    }, {
      %1 = "tf_device.cluster_func"() {step_marker_location = "STEP_MARK_AT_ENTRY", _replication_info = "cluster", _xla_compile_device_type = "TPU", allow_soft_placement = true, computation_shape = [], device = "", device_assignment = [], func = @_func, input_sharding_configuration = [], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], topology = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : () -> tensor<?xi32>
      tf_device.return %1 : tensor<?xi32>
    }) : () -> (tensor<?x!tf_type.string>, tensor<?xi32>)
    return %0#1, %0#0 : tensor<?xi32>, tensor<?x!tf_type.string>
  }
  func.func private @_func() -> (tensor<?xi32> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) {
    %0 = "tf.A"() : () -> tensor<?xi32>
    "tf._XlaHostComputeMlir"(%0) {host_mlir_module = "", recv_key = "host_compute_channel_0_retvals", send_key = "host_compute_channel_0_args"} : (tensor<?xi32>) -> ()
    %1 = "tf.C"() : () -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}

// -----

// The following xla.OpSharding is used:
// Proto debug string:
//   type : OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 1
//   tile_assignment_devices: 0
//   last_tile_dims: REPLICATED
// Serialized string:
//   "\08\03\1A\06\01\01\01\01\01\01\22\01\00B\01\00"

// Test that an input sharding with last_tile_dims REPLICATED won't generate SplitOp.
//CHECK-NOT: tf.Split
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1"]} {
  func.func @cluster_to_single_core(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster1", func = @_func, num_replica = 1, num_cores_per_replica = 1, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", device_assignment = [], input_sharding_configuration = ["\08\03\1A\06\01\01\01\01\01\01\22\01\00B\01\00"], output_sharding_configuration = [""], use_spmd_for_xla_partitioning = false, use_tpu = true} : (tensor<128xf32>) -> tensor<128xf32>
    func.return %0 : tensor<128xf32>
  }
  func.func @_func(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    func.return %arg0 : tensor<128xf32>
  }
}

// -----

// CHECK-LABEL: func @annotate_dynamic_shape_tensor
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:COMPOSITE:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1437 : i32}} {
  func.func @annotate_dynamic_shape_tensor(%arg0: tensor<512xi64> {tf._user_specified_name = "190", tf.device = "/job:localhost/replica:0/task:0/device:CPU:0"}) -> (tensor<512xi32>) {
    %0 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster_test_fn", device = ""} : () -> tensor<!tf_type.string>
    %cst = "tf.Const"() {value = dense<512> : tensor<i32>} : () -> tensor<i32>
    %2:4 = "tf_device.launch"() ({
      %4 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<512xi64>) -> tensor<512xi32>
      %5 = "tf.TPUCopyWithDynamicShape"(%4,  %cst) {operandSegmentSizes = array<i32: 1, 1>} : (tensor<512xi32>, tensor<i32>) -> tensor<512xi32>
      tf_device.return %5 : tensor<512xi32>
    }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<512xi32>, tensor<1024xi32>, tensor<1024xi32>, tensor<1024xf32>)
    // CHECK: %[[COMPILE_OUTPUT:[0-9]*]]:4 = "tf_device.launch"
    // CHECK: "tf._TPUCompileMlir"()
    // CHECK: is_bounded_dynamic_dim: true
    %3 = "tf_device.cluster_func"(%2#0) {_dynamic_arg_index = [0 : i32], _has_manual_control_dependencies = true, _replication_info = "cluster_test_fn", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [], func = @_func, host_compute_core = [], input_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], num_cores_per_replica = 1 : i64, output_sharding_configuration = ["\08\01\1A\01\01\22\01\00"], padding_map = [], step_marker_location = "STEP_MARK_AT_ENTRY", topology = "", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = false, use_tpu = true} : (tensor<512xi32>) ->  tensor<512xi32>
    return %3: tensor<512xi32>
  }
func.func private @_func(%arg0: tensor<?xi32, #mhlo.type_extensions<bounds = [512]>> {mhlo.sharding = "\08\01\1A\01\01\22\01\00"}) -> (tensor<512xi32>) {
    %0 = "tf.A"(%arg0) {} : (tensor<?xi32, #mhlo.type_extensions<bounds = [512]>>) -> tensor<512xi32>
    return %0 : tensor<512xi32>
  }
}

// -----

// The following xla.OpSharding is used:
// Proto debug string:
//   type : OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 4
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
//   tile_assignment_devices: 2
//   tile_assignment_devices: 3
//   last_tile_dims: REPLICATED
// Serialized string:
//   "\08\03\1A\03\01\01\04\22\04\00\01\02\03B\01\00"

// Test that SplitOp is not generated when an input sharding has 
// last_tile_dims REPLICATED and more tile_assignment_dimensions 
// than tensor dimenstions, even when the SPMD sharding is enabled and
// num_cores_per_replica is more than 1.
// Test that ConcatV2 Op is not generated when a output sharding has 
// last_tile_dims REPLICATED and more tile_assignment_dimensions 
// than tensor dimenstions, even when the SPMD sharding is enabled and
// num_cores_per_replica is more than 1.
//CHECK-NOT: tf.Split
// CHECK-NOT: tf.ConcatV2
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:worker/replica:0/task:0/device:TPU:0", "/job:worker/replica:0/task:0/device:TPU:1", "/job:worker/replica:0/task:0/device:TPU:2", "/job:worker/replica:0/task:0/device:TPU:3"]} {
  func.func @cluster_to_single_core(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "cluster1", func = @_func, num_replica = 1, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_ENTRY", topology = "\0A\04\02\02\01\01\10\01\18\04\22\10\00\00\00\00\01\00\00\00\00\01\00\00\01\01\00\00", device_assignment = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\03\01\01\04\22\04\00\01\02\03B\01\00"], output_sharding_configuration = ["\08\03\1A\03\01\01\04\22\04\00\01\02\03B\01\00"], use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<4x128xf32>) -> tensor<4x128xf32>
    func.return %0 : tensor<4x128xf32>
  }
  func.func @_func(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    func.return %arg0 : tensor<4x128xf32>
  }
}

// -----

// Tests _ici_weight_distribution_mlir_bridge_marker attribute can be
// produced in created Split op
module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @propagate_ici_weight_attr_to_split_op
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x10xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<*xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<*xi32>)
  func.func @propagate_ici_weight_attr_to_split_op(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<*xi32>
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:5 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      // CHECK:      %[[CONST_SPLIT_0_DIM:.*]] = "tf.Const"()
      // CHECK:      %[[SPLIT_0_OUT:[a-z0-9]+]]:2 = "tf.Split"
      // CHECK:      _ici_weight_distribution_mlir_bridge_marker = true
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:5 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#0, %[[COMPILE]]#1)
      // CHECK:          tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#0, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK:          tf_device.return %[[EXECUTE_1_OUTPUT]]
      // CHECK:        %[[LAUNCH_2_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_2_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#1, %[[COMPILE]]#3)
      // CHECK:          tf_device.return %[[EXECUTE_2_OUTPUT]]
      // CHECK:        %[[LAUNCH_3_OUTPUT:[0-9]*]] = "tf_device.launch"
      // CHECK-NEXT:     %[[EXECUTE_3_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_0_OUT]]#1, %[[COMPILE]]#4)
      // CHECK:          tf_device.return %[[EXECUTE_3_OUTPUT]]
      %1 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
        %identity = "tf.Identity"(%ri_1) {_ici_weight_distribution_mlir_bridge_marker = true} : (tensor<128x10xf32>) -> tensor<128x10xf32>
        tf_device.return %identity : tensor<128x10xf32>
      }) {_ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<128x10xf32>
      %2, %3 = "tf_device.cluster_func"(%1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 4, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\01\02\10\01\18\08\22 \00\00\00\00\00\00\00\01\01\00\00\00\01\00\00\01\00\01\00\00\00\01\00\01\01\01\00\00\01\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], input_sharding_configuration = ["\08\03\1A\03\02\01\02\22\04\00\01\02\030\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %2, %3 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
  }
}

// -----

// Tests that outputs are correctly merged and fed from TPU computation for
// tiled output sharding with padding for concat ops.

// The following OpSharding is used for TPU computation outputs in below test:
// Proto debug string:
//  output 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"
//
// output 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 0
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"


module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_tiled_output
  func.func @parallel_execute_with_tiled_output(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<128x10xi32>, %arg3: tensor<128x10xi32>) -> (tensor<128x5xi32>, tensor<10x5xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x10xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<128x10xi32>
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<128x10xi32>) {n = 2 : i32} {
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      //
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      //
      // CHECK:     %[[CONST_CONCAT3_DIM:.*]] = "tf.Const"()
      // CHECK:     %[[CONCAT3_OUTPUT:[0-9]*]] = "tf.Concat"(%[[CONST_CONCAT3_DIM]], %[[PARALLEL_EXECUTE_OUTPUT]]#0, %[[PARALLEL_EXECUTE_OUTPUT]]#2)
      // CHECK:     %[[CONST_SLICE_BEGIN:.*]] = "tf.Const"() 
      //                             dense<0>
      //                            tensor<2xi64>}> : () -> tensor<2xi64> 
      // CHECK:     %[[CONST_SLICE_SIZE:.*]] =  "tf.Const"() 
      //                           dense<[128, 5]> : tensor<2xi64>}> : () -> tensor<2xi64> 
      // CHECK:     "tf.Slice"(%[[CONCAT3_OUTPUT]], %[[CONST_SLICE_BEGIN]], %[[CONST_SLICE_SIZE]])
      //                                  : (tensor<128x6xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<128x5xi32>
      %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\00"], output_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\00"], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<128x10xi32>) -> (tensor<128x5xi32>, tensor<10x5xi1>)
      tf_device.return %1, %2 : tensor<128x5xi32>, tensor<10x5xi1>
    }
    func.return %0#0, %1#0 : tensor<128x5xi32>, tensor<10x5xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<128x10xi32>, tensor<128x10xi32>) -> (tensor<128x10xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<10x5xi1>) -> tensor<10x5xi1>
    func.return %4, %3 : tensor<128x10xi32>, tensor<10x5xi1>
  }
}

// -----

// Tests inputs are correctly split and fed into TPU computation for tiled input
// sharding with padding.

// The following OpSharding is used for TPU computation inputs in the below
// test:
// Proto debug string:
//  input 0
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//  "\08\03\1A\02\01\02\22\02\00\01"
//
// input 1
//  type: MAXIMAL
//  tile_assignment_dimensions: 1
//  tile_assignment_devices: 1
// Serialized string:
//  "\08\01\1A\01\01\22\01\01"


module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  // CHECK-LABEL: func @parallel_execute_with_tiled_input
  // CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<128x9xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<128x9xf32>, %[[ARG_2:[a-z0-9]*]]: tensor<128x10xi32>, %[[ARG_3:[a-z0-9]*]]: tensor<128x10xi32>)
  func.func @parallel_execute_with_tiled_input(%arg0: tensor<128x9xf32>, %arg1: tensor<128x9xf32>, %arg2: tensor<128x10xi32>, %arg3: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    // CHECK: tf_device.replicate
    // CHECK-SAME: [%[[ARG_0]], %[[ARG_1]]] as %[[RI_0:[a-z0-9]*]]: tensor<128x9xf32>
    // CHECK-SAME: [%[[ARG_2]], %[[ARG_3]]] as %[[RI_1:[a-z0-9]*]]: tensor<128x10xi32>
    // CHECK-SAME: devices =
    // CHECK-SAME: TPU_REPLICATED_CORE_0 = ["/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1"]
    // CHECK-SAME: TPU_REPLICATED_CORE_1 = ["/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU:0"]
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x9xf32>, [%arg2, %arg3] as %ri_2: tensor<128x10xi32>) {n = 2 : i32} {
      // CHECK:      %[[DEVICE_LAUNCH_OUT:[a-z0-9]+]] = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}>
      // CHECK:      %[[COMPILE:[a-z0-9]+]]:3 = "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf._TPUCompileMlir"
      // CHECK:      "tf_device.launch"() <{device = "/job:localhost/replica:0/task:0/device:CPU:0"}>
      // CHECK-NEXT:   "tf.TPUCompileSucceededAssert"(%[[COMPILE]]#0)
      //
      // CHECK:   %[[PAD_SHAPE:[a-z0-9]+]] =  "tf.Const"() 
      // CHECK:                                    [0, 0], [0, 1]
      // CHECK:                                   : tensor<2x2xi64>}> : () -> tensor<2x2xi64> 
      // CHECK:   %[[PAD_OUT:[a-z0-9]+]] = "tf.Pad"(%[[DEVICE_LAUNCH_OUT]], %[[PAD_SHAPE]]) {_ici_weight_distribution_mlir_bridge_marker = true} : (tensor<128x9xf32>, tensor<2x2xi64>) -> tensor<128x10xf32> 
      // CHECK:   %[[CONST_SPLIT_DIM:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> {_ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<i32> 
      // CHECK:   %[[SPLIT_OUT:[a-z0-9]+]]:2 =   "tf.Split"(%[[CONST_SPLIT_DIM]], %[[PAD_OUT]]) {_ici_weight_distribution_mlir_bridge_marker = true, num_split = 2 : i32} : (tensor<i32>, tensor<128x10xf32>) -> (tensor<128x5xf32>, tensor<128x5xf32>)
      // CHECK:      %[[PARALLEL_EXECUTE_OUTPUT:[0-9]*]]:3 = "tf_device.parallel_execute"
      // CHECK-NEXT:   %[[LAUNCH_0_OUTPUT:[0-9]*]]:2 = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_0"}>
      //
      // CHECK-NEXT:     %[[EXECUTE_0_OUTPUT:[0-9]*]]:2 = "tf.TPUExecute"(%[[SPLIT_OUT]]#0, %[[COMPILE]]#1)
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_0_OUTPUT]]
      // CHECK:        %[[LAUNCH_1_OUTPUT:[0-9]*]] = "tf_device.launch"() <{device = "TPU_REPLICATED_CORE_1"}>
      // CHECK-NEXT:     %[[EXECUTE_1_OUTPUT:[0-9]*]] = "tf.TPUExecute"(%[[SPLIT_OUT]]#1, %[[RI_1]], %[[COMPILE]]#2)
      // CHECK-NEXT:     tf_device.return %[[EXECUTE_1_OUTPUT]]
      %1 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
        %identity = "tf.Identity"(%ri_1) {_ici_weight_distribution_mlir_bridge_marker = true} : (tensor<128x9xf32>) -> tensor<128x9xf32>
        tf_device.return %identity : tensor<128x9xf32>
      }) {_ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<128x9xf32>
      %2, %3 = "tf_device.cluster_func"(%1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x9xf32>, tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
      tf_device.return %2, %3 : tensor<128x10xi32>, tensor<10x5xi1>
    }
    func.return %0#0, %1#0 : tensor<128x10xi32>, tensor<10x5xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x9xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x9xf32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<128x10xi32>, tensor<128x10xi32>) -> (tensor<128x10xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<10x5xi1>) -> tensor<10x5xi1>
    func.return %4, %3 : tensor<128x10xi32>, tensor<10x5xi1>
  }
}

// -----

// CHECK: "tf.Split"
//               : (tensor<128x1024xf32>) -> (tensor<32x1024xf32>, tensor<32x1024xf32>, tensor<32x1024xf32>, tensor<32x1024xf32>)
module attributes {tf.devices = {"/job:tpu_host_worker/replica:0/task:0/device:CPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:0", "/job:tpu_host_worker/replica:0/task:0/device:TPU:1", "/job:tpu_host_worker/replica:0/task:0/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:1/device:CPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:0", "/job:tpu_host_worker/replica:0/task:1/device:TPU:1", "/job:tpu_host_worker/replica:0/task:1/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:2/device:CPU:0", "/job:tpu_host_worker/replica:0/task:2/device:TPU:0", "/job:tpu_host_worker/replica:0/task:2/device:TPU:1", "/job:tpu_host_worker/replica:0/task:2/device:TPU_SYSTEM:0", "/job:tpu_host_worker/replica:0/task:3/device:CPU:0", "/job:tpu_host_worker/replica:0/task:3/device:TPU:0", "/job:tpu_host_worker/replica:0/task:3/device:TPU:1", "/job:tpu_host_worker/replica:0/task:3/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1857 : i32}} {
  func.func @main(%arg0: tensor<i32> {tf._user_specified_name = "steps", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg1: tensor<*x!tf_type.resource<tensor<i64>>> {tf._user_specified_name = "899", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg2: tensor<*x!tf_type.resource<tensor<i64>>> {tf._user_specified_name = "901", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg3: tensor<*x!tf_type.resource<tensor<128x1024xf32>>> {tf._user_specified_name = "903", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg4: tensor<*x!tf_type.resource<tensor<1024xf32>>> {tf._user_specified_name = "905", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg5: tensor<*x!tf_type.resource<tensor<1024x1xf32>>> {tf._user_specified_name = "907", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg6: tensor<*x!tf_type.resource<tensor<i64>>> {tf._user_specified_name = "909", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg7: tensor<*x!tf_type.resource<tensor<25001x64xf32>>> {tf._user_specified_name = "911", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg8: tensor<*x!tf_type.resource<tensor<25001x64xf32>>> {tf._user_specified_name = "913", tf.device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0"}, %arg9: tensor<*x!tf_type.resource<tensor<25001x64xf32>>> {tf._user_specified_name = "915", tf.device = "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"}, %arg10: tensor<*x!tf_type.resource<tensor<25001x64xf32>>> {tf._user_specified_name = "917", tf.device = "/job:tpu_host_worker/replica:0/task:3/device:CPU:0"}, %arg11: tensor<*x!tf_type.resource<tensor<25001x32xf32>>> {tf._user_specified_name = "919", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg12: tensor<*x!tf_type.resource<tensor<25001x32xf32>>> {tf._user_specified_name = "921", tf.device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0"}, %arg13: tensor<*x!tf_type.resource<tensor<25001x32xf32>>> {tf._user_specified_name = "923", tf.device = "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"}, %arg14: tensor<*x!tf_type.resource<tensor<25001x32xf32>>> {tf._user_specified_name = "925", tf.device = "/job:tpu_host_worker/replica:0/task:3/device:CPU:0"}, %arg15: tensor<*x!tf_type.resource<tensor<6x32xf32>>> {tf._user_specified_name = "927", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg16: tensor<*x!tf_type.resource<tensor<6x32xf32>>> {tf._user_specified_name = "929", tf.device = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0"}, %arg17: tensor<*x!tf_type.resource<tensor<6x32xf32>>> {tf._user_specified_name = "931", tf.device = "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"}, %arg18: tensor<*x!tf_type.resource<tensor<6x32xf32>>> {tf._user_specified_name = "933", tf.device = "/job:tpu_host_worker/replica:0/task:3/device:CPU:0"}, %arg19: tensor<*x!tf_type.resource<tensor<128x1024xf32>>> {tf._user_specified_name = "935", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg20: tensor<*x!tf_type.resource<tensor<1024xf32>>> {tf._user_specified_name = "937", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}, %arg21: tensor<*x!tf_type.resource<tensor<1024x1xf32>>> {tf._user_specified_name = "939", tf.device = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"}) -> tensor<i64> attributes {allow_soft_placement = false, tf.entry_function = {control_outputs = "", inputs = "steps,unknown,unknown_0,unknown_1,unknown_2,unknown_3,unknown_4,unknown_5,unknown_6,unknown_7,unknown_8,unknown_9,unknown_10,unknown_11,unknown_12,unknown_13,unknown_14,unknown_15,unknown_16,unknown_17,unknown_18,unknown_19", outputs = "statefulpartitionedcall_RetVal"}} {
    %0 = "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster__train_helper", device = ""} : () -> tensor<!tf_type.string>
    %1 = "tf.ReadVariableOp"(%arg3) : (tensor<*x!tf_type.resource<tensor<128x1024xf32>>>) -> tensor<128x1024xf32>
    %2 = "tf.ReadVariableOp"(%arg4) : (tensor<*x!tf_type.resource<tensor<1024xf32>>>) -> tensor<1024xf32>
    %3:2 = tf_device.replicate {n = 2 : i32} {
      %6 = "tf_device.cluster_func"(%1, %2) <{func = @_func}> {_dynamic_arg_index = [], _has_manual_control_dependencies = true, _replication_info = "cluster__train_helper", _xla_compile_device_type = "TPU", allow_soft_placement = false, computation_shape = [], device = "", device_assignment = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0], host_compute_core = [], input_sharding_configuration = ["\08\03\1A\01\04\22\04\00\01\02\03", ""], num_cores_per_replica = 4 : i64, output_sharding_configuration = [""], padding_map = [], step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\02\02\02\01\10\04\18\02\22 \00\00\00\00\00\01\00\00\01\00\00\00\01\01\00\00\00\00\01\00\00\01\01\00\01\00\01\00\01\01\01\00*\02\08\01", tpu_compile_options_proto = "", use_spmd_for_xla_partitioning = true, use_tpu = true} : (tensor<128x1024xf32>, tensor<1024xf32>) -> tensor<*xf32>
      tf_device.return %6 : tensor<*xf32>
    }
    %4 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf_type.resource<tensor<i64>>>) -> tensor<i64>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<i64>) -> tensor<i64>
    return %5 : tensor<i64>
  }
  func.func private @_func(%arg0: tensor<128x1024xf32> {mhlo.is_same_data_across_replicas = true, mhlo.sharding = "\08\03\1A\01\04\22\04\00\01\02\03"}, %arg1: tensor<1024xf32> {mhlo.is_same_data_across_replicas = true, mhlo.sharding = ""}) -> (tensor<*xf32> {mhlo.sharding = ""}) {
    %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "\08\03\1A\01\04\22\04\00\01\02\03", sharding = "\08\03\1A\01\04\22\04\00\01\02\03"}> {unspecified_dims = []} : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
    %1 = "tf.MatMul"(%0, %arg1) : (tensor<128x1024xf32>, tensor<1024xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }
}
