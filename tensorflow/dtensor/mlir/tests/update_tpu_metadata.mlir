// RUN: dtensor-opt %s -split-input-file -dtensor-update-tpu-metadata -verify-diagnostics | FileCheck %s

// Check that TPUCompileMetadata proto is updated with correct number of replicas.
// CHECK-LABEL: func @main
func.func @main() {
  "tf.StatefulPartitionedCall"() {config = ":|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @f_callee} : () -> ()
  func.return
}

func.func @f_callee() {
  // CHECK:    tf_device.launch
  // CHECK:    device = ""
  // CHECK:      "tf._TPUCompileMlir"
  // CHECK-SAME:  metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\04 \01"
  %0:2 = "tf_device.launch"() ({
    %1, %2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\01 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1, %2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  })  {device = "tpu_host:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  // CHECK: "tf.TPUExecute"
  "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
  func.return
}

// -----

// Check that device placement of _TPUCompileMlir/TPUExeute operation is removed/updated properly.
// CHECK-LABEL: func @main
func.func @main() {
  "tf.StatefulPartitionedCall"() {config = "|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @f_callee} : () -> ()
  func.return
}

func.func @f_callee() {
  // CHECK:    tf_device.launch
  // CHECK:    device = ""
  // CHECK:      "tf._TPUCompileMlir"
  %0:2 = "tf_device.launch"() ({
    %1, %2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\01 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1, %2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  })  {device = "tpu_host:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  // CHECK:    tf_device.launch
  // CHECK:    device = ""
  // CHECK:      "tf.TPUExecute"
  "tf_device.launch"() ({
    "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> ()
  func.return
}

// -----

// Check that unparable TPUCompilaMetadataProto is disallowed.
func.func @main() {
  "tf.StatefulPartitionedCall"() {config = "|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @f_callee} : () -> ()
  func.return
}

func.func @f_callee() {
  %0:2 = "tf_device.launch"() ({
    // expected-error @+1 {{unable to parse TPUCompileMetadata}}
    %1, %2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      metadata = "\0A\0B\0C",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1, %2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  })  {device = "tpu_host:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  "tf_device.launch"() ({
    "tf.TPUExecute"(%0#1) : (tensor<2x!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> ()
  func.return
}

// -----

// Check for Xla Spmd mesh that TPUCompileOp has correct metadata proto and
// number of program outputs is equal to number of devices on mesh.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<12x24xf32>) -> (tensor<12x24xf32>) {
    %0 = "tf.StatefulPartitionedCall"(%arg1) {
      config = "|x=2,y=4|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/replica:0/task:0/device:TPU:0,/job:localhost/replica:0/task:0/device:TPU:1,/job:localhost/replica:0/task:0/device:TPU:2,/job:localhost/replica:0/task:0/device:TPU:3,/job:localhost/replica:0/task:0/device:TPU:4,/job:localhost/replica:0/task:0/device:TPU:5,/job:localhost/replica:0/task:0/device:TPU:6,/job:localhost/replica:0/task:0/device:TPU:7|use_xla_spmd",
      config_proto = "",
      executor_type = "",
      f = @_xla_spmd_func} : (tensor<12x24xf32>) -> tensor<12x24xf32>
    return %0 : tensor<12x24xf32>
  }

func.func private @_xla_spmd_func(%arg0: tensor<12x24xf32>) -> tensor<12x24xf32> {
  // CHECK:    tf_device.launch
  // CHECK:    device = ""
  // CHECK:      %compilation_status, %program:8 = "tf._TPUCompileMlir"
  // CHECK-SAME:  metadata = "\0A\10\08\01\12\08\12\02\08\0C\12\02\08\18\18\01\22\00\12\02\0A\00\18\01 \08x\01\88\01\ED\91\DC\F5\C3\8C\95\B5\90\01"
  %0:2 = "tf_device.launch"() ({
    %compilation_status, %program = "tf._TPUCompileMlir"() {metadata = "\0A\18\08\01\12\08\12\02\08\0C\12\02\08\18\18\01\22\08\08\01\1A\01\01\22\01\00\12\0A\0A\08\08\01\1A\01\01\22\01\00\18\01 \01\88\01\ED\91\DC\F5\C3\8C\95\B5\90\01", mlir_module = "#loc = loc(unknown)\0Amodule attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1345 : i32}} {\0A  func.func @main(%arg0: tensor<12x24xf32> {mhlo.sharding = \22\22} loc(unknown)) -> (tensor<12x24xf32> {mhlo.sharding = \22\22}) {\0A    %0 = \22tf.Identity\22(%arg0) : (tensor<12x24xf32>) -> tensor<12x24xf32> loc(#loc)\0A    return %0 : tensor<12x24xf32> loc(#loc)\0A  } loc(#loc)\0A} loc(#loc)\0A"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
    tf_device.return %compilation_status, %program : tensor<!tf_type.string>, tensor<3x!tf_type.string>
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf_device.launch"() ({
    "tf.TPUCompileSucceededAssert"(%0#0) : (tensor<!tf_type.string>) -> ()
    tf_device.return
  }) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> ()
  %1 = "tf_device.launch"() ({
    // CHECK: "tf.TPUExecute"
    %2 = "tf.TPUExecute"(%arg0, %0#1) : (tensor<12x24xf32>, tensor<3x!tf_type.string>) -> tensor<12x24xf32>
    tf_device.return %2 : tensor<12x24xf32>
  }) {device = "/job:localhost/replica:0/task:0/device:TPU:0"} : () -> tensor<12x24xf32>
  return %1 : tensor<12x24xf32>
}

func.func private @_func(%arg0: tensor<12x24xf32> {mhlo.sharding = ""}) -> (tensor<12x24xf32> {mhlo.sharding = ""}) {
  %0 = "tf.Identity"(%arg0) : (tensor<12x24xf32>) -> tensor<12x24xf32>
  return %0 : tensor<12x24xf32>
}

