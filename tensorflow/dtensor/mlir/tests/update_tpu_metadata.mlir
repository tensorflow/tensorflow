// RUN: dtensor-opt %s -split-input-file -dtensor-update-tpu-metadata -verify-diagnostics | FileCheck %s

// Check that TPUCompileMetadata proto is updated with correct number of replicas.
// CHECK-LABEL: func @main
func.func @main() {
  "tf.StatefulPartitionedCall"() {config = ":|x=2,y=2|*TPU", config_proto = "", executor_type = "", f = @f_callee} : () -> ()
  func.return
}

func.func @f_callee() {
  // CHECK:    tf_device.launch
  // CHECK:      "tf._TPUCompileMlir"
  // CHECK-SAME:  metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\04 \01"
  // CHECK:    device = ""
  %0:2 = "tf_device.launch"() ({
    %1, %2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\01 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1, %2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  })  {device = "tpu_host:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  // CHECK-NEXT: "tf.TPUExecute"
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
  // CHECK:      "tf._TPUCompileMlir"
  // CHECK:    device = ""
  %0:2 = "tf_device.launch"() ({
    %1, %2 = "tf._TPUCompileMlir"() {
      NumDynamicShapes = 0 : i64,
      metadata = "\0A\09\08\01\12\05\12\03\08\80\01\18\01 \01",
      mlir_module = "..."} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)
    tf_device.return %1, %2 : tensor<!tf_type.string>, tensor<2x!tf_type.string>
  })  {device = "tpu_host:0"} : () -> (tensor<!tf_type.string>, tensor<2x!tf_type.string>)

  // CHECK:    tf_device.launch
  // CHECK:      "tf.TPUExecute"
  // CHECK:    device = ""
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

