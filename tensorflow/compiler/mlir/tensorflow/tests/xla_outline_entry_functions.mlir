// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-xla-outline-entry-functions | FileCheck %s

// Check that we outline the top-level functions.

// CHECK-LABEL: func.func private @main_outlined(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @f}> {_xla_compile_device_type = "CPU"} : (tensor<i32>) -> tensor<i32>
// CHECK:         %cst = "tf.Const"() <{value = dense<5> : tensor<i32>}> : () -> tensor<i32>
// CHECK:         %1 = "tf.Add"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:         return %1 : tensor<i32>
// CHECK:       }

// CHECK:       func.func @main(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", tf.entry_function = {}} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @main_outlined}> {_xla_compile_device_type = "CPU", allow_soft_placement = true} : (tensor<i32>) -> tensor<i32>
// CHECK:         return %0 : tensor<i32>
// CHECK:       }
func.func @main(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true, tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", _xla_compile_device_type = "CPU", f = @f} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @f(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// Tests multiple entry functions.

// CHECK-LABEL: func.func private @entry1_outlined(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @f1}> {_xla_compile_device_type = "CPU"} : (tensor<i32>) -> tensor<i32>
// CHECK:         %cst = "tf.Const"() <{value = dense<5> : tensor<i32>}> : () -> tensor<i32>
// CHECK:         %1 = "tf.Add"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:         return %1 : tensor<i32>
// CHECK:       }

// CHECK-LABEL: func.func private @entry2_outlined(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @f1}> {_xla_compile_device_type = "CPU"} : (tensor<i32>) -> tensor<i32>
// CHECK:         %cst = "tf.Const"() <{value = dense<5> : tensor<i32>}> : () -> tensor<i32>
// CHECK:         %1 = "tf.Add"(%0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:         return %1 : tensor<i32>
// CHECK:       }

// CHECK:       func.func @entry1(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", tf.entry_function = {}} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @entry1_outlined}> {_xla_compile_device_type = "CPU", allow_soft_placement = true} : (tensor<i32>) -> tensor<i32>
// CHECK:         return %0 : tensor<i32>
// CHECK:       }

// CHECK:       func.func @entry2(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", tf.entry_function = {}} {
// CHECK:         %0 = "tf.StatefulPartitionedCall"(%arg0) <{config = "", config_proto = "", executor_type = "", f = @entry2_outlined}> {_xla_compile_device_type = "CPU", allow_soft_placement = true} : (tensor<i32>) -> tensor<i32>
// CHECK:         return %0 : tensor<i32>
// CHECK:       }
func.func @entry1(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true, tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", _xla_compile_device_type = "CPU", f = @f1} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @entry2(%arg0: tensor<i32>) -> tensor<i32> attributes {_xla_compile_device_type = "CPU", allow_soft_placement = true, tf.entry_function = {}} {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", _xla_compile_device_type = "CPU", f = @f1} : (tensor<i32>) -> (tensor<i32>)
  %1 = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Add"(%0, %1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)
  func.return %2 : tensor<i32>
}

func.func @f1(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
