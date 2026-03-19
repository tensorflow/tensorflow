// RUN: tf-opt %s -allow-unregistered-dialect --tf-move-tpu-compile-to-front --split-input-file | FileCheck %s

module {

// CHECK-LABEL: does_basic_reordering
func.func @does_basic_reordering() -> () {
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: X
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: Y
   // CHECK: OpA
   // CHECK: OpB
   // CHECK: OpC
   "tf.OpA"() : () -> ()
   %status_x, %program_x = "tf._TPUCompileMlir"() { metadata = "X", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpB"() : () -> ()
   %status_y, %program_y = "tf._TPUCompileMlir"() { metadata = "Y", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpC"() : () -> ()
}

// CHECK-LABEL: does_reordering_for_nested_compiles
func.func @does_reordering_for_nested_compiles() -> () {
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: Z
   // CHECK: tf_device.launch
   // CHECK-NEXT: _TPUCompileMlir
   // CHECK-SAME: X
   // CHECK: tf_device.launch
   // CHECK-NEXT: _TPUCompileMlir
   // CHECK-SAME: Y
   // CHECK: OpA
   // CHECK: OpB
   // CHECK: OpC
   "tf.OpA"() : () -> ()
   "tf_device.launch"() ({
     %status_x, %program_x = "tf._TPUCompileMlir"() { metadata = "X", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
     tf_device.return
   }) {device = ""} : () -> ()
   "tf.OpB"() : () -> ()
   "tf_device.launch"() ({
     %status_y, %program_y = "tf._TPUCompileMlir"() { metadata = "Y", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
     tf_device.return
   }) {device = ""} : () -> ()
   %status_z, %program_z = "tf._TPUCompileMlir"() { metadata = "Z", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpC"() : () -> ()
}
}
