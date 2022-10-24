// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Check that attributes that define types are exported.

// CHECK: key: "Tinputs"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_FLOAT

// CHECK: key: "Toutputs"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_FLOAT

// CHECK: "extra_type_attr"
// CHECK-NEXT:    value
// CHECK-NEXT:      list
// CHECK-NEXT:        type: DT_INT32
// CHECK-NEXT:        type: DT_FLOAT

// CHECK-LABEL: function
// CHECK: name: "plain"
// CHECK: Placeholder
// CHECK: key: "type"
// CHECK: type: DT_INT8

func.func @main(%arg0 : tensor<16xf32>) {
  tf_executor.graph {
    %1:2 = tf_executor.island wraps "tf.MlirPassthroughOp"(%arg0) {extra_type_attr = [tensor<5xi32>, tensor<16xf32>], Tinputs = [tensor<16xf32>], Toutputs = [tensor<16xf32>], mlir_module = ""} : (tensor<16xf32>) -> tensor<16xf32>
    tf_executor.fetch
  }
  func.return
}

func.func @plain() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder"() {type = i8} : () -> tensor<16xi8>
    tf_executor.fetch
  }
  func.return
}
