// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=10,17:17,19 -tf-xla-emit-use-tuple-args -tf-xla-emit-return-tuple | FileCheck %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=10,17:17,19 | FileCheck -check-prefix=NO_TUPLES %s
// RUN: tf-mlir-translate -mlir-tf-to-hlo-text-via-builder %s -tf-input-shapes=10,17:17,19 | FileCheck -check-prefix=NO_TUPLES %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main(%arg0: tensor<*xf32>, %arg1: tensor<?x19xf32>) -> tensor<?x19xf32> {
    %0 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<?x19xf32>) -> tensor<?x19xf32>
    func.return %0 : tensor<?x19xf32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       (arg_tuple.{{[0-9]+}}: (f32[10,17], f32[17,19])) -> (f32[10,19])

// NO_TUPLES-LABEL: HloModule main
// NO_TUPLES:       ({{.+}}: f32[10,17], {{.+}}: f32[17,19]) -> f32[10,19]
