// RUN: not tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

// CHECK: Opaque elements attr not supported
func @main() {
  %0 = "tf.Const"() {value = opaque<"tf", "0x0123456789ABCDEF"> : tensor<4xf32>} : () -> tensor<4xf32>
  return
}
