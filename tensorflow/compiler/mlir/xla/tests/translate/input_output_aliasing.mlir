// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text -emit-return-tuple %s | FileCheck %s
// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE-ARG

// CHECK-LABEL: ENTRY %main
// CHECK: // OutputIndex {0} aliases with input 0 at {}
// TUPLE-ARG-LABEL: ENTRY %main
// TUPLE-ARG: // OutputIndex {0} aliases with input 0 at {0}
func @main(%arg0: tensor<1xf32> {tf.aliasing_output = 0 : i64}) -> (tensor<1xf32>) {
  %0 = xla_hlo.constant dense<4.200000e+01> : tensor<1xf32>
  %1 = xla_hlo.add %arg0, %0 : tensor<1xf32>
  return %1 : tensor<1xf32>
}
