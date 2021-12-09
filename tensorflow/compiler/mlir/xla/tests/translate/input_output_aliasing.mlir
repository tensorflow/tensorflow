// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text -emit-return-tuple %s | FileCheck %s
// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE-ARG
// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text  %s | FileCheck %s --check-prefix=NO-RETURN-TUPLE

// CHECK-LABEL: ENTRY %main
// CHECK: // OutputIndex {0} aliases with input 0 at {}
// TUPLE-ARG-LABEL: ENTRY %main
// TUPLE-ARG: // OutputIndex {0} aliases with input 0 at {0}
// NO-RETURN-TUPLE-LABEL: ENTRY %main
// NO-RETURN-TUPLE: // OutputIndex {} aliases with input 0 at {}
func @main(%arg0: tensor<1xf32> {tf.aliasing_output = 0 : i64}) -> (tensor<1xf32>) {
  %0 = mhlo.constant dense<4.200000e+01> : tensor<1xf32>
  %1 = mhlo.add %arg0, %0 : tensor<1xf32>
  return %1 : tensor<1xf32>
}
