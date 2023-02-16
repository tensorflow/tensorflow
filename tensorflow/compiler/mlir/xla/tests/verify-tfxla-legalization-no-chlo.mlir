// RUN: tf-opt "-tfxla-verify-legalization=legalize-chlo=false" -verify-diagnostics -split-input-file %s | FileCheck %s --dump-input=fail
// Tests the VerifyTFXLALegalization Pass, that just ensures we don't have
// any illegal ops at the end of the pipeline. This runs with
// legalize-chlo=false since errors can't be mixed with the legalize-chlo=True
// version.

// CHECK-LABEL: allows_chlo
func.func @allows_chlo(%arg0: tensor<1x32x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x32x10x32xi32> {
  %0 = "chlo.broadcast_add"(%arg0, %arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<1x32x10x32xi32>, tensor<32xi32>) -> tensor<1x32x10x32xi32>
  func.return %0 : tensor<1x32x10x32xi32>
}
