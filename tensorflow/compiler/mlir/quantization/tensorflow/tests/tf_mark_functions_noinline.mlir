// RUN: tf-quant-opt %s -tf-mark-functions-noinline='noinline-functions=noinline0' \
// RUN:     -allow-unregistered-dialect -mlir-disable-threading \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// Tests that the function is marked tf._noinline = true.

// CHECK-LABEL: @noinline0
// CHECK-SAME: attributes {{{.*tf._noinline = true.*}}}
func.func @noinline0() -> (tensor<0xf32>) {
  %cst = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
  return %cst : tensor<0xf32>
}

// -----

// Tests that the function not listed in the option `noinline-functions`
// is not marked tf._noinline = true.

// CHECK-LABEL: @inline
// CHECK-NOT: tf._noinline
func.func @inline() -> (tensor<0xf32>) {
  %cst = "tf.Const"() {value = dense<1.0> : tensor<0xf32>} : () -> tensor<0xf32>
  return %cst : tensor<0xf32>
}
