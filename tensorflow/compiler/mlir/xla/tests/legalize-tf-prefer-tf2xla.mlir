// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false use-tf2xla-fallback=true prefer-tf2xla=true" %s | FileCheck %s
// RUN: tf-opt "-xla-legalize-tf=allow-partial-conversion device-type=XLA_CPU_JIT legalize-chlo=false prefer-tf2xla=true" %s | FileCheck --check-prefix NOFALLBACK %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: @abs
func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK:  "mhlo.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: bessel_i0e
func @bessel_i0e(%arg0: tensor<3xf16>, %arg1: tensor<3xf32>, %arg2: tensor<3xf64>) -> (tensor<3xf16>, tensor<3xf32>, tensor<3xf64>) {
  // CHECK-NOT: tf.BesselI0e
  %0 = "tf.BesselI0e"(%arg0) : (tensor<3xf16>) -> (tensor<3xf16>)
  %1 = "tf.BesselI0e"(%arg1) : (tensor<3xf32>) -> (tensor<3xf32>)
  %2 = "tf.BesselI0e"(%arg2) : (tensor<3xf64>) -> (tensor<3xf64>)
  return %0, %1, %2 : tensor<3xf16>, tensor<3xf32>, tensor<3xf64>
}

// NOFALLBACK-LABEL: @xla_svd
func @xla_svd(%arg0: tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
  // NOFALLBACK: XlaSvd
  %s, %u, %v = "tf.XlaSvd"(%arg0) {max_iter = 1, epsilon = 1.0E-09 : f32, precision_config = ""} : (tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
  return %s, %u, %v : tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
}

}