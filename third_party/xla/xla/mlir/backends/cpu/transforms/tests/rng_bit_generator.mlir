// RUN: xla-cpu-opt %s -xla-legalize-library-ops | FileCheck %s

func.func @rng_bit_generator(%state: tensor<2xui64>) -> (tensor<2xui64>, tensor<10xui32>) {
  %new_state, %output = "mhlo.rng_bit_generator"(%state) {
    rng_algorithm = #mhlo.rng_algorithm<DEFAULT>
  } : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10xui32>)
  func.return %new_state, %output : tensor<2xui64>, tensor<10xui32>
}

// CHECK-LABEL: @rng_bit_generator
//  CHECK-SAME: %[[ARG0:.*]]: tensor
//       CHECK: %[[STATE_INIT:.*]] = tensor.empty() : tensor<2xui64>
//       CHECK: %[[DST_INIT:.*]] = tensor.empty() : tensor<10xui32>
//       CHECK: "xla_cpu.rng_bit_generator"(%[[ARG0]], %[[STATE_INIT]], %[[DST_INIT]])
//  CHECK-SAME:   {rng_algorithm = #mhlo.rng_algorithm<DEFAULT>} :
//  CHECK-SAME:   (tensor<2xui64>, tensor<2xui64>, tensor<10xui32>) -> (tensor<2xui64>, tensor<10xui32>)
