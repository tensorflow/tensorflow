// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-import -split-input-file -verify-diagnostics

sdy.mesh @mesh = <["a"=2]>

func.func @using_same_body_func(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%arg0) {called_computations = [@local_xla.sdy.manual_computation_body_0], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // expected-error @+2 {{'stablehlo.custom_call' op expected a unique FuncOp per @local_xla.sdy.ManualComputation custom call}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.custom_call'}}
  %1 = stablehlo.custom_call @local_xla.sdy.ManualComputation(%0) {called_computations = [@local_xla.sdy.manual_computation_body_0], mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\\\22a\\\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh, [{\\\22a\\\22}, {}]>]>"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

func.func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  return %arg0 : tensor<2x8xf32>
}
