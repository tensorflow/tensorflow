// RUN: sdy_opt %s -xla-mhlo-round-trip-shard-map-import -split-input-file -verify-diagnostics

sdy.mesh @mesh_1 = <["a"=4, "b"=2]>
sdy.mesh @mesh_2 = <["a"=4, "b"=2, "c"=3]>

func.func public @multiple_meshes(%arg0: tensor<16x16xf32>) -> tensor<32x4xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b"}, {"a"}]>]>} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x16xf32>) -> tensor<8x4xf32>
  // expected-error @+1 {{Multiple meshes in a single manual computation.}}
  %2 = call @shmap_body_0(%1) : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<8x4xf32>) -> tensor<8x4xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"a"}, {}], replicated={"c"}>]>} : (tensor<8x4xf32>) -> tensor<32x4xf32>
  return %4 : tensor<32x4xf32>
}
func.func private @shmap_body_0(%arg0: tensor<8x4xf32>) -> (tensor<8x4xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{expecting CustomCallOp as operand}}
  %0 = call @shmap_body_1(%arg0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @Sharding(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2 = stablehlo.custom_call @SPMDShardToFullShape(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %2 : tensor<16x32xf32>
}
func.func private @shmap_body_1(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting SPMDFullToShardShape custom call as operand}}
  %1 = call @shmap_body_1(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2 = stablehlo.custom_call @Sharding(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %3 : tensor<16x32xf32>
}
func.func private @shmap_body_1(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @SPMDFullToShardShape(%arg0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting CustomCallOp as operand of SPMDFullToShardShape}}
  %1 = call @shmap_body(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %2 = stablehlo.custom_call @Sharding(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %3 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @SPMDFullToShardShape(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting Sharding CustomCallOp as operand of SPMDFullToShardShape}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting each result of shmap_body to have one or no uses}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  stablehlo.custom_call @SPMDShardToFullShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %3 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting Sharding CustomCallOp user of the result to have one use}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %4, %3 : tensor<16x32xf32>, tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting CustomCallOp as the use of the result of the CallOp}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %2 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting Sharding CustomCallOp as the use of the result of the CallOp}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting CustomCallOp as the use of Sharding CustomCallOp}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %3 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4]>

func.func public @pattern_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  // expected-error @+1 {{expecting SPMDShardToFullShape CustomCallOp as the use of Sharding CustomCallOp}}
  %2 = call @shmap_body(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %3 = stablehlo.custom_call @Sharding(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %4 = stablehlo.custom_call @Sharding(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  return %4 : tensor<16x32xf32>
}
func.func private @shmap_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
