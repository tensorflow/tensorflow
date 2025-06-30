// RUN: sdy_opt %s -xla-sdy-stablehlo-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>
sdy.mesh @mesh_2 = <["a"=4, "b"=4, "c"=4]>
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {"b"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<8x16xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}, {}], replicated={"a"}>]>} : tensor<16x32xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %4 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %5 = stablehlo.dot %4, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %6 = "stablehlo.all_reduce"(%5) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<{{\[\[}}0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
  // CHECK-NEXT: ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  // CHECK-NEXT:   %9 = stablehlo.add %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %9 {xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<f32>
  // CHECK-NEXT: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %7 = mhlo.copy %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x32xf32>
  // CHECK-NEXT: %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}], replicated={"b"}>]>} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %8 : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"b"}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDShardToFullShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %4 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"b"}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : (tensor<8x8xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %6 = mhlo.copy %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"b"}>} : tensor<8x4xf32>
  // CHECK-NEXT: %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {"b"}]>]>} : (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %7 : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"b"}]>] out_shardings=[<@mesh_0, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_in_manual_computation_body
func.func @sharding_in_manual_computation_body(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {"b"}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<8x16xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"b"}, {}], replicated={"a"}>]>} : tensor<16x32xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: %4 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<4x8xf32>
  // CHECK-NEXT: %5 = stablehlo.dot %4, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"d"}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
  // CHECK-NEXT: %6 = mhlo.copy %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<4x32xf32>
  // CHECK-NEXT: %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}], replicated={"b"}>]>} : (tensor<4x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %7 : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_1, [{"a"}, {"b"}]>, <@mesh_1, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_1, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<4x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>} : tensor<4x8xf32>
    %2 = stablehlo.dot %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"d"}, {"c"}]>]>} : (tensor<4x8xf32>, tensor<8x32xf32>) -> tensor<4x32xf32>
    sdy.return %2 : tensor<4x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @call_op_with_no_operands_or_results
func.func @call_op_with_no_operands_or_results() {
  // CHECK-LABEL: %cst = stablehlo.constant
  // CHECK-NOT:     sdy.sharding
  // CHECK-NOT:     mhlo.sharding
  // CHECK-NEXT: %0 = stablehlo.add %cst, %cst {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<2x2xf32>
  // CHECK-NEXT: return
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<2x2xf32>
    sdy.return
  } : () -> ()
  return
}

// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}, {"c"}]>}) -> (tensor<4x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}, {?}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}, {}]>]>} : tensor<4x8x16xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8x16xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  // CHECK-NEXT: %4 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4x16xf32>
  // CHECK-NEXT: %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : (tensor<2x4x16xf32>) -> tensor<2x4x8xf32>
  // CHECK-NEXT: %6 = stablehlo.multiply %5, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : tensor<2x4x8xf32>
  // CHECK-NEXT: %7 = mhlo.copy %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b", "c"}>} : tensor<2x4x8xf32>
  // CHECK-NEXT: %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {"c"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  // CHECK-NEXT: %9 = mhlo.copy %8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4x16xf32>
  // CHECK-NEXT: %10 = stablehlo.custom_call @SPMDShardToFullShape(%9) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x16xf32>) -> tensor<2x8x16xf32>
  // CHECK-NEXT: %11 = mhlo.copy %10 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8x16xf32>
  // CHECK-NEXT: %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}, {}]>]>} : (tensor<2x8x16xf32>) -> tensor<4x8x16xf32>
  // CHECK-NEXT: return %12 : tensor<4x8x16xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8x16xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}, {}]>] out_shardings=[<@mesh_1, [{}, {"b"}, {}]>] manual_axes={"b"} (%arg2: tensor<2x4x16xf32>) {
      %2 = sdy.manual_computation(%arg2) in_shardings=[<@mesh_1, [{}, {}, {"c"}]>] out_shardings=[<@mesh_1, [{}, {}, {"c"}]>] manual_axes={"c"} (%arg3: tensor<2x4x8xf32>) {
        %3 = stablehlo.multiply %arg3, %arg3 : tensor<2x4x8xf32>
        sdy.return %3 : tensor<2x4x8xf32>
      } : (tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
      sdy.return %2 : tensor<2x4x16xf32>
    } : (tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
    sdy.return %1 : tensor<2x8x16xf32>
  } : (tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %2 = mhlo.copy %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2x8xf32>) -> tensor<2x4xf32>
  // CHECK-NEXT: %4 = stablehlo.multiply %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
  // CHECK-NEXT: %5 = stablehlo.add %4, %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
  // CHECK-NEXT: %6 = stablehlo.subtract %5, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c", "d"}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
  // CHECK-NEXT: %7 = mhlo.copy %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<2x4xf32>
  // CHECK-NEXT: %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %9 = stablehlo.add %8, %8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %10 = mhlo.copy %9 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<2x8xf32>
  // CHECK-NEXT: %11 = stablehlo.custom_call @SPMDShardToFullShape(%10) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"a"}, {}]>]>} : (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %11 : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      %3 = stablehlo.add %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c"}, {}]>]>} : tensor<2x4xf32>
      %4 = stablehlo.subtract %3, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1, [{"c", "d"}, {}]>]>} : tensor<2x4xf32>
      sdy.return %4 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %5 = stablehlo.add %1, %1 : tensor<2x8xf32>
    sdy.return %5 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @multiple_manual_computation_uses
func.func @multiple_manual_computation_uses(%arg0: tensor<2x4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {"a"}]>}, %arg1: tensor<32x16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {}, {"a"}]>}) -> (tensor<131x4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_0, [{?}, {?}, {"a"}]>}) {
  // CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<2x4x8xi32>
  // CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x8xi32>) -> tensor<2x4x2xi32>
  // CHECK-NEXT: %2 = stablehlo.custom_call @sdy_testonly(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %3 = mhlo.copy %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<3x4x2xi32>
  // CHECK-NEXT: %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<3x4x2xi32>) -> tensor<3x4x8xi32>
  // CHECK-NEXT: %5 = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<32x16x8xi32>
  // CHECK-NEXT: %6 = stablehlo.custom_call @SPMDFullToShardShape(%5) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32x16x8xi32>) -> tensor<32x16x2xi32>
  // CHECK-NEXT: %7 = stablehlo.reshape %6 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32>
  // CHECK-NEXT: %8 = mhlo.copy %7 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<128x4x2xi32>
  // CHECK-NEXT: %9 = stablehlo.custom_call @SPMDShardToFullShape(%8) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<128x4x2xi32>) -> tensor<128x4x8xi32>
  // CHECK-NEXT: %10 = mhlo.copy %4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<3x4x8xi32>
  // CHECK-NEXT: %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<3x4x8xi32>) -> tensor<3x4x2xi32>
  // CHECK-NEXT: %12 = mhlo.copy %9 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : tensor<128x4x8xi32>
  // CHECK-NEXT: %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<128x4x8xi32>) -> tensor<128x4x2xi32>
  // CHECK-NEXT: %14 = stablehlo.concatenate %13, %11, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<128x4x2xi32>, tensor<3x4x2xi32>) -> tensor<131x4x2xi32>
  // CHECK-NEXT: %15 = mhlo.copy %14 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<131x4x2xi32>
  // CHECK-NEXT: %16 = stablehlo.custom_call @SPMDShardToFullShape(%15) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}, {}, {"a"}]>]>} : (tensor<131x4x2xi32>) -> tensor<131x4x8xi32>
  // CHECK-NEXT: return %16 : tensor<131x4x8xi32>
  %1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<2x4x2xi32>) {
    %4 = stablehlo.custom_call @sdy_testonly(%arg2) : (tensor<2x4x2xi32>) -> tensor<3x4x2xi32>
    sdy.return %4 : tensor<3x4x2xi32>
  } : (tensor<2x4x8xi32>) -> tensor<3x4x8xi32>
  %2 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<32x16x2xi32>) {
    %4 = stablehlo.reshape %arg2 : (tensor<32x16x2xi32>) -> tensor<128x4x2xi32>
    sdy.return %4 : tensor<128x4x2xi32>
  } : (tensor<32x16x8xi32>) -> tensor<128x4x8xi32>
  %3 = sdy.manual_computation(%1, %2) in_shardings=[<@mesh_0, [{}, {}, {"a"}]>, <@mesh_0, [{}, {}, {"a"}]>] out_shardings=[<@mesh_0, [{}, {}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<3x4x2xi32>, %arg3: tensor<128x4x2xi32>) {
    %4 = stablehlo.concatenate %arg3, %arg2, dim = 0 : (tensor<128x4x2xi32>, tensor<3x4x2xi32>) -> tensor<131x4x2xi32>
    sdy.return %4 : tensor<131x4x2xi32>
  } : (tensor<3x4x8xi32>, tensor<128x4x8xi32>) -> tensor<131x4x8xi32>
  return %3 : tensor<131x4x8xi32>
}

// CHECK-LABEL: func @named_computation_in_manual_computation
func.func @named_computation_in_manual_computation(%arg0: tensor<32xi32>) -> (tensor<32xi32>, tensor<32xi32>, tensor<16xi32>) {
// CHECK-NEXT: %0 = mhlo.copy %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : tensor<32xi32>
// CHECK-NEXT: %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<32xi32>) -> tensor<8xi32>
// CHECK-NEXT: %2:2 = sdy.named_computation<"foo">(%1) in_shardings=[<@mesh_0, [{}]>] out_shardings=[<@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>] (%arg1: tensor<8xi32>) {
// CHECK-NEXT:   %10 = stablehlo.multiply %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT:   %11 = stablehlo.negate %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT:   sdy.return {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} %10, %11 : tensor<8xi32>, tensor<8xi32>
// CHECK-NEXT: } {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
// CHECK-NEXT: %3 = sdy.named_computation<"no_input_named_computation">() in_shardings=[] out_shardings=[<@mesh_0, [{}]>] () {
// CHECK-NEXT:   %c = stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-NEXT:   %10 = stablehlo.negate %c {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<4xi32>
// CHECK-NEXT:   sdy.return {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} %10 : tensor<4xi32>
// CHECK-NEXT: } {xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : () -> tensor<4xi32>
// CHECK-NEXT: %4 = mhlo.copy %2#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT: %5 = stablehlo.custom_call @SPMDShardToFullShape(%4) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<8xi32>) -> tensor<32xi32>
// CHECK-NEXT: %6 = mhlo.copy %2#1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<8xi32>
// CHECK-NEXT: %7 = stablehlo.custom_call @SPMDShardToFullShape(%6) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<8xi32>) -> tensor<32xi32>
// CHECK-NEXT: %8 = mhlo.copy %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a"}>} : tensor<4xi32>
// CHECK-NEXT: %9 = stablehlo.custom_call @SPMDShardToFullShape(%8) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"a", "b"}]>]>} : (tensor<4xi32>) -> tensor<16xi32>
// CHECK-NEXT: return %5, %7, %9 : tensor<32xi32>, tensor<32xi32>, tensor<16xi32>
  %0:3 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a", "b"}]>] out_shardings=[<@mesh_0, [{"a", "b"}]>, <@mesh_0, [{"a", "b"}]>, <@mesh_0, [{"a", "b"}]>] manual_axes={"a"} (%arg1: tensor<8xi32>) {
    %1:2 = sdy.named_computation<"foo">(%arg1) out_shardings=[<@mesh_0, [{"b"}]>, <@mesh_0, [{"b"}]>] (%arg2: tensor<8xi32>) {
      %2 = stablehlo.multiply %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : tensor<8xi32>
      %3 = stablehlo.negate %arg2 : tensor<8xi32>
      sdy.return %2, %3 : tensor<8xi32>, tensor<8xi32>
    } : (tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
    %4 = sdy.named_computation<"no_input_named_computation">() () {
      %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
      %5 = stablehlo.negate %c : tensor<4xi32>
      sdy.return %5 : tensor<4xi32>
    } : () -> (tensor<4xi32>)
    sdy.return %1#0, %1#1, %4 : tensor<8xi32>, tensor<8xi32>, tensor<4xi32>
  } : (tensor<32xi32>) -> (tensor<32xi32>, tensor<32xi32>, tensor<16xi32>)
  return %0#0, %0#1, %0#2 : tensor<32xi32>, tensor<32xi32>, tensor<16xi32>
}

// CHECK-LABEL: func @manual_computation_with_tokens
func.func @manual_computation_with_tokens(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh_0, []>},
    %arg1: tensor<2xi64> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b"}]>}
) -> (!stablehlo.token, tensor<2xi64>) {
  // CHECK-NEXT: %[[COPY_OPERAND:.*]] = mhlo.copy %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : tensor<2xi64>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY_OPERAND]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (tensor<2xi64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[TOKEN_CALL:.*]] = stablehlo.custom_call @sdy_testonly(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (!stablehlo.token) -> !stablehlo.token
  // CHECK-NEXT: stablehlo.custom_call @sdy_testonly(%[[TOKEN_CALL]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, []>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : (!stablehlo.token) -> ()
  // CHECK-NEXT: %[[COPY_RESULT:.*]] = mhlo.copy %[[FULL_TO_SHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{}]>]>, xla.sdy.manual_axes = #sdy<manual_axes{"a", "b"}>} : tensor<1xi64>
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_RESULT]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_0, [{"b"}]>]>} : (tensor<1xi64>) -> tensor<2xi64>
  // CHECK-NEXT: return %[[TOKEN_CALL]], %[[SHARD_TO_FULL]] : !stablehlo.token, tensor<2xi64>
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
      out_shardings=[<@mesh_0, []>, <@mesh_0, [{"b"}]>]
      manual_axes={"a", "b"} (%arg2: !stablehlo.token, %arg3: tensor<1xi64>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg2) : (!stablehlo.token) -> (!stablehlo.token)
    stablehlo.custom_call @sdy_testonly(%1) {sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh_0, []>]>} : (!stablehlo.token) -> ()
    sdy.return %1, %arg3 : !stablehlo.token, tensor<1xi64>
  } : (!stablehlo.token, tensor<2xi64>) -> (!stablehlo.token, tensor<2xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<2xi64>
}
