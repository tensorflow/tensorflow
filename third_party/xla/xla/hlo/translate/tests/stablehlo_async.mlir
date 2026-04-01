// RUN: xla-translate --stablehlo-to-hlo-text --print-sugar=false -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false -split-input-file %s | FileCheck %s --check-prefix CHECK-DIRECT

// Tests for StableHLO async ops to validate StableHLO -> HLO conversion.

// CHECK-LABEL: HloModule main
// CHECK: ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[AGS:.*]] = (f32[4,2], f32[4,4]) all-gather-start(%[[ARG]]), channel_id=1,
// CHECK-SAME{LITERAL}: replica_groups={{0,1},{2,3}},
// CHECK-SAME:  dimensions={1}, use_global_device_ids=true,
// CHECK-NEXT:  ROOT %{{.*}} = f32[4,4] all-gather-done(%[[AGS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<4x4xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, use_global_device_ids}> : (tensor<4x2xf32>) -> tensor<4x4xf32>
    stablehlo.return %1 : tensor<4x4xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x4xf32>>
  %2 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x4xf32>>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.all_gather
// CHECK-DIRECT: stablehlo.async_done

// -----

// CHECK-LABEL: HloModule main
// CHECK: ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[ARS:.*]] = f32[4,2] all-reduce-start(%[[ARG]]), channel_id=1,
// CHECK-SAME{LITERAL}: replica_groups={{0,1},{2,3}},
// CHECK-SAME:  to_apply=%{{.*}},
// CHECK-NEXT:  ROOT %{{.*}} = f32[4,2] all-reduce-done(%[[ARS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    stablehlo.return %1 : tensor<4x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x2xf32>>
  %3 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x2xf32>>) -> tensor<4x2xf32>
  return %3 : tensor<4x2xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.all_reduce
// CHECK-DIRECT: stablehlo.async_done

// -----

// CHECK-LABEL: HloModule main
// CHECK:       %[[COMP:.*]] ({{.*}}: f32[4,2]) -> f32[4,2] {
// CHECK:         ROOT %{{.*}} = f32[4,2] all-to-all(
// CHECK:       ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[AS:.*]] = ((f32[4,2]), f32[4,2]) async-start(%[[ARG]]), calls=%[[COMP]]
// CHECK-NEXT:  ROOT %{{.*}} = f32[4,2] async-done(%[[AS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.all_to_all"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
    stablehlo.return %1 : tensor<4x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x2xf32>>
  %2 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x2xf32>>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.all_to_all
// CHECK-DIRECT: stablehlo.async_done

// -----

// CHECK-LABEL: HloModule main
// CHECK: ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[CPS:.*]] = (f32[4,2], f32[4,2]) collective-permute-start(%[[ARG]]),
// CHECK-SAME{LITERAL}: source_target_pairs={{0,1},{1,2},{2,3}},
// CHECK-NEXT:  ROOT %{{.*}} = f32[4,2] collective-permute-done(%[[CPS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.collective_permute"(%arg1) <{source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
    stablehlo.return %1 : tensor<4x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x2xf32>>
  %2 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x2xf32>>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.collective_permute
// CHECK-DIRECT: stablehlo.async_done

// -----

// CHECK-LABEL: HloModule main
// CHECK:       %[[COMP:.*]] ({{.*}}: f32[4,2]) -> f32[2,2] {
// CHECK:         ROOT %{{.*}} = f32[2,2] reduce-scatter(
// CHECK:       ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[AS:.*]] = ((f32[4,2]), f32[2,2]) async-start(%[[ARG]]), calls=%[[COMP]]
// CHECK-NEXT:  ROOT %{{.*}} = f32[2,2] async-done(%[[AS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<2x2xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<4x2xf32>) -> tensor<2x2xf32>
    stablehlo.return %1 : tensor<2x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<2x2xf32>>
  %3 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<2x2xf32>>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.reduce_scatter
// CHECK-DIRECT: stablehlo.async_done

// -----

// CHECK-LABEL: HloModule main
// CHECK:       %[[COMP:.*]] ({{.*}}: f32[4,2]) -> f32[4,2] {
// CHECK:         ROOT %{{.*}} = f32[4,2] collective-broadcast(
// CHECK:       ENTRY
// CHECK-NEXT:  %[[ARG:.*]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[AS:.*]] = ((f32[4,2]), f32[4,2]) async-start(%[[ARG]]), calls=%[[COMP]]
// CHECK-NEXT:  ROOT %{{.*}} = f32[4,2] async-done(%[[AS]])
func.func @main(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %0 = "stablehlo.async_start"(%arg0) ({
  ^bb0(%arg1: tensor<4x2xf32>):
    %1 = "stablehlo.collective_broadcast"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : (tensor<4x2xf32>) -> tensor<4x2xf32>
    stablehlo.return %1 : tensor<4x2xf32>
  }) : (tensor<4x2xf32>) -> !stablehlo.future<tensor<4x2xf32>>
  %2 = "stablehlo.async_done"(%0) : (!stablehlo.future<tensor<4x2xf32>>) -> tensor<4x2xf32>
  return %2 : tensor<4x2xf32>
}
// CHECK-DIRECT: stablehlo.async_start
// CHECK-DIRECT: stablehlo.collective_broadcast
// CHECK-DIRECT: stablehlo.async_done
