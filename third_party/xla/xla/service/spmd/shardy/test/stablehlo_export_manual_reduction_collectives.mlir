// RUN: sdy_opt %s --split-input-file -xla-sdy-stablehlo-export-manual-reduction-collectives 2>&1 | FileCheck %s

sdy.mesh @mesh_x_2_y_2 = <["x"=2, "y"=2]>
sdy.mesh @mesh_x_2_y_4_z_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_x_4_y_6 = <["x"=4, "y"=6]>
sdy.mesh @mesh_x_2_y_2_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>

// CHECK-LABEL: func @all_reduce_input_no_unreduced_axes
func.func @all_reduce_input_no_unreduced_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{"y"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{"y"}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x8xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{"y"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_f32
func.func @all_reduce_f32(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{"y"}, {}], unreduced={"x"}>}) -> tensor<8x8xf32> {
  // CHECK-NEXT:          %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:              in_shardings=[<@mesh_x_2_y_2, [{"y"}, {}], unreduced={"x"}>]
  // CHECK-SAME:              out_shardings=[<@mesh_x_2_y_2, [{"y"}, {}]>]
  // CHECK-SAME:              manual_axes={"x", "y"} (%arg1: tensor<4x8xf32>) {
  // CHECK-NEXT:            %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%arg1) <{
  // CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // CHECK-SAME:              use_global_device_ids}> ({
  // CHECK-NEXT:            ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  // CHECK-NEXT:              %[[ADD:.*]] = stablehlo.add %arg2, %arg3 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT:          sdy.return %[[ALL_REDUCE]] : tensor<4x8xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MANUAL_COMP]] : tensor<8x8xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{"y"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_i64
func.func @all_reduce_i64(%arg0: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{}, {"x"}], unreduced={"y"}>}) -> tensor<8x8xi64> {
  // CHECK-NEXT:          %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:              in_shardings=[<@mesh_x_2_y_2, [{}, {"x"}], unreduced={"y"}>]
  // CHECK-SAME:              out_shardings=[<@mesh_x_2_y_2, [{}, {"x"}]>]
  // CHECK-SAME:              manual_axes={"x", "y"} (%arg1: tensor<8x4xi64>) {
  // CHECK-NEXT:            %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%arg1) <{
  // CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
  // CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
  // CHECK-SAME:              use_global_device_ids}> ({
  // CHECK-NEXT:            ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
  // CHECK-NEXT:              %[[ADD:.*]] = stablehlo.add %arg2, %arg3 : tensor<i64>
  // CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<i64>
  // CHECK-NEXT:            }) : (tensor<8x4xi64>) -> tensor<8x4xi64>
  // CHECK-NEXT:          sdy.return %[[ALL_REDUCE]] : tensor<8x4xi64>
  // CHECK-NEXT:          } : (tensor<8x8xi64>) -> tensor<8x8xi64>
  // CHECK-NEXT:          return %[[MANUAL_COMP]] : tensor<8x8xi64>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_x_2_y_2, [{}, {"x"}]> : tensor<8x8xi64>
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: func @all_reduce_to_fully_replicated
func.func @all_reduce_to_fully_replicated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>}) -> tensor<8x8xf32> {
  // CHECK-NEXT:          %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:              in_shardings=[<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>]
  // CHECK-SAME:              out_shardings=[<@mesh_x_2_y_2, [{}, {}]>]
  // CHECK-SAME:              manual_axes={"x", "y"} (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:            %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%arg1) <{
  // CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  // CHECK-SAME:              use_global_device_ids}> ({
  // CHECK-NEXT:            ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  // CHECK-NEXT:              %[[ADD:.*]] = stablehlo.add %arg2, %arg3 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          sdy.return %[[ALL_REDUCE]] : tensor<8x8xf32>
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MANUAL_COMP]] : tensor<8x8xf32>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh_x_2_y_2, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_single_axis
func.func @all_reduce_single_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{"y"}, {}], unreduced={"x"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 2], [1, 3]]>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{"y"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}


// CHECK-LABEL: func @all_reduce_single_axis_2
func.func @all_reduce_single_axis_2(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{"x"}, {}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 1], [2, 3]]>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_x_2_y_2, [{"x"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_single_axis_3
func.func @all_reduce_single_axis_3(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_4_z_2, [{"x"}, {"z"}], unreduced={"y"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_x_2_y_4_z_2, [{"x"}, {"z"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @al_reduce_multiple_axes
func.func @al_reduce_multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 1, 2, 3]]>
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh_x_2_y_2, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @al_reduce_multiple_axes_2
func.func @al_reduce_multiple_axes_2(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 2, 1, 3]]>
  %0 = sdy.all_reduce {"y", "x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_non_iota_device_order
func.func @all_reduce_non_iota_device_order(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2_non_iota, [{"y"}, {}], unreduced={"x"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[3, 1], [2, 0]]>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2_non_iota, [{"y"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @all_reduce_sub_axis
func.func @all_reduce_sub_axis(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_6, [{}, {}], unreduced={"x", "y"}>}) -> tensor<8x8xf32> {
  // CHECK{LITERAL}: replica_groups = dense<[[0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20], [3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23]]>
  %0 = sdy.all_reduce {"y":(2)3, "x"} %arg0 out_sharding=<@mesh_x_4_y_6, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @partial_all_reduce
func.func @partial_all_reduce(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>}) -> tensor<8x8xf32> {
  // CHECK-NEXT:          %[[MANUAL_COMP_X:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:              in_shardings=[<@mesh_x_2_y_2, [{}, {}], unreduced={"x", "y"}>]
  // CHECK-SAME:              out_shardings=[<@mesh_x_2_y_2, [{}, {}], unreduced={"y"}>]
  // CHECK-SAME:              manual_axes={"x", "y"} (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:            %[[ALL_REDUCE_X:.*]] = "stablehlo.all_reduce"(%arg1) <{
  // CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>,
  // CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 2], [1, 3]]>
  // CHECK-SAME:              use_global_device_ids}>
  // CHECK:                 sdy.return %[[ALL_REDUCE_X]]
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          %[[ADD:.*]] = stablehlo.add %[[MANUAL_COMP_X]], %[[MANUAL_COMP_X]]
  // CHECK-SAME:            {mhlo.frontend_attributes = {sdy.has_unreduced_axis = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_2_y_2, [{}, {}], unreduced={"y"}>]>
  // CHECK-NEXT:          %[[MANUAL_COMP_Y:.*]] = sdy.manual_computation(%[[ADD]])
  // CHECK-SAME:              in_shardings=[<@mesh_x_2_y_2, [{}, {}], unreduced={"y"}>]
  // CHECK-SAME:              out_shardings=[<@mesh_x_2_y_2, [{}, {}]>]
  // CHECK-SAME:              manual_axes={"x", "y"} (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:            %[[ALL_REDUCE_Y:.*]] = "stablehlo.all_reduce"(%arg1) <{
  // CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>,
  // CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 1], [2, 3]]>
  // CHECK-SAME:              use_global_device_ids}>
  // CHECK:                 sdy.return %[[ALL_REDUCE_X]]
  // CHECK-NEXT:          } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:          return %[[MANUAL_COMP_Y]]
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_x_2_y_2, [{}, {}], unreduced={"y"}> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_2_y_2, [{}, {}], unreduced={"y"}>]>} : tensor<8x8xf32>
  %2 = sdy.all_reduce {"y"} %1 out_sharding=<@mesh_x_2_y_2, [{}, {}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @unique_channel_handle
func.func @unique_channel_handle(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>},
    %arg1: !stablehlo.token) -> tensor<16x16xf32> {
  %token = "stablehlo.send"(%arg0, %arg1) <{
      channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>,
      is_host_transfer = true}> :
      (tensor<8x8xf32>, !stablehlo.token) -> !stablehlo.token
  // CHECK:               "stablehlo.all_reduce"
  // CHECK-SAME:            channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = "stablehlo.all_gather"(%1) <{
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    }> : (tensor<8x8xf32>) -> tensor<16x8xf32>
  %3 = "stablehlo.all_gather"(%2) <{
      all_gather_dim = 1 : i64,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
      replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
      use_global_device_ids
    }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : (tensor<16x8xf32>) -> tensor<16x16xf32>
  // CHECK:               "stablehlo.all_reduce"
  // CHECK-SAME:            channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, use_global_device_ids
  %4 = sdy.all_reduce {"y"} %3 out_sharding=<@mesh, [{}, {}]> : tensor<16x16xf32>
  return %4 : tensor<16x16xf32>
}

// CHECK-LABEL: func @unique_channel_handle_2
func.func @unique_channel_handle_2(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) -> tensor<8x8xf32> {
  // CHECK:               "stablehlo.all_reduce"
  // CHECK-SAME:            channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, use_global_device_ids
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

//===----------------------------------------------------------------------===//
// Unreduced frontend attribute tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unreduced_func_input
func.func @unreduced_func_input(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg0, %arg1
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unreduced_op
func.func @unreduced_op(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {mhlo.frontend_attributes = {sdy.has_unreduced_axis = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {mhlo.frontend_attributes = {sdy.has_unreduced_axis = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}
