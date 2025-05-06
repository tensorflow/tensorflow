// RUN: sdy_opt %s -xla-sdy-round-trip-shard-map-export 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>
sdy.mesh @mesh_1 = <["a"=2, "b"=2, "c"=2, "d"=2]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a", ?}, {"b", ?}]>}, %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"b", ?}, {?}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]]:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1)
  // CHECK-SAME:   {has_side_effect = true
  // CHECK-SAME:    mhlo.frontend_attributes = {
  // CHECK-SAME:      xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>",
  // CHECK-SAME:      xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>"}}
  // CHECK-SAME: : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK-NEXT: %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body(%[[GLOBAL_TO_LOCAL]]#0, %[[GLOBAL_TO_LOCAL]]#1)
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}}
  // CHECK-SAME: : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {
  // CHECK-SAME:      xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>",
  // CHECK-SAME:      xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}}
  // CHECK-SAME: (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh_0, [{"a"}, {"b"}]>, <@mesh_0, [{"b"}, {}], replicated={"a"}>] out_shardings=[<@mesh_0, [{"a"}, {}], replicated={"b"}>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<2x8xf32>
    %2 = stablehlo.dot %1, %arg3 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
    %3 = "stablehlo.all_reduce"(%2) ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %4 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) {
      replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
    } : (tensor<2x32xf32>) -> tensor<2x32xf32>
    sdy.return %3 : tensor<2x32xf32>
  } : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_comp_using_another
func.func @manual_comp_using_another(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_0, [{}, {"b"}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL_0:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME:   {has_side_effect = true
  // CHECK-SAME:    mhlo.frontend_attributes = {
  // CHECK-SAME:      xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}]>]>",
  // CHECK-SAME:      xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>"}}
  // CHECK-SAME: : (tensor<8x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP_0:.*]] = call @local_xla.sdy.manual_computation_body_0(%[[GLOBAL_TO_LOCAL_0]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL_0:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP_0]])
  // CHECK-SAME:   {mhlo.frontend_attributes = {
  // CHECK-SAME:      xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME:      xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: : (tensor<2x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL_1:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%[[LOCAL_TO_GLOBAL_0]])
  // CHECK-SAME: {has_side_effect = true
  // CHECK-SAME:  mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\22b\22}]>]>",
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>"}}
  // CHECK-SAME: (tensor<8x8xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[SHMAP_1:.*]] = call @local_xla.sdy.manual_computation_body_1(%[[GLOBAL_TO_LOCAL_1]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\22b\22}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\22b\22}]>]>"}}
  // CHECK-SAME: : (tensor<8x4xf32>) -> tensor<8x4xf32
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL_1:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP_1]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
  // CHECK-SAME:    xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{}, {\22b\22}]>]>"}}
  // CHECK-SAME: (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL_1]] : tensor<8x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"a"}, {}]>] out_shardings=[<@mesh_0, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    sdy.return %arg1 : tensor<2x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>

  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_0, [{}, {"b"}]>] out_shardings=[<@mesh_0, [{}, {"b"}]>] manual_axes={"b"} (%arg1: tensor<8x4xf32>) {
    sdy.return %arg1 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @nested_shmaps
func.func @nested_shmaps(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME: {has_side_effect = true,
  // CHECK-SAME:  mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>",
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>"}}
  // CHECK-SAME: (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body_3(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: : (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME:    xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    sdy.return %1 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @nested_shmaps_extra_op
func.func @nested_shmaps_extra_op(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a"}, {"b"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME: {has_side_effect = true,
  // CHECK-SAME:  mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>",
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>"}}
  // CHECK-SAME: (tensor<4x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body_5(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: (tensor<2x8xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22}>",
  // CHECK-SAME:    xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{\22a\22}, {}]>]>"}}
  // CHECK-SAME: (tensor<2x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_1, [{"a"}, {}]>] out_shardings=[<@mesh_1, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh_1, [{}, {"b"}]>] out_shardings=[<@mesh_1, [{}, {"b"}]>] manual_axes={"b"} (%arg2: tensor<2x4xf32>) {
      %2 = stablehlo.multiply %arg2, %arg2 : tensor<2x4xf32>
      sdy.return %2 : tensor<2x4xf32>
    } : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %3 = stablehlo.add %1, %1 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @manual_computation_no_inputs
func.func @manual_computation_no_inputs() -> tensor<4xi64> {
  // CHECK-NEXT: %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body_6()
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22b\22}]>]>"}}
  // CHECK-SAME: () -> tensor<2xi64>
  // CHECK-NEXT: %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
  // CHECK-SAME:    xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22b\22}]>]>"}}
  // CHECK-SAME: (tensor<2xi64>) -> tensor<4xi64>
  // CHECK-NEXT: return %[[LOCAL_TO_GLOBAL]] : tensor<4xi64>
  %0 = sdy.manual_computation() in_shardings=[] out_shardings=[<@mesh_0, [{"b"}]>] manual_axes={"b"} () {
    %1 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
    sdy.return %1 : tensor<2xi64>
  } : () -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// CHECK-LABEL: func @manual_computation_no_outputs
func.func @manual_computation_no_outputs(%arg0: tensor<4xi64>) {
  // CHECK-NEXT: %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
  // CHECK-SAME: {has_side_effect = true,
  // CHECK-SAME:  mhlo.frontend_attributes = {
  // CHECK-SAME:    xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22b\22}]>]>",
  // CHECK-SAME:    xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>"}}
  // CHECK-SAME: (tensor<4xi64>) -> tensor<2xi64>
  // CHECK-NEXT: call @local_xla.sdy.manual_computation_body_7(%[[GLOBAL_TO_LOCAL]])
  // CHECK-SAME: {mhlo.frontend_attributes = {
  // CHECK-SAME: xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22b\22}]>]>",
  // CHECK-SAME: xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
  // CHECK-SAME: xla.sdy.out_shardings = "#sdy.sharding_per_value<[]>"}}
  // CHECK-SAME: : (tensor<2xi64>) -> ()
  // CHECK-NEXT: return
  sdy.manual_computation(%arg0) in_shardings=[<@mesh_0, [{"b"}]>] out_shardings=[] manual_axes={"b"} (%arg1: tensor<2xi64>) {
    stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<2xi64>) -> ()
    sdy.return
  } : (tensor<4xi64>) -> ()
  func.return
}

// CHECK-LABEL: func @manual_computation_no_inputs_no_outputs
func.func @manual_computation_no_inputs_no_outputs() {
  // CHECK-NEXT: call @local_xla.sdy.manual_computation_body_8() {mhlo.frontend_attributes = {
  // CHECK-SAME:   xla.sdy.in_shardings = "#sdy.sharding_per_value<[]>",
  // CHECK-SAME:   xla.sdy.manual_axes = "#sdy<manual_axes{}>",
  // CHECK-SAME:   xla.sdy.out_shardings = "#sdy.sharding_per_value<[]>"
  // CHECK-SAME: }} : () -> ()
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    sdy.return
  } : () -> ()
  func.return
}

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:    stablehlo.add
// CHECK-NEXT:    stablehlo.dot
// CHECK-NEXT:    "stablehlo.all_reduce"

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:    return %arg0 : tensor<2x8xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_1(%arg0: tensor<8x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT:    return %arg0 : tensor<8x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_2(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:    stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_3(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32
// CHECK-NEXT:   %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
// CHECK-SAME:     {has_side_effect = true,
// CHECK-SAME:      mhlo.frontend_attributes = {
// CHECK-SAME:        xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>",
// CHECK-SAME:        xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>"}}
// CHECK-SAME:   (tensor<2x8xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body_2(%[[GLOBAL_TO_LOCAL]])
// CHECK-SAME:     {mhlo.frontend_attributes = {
// CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>",
// CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
// CHECK-SAME:     xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>"}}
// CHECK-SAME:     : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
// CHECK-SAME:     {mhlo.frontend_attributes = {
// CHECK-SAME:        xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
// CHECK-SAME:        xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>"}}
// CEHECK-SAME:  (tensor<2x4xf32>) -> tensor<2x8xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_4(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:    stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_5(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[GLOBAL_TO_LOCAL:.*]] = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0)
// CHECK-SAME:     {has_side_effect = true,
// CHECK-SAME:      mhlo.frontend_attributes = {
// CHECK-SAME:        xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>",
// CHECK-SAME:        xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>"}}
// CHECK-SAME:     (tensor<2x8xf32>) -> tensor<2x4xf32
// CHECK-NEXT:   %[[SHMAP:.*]] = call @local_xla.sdy.manual_computation_body_4(%[[GLOBAL_TO_LOCAL]])
// CHECK-SAME:     {mhlo.frontend_attributes = {
// CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>",
// CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
// CHECK-SAME:     xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>"}}
// CHECK-SAME:      : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT:   %[[LOCAL_TO_GLOBAL:.*]] = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%[[SHMAP]])
// CHECK-SAME:     {mhlo.frontend_attributes = {
// CHECK-SAME:        xla.sdy.manual_axes = "#sdy<manual_axes{\22b\22}>",
// CHECK-SAME:        xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_1, [{}, {\22b\22}]>]>"}}
// CHECK-SAME:   (tensor<2x4xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LOCAL_TO_GLOBAL]], %[[LOCAL_TO_GLOBAL]] : tensor<2x8xf32>
// CHECK-NEXT:   return %[[ADD]] : tensor<2x8xf32>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_6() -> tensor<2xi64> {
// CHECK-NEXT:    %[[C:.*]] = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
// CHECK-NEXT:    return %[[C]] : tensor<2xi64>

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_7(%arg0: tensor<2xi64>) {
// CHECK-NEXT:    stablehlo.custom_call @sdy_testonly(%arg0) : (tensor<2xi64>) -> ()
// CHECK-NEXT:    return

// CHECK-LABEL: func @local_xla.sdy.manual_computation_body_8() {
// CHECK-NEXT:    return
