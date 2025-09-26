// RUN: sdy_opt %s -xla-sdy-round-trip-clone-manual-computation-calls -split-input-file 2>&1 | FileCheck %s

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @single_manual_comp
func.func @single_manual_comp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  %0:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>"}} : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK: call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %1 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %2 : tensor<8x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @using_same_body_func
func.func @using_same_body_func(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  %0:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>"}} : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK: call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %1 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK: call @local_xla.sdy.manual_computation_body_0(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %3 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %4 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%3) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %4 : tensor<8x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }

// -----

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @using_different_body_func
func.func @using_different_body_func(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  %0:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>"}} : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK: call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %1 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK: call @local_xla.sdy.manual_computation_body_another(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %3 = call @local_xla.sdy.manual_computation_body_another(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %4 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%3) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %4 : tensor<8x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body_another(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body_another(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----

sdy.mesh @mesh_0 = <["a"=4, "b"=2]>

// CHECK-LABEL: func @using_same_body_func_potential_name_collision
func.func @using_same_body_func_potential_name_collision(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> (tensor<8x32xf32>) {
  %0:2 = stablehlo.custom_call @local_xla.sdy.GlobalToLocalShape(%arg0, %arg1) {mhlo.frontend_attributes = {xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {\22b\22}]>, <@mesh_0, [{\22b\22}, {}], replicated={\22a\22}>]>", xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>"}} : (tensor<8x16xf32>, tensor<16x32xf32>) -> (tensor<2x8xf32>, tensor<8x32xf32>)
  // CHECK: call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %1 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %2 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%1) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK: call @local_xla.sdy.manual_computation_body_0(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %3 = call @local_xla.sdy.manual_computation_body_0(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %4 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%3) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  // CHECK: call @local_xla.sdy.manual_computation_body_1(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %5 = call @local_xla.sdy.manual_computation_body(%0#0, %0#1) : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  %6 = stablehlo.custom_call @local_xla.sdy.LocalToGlobalShape(%5) {mhlo.frontend_attributes = {xla.sdy.manual_axes = "#sdy<manual_axes{\22a\22, \22b\22}>", xla.sdy.out_shardings = "#sdy.sharding_per_value<[<@mesh_0, [{\22a\22}, {}], replicated={\22b\22}>]>"}} : (tensor<2x32xf32>) -> tensor<8x32xf32>
  return %6 : tensor<8x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }
func.func @local_xla.sdy.manual_computation_body_0(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK:      func.func @local_xla.sdy.manual_computation_body_1(%arg0: tensor<2x8xf32>, %arg1: tensor<8x32xf32>) -> tensor<2x32xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x8xf32>, tensor<8x32xf32>) -> tensor<2x32xf32>
// CHECK-NEXT:   return %0 : tensor<2x32xf32>
// CHECK-NEXT: }

