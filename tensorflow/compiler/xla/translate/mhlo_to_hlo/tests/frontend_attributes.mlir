// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Tests parameters with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<3x4xf32> {mhlo.frontend_attributes = {_test = "a"}}, %arg1: tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tuple<tensor<3x4xf32>, tensor<3x4xf32>>
  func.return %0 : tuple<tensor<3x4xf32>, tensor<3x4xf32>>
}

// CHECK:  ENTRY
// CHECK:  %[[P0:.*]] = f32[3,4] parameter(0), frontend_attributes={_test="a"}
// CHECK:  %[[P1:.*]] = f32[3,4] parameter(1)
// CHECK-NOT: frontend_attributes=

// -----

// Tests call ops with frontend_attributes have such attributes set correctly
// in HloModule

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = call @add(%arg0, %arg0) {mhlo.frontend_attributes = {_xla_compute_type = "dense"}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
func.func private @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:       %[[CALL:.*]] = f32[] call
// CHECK-SAME:  frontend_attributes={_xla_compute_type="dense"}