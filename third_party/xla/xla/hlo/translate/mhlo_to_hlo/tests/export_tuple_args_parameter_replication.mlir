// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -emit-use-tuple-args -verify-diagnostics %s | FileCheck %s

// CHECK: ENTRY
// CHECK:  [[VAL_1:%.*]] = (f32[], (f32[2,4], (f32[2,4]))) parameter(0), parameter_replication={true,false,true}
func.func @main(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false, true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

// CHECK: ENTRY
// CHECK:  [[VAL_1:%.*]] = (f32[], (f32[2,4], (f32[2,4]))) parameter(0), parameter_replication={false,false,true}
func.func @main(%arg0: tensor<f32>, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false, true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}
