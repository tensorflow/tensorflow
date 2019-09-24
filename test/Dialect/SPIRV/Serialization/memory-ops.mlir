// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:           func {{@.*}}([[ARG1:%.*]]: !spv.ptr<f32, Input>, [[ARG2:%.*]]: !spv.ptr<f32, Output>) {
// CHECK-NEXT:        [[VALUE:%.*]] = spv.Load "Input" [[ARG1]] : f32
// CHECK-NEXT:        spv.Store "Output" [[ARG2]], [[VALUE]] : f32

spv.module "Logical" "GLSL450" {
  func @load_store(%arg0 : !spv.ptr<f32, Input>, %arg1 : !spv.ptr<f32, Output>) {
    %1 = spv.Load "Input" %arg0 : f32
    spv.Store "Output" %arg1, %1 : f32
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @access_chain(%arg0 : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>,
                     %arg1 : i32, %arg2 : i32) {
    // CHECK: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    // CHECK-NEXT: {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}, {{%.*}}] : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, Function>
    %1 = spv.AccessChain %arg0[%arg1] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    %2 = spv.AccessChain %arg0[%arg1, %arg2] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    spv.Return
  }
}
