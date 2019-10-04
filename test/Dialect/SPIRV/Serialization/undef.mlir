// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    // CHECK: {{%.*}} = spv.Undef : f32
    %0 = spv.Undef : f32
    // CHECK: {{%.*}} = spv.Undef : vector<4xi32>
    %1 = spv.Undef : vector<4xi32>
    // CHECK: {{%.*}} = spv.Undef : !spv.array<4 x !spv.array<4 x i32>>
    %2 = spv.Undef : !spv.array<4x!spv.array<4xi32>>
    // CHECK: {{%.*}} = spv.Undef : !spv.ptr<!spv.struct<f32>, StorageBuffer>
    %3 = spv.Undef : !spv.ptr<!spv.struct<f32>, StorageBuffer>
    spv.Return
  }
}