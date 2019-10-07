// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    // CHECK: {{%.*}} = spv.undef : f32
    // CHECK-NEXT: {{%.*}} = spv.undef : f32
    %0 = spv.undef : f32
    %1 = spv.undef : f32
    %2 = spv.FAdd %0, %1 : f32
    // CHECK: {{%.*}} = spv.undef : vector<4xi32>
    %3 = spv.undef : vector<4xi32>
    %4 = spv.CompositeExtract %3[1 : i32] : vector<4xi32>
    // CHECK: {{%.*}} = spv.undef : !spv.array<4 x !spv.array<4 x i32>>
    %5 = spv.undef : !spv.array<4x!spv.array<4xi32>>
    %6 = spv.CompositeExtract %5[1 : i32, 2 : i32] : !spv.array<4x!spv.array<4xi32>>
    // CHECK: {{%.*}} = spv.undef : !spv.ptr<!spv.struct<f32>, StorageBuffer>
    %7 = spv.undef : !spv.ptr<!spv.struct<f32>, StorageBuffer>
    %8 = spv.constant 0 : i32
    %9 = spv.AccessChain %7[%8] : !spv.ptr<!spv.struct<f32>, StorageBuffer>
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  // CHECK: func {{@.*}}
  func @ignore_unused_undef() -> () {
    // CHECK-NEXT: spv.Return
    %0 = spv.undef : f32
    spv.Return
  }
}