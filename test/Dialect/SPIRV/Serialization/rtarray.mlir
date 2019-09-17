// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable {{@.*}} : !spv.ptr<!spv.rtarray<f32>, StorageBuffer>
  spv.globalVariable @var0 : !spv.ptr<!spv.rtarray<f32>, StorageBuffer>
  // CHECK: spv.globalVariable {{@.*}} : !spv.ptr<!spv.rtarray<vector<4xf16>>, Input>
  spv.globalVariable @var1 : !spv.ptr<!spv.rtarray<vector<4xf16>>, Input>
}