// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  spv.specConstant @condition_scalar = true
  func @select() -> () {
    %0 = spv.constant 4.0 : f32
    %1 = spv.constant 5.0 : f32
    %2 = spv._reference_of @condition_scalar : i1
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : i1, f32
    %3 = spv.Select %2, %0, %1 : i1, f32
    %4 = spv.constant dense<[2.0, 3.0, 4.0, 5.0]> : vector<4xf32>
    %5 = spv.constant dense<[6.0, 7.0, 8.0, 9.0]> : vector<4xf32>
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : i1, vector<4xf32>
    %6 = spv.Select %2, %4, %5 : i1, vector<4xf32>
    %7 = spv.constant dense<[true, true, true, true]> : vector<4xi1>
    // CHECK: spv.Select {{.*}}, {{.*}}, {{.*}} : vector<4xi1>, vector<4xf32>
    %8 = spv.Select %7, %4, %5 : vector<4xi1>, vector<4xf32>
    spv.Return
  }
}