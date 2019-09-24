// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @iequal_scalar(%arg0: i32, %arg1: i32)  {
    // CHECK: {{.*}} = spv.IEqual {{.*}}, {{.*}} : i32
    %0 = spv.IEqual %arg0, %arg1 : i32
    spv.Return
  }
  func @inotequal_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.INotEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.INotEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @sgt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.SGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SGreaterThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @sge_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.SGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @slt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.SLessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SLessThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @slte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.SLessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.SLessThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @ugt_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.UGreaterThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.UGreaterThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @ugte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.UGreaterThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.UGreaterThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @ult_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
    // CHECK: {{.*}} = spv.ULessThan {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.ULessThan %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @ulte_vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>)  {
    // CHECK: {{.*}} = spv.ULessThanEqual {{.*}}, {{.*}} : vector<4xi32>
    %0 = spv.ULessThanEqual %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
}

// -----

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
