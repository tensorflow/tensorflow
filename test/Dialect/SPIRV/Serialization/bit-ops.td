// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @bitcount(%arg: i32) -> i32 {
    // CHECK: spv.BitCount {{%.*}} : i32
    %0 = spv.BitCount %arg : i32
    spv.ReturnValue %0 : i32
  }
  func @bitreverse(%arg: i32) -> i32 {
    // CHECK: spv.BitReverse {{%.*}} : i32
    %0 = spv.BitReverse %arg : i32
    spv.ReturnValue %0 : i32
  }
  func @not(%arg: i32) -> i32 {
    // CHECK: spv.Not {{%.*}} : i32
    %0 = spv.Not %arg : i32
    spv.ReturnValue %0 : i32
  }
  func @shift_left_logical(%arg0: i32, %arg1 : i16) -> i32 {
    // CHECK: {{%.*}} = spv.ShiftLeftLogical {{%.*}}, {{%.*}} : i32, i16
    %0 = spv.ShiftLeftLogical %arg0, %arg1: i32, i16
    spv.ReturnValue %0 : i32
  }
  func @shift_right_aritmethic(%arg0: vector<4xi32>, %arg1 : vector<4xi8>) -> vector<4xi32> {
    // CHECK: {{%.*}} = spv.ShiftRightArithmetic {{%.*}}, {{%.*}} : vector<4xi32>, vector<4xi8>
    %0 = spv.ShiftRightArithmetic %arg0, %arg1: vector<4xi32>, vector<4xi8>
    spv.ReturnValue %0 : vector<4xi32>
  }
  func @shift_right_logical(%arg0: vector<2xi32>, %arg1 : vector<2xi8>) -> vector<2xi32> {
    // CHECK: {{%.*}} = spv.ShiftRightLogical {{%.*}}, {{%.*}} : vector<2xi32>, vector<2xi8>
    %0 = spv.ShiftRightLogical %arg0, %arg1: vector<2xi32>, vector<2xi8>
    spv.ReturnValue %0 : vector<2xi32>
  }
}
