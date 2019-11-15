// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @bitcount(%arg: i32) -> i32 {
    // CHECK: spv.BitCount {{%.*}} : i32
    %0 = spv.BitCount %arg : i32
    spv.ReturnValue %0 : i32
  }
  func @bit_field_insert(%base: vector<3xi32>, %insert: vector<3xi32>, %offset: i32, %count: i16) -> vector<3xi32> {
    // CHECK: {{%.*}} = spv.BitFieldInsert {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i32, i16
    %0 = spv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i32, i16
    spv.ReturnValue %0 : vector<3xi32>
  }
  func @bit_field_s_extract(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
    // CHECK: {{%.*}} = spv.BitFieldSExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
    %0 = spv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
    spv.ReturnValue %0 : vector<3xi32>
  }
  func @bit_field_u_extract(%base: vector<3xi32>, %offset: i8, %count: i8) -> vector<3xi32> {
    // CHECK: {{%.*}} = spv.BitFieldUExtract {{%.*}}, {{%.*}}, {{%.*}} : vector<3xi32>, i8, i8
    %0 = spv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
    spv.ReturnValue %0 : vector<3xi32>
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
