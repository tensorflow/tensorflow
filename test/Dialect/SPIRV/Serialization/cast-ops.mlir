// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @bit_cast(%arg0 : f32) {
    // CHECK: {{%.*}} = spv.Bitcast {{%.*}} : f32 to i32
    %0 = spv.Bitcast %arg0 : f32 to i32
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @convert_f_to_s(%arg0 : f32) -> i32 {
    // CHECK: {{%.*}} = spv.ConvertFToS {{%.*}} : f32 to i32
    %0 = spv.ConvertFToS %arg0 : f32 to i32
    spv.ReturnValue %0 : i32
  }
  func @convert_f_to_u(%arg0 : f32) -> i32 {
    // CHECK: {{%.*}} = spv.ConvertFToU {{%.*}} : f32 to i32
    %0 = spv.ConvertFToU %arg0 : f32 to i32
    spv.ReturnValue %0 : i32
  }
  func @convert_s_to_f(%arg0 : i32) -> f32 {
    // CHECK: {{%.*}} = spv.ConvertSToF {{%.*}} : i32 to f32
    %0 = spv.ConvertSToF %arg0 : i32 to f32
    spv.ReturnValue %0 : f32
  }
  func @convert_u_to_f(%arg0 : i32) -> f32 {
    // CHECK: {{%.*}} = spv.ConvertUToF {{%.*}} : i32 to f32
    %0 = spv.ConvertUToF %arg0 : i32 to f32
    spv.ReturnValue %0 : f32
  }
  func @f_convert(%arg0 : f32) -> f64 {
    // CHECK: {{%.*}} = spv.FConvert {{%.*}} : f32 to f64
    %0 = spv.FConvert %arg0 : f32 to f64
    spv.ReturnValue %0 : f64
  }
  func @s_convert(%arg0 : i32) -> i64 {
    // CHECK: {{%.*}} = spv.SConvert {{%.*}} : i32 to i64
    %0 = spv.SConvert %arg0 : i32 to i64
    spv.ReturnValue %0 : i64
  }
  func @u_convert(%arg0 : i32) -> i64 {
    // CHECK: {{%.*}} = spv.UConvert {{%.*}} : i32 to i64
    %0 = spv.UConvert %arg0 : i32 to i64
    spv.ReturnValue %0 : i64
  }
}
