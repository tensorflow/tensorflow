// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @fmul(%arg0 : f32, %arg1 : f32) {
    // CHECK: {{%.*}}= spv.FMul {{%.*}}, {{%.*}} : f32
    %0 = spv.FMul %arg0, %arg1 : f32
    spv.Return
  }
  func @fadd(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FAdd {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FAdd %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @fdiv(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FDiv {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FDiv %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @fmod(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FMod {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FMod %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @fnegate(%arg0 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FNegate {{%.*}} : vector<4xf32>
    %0 = spv.FNegate %arg0 : vector<4xf32>
    spv.Return
  }
  func @fsub(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FSub {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FSub %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @frem(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) {
    // CHECK: {{%.*}} = spv.FRem {{%.*}}, {{%.*}} : vector<4xf32>
    %0 = spv.FRem %arg0, %arg1 : vector<4xf32>
    spv.Return
  }
  func @iadd(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.IAdd {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.IAdd %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @isub(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.ISub {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.ISub %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @imul(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.IMul {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.IMul %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @udiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.UDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.UDiv %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @umod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.UMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.UMod %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @sdiv(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.SDiv {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SDiv %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @smod(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.SMod {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SMod %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
  func @srem(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
    // CHECK: {{%.*}} = spv.SRem {{%.*}}, {{%.*}} : vector<4xi32>
    %0 = spv.SRem %arg0, %arg1 : vector<4xi32>
    spv.Return
  }
}
