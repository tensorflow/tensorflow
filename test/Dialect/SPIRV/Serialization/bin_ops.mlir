// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_bin_ops() -> () {
  spv.module "Logical" "VulkanKHR" {
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
  return
}
