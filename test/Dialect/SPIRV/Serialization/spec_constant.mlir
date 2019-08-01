// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_module() -> () {
  spv.module "Logical" "GLSL450" {

    // CHECK: spv.constant spec true
    %0 = spv.constant spec true
    // CHECK: spv.constant spec false
    %1 = spv.constant spec false

    // CHECK: spv.constant spec -5 : i32
    %2 = spv.constant spec -5 : i32

    // CHECK: spv.constant spec 1.000000e+00 : f32
    %3 = spv.constant spec 1. : f32

    // Bool vector
    // CHECK: spv.constant spec dense<false> : vector<2xi1>
    %4 = spv.constant spec dense<false> : vector<2xi1>
    // CHECK: spv.constant spec dense<[true, true, true]> : vector<3xi1>
    %5 = spv.constant spec dense<true> : vector<3xi1>
    // CHECK: spv.constant spec dense<[false, true]> : vector<2xi1>
    %6 = spv.constant spec dense<[false, true]> : vector<2xi1>

    // Integer vector
    // CHECK: spv.constant spec dense<0> : vector<2xi32>
    %7 = spv.constant spec dense<0> : vector<2xi32>
    // CHECK: spv.constant spec dense<1> : vector<3xi32>
    %8 = spv.constant spec dense<1> : vector<3xi32>
    // CHECK: spv.constant spec dense<[2, -3, 4]> : vector<3xi32>
    %9 = spv.constant spec dense<[2, -3, 4]> : vector<3xi32>

    // Fp vector
    // CHECK: spv.constant spec dense<0.000000e+00> : vector<4xf32>
    %10 = spv.constant spec dense<0.> : vector<4xf32>
    // CHECK: spv.constant spec dense<-1.500000e+01> : vector<4xf32>
    %11 = spv.constant spec dense<-15.> : vector<4xf32>
    // CHECK: spv.constant spec dense<[7.500000e-01, -2.500000e-01, 1.000000e+01, 4.200000e+01]> : vector<4xf32>
    %12 = spv.constant spec dense<[0.75, -0.25, 10., 42.]> : vector<4xf32>

    // Array
    // CHECK: spv.constant spec [dense<3.000000e+00> : vector<2xf32>, dense<[4.000000e+00, 5.000000e+00]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
    %13 = spv.constant spec [dense<3.0> : vector<2xf32>, dense<[4., 5.]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
  }
  return
}

