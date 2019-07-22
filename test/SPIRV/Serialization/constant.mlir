// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_module() -> () {
  spv.module "Logical" "GLSL450" {

    // Bool
    // CHECK: spv.constant true
    %0 = spv.constant true
    // CHECK: spv.constant false
    %1 = spv.constant false

    // 32-bit integer
    // CHECK: spv.constant 0 : i32
    %2 = spv.constant  0 : i32
    // CHECK: spv.constant 10 : i32
    %3 = spv.constant 10 : i32
    // CHECK: spv.constant -5 : i32
    %4 = spv.constant -5 : i32

    // 64-bit integer
    // CHECK: spv.constant 4294967296 : i64
    %5 = spv.constant           4294967296 : i64 //  2^32
    // CHECK: spv.constant -4294967296 : i64
    %6 = spv.constant          -4294967296 : i64 // -2^32
    // CHECK: spv.constant 9223372036854775807 : i64
    %7 = spv.constant  9223372036854775807 : i64 //  2^63 - 1
    // CHECK: spv.constant -9223372036854775808 : i64
    %8 = spv.constant -9223372036854775808 : i64 // -2^63

    // 16-bit integer
    // CHECK: spv.constant -32768 : i16
    %9 = spv.constant -32768 : i16 // -2^15
    // CHECK: spv.constant 32767 : i16
    %10 = spv.constant 32767 : i16 //  2^15 - 1

    // float
    // CHECK: spv.constant 0.000000e+00 : f32
    %11 = spv.constant 0. : f32
    // CHECK: spv.constant 1.000000e+00 : f32
    %12 = spv.constant 1. : f32
    // CHECK: spv.constant -0.000000e+00 : f32
    %13 = spv.constant -0. : f32
    // CHECK: spv.constant -1.000000e+00 : f32
    %14 = spv.constant -1. : f32
    // CHECK: spv.constant 7.500000e-01 : f32
    %15 = spv.constant 0.75 : f32
    // CHECK: spv.constant -2.500000e-01 : f32
    %16 = spv.constant -0.25 : f32

    // double
    // TODO(antiagainst): test range boundary values
    // CHECK: spv.constant 1.024000e+03 : f64
    %17 = spv.constant 1024. : f64
    // CHECK: spv.constant -1.024000e+03 : f64
    %18 = spv.constant -1024. : f64

    // half
    // CHECK: spv.constant 5.120000e+02 : f16
    %19 = spv.constant 512. : f16
    // CHECK: spv.constant -5.120000e+02 : f16
    %20 = spv.constant -512. : f16

    // Bool vector
    // CHECK: spv.constant dense<false> : vector<2xi1>
    %21 = spv.constant dense<false> : vector<2xi1>
    // CHECK: spv.constant dense<[true, true, true]> : vector<3xi1>
    %22 = spv.constant dense<true> : vector<3xi1>
    // CHECK: spv.constant dense<[false, true]> : vector<2xi1>
    %23 = spv.constant dense<[false, true]> : vector<2xi1>

    // Integer vector
    // CHECK: spv.constant dense<0> : vector<2xi32>
    %24 = spv.constant dense<0> : vector<2xi32>
    // CHECK: spv.constant dense<1> : vector<3xi32>
    %25 = spv.constant dense<1> : vector<3xi32>
    // CHECK: spv.constant dense<[2, -3, 4]> : vector<3xi32>
    %26 = spv.constant dense<[2, -3, 4]> : vector<3xi32>

    // Fp vector
    // CHECK: spv.constant dense<0.000000e+00> : vector<4xf32>
    %27 = spv.constant dense<0.> : vector<4xf32>
    // CHECK: spv.constant dense<-1.500000e+01> : vector<4xf32>
    %28 = spv.constant dense<-15.> : vector<4xf32>
    // CHECK: spv.constant dense<[7.500000e-01, -2.500000e-01, 1.000000e+01, 4.200000e+01]> : vector<4xf32>
    %29 = spv.constant dense<[0.75, -0.25, 10., 42.]> : vector<4xf32>

    // Array
    // CHECK: spv.constant [dense<3.000000e+00> : vector<2xf32>, dense<[4.000000e+00, 5.000000e+00]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
    %30 = spv.constant [dense<3.0> : vector<2xf32>, dense<[4., 5.]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
  }
  return
}
