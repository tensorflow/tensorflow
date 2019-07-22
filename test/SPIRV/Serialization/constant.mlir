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
  }
  return
}
