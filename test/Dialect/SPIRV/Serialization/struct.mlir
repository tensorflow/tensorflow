// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK: !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Input>
  spv.globalVariable @var0 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Input>

  // CHECK: !spv.ptr<!spv.struct<f32 [0], !spv.struct<f32 [0], !spv.array<16 x f32 [4]> [4]> [4]>, Input>
  spv.globalVariable @var1 bind(0, 2) : !spv.ptr<!spv.struct<f32 [0], !spv.struct<f32 [0], !spv.array<16 x f32 [4]> [4]> [4]>, Input>

  // CHECK: !spv.ptr<!spv.struct<f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38]>, StorageBuffer>
  spv.globalVariable @var2 : !spv.ptr<!spv.struct<f32 [0], i32 [4], f64 [8], i64 [16], f32 [24], i32 [30], f32 [34], i32 [38]>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<!spv.array<128 x !spv.struct<!spv.array<128 x f32 [4]> [0]> [4]> [0]>, StorageBuffer>
  spv.globalVariable @var3 : !spv.ptr<!spv.struct<!spv.array<128 x !spv.struct<!spv.array<128 x f32 [4]> [0]> [4]> [0]>, StorageBuffer>

  // CHECK: !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Input>,
  // CHECK-SAME: !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Output>
  func @kernel_1(%arg0: !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Input>, %arg1: !spv.ptr<!spv.struct<!spv.array<128 x f32 [4]> [0]>, Output>) -> () {
    spv.Return
  }
}
