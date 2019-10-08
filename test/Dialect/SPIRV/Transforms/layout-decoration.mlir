// RUN: mlir-opt -decorate-spirv-composite-type-layout -split-input-file -verify-diagnostics %s -o - | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @var0 bind(0, 1) : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
  spv.globalVariable @var0 bind(0,1) : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>

  // CHECK: spv.globalVariable @var1 bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<64 x i32 [4]> [0], f32 [256]>, StorageBuffer>
  spv.globalVariable @var1 bind(0,2) : !spv.ptr<!spv.struct<!spv.array<64xi32>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var2 bind(1, 0) : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64 x i32 [4]> [0], f32 [256]> [0], i32 [260]>, StorageBuffer>
  spv.globalVariable @var2 bind(1,0) : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64xi32>, f32>, i32>, StorageBuffer>

  // CHECK: spv.globalVariable @var3 : !spv.ptr<!spv.struct<!spv.array<16 x !spv.struct<f32 [0], f32 [4], !spv.array<16 x f32 [4]> [8]> [72]> [0], f32 [1152]>, StorageBuffer>
  spv.globalVariable @var3 : !spv.ptr<!spv.struct<!spv.array<16x!spv.struct<f32, f32, !spv.array<16xf32>>>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var4 bind(1, 2) : !spv.ptr<!spv.struct<!spv.struct<!spv.struct<i1 [0], i8 [1], i16 [2], i32 [4], i64 [8]> [0], f32 [16], i1 [20]> [0], i1 [24]>, StorageBuffer>
  spv.globalVariable @var4 bind(1,2) : !spv.ptr<!spv.struct<!spv.struct<!spv.struct<i1, i8, i16, i32, i64>, f32, i1>, i1>, StorageBuffer>

  // CHECK: spv.globalVariable @var5 bind(1, 3) : !spv.ptr<!spv.struct<!spv.array<256 x f32 [4]> [0]>, StorageBuffer>
  spv.globalVariable @var5 bind(1,3) : !spv.ptr<!spv.struct<!spv.array<256xf32>>, StorageBuffer>

  func @kernel() -> () {
    %c0 = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv._address_of @var0 : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
    %0 = spv._address_of @var0 : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>
    // CHECK:  {{%.*}} = spv.AccessChain {{%.*}}[{{%.*}}] : !spv.ptr<!spv.struct<i32 [0], !spv.struct<f32 [0], i32 [4]> [4], f32 [12]>, Uniform>
    %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<i32, !spv.struct<f32, i32>, f32>, Uniform>
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @var0 : !spv.ptr<!spv.struct<!spv.struct<!spv.struct<!spv.struct<!spv.struct<i1 [0], i1 [1], f64 [8]> [0], i1 [16]> [0], i1 [24]> [0], i1 [32]> [0], i1 [40]>, Uniform>
  spv.globalVariable @var0 : !spv.ptr<!spv.struct<!spv.struct<!spv.struct<!spv.struct<!spv.struct<i1, i1, f64>, i1>, i1>, i1>, i1>, Uniform>

  // CHECK: spv.globalVariable @var1 : !spv.ptr<!spv.struct<!spv.struct<i16 [0], !spv.struct<i1 [0], f64 [8]> [8], f32 [24]> [0], f32 [32]>, Uniform>
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<!spv.struct<i16, !spv.struct<i1, f64>, f32>, f32>, Uniform>

  // CHECK: spv.globalVariable @var2 : !spv.ptr<!spv.struct<!spv.struct<i16 [0], !spv.struct<i1 [0], !spv.array<16 x !spv.array<16 x i64 [8]> [128]> [8]> [8], f32 [2064]> [0], f32 [2072]>, Uniform>
  spv.globalVariable @var2 : !spv.ptr<!spv.struct<!spv.struct<i16, !spv.struct<i1, !spv.array<16x!spv.array<16xi64>>>, f32>, f32>, Uniform>

  // CHECK: spv.globalVariable @var3 : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64 x i64 [8]> [0], i1 [512]> [0], i1 [520]>, Uniform>
  spv.globalVariable @var3 : !spv.ptr<!spv.struct<!spv.struct<!spv.array<64xi64>, i1>, i1>, Uniform>

  // CHECK: spv.globalVariable @var4 : !spv.ptr<!spv.struct<i1 [0], !spv.struct<i64 [0], i1 [8], i1 [9], i1 [10], i1 [11]> [8], i1 [24]>, Uniform>
  spv.globalVariable @var4 : !spv.ptr<!spv.struct<i1, !spv.struct<i64, i1, i1, i1, i1>, i1>, Uniform>

  // CHECK: spv.globalVariable @var5 : !spv.ptr<!spv.struct<i1 [0], !spv.struct<i1 [0], i1 [1], i1 [2], i1 [3], i64 [8]> [8], i1 [24]>, Uniform>
  spv.globalVariable @var5 : !spv.ptr<!spv.struct<i1, !spv.struct<i1, i1, i1, i1, i64>, i1>, Uniform>

  // CHECK: spv.globalVariable @var6 : !spv.ptr<!spv.struct<i1 [0], !spv.struct<i64 [0], i32 [8], i16 [12], i8 [14], i1 [15]> [8], i1 [24]>, Uniform>
  spv.globalVariable @var6 : !spv.ptr<!spv.struct<i1, !spv.struct<i64, i32, i16, i8, i1>, i1>, Uniform>

  // CHECK: spv.globalVariable @var7 : !spv.ptr<!spv.struct<i1 [0], !spv.struct<!spv.struct<i1 [0], i64 [8]> [0], i1 [16]> [8], i1 [32]>, Uniform>
  spv.globalVariable @var7 : !spv.ptr<!spv.struct<i1, !spv.struct<!spv.struct<i1, i64>, i1>, i1>, Uniform>
}

// -----

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @var0 : !spv.ptr<!spv.struct<vector<2xi32> [0], f32 [8]>, StorageBuffer>
  spv.globalVariable @var0 : !spv.ptr<!spv.struct<vector<2xi32>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var1 : !spv.ptr<!spv.struct<vector<3xi32> [0], f32 [12]>, StorageBuffer>
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<vector<3xi32>, f32>, StorageBuffer>

  // CHECK: spv.globalVariable @var2 : !spv.ptr<!spv.struct<vector<4xi32> [0], f32 [16]>, StorageBuffer>
  spv.globalVariable @var2 : !spv.ptr<!spv.struct<vector<4xi32>, f32>, StorageBuffer>
}

// -----

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @emptyStructAsMember : !spv.ptr<!spv.struct<!spv.struct<> [0]>, StorageBuffer>
  spv.globalVariable @emptyStructAsMember : !spv.ptr<!spv.struct<!spv.struct<>>, StorageBuffer>

  // CHECK: spv.globalVariable @arrayType : !spv.ptr<!spv.array<4 x !spv.array<4 x f32>>, StorageBuffer>
  spv.globalVariable @arrayType : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, StorageBuffer>

  // CHECK: spv.globalVariable @InputStorage : !spv.ptr<!spv.struct<!spv.array<256 x f32>>, Input>
  spv.globalVariable @InputStorage : !spv.ptr<!spv.struct<!spv.array<256xf32>>, Input>

  // CHECK: spv.globalVariable @customLayout : !spv.ptr<!spv.struct<f32 [256], i32 [512]>, Uniform>
  spv.globalVariable @customLayout : !spv.ptr<!spv.struct<f32 [256], i32 [512]>, Uniform>

  // CHECK:  spv.globalVariable @emptyStruct : !spv.ptr<!spv.struct<>, Uniform>
  spv.globalVariable @emptyStruct : !spv.ptr<!spv.struct<>, Uniform>
}

// -----

spv.module "Logical" "GLSL450" {
  // CHECK: spv.globalVariable @var0 : !spv.ptr<!spv.struct<i32 [0]>, PushConstant>
  spv.globalVariable @var0 : !spv.ptr<!spv.struct<i32>, PushConstant>
  // CHECK: spv.globalVariable @var1 : !spv.ptr<!spv.struct<i32 [0]>, PhysicalStorageBuffer>
  spv.globalVariable @var1 : !spv.ptr<!spv.struct<i32>, PhysicalStorageBuffer>
}
