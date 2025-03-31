// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding | FileCheck %s
// Test that gather/scatter are assigned swizzled encodings

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_gather(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked> ) -> tensor<32x32xi8, #blocked1> {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <tensor<1x32xi8, #[[NVMMA_32]]>>
  // CHECK: tt.descriptor_gather {{.*}} : (!tt.tensordesc<tensor<1x32xi8, #[[NVMMA_32]]>>
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <tensor<1x32xi8>>
  %2 = tt.descriptor_gather %1[%arg3, %c8_i32] : (!tt.tensordesc<tensor<1x32xi8>>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>
  tt.return %2 : tensor<32x32xi8, #blocked1>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_scatter(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %arg4: tensor<32x32xi8, #blocked1>) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <tensor<1x32xi8, #[[NVMMA_32]]>>
  // CHECK: tt.descriptor_scatter {{.*}} : !tt.tensordesc<tensor<1x32xi8, #[[NVMMA_32]]>>, {{.*}}
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <tensor<1x32xi8>>
  tt.descriptor_scatter %1[%arg3, %c8_i32], %arg4 : !tt.tensordesc<tensor<1x32xi8>>, tensor<32xi32, #blocked>, i32, tensor<32x32xi8, #blocked1>
  tt.return
}
}
