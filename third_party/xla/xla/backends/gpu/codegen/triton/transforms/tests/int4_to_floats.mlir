// RUN: xla-opt --split-input-file --int4-to-packed-int4-rewrite --canonicalize %s | FileCheck %s

tt.func @gemm_fusion_dot_impl(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}) -> (tensor<16x32xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %c32_i64 = arith.constant 32 : i64
  %11 = tt.make_tensor_ptr %arg0, [%c32_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xi4>>
  %12 = tt.advance %11, [%c0_i32, %c0_i32] : <tensor<16x32xi4>>
  %16 = tt.load %12 : !tt.ptr<tensor<16x32xi4>>
  // CHECK: tt.elementwise_inline_asm
  // CHECK: tt.join
  // CHECK: tt.reshape
  // CHECK: arith.extf
  %18 = arith.extsi %16 : tensor<16x32xi4> to tensor<16x32xi8>
  %19 = arith.sitofp %18 : tensor<16x32xi8> to tensor<16x32xf32>
  tt.return %19 : tensor<16x32xf32>
}

// -----

tt.func @gemm_fusion_dot_impl(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}) -> (tensor<16x32xbf16>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %c32_i64 = arith.constant 32 : i64
  %11 = tt.make_tensor_ptr %arg0, [%c32_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xi4>>
  %12 = tt.advance %11, [%c0_i32, %c0_i32] : <tensor<16x32xi4>>
  %16 = tt.load %12 : !tt.ptr<tensor<16x32xi4>>
  // CHECK: tt.elementwise_inline_asm
  // CHECK: tt.join
  // CHECK: tt.reshape
  // CHECK-NOT: arith.extf
  %18 = arith.extsi %16 : tensor<16x32xi4> to tensor<16x32xi8>
  %19 = arith.sitofp %18 : tensor<16x32xi8> to tensor<16x32xbf16>
  tt.return %19 : tensor<16x32xbf16>
}

// -----

tt.func @gemm_fusion_dot_impl(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}) -> (tensor<16x32xf16>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %c32_i64 = arith.constant 32 : i64
  %11 = tt.make_tensor_ptr %arg0, [%c32_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xi4>>
  %12 = tt.advance %11, [%c0_i32, %c0_i32] : <tensor<16x32xi4>>
  %16 = tt.load %12 : !tt.ptr<tensor<16x32xi4>>
  // CHECK: arith.shli
  // CHECK: arith.shrsi
  // CHECK: arith.shrsi
  // CHECK: tt.join
  %18 = arith.extsi %16 : tensor<16x32xi4> to tensor<16x32xi8>
  // CHECK: arith.sitofp
  %19 = arith.sitofp %18 : tensor<16x32xi8> to tensor<16x32xf16>
  tt.return %19 : tensor<16x32xf16>
}
