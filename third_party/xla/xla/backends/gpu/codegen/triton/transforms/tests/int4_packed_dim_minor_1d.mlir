// RUN: xla-opt --int4-to-packed-int4-rewrite --canonicalize -- %s | FileCheck --dump-input=fail %s

module {
  tt.func @minor_1d(%arg0: !tt.ptr<i4> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i64 = arith.constant 128 : i64
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0> : tensor<64x64xi8>

                    %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xi4>>
// CHECK:           %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xi8>>

                    %1 = tt.advance %0, [%c0_i32, %c64_i32] : <tensor<64x64xi4>>
// CHECK-NEXT:      %1 = tt.advance %0, [%c0_i32, %c64_i32] : <tensor<32x64xi8>>

                    %2:2 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg3 = %1, %arg4 = %cst) -> (!tt.ptr<tensor<64x64xi4>>, tensor<64x64xi8>)  : i32 {
// CHECK-NEXT:      %2:2 = scf.for %arg2 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg3 = %1, %arg4 = %cst_0) -> (!tt.ptr<tensor<32x64xi8>>, tensor<64x64xi8>)  : i32 {

                    %4 = tt.load %arg3 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<64x64xi4>>
// CHECK-NEXT:      %4 = tt.load %arg3 {boundaryCheck = array<i32: 1>, padding = 1 : i32} : !tt.ptr<tensor<32x64xi8>>

                    %5 = tt.advance %arg3, [%c64_i32, %c0_i32] : <tensor<64x64xi4>>
// CHECK-NEXT:      %5 = tt.advance %arg3, [%c32_i32, %c0_i32] : <tensor<32x64xi8>>

                    %6 = arith.extsi %4 : tensor<64x64xi4> to tensor<64x64xi8>
// CHECK-NEXT:      %6 = arith.shli %4, %cst : tensor<32x64xi8>
// CHECK-NEXT:      %7 = arith.shrsi %6, %cst : tensor<32x64xi8>
// CHECK-NEXT:      %8 = arith.shrsi %4, %cst : tensor<32x64xi8>
// CHECK-NEXT:      %9 = tt.join %7, %8 : tensor<32x64xi8> -> tensor<32x64x2xi8>
// CHECK-NEXT:      %10 = tt.trans %9 {order = array<i32: 0, 2, 1>} : tensor<32x64x2xi8> -> tensor<32x2x64xi8>
// CHECK-NEXT:      %11 = tt.reshape %10 : tensor<32x2x64xi8> -> tensor<64x64xi8>

                    scf.yield %5, %6 : !tt.ptr<tensor<64x64xi4>>, tensor<64x64xi8>
// CHECK-NEXT:      scf.yield %5, %11 : !tt.ptr<tensor<32x64xi8>>, tensor<64x64xi8>
    }
    %3 = tt.make_tensor_ptr %arg1, [%c128_i64, %c1_i64], [%c1_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xi8>>
    tt.store %3, %2#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x64xi8>>
    tt.return
  }
}


