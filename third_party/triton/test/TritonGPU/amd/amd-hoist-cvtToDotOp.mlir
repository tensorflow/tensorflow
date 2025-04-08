// RUN: triton-opt %s -split-input-file -tritonamdgpu-hoist-layout-conversions | FileCheck %s

// Hoist convert_layout out of the loop since the defining op of the src is out of the loop

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
// CHECK-LABEL: hoist_cvtToDotOp
//       CHECK: %[[AF16:.*]] = arith.truncf
//  CHECK-NEXT: %[[opA:.*]] = ttg.convert_layout %[[AF16]]
//  CHECK-NEXT: scf.for
//       CHECK: tt.dot %[[opA]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @hoist_cvtToDotOp(%opA: tensor<256x128xf32, #blocked>, %opB: tensor<128x256xf16, #dotOp1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = arith.truncf %opA : tensor<256x128xf32, #blocked> to tensor<256x128xf16, #blocked>
    %1:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %2 = ttg.convert_layout %0 : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #dotOp0>
      %3 = tt.dot %2, %opB, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %1#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Keep convert_layout inside the loop since the defining op of the src is inside the loop

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
// CHECK-LABEL: defOp_in_loop
//       CHECK: scf.for
//       CHECK: %[[AF16:.*]] = arith.truncf
//  CHECK-NEXT: %[[opA:.*]] = ttg.convert_layout %[[AF16]]
//       CHECK: tt.dot %[[opA]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @defOp_in_loop(%opA: tensor<256x128xf32, #blocked>, %opB: tensor<128x256xf16, #dotOp1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %1:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %0 = arith.truncf %opA : tensor<256x128xf32, #blocked> to tensor<256x128xf16, #blocked>
      %2 = ttg.convert_layout %0 : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #dotOp0>
      %3 = tt.dot %2, %opB, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %1#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Keep convert_layout inside the loop since the defining op is a block argument of the loop

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
// CHECK-LABEL: defOp_blockArg
//       CHECK: scf.for
//  CHECK-NEXT: %[[opA:.*]] = ttg.convert_layout
//       CHECK: tt.dot %[[opA]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @defOp_blockArg(%opA: tensor<256x128xf16, #blocked>, %opB: tensor<128x256xf16, #dotOp1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %1:2 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst, %arg2 = %opA) -> (tensor<256x256xf32, #mma>, tensor<256x128xf16, #blocked>) : i32 {
      %2 = ttg.convert_layout %arg2 : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #dotOp0>
      %3 = tt.dot %2, %opB, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      scf.yield %3, %arg2 : tensor<256x256xf32, #mma>, tensor<256x128xf16, #blocked>
    }
    tt.store %C_ptr, %1#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}
