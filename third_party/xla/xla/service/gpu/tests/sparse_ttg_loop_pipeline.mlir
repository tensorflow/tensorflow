// RUN: xla-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliced = #triton_gpu.slice<{parent=#blocked, dim=0}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth=2}>
#dot_meta_enc = #triton_gpu.sparse_dot_meta<{parent=#mma}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func @sparse_dot_loop(%lb : index, %ub : index, %step : index,
        %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
        %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
        %A_meta : !tt.ptr<i16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    // CHECK-COUNT-6: triton_gpu.async_copy_global_to_local
    // CHECK: triton_gpu.async_wait {{.+}}, {{.+}} {num = 3 : i32}
    %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #sliced>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #sliced> -> tensor<1x32xi32, #blocked>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #blocked> -> tensor<128x32xi32, #blocked>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>

    %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #sliced>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #sliced> -> tensor<1x128xi32, #blocked>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>

    %meta_ptr_splat = tt.splat %A_meta : !tt.ptr<i16> -> tensor<128x4x!tt.ptr<i16>, #blocked>
    %meta_tmp0 = tt.make_range {end = 4: i32, start = 0: i32} : tensor<4xi32, #sliced>
    %meta_tmp1 = tt.expand_dims %meta_tmp0 {axis = 0 : i32} : tensor<4xi32, #sliced> -> tensor<1x4xi32, #blocked>
    %meta_offs = tt.broadcast %meta_tmp1 : tensor<1x4xi32, #blocked> -> tensor<128x4xi32, #blocked>
    %meta_ptr_init = tt.addptr %meta_ptr_splat, %meta_offs : tensor<128x4x!tt.ptr<i16>, #blocked>, tensor<128x4xi32, #blocked>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #blocked>
    %b_off = arith.constant dense<4> : tensor<64x128xi32, #blocked>
    %meta_off = arith.constant dense<4> : tensor<128x4xi32, #blocked>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>

    // CHECK: scf.for
    %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %c = %c_init, %meta_ptr = %meta_ptr_init)
        -> (tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, tensor<128x4x!tt.ptr<i16>, #blocked>) {
      // CHECK-COUNT-3: triton_gpu.local_load
      // CHECK: triton_xla.sparse_dot
      // CHECK-COUNT-3: triton_gpu.async_copy_global_to_local
      %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16>, #blocked>
      %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #dot_operand_a>
      %b_ = tt.load %b_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128x!tt.ptr<f16>, #blocked>
      %b = triton_gpu.convert_layout %b_ : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #dot_operand_b>
      %meta_ = tt.load %meta_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x4x!tt.ptr<i16>, #blocked>
      %meta = triton_gpu.convert_layout %meta_ : tensor<128x4xi16, #blocked> -> tensor<128x4xi16, #dot_meta_enc>
      %d = triton_xla.sparse_dot %a, %b, %c, %meta : tensor<128x32xf16, #dot_operand_a> meta tensor<128x4xi16, #dot_meta_enc> * tensor<64x128xf16, #dot_operand_b> -> tensor<128x128xf32, #mma>

      %a_ptr_next = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<128x32xi32, #blocked>
      %b_ptr_next = tt.addptr %b_ptr, %b_off : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
      %meta_ptr_next = tt.addptr %meta_ptr, %meta_off : tensor<128x4x!tt.ptr<i16>, #blocked>, tensor<128x4xi32, #blocked>
      scf.yield %a_ptr_next, %b_ptr_next, %d, %meta_ptr_next : tensor<128x32x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, tensor<128x4x!tt.ptr<i16>, #blocked>
    }
    tt.return %loop#2: tensor<128x128xf32, #mma>
  }
}
