// RUN: triton-opt %s -mlir-disable-threading -test-print-alias -verify-diagnostics -o /dev/null

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B_DOT = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// There shouldn't be any aliasing with the dot op encoding.
tt.func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A_DOT>
    %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B_DOT>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A_DOT> * tensor<32x128xf16, #B_DOT> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return
}

tt.func @alloc(%A : !tt.ptr<f16>) {
  // expected-remark @below {{%0 -> %0}}
  %cst2 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

tt.func @alloc_init(%A : !tt.ptr<f16>) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #AL>
  // expected-remark @below {{%0 -> %0}}
  %cst1 = ttg.local_alloc %cst0 : (tensor<16x16xf16, #AL>) -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory>
  tt.return
}

tt.func @trans(%A : !tt.ptr<f16>) {
  // expected-remark @below {{%0 -> %0}}
  %tensor = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %0}}
  %b = ttg.memdesc_trans %tensor {order=array<i32: 1,0>} : !ttg.memdesc<16x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x16xf16, #A_SHARED_T, #ttg.shared_memory, mutable>
  tt.return
}

tt.func @subview(%A : !ttg.memdesc<1x16x16xf16, #A_SHARED, #ttg.shared_memory>) {
  %index = arith.constant 0 : i32
  // expected-remark @below {{%0 -> %0}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<1x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %0}}
  %cst1 = ttg.memdesc_subview %a[%index, %index, %index] : !ttg.memdesc<1x16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

tt.func @if_alias(%i1 : i1) {
  // expected-remark @below {{%0 -> %0}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %1}}
  %b = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%2 -> %0,%1}}
  %cst2 = scf.if %i1 -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable> {
    scf.yield %a : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  } else {
    scf.yield %b : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  }
  tt.return
}

tt.func @for(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  // expected-remark @below {{%0 -> %0}}
  %a = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %1}}
  %b = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%2 -> %2}}
  %c = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%arg6 -> %0}}
  // expected-remark @below {{%arg7 -> %1}}
  // expected-remark @below {{%arg8 -> %2}}
  // expected-remark @below {{%3#0 -> %0,%1}}
  // expected-remark @below {{%3#1 -> %0,%1}}
  // expected-remark @below {{%3#2 -> %0,%1,%2}}
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a, %b_shared = %b, %c_shared = %c) ->
  (!ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>) {
    scf.yield %b_shared, %a_shared, %a_shared : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  }
  tt.return
}

tt.func @for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // expected-remark @below {{%0 -> %0}}
  %a_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %1}}
  %b_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%2 -> %2}}
  %c_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%arg7 -> %0}}
  // expected-remark @below {{%arg8 -> %1}}
  // expected-remark @below {{%arg9 -> %2}}
  // expected-remark @below {{%3#0 -> %0,%1}}
  // expected-remark @below {{%3#1 -> %0,%1}}
  // expected-remark @below {{%3#2 -> %0,%1,%2}}
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) ->
  (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>) {
    scf.if %i1 {
      %index = arith.constant 8 : i32
      // expected-remark @below {{%4 -> %0,%1}}
      %cst0 = ttg.memdesc_subview %a_shared[%index, %index] : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<32xf16, #A_SHARED, #ttg.shared_memory, mutable>
      scf.yield
    }
    scf.yield %b_shared, %a_shared, %a_shared : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  }
  tt.return
}

tt.func @for_for_if(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>, %i1 : i1) {
  // expected-remark @below {{%0 -> %0}}
  %a_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %1}}
  %b_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%2 -> %2}}
  %c_shared_init = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%arg7 -> %0}}
  // expected-remark @below {{%arg8 -> %1}}
  // expected-remark @below {{%arg9 -> %2}}
  // expected-remark @below {{%3#0 -> %0}}
  // expected-remark @below {{%3#1 -> %1}}
  // expected-remark @below {{%3#2 -> %2,%6,%6}}
  %a_shared, %b_shared, %c_shared = scf.for %iv = %lb to %ub step %step iter_args(%a_shared = %a_shared_init, %b_shared = %b_shared_init, %c_shared = %c_shared_init) ->
  (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>) {
    // expected-remark @below {{%arg11 -> %2,%6,%6}}
    // expected-remark @below {{%4 -> %2,%6,%6}}
    %c_shared_next = scf.for %jv = %lb to %ub step %step iter_args(%c_shared_next = %c_shared) -> (!ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>) {
      // expected-remark @below {{%5 -> %6,%6}}
      %c_shared_next_next = scf.if %i1 -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> {
        // expected-remark @below {{%6 -> %6}}
        %cst0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
        scf.yield %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
      } else {
        // expected-remark @below {{%6 -> %6}}
        %cst0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
        scf.yield %cst0 : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
      }
      scf.yield %c_shared_next_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
    }
    scf.yield %a_shared, %b_shared, %c_shared_next : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  }
  tt.return
}

tt.func @cf_for(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) {
  %idx = arith.constant 0 : i32
  // expected-remark @below {{%0 -> %0}}
  %cst = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%1 -> %1}}
  %cst_0 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // expected-remark @below {{%2 -> %0}}
  %0 = ttg.memdesc_subview %cst[%idx, %idx] : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  gpu.barrier
  // expected-remark @below {{%3 -> %3}}
  %cst_1 = ttg.local_alloc : () -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  cf.br ^bb1(%arg0, %cst, %cst_0, %cst_1 : index, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>)
^bb1(%1: index, %2: !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, %3: !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, %4: !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>):  // 2 preds: ^bb0, ^bb2
  %5 = arith.cmpi slt, %1, %arg1 : index
  // expected-remark @below {{%5 -> %0,%1,%3}}
  // expected-remark @below {{%6 -> %0,%1,%3}}
  // expected-remark @below {{%7 -> %0,%1,%3}}
  cf.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  gpu.barrier
  %8 = arith.addi %1, %arg2 : index
  cf.br ^bb1(%8, %4, %2, %3 : index, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>, !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>)
^bb3:  // pred: ^bb1
  gpu.barrier
  // expected-remark @below {{%10 -> %0}}
  %9 = ttg.memdesc_subview %0[%idx, %idx] : !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable> -> !ttg.memdesc<128x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  tt.return
}

}  // module
