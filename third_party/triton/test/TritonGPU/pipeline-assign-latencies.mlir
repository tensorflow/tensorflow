// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-assign-latencies=num-stages=3 -canonicalize | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @default_stages
tt.func @default_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @small_load
// We should *not* assign latency to the load of b_ptr.
tt.func @small_load(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}}
    // CHECK-NOT: tt.latency
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @load_into_shared
tt.func @load_into_shared(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @load_into_lt_4b
tt.func @load_into_lt_4b(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // Do not pipeline if cp.async would read less than 4 consecutive bytes
    // CHECK: tt.load
    // CHECK-NOT: {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load
tt.func @indirect_load(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @mixed_loads
tt.func @mixed_loads(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @per_loop_stages
tt.func @per_loop_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> (tensor<128x128xf32, #C>, tensor<128x128xf32, #C>) {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop_cust_stages:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 4 : i32}

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop_cust_stages#2, %loop#2: tensor<128x128xf32, #C>, tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_cust_stages
tt.func @indirect_load_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 5 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_few_stages
tt.func @indirect_load_few_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 2 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @non_dot_pipeline
tt.func @non_dot_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @no_pipeline
tt.func @no_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  }
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#2: tensor<128x128xf32, #C>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma
tt.func @tc_gen5_mma(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma
tt.func @tc_gen5_mma(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_defined_outside1
tt.func @tc_gen5_mma_defined_outside1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_defined_outside2
tt.func @tc_gen5_mma_defined_outside2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand1
tt.func @tc_gen5_mma_non_load_operand1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = "producer"() : () -> tensor<128x128xf16, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand2
tt.func @tc_gen5_mma_non_load_operand2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = "producer"() : () -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled
tt.func @tc_gen5_mma_scaled(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>,
                  %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %B_sc_sh = ttg.local_alloc %B_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled_tmem_scales
tt.func @tc_gen5_mma_scaled_tmem_scales(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>,
                  %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>,
                  %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> () {
  %true = arith.constant true
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %B_sc_tm = ttng.tmem_alloc %B_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x328x4xi8, #tmem_scales, #ttng.tensor_memory>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}
    // CHECK-NOT: tt.latency
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_tm, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x328x4xi8, #tmem_scales, #ttng.tensor_memory>, i1, i1) -> ()
    "use"(%acc_tm) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[32, 0], [64, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @block_scale_mxfp_matmul
  tt.func public @block_scale_mxfp_matmul(%lb : index, %ub : index, %step : index, %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked4> {
    %true = arith.constant true
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked4>
    %incr_A = arith.constant dense<4> : tensor<128x256xi32, #blocked>
    %incr_B = arith.constant dense<4> : tensor<256x128xi32, #blocked1>
    %incr_scale = arith.constant dense<4> : tensor<1x2x32x4x4xi32, #blocked2>

    %arg0_splat = tt.splat %arg0: !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
    %arg1_splat = tt.splat %arg1: !tt.ptr<f8E5M2> -> tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
    %arg3_splat = tt.splat %arg3: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %arg4_splat = tt.splat %arg4: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

    %76 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %77 = tt.expand_dims %76 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %79 = tt.broadcast %77 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %arg0_init = tt.addptr %arg0_splat, %79 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>

    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %88 = tt.broadcast %84 : tensor<1x128xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %arg1_init = tt.addptr %arg1_splat, %88 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>

    %44 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>>
    %48 = tt.expand_dims %46 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>>
    %50 = tt.expand_dims %48 {axis = 2 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>> -> tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>}>>
    %56 = tt.expand_dims %50 {axis = 3 : i32} : tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked2}>> -> tensor<1x1x1x1x4xi32, #blocked2>
    %57 = tt.broadcast %56 : tensor<1x1x1x1x4xi32, #blocked2> -> tensor<1x2x32x4x4xi32, #blocked2>

    %arg3_init = tt.addptr %arg3_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
    %arg4_init = tt.addptr %arg4_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>

    %99:5 = scf.for %iv = %lb to %ub step %step iter_args(%arg15 = %cst_1, %arg16 = %arg0_init, %arg17 = %arg1_init, %arg18 = %arg3_init, %arg19 = %arg4_init) -> (tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>) {
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %117 = tt.load %arg16 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
      %118 = ttg.local_alloc %117 : (tensor<128x256xf8E5M2, #blocked>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %119 = tt.load %arg17 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
      %120 = ttg.local_alloc %119 : (tensor<256x128xf8E5M2, #blocked1>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %121 = tt.load %arg18 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %122 = tt.load %arg19 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

      %137 = ttg.local_alloc %121 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %139 = ttg.local_alloc %122 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %127 = ttng.tmem_alloc %arg15 : (tensor<128x128xf32, #blocked4>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {tt.latency = 1 : i32}
      ttng.tc_gen5_mma_scaled %118, %120, %127, %137, %139, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, i1, i1) -> ()
      %132 = ttng.tmem_load %127 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked4>

      %133 = tt.addptr %arg16, %incr_A : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
      %134 = tt.addptr %arg17, %incr_B : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>
      %135 = tt.addptr %arg18, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      %136 = tt.addptr %arg19, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      scf.yield %132, %133, %134, %135, %136 : tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    } {tt.num_stages = 3 : i32}
     tt.return %99#0 : tensor<128x128xf32, #blocked4>
  }
}
