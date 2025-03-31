// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @unscheduled_loop
// CHECK: scf.for
// CHECK:   tt.load
// CHECK:   "use"
tt.func @unscheduled_loop(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) : (tensor<128x32xf16, #A>) -> ()
  }
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_async
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]]
// CHECK-DAG:   ttg.local_dealloc %[[A]]
// CHECK-DAG:   ttg.async_wait  {num = 0 : i32}

tt.func @one_dep_async(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @different_use_stages
// CHECK: scf.for
// CHECK:   ttg.async_copy_global_to_local %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttg.async_wait {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   ttg.memdesc_subview {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use1"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use2"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 3 : i32}
tt.func @different_use_stages(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use1"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    "use2"(%a) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @used_by_if_yield
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK: scf.for
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   ttg.local_load {{.*}} token %[[A_TOK3]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"{{.*}} {loop.cluster = 0 : i32, loop.stage = 3 : i32}

tt.func @used_by_if_yield(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>,
                 %init_a : tensor<128x32xf16, #A>,
                 %cnd : i1) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %a_if = scf.if %cnd -> tensor<128x32xf16, #A> {
      scf.yield %a : tensor<128x32xf16, #A>
    } else {
      scf.yield %init_a : tensor<128x32xf16, #A>
    } {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    "use"(%a_if) {loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}
// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dist1_load
tt.func @dist1_load(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>,
                 %init_a : tensor<128x32xf16, #A>) -> () {
  %_ = scf.for %iv = %lb to %ub step %step iter_args(%prev_a = %init_a) -> (tensor<128x32xf16, #A>) : index {
    "use_next_iter"(%prev_a) {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (tensor<128x32xf16, #A>) -> ()
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    scf.yield %a : tensor<128x32xf16, #A>
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_dep_sync
// CHECK: scf.for
// CHECK:   tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
tt.func @one_dep_sync(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<1x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x!tt.ptr<f16>, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK: #[[SHARED:.*]] = #ttg.swizzled_shared
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #[[SHARED]], #
// CHECK:   "use"(%[[A_VAL]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]]
// CHECK-DAG:   ttg.local_dealloc %[[A]]
// CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
tt.func @one_dep_local_alloc(%lb : index, %ub : index, %step : index,
                 %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #A>) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f16>, #A>
    %a_alloc = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable>
    %a_load = ttg.local_load %a_alloc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory, mutable> -> tensor<128x32xf16, #A>
    "use"(%a_load) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @one_load_group
tt.func @one_load_group(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>,
                       %b_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // Only one insert and extract index is used.
  // CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]]) ->
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]]
    // CHECK: %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]]
    // CHECK: %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]]
    // CHECK: %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]]
    // CHECK: %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]]
    // CHECK: %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]]
    %a = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = tt.load %a_ptr_init {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%a){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use2"(%b){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @two_load_groups
tt.func @two_load_groups(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>,
                       %b_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>,
                       %c_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS2:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK-DAG: %[[NUM_BUFS3:.*]] = arith.constant {{.*}} 3 : i32
  // Two insert and extract indices are used.
  // CHECK: scf.for {{.*}} iter_args(%[[INS2:.*]] = %[[MINUS_ONE]], %[[EXT2:.*]] = %[[MINUS_ONE]], %[[INS3:.*]] = %[[MINUS_ONE]], %[[EXT3:.*]] = %[[MINUS_ONE]]) ->
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK-DAG: %[[INS3_P1:.*]] = arith.addi %[[INS3]], %[[ONE]]
    // CHECK-DAG: %[[INS3_CMP:.*]] = arith.cmpi slt, %[[INS3_P1]], %[[NUM_BUFS3]]
    // CHECK-DAG: %[[INS3_NEXT:.*]] = arith.select %[[INS3_CMP]], %[[INS3_P1]], %[[ZERO]]
    // CHECK-DAG: %[[EXT3_P1:.*]] = arith.addi %[[EXT3]], %[[ONE]]
    // CHECK-DAG: %[[EXT3_CMP:.*]] = arith.cmpi slt, %[[EXT3_P1]], %[[NUM_BUFS3]]
    // CHECK-DAG: %[[EXT3_NEXT:.*]] = arith.select %[[EXT3_CMP]], %[[EXT3_P1]], %[[ZERO]]
    // CHECK-DAG: %[[INS2_P1:.*]] = arith.addi %[[INS2]], %[[ONE]]
    // CHECK-DAG: %[[INS2_CMP:.*]] = arith.cmpi slt, %[[INS2_P1]], %[[NUM_BUFS2]]
    // CHECK-DAG: %[[INS2_NEXT:.*]] = arith.select %[[INS2_CMP]], %[[INS2_P1]], %[[ZERO]]
    // CHECK-DAG: %[[EXT2_P1:.*]] = arith.addi %[[EXT2]], %[[ONE]]
    // CHECK-DAG: %[[EXT2_CMP:.*]] = arith.cmpi slt, %[[EXT2_P1]], %[[NUM_BUFS2]]
    // CHECK-DAG: %[[EXT2_NEXT:.*]] = arith.select %[[EXT2_CMP]], %[[EXT2_P1]], %[[ZERO]]
    %a = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %a_ptr_init {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%a){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use2"(%b){loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> ()
    "use3"(%c){loop.cluster = 0 : i32, loop.stage = 3 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 3 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dependent_loads
tt.func @dependent_loads(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: %[[C:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]]) ->
  // CHECK: %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK: %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[B:.*]] = "pointerize"(%[[A_VAL]]) {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_INS:.*]] = ttg.memdesc_subview %[[C]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK:.*]] = ttg.async_copy_global_to_local %[[B]], %[[C_INS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK2:.*]] = ttg.async_commit_group %[[C_TOK]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK3:.*]] = ttg.async_wait %[[C_TOK2]] {loop.cluster = 0 : i32, loop.stage = 4 : i32, num = 0 : i32}
  // CHECK: %[[C_EXT:.*]] = ttg.memdesc_subview %[[C]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: %[[C_VAL:.*]] = ttg.local_load %[[C_EXT]] token %[[C_TOK3]] {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: "use1"(%[[C_VAL]]) {loop.cluster = 0 : i32, loop.stage = 4 : i32}
  // CHECK: scf.yield
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[C]]
  // CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = "pointerize"(%a) {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %b {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%c){loop.cluster = 0 : i32, loop.stage = 4 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 4 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @dependent_loads_asymmetric
// Loads have different latencies, should create two load groups.
tt.func @dependent_loads_asymmetric(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>) -> () {
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS2:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK-DAG: %[[NUM_BUFS3:.*]] = arith.constant {{.*}} 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf32
  // CHECK: %[[C:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf32
  // CHECK: scf.for {{.*}} iter_args(%[[INS2:.*]] = %[[MINUS_ONE]], %[[EXT2:.*]] = %[[MINUS_ONE]], %[[INS3:.*]] = %[[MINUS_ONE]], %[[EXT3:.*]] = %[[MINUS_ONE]]) ->
  // CHECK-DAG: %[[INS3_P1:.*]] = arith.addi %[[INS3]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[INS3_CMP:.*]] = arith.cmpi slt, %[[INS3_P1]], %[[NUM_BUFS3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[INS3_NEXT:.*]] = arith.select %[[INS3_CMP]], %[[INS3_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT3_P1:.*]] = arith.addi %[[EXT3]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[EXT3_CMP:.*]] = arith.cmpi slt, %[[EXT3_P1]], %[[NUM_BUFS3]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[EXT3_NEXT:.*]] = arith.select %[[EXT3_CMP]], %[[EXT3_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK-DAG: %[[INS2_P1:.*]] = arith.addi %[[INS2]], %[[ONE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[INS2_CMP:.*]] = arith.cmpi slt, %[[INS2_P1]], %[[NUM_BUFS2]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[INS2_NEXT:.*]] = arith.select %[[INS2_CMP]], %[[INS2_P1]], %[[ZERO]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK-DAG: %[[EXT2_P1:.*]] = arith.addi %[[EXT2]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT2_CMP:.*]] = arith.cmpi slt, %[[EXT2_P1]], %[[NUM_BUFS2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK-DAG: %[[EXT2_NEXT:.*]] = arith.select %[[EXT2_CMP]], %[[EXT2_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS2_NEXT]]{{.*}} {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 4 : i32, loop.stage = 0 : i32}
  // CHECK: %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK: %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT2_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[A_VAL:.*]] = ttg.local_load %[[A_EXT]] token %[[A_TOK3]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[B:.*]] = "pointerize"(%[[A_VAL]]) {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_INS:.*]] = ttg.memdesc_subview %[[C]][%[[INS3_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK:.*]] = ttg.async_copy_global_to_local %[[B]], %[[C_INS]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK2:.*]] = ttg.async_commit_group %[[C_TOK]] {loop.cluster = 2 : i32, loop.stage = 2 : i32}
  // CHECK: %[[C_TOK3:.*]] = ttg.async_wait %[[C_TOK2]] {loop.cluster = 0 : i32, loop.stage = 5 : i32, num = 0 : i32}
  // CHECK: %[[C_EXT:.*]] = ttg.memdesc_subview %[[C]][%[[EXT3_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: %[[C_VAL:.*]] = ttg.local_load %[[C_EXT]] token %[[C_TOK3]] {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: "use1"(%[[C_VAL]]) {loop.cluster = 0 : i32, loop.stage = 5 : i32}
  // CHECK: scf.yield
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[C]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.load %a_ptr_init {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    %b = "pointerize"(%a) {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x32xf32, #A>) -> tensor<128x32x!tt.ptr<f32>, #A>
    %c = tt.load %b {loop.cluster = 2 : i32, loop.stage = 2 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "use1"(%c){loop.cluster = 0 : i32, loop.stage = 5 : i32} : (tensor<128x32xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 5 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @unused_load
tt.func @unused_load(%lb : index, %ub : index, %step : index,
                       %a_ptr_init : tensor<128x32x!tt.ptr<f32>, #A>) -> () {
  // CHECK: scf.for
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: dummy
    %a = tt.load %a_ptr_init {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x32x!tt.ptr<f32>, #A>
    "dummy"() : () -> ()
  } {tt.scheduled_max_stage = 1 : i32}
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav3
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_EXT]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  tt.func public @shmem_pipelining_mmav3(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // The combination of blocked and shared layouts for operand B would result in cp.async with less than 4 bytes size.
  // We can't pipeline that using shared memory buffer.
  // CHECK-LABEL: @no_shmem_pipelining_incompat_layout
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B:.*]] = tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_SH:.*]] = ttg.local_alloc %[[B]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_EXT]], %[[B_SH]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG:   ttg.local_dealloc %[[A]]
  // CHECK-DAG:   ttg.async_wait  {num = 0 : i32}
  tt.func public @no_shmem_pipelining_incompat_layout(
                    %lb : index, %ub : index, %step : index,
                    %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                    %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf32, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %res : tensor<128x128xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // non-zero "other" value is used in the load, while cp.async does not support it.
  // We can't feed the shared memory values directly to mma, we need other values being filled in the registers.
  // CHECK-LABEL: @no_shmem_pipelining_other_used
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:[^,]*]] = %[[INIT]], %[[INS:[^,]*]] = %[[MINUS_ONE]], %[[EXT:[^,]*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_MASKED:.*]] = arith.select {{.*}}, %[[A_LOAD]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_LOAD:.*]] = ttg.local_load %[[B_EXT]] {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_MASKED:.*]] = arith.select {{.*}}, %[[B_LOAD]], {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[A_SH:.*]] = ttg.local_alloc %[[A_MASKED]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_SH:.*]] = ttg.local_alloc %[[B_MASKED]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.warp_group_dot %[[A_SH]], %[[B_SH]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  tt.func public @no_shmem_pipelining_other_used(
                      %lb : index, %ub : index, %step : index,
                      %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                      %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                      %mask: tensor<128x128xi1, #blocked1>,
                      %other: tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.load %A_ptr, %mask, %other  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr, %mask, %other {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shmem_pipelining_mmav5
  // CHECK-DAG: %[[INIT:.*]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant 3 : i32
  // CHECK: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: %[[B:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]], %[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]])
  // CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}  : i32
  // CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}  : i32
  // CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[A_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK2:.*]] = ttg.async_commit_group %[[A_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[A_TOK3:.*]] = ttg.async_wait %[[A_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   %[[B_INS:.*]] = ttg.memdesc_subview %[[B]][%[[INS_NEXT]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK:.*]] = ttg.async_copy_global_to_local %{{.*}}, %[[B_INS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK2:.*]] = ttg.async_commit_group %[[B_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  // CHECK:   %[[B_TOK3:.*]] = ttg.async_wait %[[B_TOK2]] {loop.cluster = 0 : i32, loop.stage = 2 : i32, num = 0 : i32}
  // CHECK:   %[[B_EXT:.*]] = ttg.memdesc_subview %[[B]][%[[EXT_NEXT]]{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   ttng.tc_gen5_mma %[[A_EXT]], %[[B_EXT]], %{{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
  // CHECK:   scf.yield {{.*}}, %[[INS_NEXT]], %[[EXT_NEXT]]
  // CHECK-DAG: ttg.local_dealloc %[[A]]
  // CHECK-DAG: ttg.local_dealloc %[[B]]
  // CHECK-DAG: ttg.async_wait  {num = 0 : i32}
  tt.func public @shmem_pipelining_mmav5(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>) -> tensor<128x128xf16, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_tm = ttng.tmem_alloc %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_load_lowering
// CHECK-DAG: %[[TRUE:.*]] = arith.constant {{.*}} true
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32
// CHECK-DAG: %[[BARRIER:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// CHECK: %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK: ttng.init_barrier %[[BAR1_VIEW]], 1
// CHECK: %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK: ttng.init_barrier %[[BAR2_VIEW]], 1
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]], %[[PHASE:.*]] = %[[ZERO]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_XOR:.*]] = arith.xori %[[PHASE]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[PHASE]], %[[PHASE_XOR]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[BAR_INS:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[INS_NEXT]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.barrier_expect %[[BAR_INS]], 8192 {loop.cluster = 2 : i32, loop.stage = 0 : i32}, %[[TRUE]]
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[TMA_PTR:.*]] = ttng.tensor_desc_to_tma_ptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.async_tma_copy_global_to_local %[[TMA_PTR]][{{.*}}] %[[A_INS]], %[[BAR_INS]], %[[TRUE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[BAR_EXT:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[EXT_NEXT]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   ttng.wait_barrier %[[BAR_EXT]], %[[PHASE_NEXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_LOAD]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]], %[[PHASE_NEXT]] : i32, i32, i32
// CHECK:  %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK:  ttng.inval_barrier %[[BAR1_VIEW]]
// CHECK:  %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK:  ttng.inval_barrier %[[BAR2_VIEW]]
// CHECK:  ttg.local_dealloc %[[BARRIER]]
// CHECK:  ttg.local_dealloc %[[A]]
tt.func @tma_load_lowering(%lb : index, %ub : index, %step : index,
                 %desc : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %offs : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_load %desc[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#offsets = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_gather_lowering
// CHECK-DAG: %[[TRUE:.*]] = arith.constant {{.*}} true
// CHECK-DAG: %[[MINUS_ONE:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[NUM_BUFS:.*]] = arith.constant {{.*}} 2 : i32
// CHECK-DAG: %[[A:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x32x128
// CHECK-DAG: %[[BARRIER:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
// CHECK: %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK: ttng.init_barrier %[[BAR1_VIEW]], 1
// CHECK: %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK: ttng.init_barrier %[[BAR2_VIEW]], 1
// CHECK: scf.for {{.*}} iter_args(%[[INS:.*]] = %[[MINUS_ONE]], %[[EXT:.*]] = %[[MINUS_ONE]], %[[PHASE:.*]] = %[[ZERO]])
// CHECK:   %[[INS_P1:.*]] = arith.addi %[[INS]], %[[ONE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_CMP:.*]] = arith.cmpi slt, %[[INS_P1]], %[[NUM_BUFS]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[INS_NEXT:.*]] = arith.select %[[INS_CMP]], %[[INS_P1]], %[[ZERO]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[EXT_P1:.*]] = arith.addi %[[EXT]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_CMP:.*]] = arith.cmpi slt, %[[EXT_P1]], %[[NUM_BUFS]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[EXT_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[EXT_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_XOR:.*]] = arith.xori %[[PHASE]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[PHASE_NEXT:.*]] = arith.select %[[EXT_CMP]], %[[PHASE]], %[[PHASE_XOR]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[BAR_INS:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[INS_NEXT]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.barrier_expect %[[BAR_INS]], 16384 {loop.cluster = 2 : i32, loop.stage = 0 : i32}, %[[TRUE]]
// CHECK:   %[[A_INS:.*]] = ttg.memdesc_subview %[[A]][%[[INS_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[TMA_PTR:.*]] = ttng.tensor_desc_to_tma_ptr {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   ttng.async_tma_gather %[[TMA_PTR]][{{.*}}] %[[A_INS]], %[[BAR_INS]], %[[TRUE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32}
// CHECK:   %[[BAR_EXT:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[EXT_NEXT]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   ttng.wait_barrier %[[BAR_EXT]], %[[PHASE_NEXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_EXT:.*]] = ttg.memdesc_subview %[[A]][%[[EXT_NEXT]], %[[ZERO]], %[[ZERO]]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   %[[A_LOAD:.*]] = ttg.local_load %[[A_EXT]] {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   "use"(%[[A_LOAD]]) {loop.cluster = 0 : i32, loop.stage = 2 : i32}
// CHECK:   scf.yield %[[INS_NEXT]], %[[EXT_NEXT]], %[[PHASE_NEXT]] : i32, i32, i32
// CHECK:  %[[BAR1_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ZERO]]]
// CHECK:  ttng.inval_barrier %[[BAR1_VIEW]]
// CHECK:  %[[BAR2_VIEW:.*]] = ttg.memdesc_subview %[[BARRIER]][%[[ONE]]]
// CHECK:  ttng.inval_barrier %[[BAR2_VIEW]]
// CHECK-DAG: ttg.local_dealloc %[[BARRIER]]
// CHECK-DAG: ttg.local_dealloc %[[A]]
tt.func @tma_gather_lowering(%lb : index, %ub : index, %step : index,
                 %desc : !tt.tensordesc<tensor<1x128xf32, #nvmma_128>>,
                 %x : tensor<32xi32, #offsets>,
                 %y : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_gather %desc[%x, %y] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : (!tt.tensordesc<tensor<1x128xf32, #nvmma_128>>, tensor<32xi32, #offsets>, i32) -> tensor<32x128xf32, #A>
    "use"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<32x128xf32, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#A = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tma_reuse_barrier
// CHECK: scf.for
// CHECK:   ttng.barrier_expect {{.*}}, 16384
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK-NOT: ttng.wait_barrier
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK:   ttng.wait_barrier
// CHECK:   "use1"
// CHECK:   "use2"
// CHECK:   ttng.barrier_expect {{.*}}, 8192
// CHECK:   ttng.async_tma_copy_global_to_local
// CHECK:   ttng.wait_barrier
// CHECK:   "use3"
tt.func @tma_reuse_barrier(%lb : index, %ub : index, %step : index,
                 %descA : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %descB : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %descC : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>>,
                 %offs : i32) -> () {
  scf.for %iv = %lb to %ub step %step : index {
    %a = tt.descriptor_load %descA[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    %b = tt.descriptor_load %descB[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use1"(%a) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    "use2"(%b) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
    %c = tt.descriptor_load %descC[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x32xf16, #nvmma_64>> -> tensor<128x32xf16, #A>
    "use3"(%c) {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> ()
  } {tt.scheduled_max_stage = 2 : i32}
  tt.return
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, instrShape = [16, 16, 16]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_pipelining_mmav3
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3xi64
  // CHECK: scf.for
  // CHECK:   ttng.barrier_expect
  // CHECK:   ttng.async_tma_copy_global_to_local
  // CHECK-NOT: ttng.wait_barrier
  // CHECK:   ttng.async_tma_copy_global_to_local
  // CHECK:   ttng.wait_barrier
  // CHECK-NOT: ttg.local_alloc
  // CHECK:   ttng.warp_group_dot
  tt.func public @tma_pipelining_mmav3(%lb : index, %ub : index, %step : index,
                                              %descA : !tt.tensordesc<tensor<128x128xf16, #shared>>,
                                              %descB : !tt.tensordesc<tensor<128x128xf16, #shared>>,
                                              %offs : i32) -> tensor<128x128xf16, #mma> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : index {
      %A = tt.descriptor_load %descA[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %B = tt.descriptor_load %descB[%offs, %offs] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
      %acc_res = ttng.warp_group_dot %A_sh, %B_sh, %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %acc_res : tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    tt.return %res_f16 : tensor<128x128xf16, #mma>
  }
}

// -----
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_descriptor_lowering
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[NUM_STAGES:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[_128:.*]] = arith.constant{{.*}} 128 : i32
  // CHECK: %[[GLOBAL_ALLOC:.*]] = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 256 : i32} : !tt.ptr<i8>
  // CHECK: scf.for {{.*}} iter_args(%[[IDX:.*]] = %[[ZERO]])
  // CHECK:   %[[OFFS:.*]] = arith.muli %[[IDX]], %[[_128]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[DESC_PTR:.*]] = tt.addptr %[[GLOBAL_ALLOC]], %[[OFFS]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   tt.experimental_tensormap_create %[[DESC_PTR]]{{.*}} loop.cluster = 0 : i32, loop.stage = 1 : i32
  // CHECK:   tt.experimental_tensormap_fenceproxy_acquire %[[DESC_PTR]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[DESC:.*]] = tt.reinterpret_tensor_descriptor %[[DESC_PTR]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_P1:.*]] = arith.addi %[[IDX]], %[[ONE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_CMP:.*]] = arith.cmpi slt, %[[IDX_P1]], %[[NUM_STAGES]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   %[[IDX_NEXT:.*]] = arith.select %[[IDX_CMP]], %[[IDX_P1]], %[[ZERO]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  // CHECK:   "use"(%[[DESC]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32}
  tt.func @tensor_descriptor_lowering(
    %lb : index, %ub : index, %step : index,
    %A: !tt.ptr<f16>,
    %shape_x: i32,
    %shape_y: i32,
    %strides_x: i64,
    %strides_y: i64) -> (){
    scf.for %iv = %lb to %ub step %step : index {
      %desc = tt.make_tensor_descriptor %A, [%shape_x, %shape_y], [%strides_x, %strides_y] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <f16>, <tensor<128x128xf16, #nvmma_128>>
      "use"(%desc) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (!tt.tensordesc<tensor<128x128xf16, #nvmma_128>>) -> ()
    } {tt.scheduled_max_stage = 1 : i32}
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipelining_mmav5_scaled
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E5M2
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E5M2
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
  tt.func public @pipelining_mmav5_scaled(%lb : index, %ub : index, %step : index,
                                              %A_ptr: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>,
                                              %B_ptr: tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>,
                                              %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>,
                                              %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>) -> tensor<128x128xf32, #blocked> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>) : index {
      %A = tt.load %A_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
      %B = tt.load %B_ptr  {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f8E5M2>, #blocked1>
      %A_sh = ttg.local_alloc %A {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E5M2, #blocked1>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>
      %B_sh = ttg.local_alloc %B {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E5M2, #blocked1>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>

      %A_sc = tt.load %A_sc_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %A_sc_sh = ttg.local_alloc %A_sc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %B_sc = tt.load %B_sc_ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %B_sc_sh = ttg.local_alloc %B_sc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %acc_tm = ttng.tmem_alloc %acc {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (!ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, i1, i1) -> ()
      %acc_res = ttng.tmem_load %acc_tm {loop.cluster = 0 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %res : tensor<128x128xf32, #blocked>
  }
}
