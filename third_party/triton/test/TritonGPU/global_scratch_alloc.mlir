// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation | FileCheck %s

// CHECK: module attributes {ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK: @test_alloc{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32
  tt.func public @test_alloc() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    // CHECK:  ttg.global_scratch_memory_offset = 128
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}

// -----

// CHECK: module attributes {ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK: @helper1{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 128 : i32
  tt.func private @helper1() -> (!tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    tt.return %0 : !tt.ptr<i8>
  }

// CHECK: @test_function{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32
  tt.func public @test_function() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    // CHECK:  ttg.global_scratch_memory_offset = 128
    %1 = tt.call @helper1() : () -> (!tt.ptr<i8>)
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}
