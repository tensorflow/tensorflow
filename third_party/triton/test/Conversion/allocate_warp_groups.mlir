// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {
}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 17 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @kernel() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 16, 4, 12>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {"ttg.num-warps" = 2 : i32, "ttg.total-num-warps" = 11 : i32}
module attributes {"ttg.num-warps" = 2 : i32} {

tt.func @two_warp_specialize() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 2, 4>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()

  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 10, 2>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  } : () -> ()

  tt.return
}

}
