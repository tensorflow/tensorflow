// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -convert-warp-specialize-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @rewrite_barriers
llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
  // CHECK: barrier.sync.aligned 2, 128
  // CHECK: barrier.sync.aligned 3, 64
  // CHECK: barrier.sync.aligned 4, 32

  // CHECK: bb{{[0-9]+}}:
  // CHECK-NEXT: barrier.sync.aligned 0, 128
  nvvm.barrier0
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    // CHECK: barrier.sync.aligned 0, 128
    nvvm.barrier0
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    nvvm.barrier0
    ttg.warp_return
  }
  partition1() num_warps(2) {
    nvvm.barrier0
    ttg.warp_return
  }
  partition2() num_warps(1) {
    nvvm.barrier0
    ttg.warp_return
  } : () -> ()
  // CHECK: barrier.sync.aligned 0, 128
  nvvm.barrier0
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @generate_switch_loop
llvm.func @generate_switch_loop() attributes {allocation.offset = 32 : i32} {
  // CHECK-NEXT: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK-NEXT: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK-NEXT: [[WID:%.*]] = llvm.udiv [[TIDX]], [[C32]]
  // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: [[C31:%.*]] = llvm.mlir.constant(31 : i32)
  // CHECK-NEXT: [[CNEG1:%.*]] = llvm.mlir.constant(-1 : i32)
  // CHECK-NEXT: [[WARP_ID:%.*]] = nvvm.shfl.sync idx [[CNEG1]], [[WID]], [[C0]], [[C31]]
  // CHECK-NEXT: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-NEXT: [[IS_DEFAULT:%.*]] = llvm.icmp "ult" [[WARP_ID]], [[C4]]
  // CHECK-NEXT: llvm.cond_br [[IS_DEFAULT]], [[BODY:\^.*]], [[SWITCH_LOOP:\^.*]]

  // CHECK: [[SWITCH_LOOP]]:
  // CHECK-NEXT: "barrier.sync 1 ;"
  // CHECK-NEXT: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK-NEXT: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][[[C32]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
  // CHECK-NEXT: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-NEXT: [[REL_WID:%.*]] = llvm.sub [[WARP_ID]], [[C4]]

  // CHECK-NEXT: [[STATE_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][[[REL_WID]]]
  // CHECK-NEXT: [[STATE:%.*]] = llvm.load [[STATE_PTR]]
  // CHECK-NEXT: llvm.switch [[STATE]] : i8, [[DEFAULT:\^.*]] [
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^.*]]

  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[EXIT]]:
  // CHECK-NEXT: llvm.return

  // CHECK: [[PARTITION0]]:
  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: "partition0"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[PARTITION1]]:
  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: "partition1"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[PARTITION2]]:
  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: "partition2"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: [[BODY]]:
  // CHECK-NEXT: "before"
  // CHECK-NEXT: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK-NEXT: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][[[C32]]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]

  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[DEFAULT_PARTITION:\^.*]]
  // CHECK: [[DEFAULT_PARTITION]]:
  // CHECK-NEXT: "default"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]
  "before"() : () -> ()
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    "default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(1) {
    "partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  // CHECK: [[AFTER]]:
  // CHECK-NEXT: "after"

  // CHECK-NEXT: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK-NEXT: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][[[C32]]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(3 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.return
  "after"() : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @pass_captures
llvm.func @pass_captures(%arg0: i32, %arg1: i64) attributes {allocation.offset = 32 : i32} {
  // CHECK: ^bb4:
  // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][[[C0]]]

  // CHECK-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG0:%.*]] = llvm.load [[ARG0_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: [[ARG1:%.*]] = llvm.load [[ARG1_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: "use"([[ARG0]], [[ARG1]])
  // CHECK-NEXT: barrier.sync 1 ;

  // CHECK: ^bb5:
  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: [[SMEM_ADDR:%.*]] = llvm.mlir.addressof @global_smem
  // CHECK-NEXT: [[SMEM_BASE:%.*]] = llvm.getelementptr [[SMEM_ADDR]][[[C0]]]
  // CHECK-NEXT: [[ARG0_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0, 0] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store %arg0, [[ARG0_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: [[ARG1_PTR:%.*]] = llvm.getelementptr [[SMEM_BASE]][0, 1] : (!llvm.ptr<3>) -> !llvm.ptr<3>, !llvm.struct<packed (i32, i64)>
  // CHECK-NEXT: llvm.store %arg1, [[ARG1_PTR]] {alignment = 1 : i64}
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: barrier.sync 1 ;
  ttg.warp_specialize(%arg0, %arg1) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg2: i32, %arg3: i64) num_warps(4) {
    "use"(%arg2, %arg3) : (i32, i64) -> ()
    ttg.warp_return
  } : (i32, i64) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 18 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @partition_warpid_order
llvm.func @partition_warpid_order() attributes {allocation.offset = 32 : i32} {
  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[EXIT:\^.*]]

  // CHECK: [[PARTITION0]]:
  // CHECK: "ws0_partition0"
  // CHECK: [[PARTITION1]]:
  // CHECK: "ws0_partition1"
  // CHECK: [[PARTITION2]]:
  // CHECK: "ws0_partition2"

  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK-NEXT: getelementptr

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]

  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[8]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[9]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[10]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[11]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[12]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[13]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 6, 4, 10>}
  default {
    "ws0_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws0_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "ws0_partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(8) {
    "ws0_partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 12 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @multiple_specialize
llvm.func @multiple_specialize() attributes {allocation.offset = 32 : i32} {
  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[WS0_PARTITION0:\^.*]],
  // CHECK-NEXT: 1: [[WS0_PARTITION1:\^.*]],
  // CHECK-NEXT: 2: [[WS0_PARTITION2:\^.*]],
  // CHECK-NEXT: 3: [[WS1_PARTITION0:\^.*]],
  // CHECK-NEXT: 4: [[WS1_PARTITION1:\^.*]],
  // CHECK-NEXT: 5: [[WS3_PARTITION0:\^.*]],
  // CHECK-NEXT: 6: [[EXIT:\^.*]]

  // CHECK: [[WS0_PARTITION0]]:
  // CHECK: "ws0_partition0"
  // CHECK: [[WS0_PARTITION1]]:
  // CHECK: "ws0_partition1"
  // CHECK: [[WS0_PARTITION2]]:
  // CHECK: "ws0_partition2"
  // CHECK: [[WS1_PARTITION0]]:
  // CHECK: "ws1_partition0"
  // CHECK: [[WS1_PARTITION1]]:
  // CHECK: "ws1_partition1"
  // CHECK: [[WS3_PARTITION0]]:
  // CHECK: "ws3_partition0"

  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(2 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK: barrier.sync 1 ;
  // CHECK: "ws0_default"
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4, 8, 10>}
  default {
    "ws0_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws0_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    "ws0_partition1"() : () -> ()
    ttg.warp_return
  }
  partition2() num_warps(1) {
    "ws0_partition2"() : () -> ()
    ttg.warp_return
  } : () -> ()

  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(4 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(4 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(4 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(4 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(3 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(3 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(3 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(3 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK: barrier.sync 1 ;
  // CHECK: "ws1_default"
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 8, 4>}
  default {
    "ws1_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "ws1_partition0"() : () -> ()
    ttg.warp_return
  }
  partition1() num_warps(4) {
    "ws1_partition1"() : () -> ()
    ttg.warp_return
  } : () -> ()

  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(-1 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK: barrier.sync 1 ;
  // CHECK: "ws2_default"
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32>}
  default {
    "ws2_default"() : () -> ()
    ttg.warp_yield
  } : () -> ()

  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[0]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[1]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[2]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[3]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[4]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[5]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[6]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK-NEXT: [[CX:%.*]] = llvm.mlir.constant(5 : i8)
  // CHECK-NEXT: [[PTR:%.*]] = llvm.getelementptr %{{[0-9]+}}[7]
  // CHECK-NEXT: llvm.store [[CX]], [[PTR]]
  // CHECK: barrier.sync 1 ;
  // CHECK: "ws3_default"
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "ws3_default"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    "ws3_partition0"() : () -> ()
    ttg.warp_return
  }: () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @cfg
llvm.func @cfg() attributes {allocation.offset = 32 : i32} {
  // CHECK: [[SWITCH_LOOP:\^bb1]]:
  // CHECK: llvm.switch
  // CHECK-NEXT: 0: [[PARTITION:\^.*]],
  // CHECK-NEXT: 1: [[EXIT:\^.*]]

  // CHECK: [[PARTITION]]:
  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[SWITCH_LOOP]]

  // CHECK: barrier.sync 1 ;
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK: llvm.br [[DEFAULT:\^.*]]
  // CHECK: [[DEFAULT]]:
  // CHECK-NEXT: "something"()[[[A:\^.*]], [[B:\^.*]]]
  // CHECK: [[A]]:
  // CHECK-NEXT: "A"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[AFTER:\^.*]]
  // CHECK: [[B]]:
  // CHECK-NEXT: "B"
  // CHECK-NEXT: barrier.sync 1 ;
  // CHECK-NEXT: llvm.br [[AFTER]]
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "something"()[^A, ^B] : () -> ()
  ^A:
   "A"() : () -> ()
    ttg.warp_yield
  ^B:
   "B"() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    "something"()[^A, ^B] : () -> ()
  ^A:
   "A"() : () -> ()
    ttg.warp_return
  ^B:
   "B"() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @no_captures
llvm.func @no_captures() attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.total-num-warps" = 6 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @type_conversion_results
// CHECK-NOT: !tt.ptr<i32>
// CHECK-NOT: unrealized_conversion_cast
llvm.func @type_conversion_results(%arg0: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<1> to !tt.ptr<i32>
  %1 = ttg.warp_specialize(%0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    // CHECK: llvm.br [[AFTER:\^.*]](%arg0 : !llvm.ptr<1>)
    ttg.warp_yield %0 : !tt.ptr<i32>
  }
  partition0(%arg1: !tt.ptr<i32>) num_warps(2) {
    %3 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<i32> to !llvm.ptr<1>
    %4 = llvm.load %3 : !llvm.ptr<1> -> i32
    ttg.warp_return
  } : (!tt.ptr<i32>) -> !tt.ptr<i32>
  // CHECK: [[AFTER]]([[OUT:%.*]]: !llvm.ptr<1>):
  %2 = builtin.unrealized_conversion_cast %1 : !tt.ptr<i32> to !llvm.ptr<1>
  // CHECK-NEXT: "use"([[OUT]])
  "use"(%2) : (!llvm.ptr<1>) -> ()
  llvm.return
}

}
