// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-rewrite-partition-dependencies -verify-diagnostics -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @invalid_attribute(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{partition stages attribute 'ttg.partition.stages' has invalid element "a"}}
  scf.for %i = %lb to %ub step %step : i32 {
    scf.yield
  } {ttg.partition.stages = ["a"]}
  scf.for %j = %lb to %ub step %step : i32 {
    scf.yield
  }
  scf.for %k = %lb to %ub step %step : i32 {
    // expected-warning @below {{invalid partition index -1}}
    "op"() {ttg.partition = -1} : () -> ()
    scf.yield
  } {ttg.partition.stages = [2, 2]}
  tt.return
}

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{warp schedule contains a cycle}}
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{operation in partition #1 uses value defined in partition #0}}
    %1 = "op_b"(%0) {ttg.partition = 1} : (index) -> index
    // expected-note @below {{operation in partition #0 uses value defined in partition #1}}
    "op_c"(%1) {ttg.partition = 0} : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}
  // expected-warning @below {{warp schedule contains a cycle}}
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{operation in partition #1 uses value defined in partition #0}}
    %1 = "op_b"(%0) {ttg.partition = 1} : (index) -> index
    // expected-note @below {{operation in partition #2 uses value defined in partition #1}}
    %2 = "op_c"(%1) {ttg.partition = 2} : (index) -> index
    // expected-note @below {{operation in partition #0 uses value defined in partition #2}}
    "op_c"(%2) {ttg.partition = 0} : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2, 3]}
  tt.return
}

tt.func @invalid_root_partition(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-note @below {{operand defined here in partition #0 at distance 0}}
    %0 = "partition"() {ttg.partition = 0} : () -> index
    // expected-warning @below {{operation in the root partition depends on a value that originates from a non-root partition through operand #0}}
    "root"(%0) : (index) -> ()
    scf.yield
  } {ttg.partition.stages = [0, 2]}

  %c0 = arith.constant 0 : index
  scf.for %j = %lb to %ub step %step iter_args(%k = %c0) -> index : i32 {
    // expected-warning @below {{operation in the root partition depends on a value that originates from a non-root partition through operand #0}}
    "root"(%k) : (index) -> ()
    // expected-note @below {{operand defined here in partition #0 at distance 1}}
    %0 = "partition"() {ttg.partition = 0} : () -> index
    scf.yield %0 : index
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

tt.func @invalid_partition_stage(%lb: i32, %ub: i32, %step: i32) {
  // expected-warning @below {{partition #0 has stage 2 but is consumed by partition #1 with stage 0}}
  scf.for %i = %lb to %ub step %step : i32 {
    // expected-note @below {{value defined here in partition #0}}
    %0 = "op_a"() {ttg.partition = 0} : () -> index
    // expected-note @below {{use of value defined in partition #0}}
    "op_b"(%0) {ttg.partition = 1} : (index) -> ()
  } {ttg.partition.stages = [2, 0]}
  tt.return
}

tt.func @invalid_future_partition(%lb: i32, %ub: i32, %step: i32) {
  %c0 = arith.constant 0 : index
  // expected-warning @below {{partition #1 has stage 2 but is consumed by partition #0 with stage 0 at distance 1}}
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0) -> index : i32 {
    // expected-note @below {{use of value defined in partition #1 at 1 iterations in the future}}
    "op_a"(%k) {ttg.partition = 0} : (index) -> ()
    // expected-note @below {{value defined here in partition #1}}
    %0 = "op_b"() {ttg.partition = 1} : () -> index
    scf.yield %0 : index
  } {ttg.partition.stages = [0, 2]}
  tt.return
}

// CHECK-LABEL: @two_consumers
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32,
  // CHECK-NEXT: [[READY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: [[EMPTY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,

  // CHECK-NEXT: [[READY0:%.*]] = ttg.memdesc_subview [[READY_BARS]][%c0_i32]
  // CHECK-NEXT: [[EMPTY0:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[READY0]], 1
  // CHECK-NEXT: ttng.init_barrier [[EMPTY0]], 2
  // CHECK-NEXT: ttng.arrive_barrier [[EMPTY0]], 2

  // CHECK-NEXT: [[READY1:%.*]] = ttg.memdesc_subview [[READY_BARS]][%c1_i32]
  // CHECK-NEXT: [[EMPTY1:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[READY1]], 1
  // CHECK-NEXT: ttng.init_barrier [[EMPTY1]], 2
  // CHECK-NEXT: ttng.arrive_barrier [[EMPTY1]], 2

  // CHECK-NEXT: %{{[0-9]+}}:6 = scf.for %arg{{[0-9]+}} = %arg0 to %arg1 step %arg2 iter_args(
  // CHECK-SAME:   [[CONSUMER_IDX0:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME:   [[CONSUMER_PHASE0:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME:   [[CONSUMER_IDX1:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME:   [[CONSUMER_PHASE1:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME:   [[PRODUCER_IDX:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME:   [[PRODUCER_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    // CHECK-NEXT: [[OUTPUT:%.*]] = "op_a"() {ttg.partition = 0 : i32}
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK-NEXT: [[NEXT_IDX:%.*]] = arith.addi [[PRODUCER_IDX]], %c1_i32
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[PRODUCER_PHASE]], %c1_i32
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK-NEXT: [[PHASE0:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[PRODUCER_PHASE]]
    // CHECK-NEXT: [[IDX0:%.*]] = arith.select [[ROLLOVER]], %c0_i32, [[NEXT_IDX]]
    // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_subview [[BUFFERS]][[[IDX0]], %c0_i32]
    // CHECK-NEXT: [[READY:%.*]] = ttg.memdesc_subview [[READY_BARS]][[[IDX0]]]
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][[[IDX0]]]
    // CHECK-NEXT: ttng.wait_barrier [[EMPTY]], [[PHASE0]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[OUTPUT]], [[VIEW]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[READY]], 1 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[NEXT_IDX:%.*]] = arith.addi [[CONSUMER_IDX0]], %c1_i32
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[CONSUMER_PHASE0]], %c1_i32
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK-NEXT: [[PHASE1:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[CONSUMER_PHASE0]]
    // CHECK-NEXT: [[IDX1:%.*]] = arith.select [[ROLLOVER]], %c0_i32, [[NEXT_IDX]]
    // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_subview [[BUFFERS]][[[IDX1]], %c0_i32]
    // CHECK-NEXT: [[READY:%.*]] = ttg.memdesc_subview [[READY_BARS]][[[IDX1]]]
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][[[IDX1]]]
    // CHECK-NEXT: ttng.wait_barrier [[READY]], [[PHASE1]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VALUE:%.*]] = ttg.local_load [[VIEW]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[EMPTY]], 1 {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VALUE]]) {ttg.partition = 1 : i32}
    "op_b"(%0) {ttg.partition = 1} : (!ty) -> ()

    // CHECK-NEXT: [[NEXT_IDX:%.*]] = arith.addi [[CONSUMER_IDX1]], %c1_i32
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[CONSUMER_PHASE1]], %c1_i32
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK-NEXT: [[PHASE2:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[CONSUMER_PHASE1]]
    // CHECK-NEXT: [[IDX2:%.*]] = arith.select [[ROLLOVER]], %c0_i32, [[NEXT_IDX]]
    // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_subview [[BUFFERS]][[[IDX2]], %c0_i32]
    // CHECK-NEXT: [[READY:%.*]] = ttg.memdesc_subview [[READY_BARS]][[[IDX2]]]
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][[[IDX2]]]
    // CHECK-NEXT: ttng.wait_barrier [[READY]], [[PHASE2]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[VALUE:%.*]] = ttg.local_load [[VIEW]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[EMPTY]], 1 {ttg.partition = 2 : i32}
    // CHECK-NEXT: "op_d"([[VALUE]]) {ttg.partition = 2 : i32}
    "op_d"(%0) {ttg.partition = 2} : (!ty) -> ()

    // CHECK-NEXT: yield [[IDX1]], [[PHASE1]], [[IDX2]], [[PHASE2]], [[IDX0]], [[PHASE0]]

  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32]
  } {ttg.partition.stages = [0, 2, 2]}
  // CHECK-NEXT: ttg.local_dealloc [[BUFFERS]]
  // CHECK-NEXT: ttng.inval_barrier [[READY0]]
  // CHECK-NEXT: ttng.inval_barrier [[EMPTY0]]
  // CHECK-NEXT: ttng.inval_barrier [[READY1]]
  // CHECK-NEXT: ttng.inval_barrier [[EMPTY1]]
  // CHECK-NEXT: ttg.local_dealloc [[READY_BARS]]
  // CHECK-NEXT: ttg.local_dealloc [[EMPTY_BARS]]
  tt.return
}

// CHECK-LABEL: @distance_one
tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty

  // CHECK: [[BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32,
  // CHECK-NEXT: [[READY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,
  // CHECK-NEXT: [[EMPTY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64,

  // CHECK-NEXT: [[INIT:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c0_i32, %c0_i32]
  // CHECK-NEXT: ttg.local_store %cst, [[INIT]]
  // CHECK-NEXT: [[READY0:%.*]] = ttg.memdesc_subview [[READY_BARS]][%c0_i32]
  // CHECK-NEXT: [[EMPTY0:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[READY0]], 1
  // CHECK-NEXT: ttng.init_barrier [[EMPTY0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[READY0]], 1

  // CHECK-NEXT: [[READY1:%.*]] = ttg.memdesc_subview [[READY_BARS]][%c1_i32]
  // CHECK-NEXT: [[EMPTY1:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[READY1]], 1
  // CHECK-NEXT: ttng.init_barrier [[EMPTY1]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[EMPTY1]], 1

  // CHECK-NEXT: %{{[0-9]+}}:4 = scf.for %arg{{[0-9]+}} = %arg0 to %arg1 step %arg2 iter_args(
  // CHECK-SAME:   [[CONSUMER_IDX:%arg[0-9]+]] = %c-1_i32
  // CHECK-SAME:   [[CONSUMER_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME:   [[PRODUCER_IDX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME:   [[PRODUCER_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[OUTPUT:%.*]] = "op_a"() {ttg.partition = 0 : i32}
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK-NEXT: [[NEXT_IDX:%.*]] = arith.addi [[PRODUCER_IDX]], %c1_i32
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[PRODUCER_PHASE]], %c1_i32
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK-NEXT: [[PHASE0:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[PRODUCER_PHASE]]
    // CHECK-NEXT: [[IDX0:%.*]] = arith.select [[ROLLOVER]], %c1_i32, [[NEXT_IDX]]
    // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_subview [[BUFFERS]][[[IDX0]], %c0_i32]
    // CHECK-NEXT: [[READY:%.*]] = ttg.memdesc_subview [[READY_BARS]][[[IDX0]]]
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][[[IDX0]]]
    // CHECK-NEXT: ttng.wait_barrier [[EMPTY]], [[PHASE0]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttg.local_store [[OUTPUT]], [[VIEW]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[READY]], 1 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[NEXT_IDX:%.*]] = arith.addi [[CONSUMER_IDX]], %c1_i32
    // CHECK-NEXT: [[NEXT_PHASE:%.*]] = arith.xori [[CONSUMER_PHASE]], %c1_i32
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK-NEXT: [[PHASE1:%.*]] = arith.select [[ROLLOVER]], [[NEXT_PHASE]], [[CONSUMER_PHASE]]
    // CHECK-NEXT: [[IDX1:%.*]] = arith.select [[ROLLOVER]], %c1_i32, [[NEXT_IDX]]
    // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_subview [[BUFFERS]][[[IDX1]], %c0_i32]
    // CHECK-NEXT: [[READY:%.*]] = ttg.memdesc_subview [[READY_BARS]][[[IDX1]]]
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][[[IDX1]]]
    // CHECK-NEXT: ttng.wait_barrier [[READY]], [[PHASE1]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[VALUE:%.*]] = ttg.local_load [[VIEW]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[EMPTY]], 1 {ttg.partition = 1 : i32}
    // CHECK-NEXT: "op_b"([[VALUE]]) {ttg.partition = 1 : i32}
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()

    // CHECK-NEXT: yield [[IDX1]], [[PHASE1]], [[IDX0]], [[PHASE0]]
    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 0]}
  tt.return
}

// CHECK-LABEL: @complex_case
tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<6xi64,
  // CHECK: ttng.init_barrier %{{.*}}, 4
  %cst = arith.constant dense<0> : !ty
  // CHECK: iter_args
  // CHECK-SAME: [[CIDX0:%arg[0-9]+]] = %c0_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[CIDX1:%arg[0-9]+]] = %c0_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[CIDX2:%arg[0-9]+]] = %c-1_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[CIDX3:%arg[0-9]+]] = %c-1_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[PIDX0:%arg[0-9]+]] = %c1_i32, %arg{{[0-9]+}}
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
    // CHECK: op_a
    // CHECK: [[NEXT:%.*]] = arith.addi [[PIDX0]], %c1_i32
    // CHECK: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT]], %c6_i32
    // CHECK: select [[ROLLOVER]], %c2_i32, [[NEXT]]
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty

    // CHECK: arith.addi [[CIDX0]], %c1_i32
    // CHECK: select %{{.*}}, %c2_i32
    // CHECK: op_b
    "op_b"(%k) {ttg.partition = 1} : (!ty) -> ()

    // CHECK: arith.addi [[CIDX1]], %c1_i32
    // CHECK: select %{{.*}}, %c2_i32
    // CHECK-COUNT-2: op_c
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = 2} : (!ty) -> ()

    // CHECK: arith.addi [[CIDX2]], %c1_i32
    // CHECK: select %{{.*}}, %c2_i32
    // CHECK: op_d
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()

    // CHECK: arith.addi [[CIDX3]], %c1_i32
    // CHECK: select %{{.*}}, %c2_i32
    // CHECK: op_d
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [0, 2, 2]}
  tt.return
}

// CHECK-LABEL: @reuse_argument
tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty

  // CHECK: [[BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32
  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c0_i32, %c0_i32]
  // CHECK: local_store [[CST0]], [[VALUE]]
  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c1_i32, %c0_i32]
  // CHECK: local_store [[CST1]], [[VALUE]]
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    "op_d"(%l) {ttg.partition = 1} : (!ty) -> ()
    "op_d"(%l) {ttg.partition = 2} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {ttg.partition.stages = [1, 0, 0]}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch
tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-DAG: [[CST2:%.*]] = arith.constant dense<2>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK: [[BUFFERS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<6x1xi32

  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c0_i32, %c0_i32]
  // CHECK: local_store [[CST0]], [[VALUE]]
  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c1_i32, %c0_i32]
  // CHECK: local_store [[CST2]], [[VALUE]]

  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c2_i32, %c0_i32]
  // CHECK: local_store [[CST0]], [[VALUE]]
  // CHECK: [[VALUE:%.*]] = ttg.memdesc_subview [[BUFFERS]][%c3_i32, %c0_i32]
  // CHECK: local_store [[CST1]], [[VALUE]]

  // CHECK: iter_args
  // CHECK-SAME: [[CIDX0:%arg[0-9]+]] = %c0_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[CIDX1:%arg[0-9]+]] = %c1_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[CIDX2:%arg[0-9]+]] = %c-1_i32, %arg{{[0-9]+}}
  // CHECK-SAME: [[PIDX0:%arg[0-9]+]] = %c3_i32, %arg{{[0-9]+}}
  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // CHECK: [[OUT:%.*]] = "op_a"()
    %0 = "op_a"() {ttg.partition = 0} : () -> !ty
    // CHECK: [[NEXT_IDX:%.*]] = arith.addi [[PIDX0]], %c1_i32
    // CHECK: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c6_i32
    // CHECK: [[IDX:%.*]] = arith.select [[ROLLOVER]], %c4_i32, [[NEXT_IDX]]
    // CHECK: memdesc_subview [[BUFFERS]][[[IDX]], %c0_i32]

    // CHECK: [[NEXT_IDX:%.*]] = arith.addi [[CIDX0]], %c1_i32
    // CHECK: [[LAST:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c6_i32
    // CHECK: [[AT_END:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK: [[ROLLOVER:%.*]] = arith.ori [[LAST]], [[AT_END]]
    // CHECK: [[IDX:%.*]] = arith.select [[ROLLOVER]], %c4_i32, [[NEXT_IDX]]
    // CHECK: memdesc_subview [[BUFFERS]][[[IDX]], %c0_i32]
    // CHECK: op_b
    "op_b"(%a) {ttg.partition = 1}: (!ty) -> ()

    // CHECK: [[NEXT_IDX:%.*]] = arith.addi [[CIDX1]], %c1_i32
    // CHECK: [[ROLLOVER:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c6_i32
    // CHECK: [[IDX:%.*]] = arith.select [[ROLLOVER]], %c4_i32, [[NEXT_IDX]]
    // CHECK: memdesc_subview [[BUFFERS]][[[IDX]], %c0_i32]
    // CHECK: op_c
    "op_c"(%b) {ttg.partition = 2}: (!ty) -> ()

    // CHECK: [[NEXT_IDX:%.*]] = arith.addi [[CIDX2]], %c1_i32
    // CHECK: [[LAST:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c6_i32
    // CHECK: [[AT_END:%.*]] = arith.cmpi eq, [[NEXT_IDX]], %c2_i32
    // CHECK: [[ROLLOVER:%.*]] = arith.ori [[LAST]], [[AT_END]]
    // CHECK: [[IDX:%.*]] = arith.select [[ROLLOVER]], %c4_i32, [[NEXT_IDX]]
    // CHECK: memdesc_subview [[BUFFERS]][[[IDX]], %c0_i32]
    // CHECK: op_d
    "op_d"(%c) {ttg.partition = 3}: (!ty) -> ()

    scf.yield %0, %a, %a : !ty, !ty, !ty
  } {ttg.partition.stages = [0, 0, 0, 0]}
  tt.return
}

// CHECK-LABEL: @self_recursion
tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NOT: ttg.local_alloc
  %cst = arith.constant dense<0> : !ty
  // CHECK: iter_args([[ARG:%arg[0-9]+]] = %cst)
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[OUT:%.*]] = "op_a"([[ARG]])
    %0 = "op_a"(%k) {ttg.partition = 0} : (!ty) -> !ty
    // CHECK: yield [[OUT]]
    scf.yield %0 : !ty
  } {ttg.partition.stages = [0]}
  tt.return
}

// CHECK-LABEL: @self_recursion_and_use
tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  // CHECK: iter_args([[ARG:%arg[0-9]+]] = %cst,
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[OUT:%.*]] = "op_a"([[ARG]])
    // CHECK: local_store [[OUT]]
    %0 = "op_a"(%k) {ttg.partition = 0} : (!ty) -> !ty

    // CHECK: [[VALUE:%.*]] = ttg.local_load
    // CHECK: "op_b"([[VALUE]])
    "op_b"(%0) {ttg.partition = 1} : (!ty) -> !ty

    // CHECK: yield [[OUT]]
    scf.yield %0 : !ty
  } {ttg.partition.stages = [0, 1]}
  tt.return
}

}
