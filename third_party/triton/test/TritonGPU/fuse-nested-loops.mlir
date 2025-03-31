// RUN: triton-opt %s --allow-unregistered-dialect --tritongpu-fuse-nested-loops -cse | FileCheck %s

// CHECK-LABEL: @empty_function
tt.func @empty_function() {
  tt.return
}

// CHECK-LABEL: @no_fusion
tt.func @no_fusion(%lb: index, %ub: index, %step: index) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: before.loop
  "before.loop"() : () -> ()
  // CHECK-NEXT: scf.for
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %c0) -> index {
    // CHECK-NEXT: body
    %1 = "body"(%i, %k) : (index, index) -> index
    // CHECK-NEXT: yield
    scf.yield %1 : index
  // CHECK-NEXT: }
  } {"ttg.always-fuse"}
  // CHECK-NEXT: after.loop
  "after.loop"() : () -> ()
  tt.return %0 : index
}

// CHECK-LABEL: @fuse_one_level_simple
// CHECK-SAME: [[LBI:%.*]]: i64, [[UBI:%.*]]: i64, [[STEPI:%.*]]: i64, [[LBJ:%.*]]: i64, [[UBJ:%.*]]: i64, [[STEPJ:%.*]]: i64
tt.func @fuse_one_level_simple(%lbi: i64, %ubi: i64, %stepi: i64, %lbj: i64, %ubj: i64, %stepj: i64) {
  // len_i = len(range(lbi, ubi, stepi))
  //
  // CHECK-NEXT: [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]]
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]]

  // len_j = len(range(lbj0, ubj0, stepj0))
  //
  // CHECK-NEXT: [[DIFF_J:%.*]] = arith.subi [[UBJ]], [[LBJ]]
  // CHECK-NEXT: [[LEN_J:%.*]] = arith.ceildivsi [[DIFF_J]], [[STEPJ]]

  // inner_len = max(1, len_j0)
  //
  // CHECK-NEXT: [[PLEN0:%.*]] = arith.constant 0 : i64
  // CHECK:      [[LEN_J_CLAMP:%.*]] = arith.maxsi %c1_i64, [[LEN_J]]
  // CHECK-NEXT: [[PLEN1:%.*]] = arith.addi [[PLEN0]], [[LEN_J_CLAMP]]
  // CHECK-NEXT: [[INNER_LEN:%.*]] = arith.subi [[PLEN1]], %c0_i64

  // total_iters = len_i * max(1, inner_len)
  //
  // CHECK: [[TOTAL_ITERS:%.*]] = arith.muli [[LEN_I]], [[INNER_LEN]]

  // T = -1
  // i = lbi - stepi
  // j = None
  // for _ in range(total_iters):
  //
  // CHECK: [[I_INIT:%.*]] = arith.subi [[LBI]], [[STEPI]]
  // CHECK: scf.for %{{.*}} = %c0_i64 to [[TOTAL_ITERS]] step %c1_i64 iter_args(
  // CHECK-SAME: [[T_ARG:%.*]] = %c-1_i64, [[I_ARG:%.*]] = [[I_INIT]], [[J_ARG:%.*]] = %c0_i64) -> (i64, i64, i64) : i64 {
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    // T = 0 if T == (inner_len - 1) else T + 1
    //
    // CHECK:      [[T_PLUS_1:%.*]] = arith.addi [[T_ARG]], %c1_i64
    // CHECK-NEXT: [[T_END:%.*]] = arith.subi [[INNER_LEN]], %c1_i64
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[T_ARG]], [[T_END]]
    // CHECK-NEXT: [[T:%.*]] = arith.select [[ROLLOVER]], %c0_i64, [[T_PLUS_1]]

    // if T == 0:
    //   i += stepi
    //   prologue(i)
    //   j = lbj
    //
    // CHECK:      [[START:%.*]] = arith.subi %c0_i64, %c0_i64 : i64
    // CHECK-NEXT: [[PROLOGUE_COND:%.*]] = arith.cmpi eq, [[T]], [[START]]
    // CHECK-NEXT: [[JI:%.*]]:2 = scf.if [[PROLOGUE_COND]] -> (i64, i64) {
    // CHECK-NEXT:   [[I:%.*]] = arith.addi [[I_ARG]], [[STEPI]]
    // CHECK-NEXT:   "prologue"([[I]]) : (i64) -> ()
    // CHECK-NEXT:   yield [[LBJ]], [[I]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[J_ARG]], [[I_ARG]]
    // CHECK-NEXT: }
    "prologue"(%i) : (i64) -> ()

    // if T >= 0 and T < len_j:
    //   body(i, j)
    //   j += stepj
    //
    // CHECK:      [[END:%.*]] = arith.addi [[START]], [[LEN_J]]
    // CHECK-NEXT: [[GE:%.*]] = arith.cmpi sge, [[T]], [[START]]
    // CHECK-NEXT: [[LT:%.*]] = arith.cmpi slt, [[T]], [[END]]
    // CHECK-NEXT: [[COND:%.*]] = arith.andi [[GE]], [[LT]]
    // CHECK-NEXT: [[J_NEXT:%.*]] = scf.if [[COND]] -> (i64) {
    // CHECK-NEXT:   "body"([[JI]]#1, [[JI]]#0) : (i64, i64) -> ()
    // CHECK-NEXT:   [[J_INCR:%.*]] = arith.addi [[JI]]#0, [[STEPJ]]
    // CHECK-NEXT:   yield [[J_INCR]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[JI]]#0
    // CHECK-NEXT: }
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }

    // if T == max(1, len_j) - 1:
    //   epilogue(i)
    //   i += stepi
    //
    // CHECK-NEXT: [[EPILOGUE_COND:%.*]] = arith.cmpi eq, [[T]], [[T_END]]
    // CHECK-NEXT: scf.if [[EPILOGUE_COND]] {
    // CHECK-NEXT:   "epilogue"([[JI]]#1) : (i64) -> ()
    // CHECK-NEXT: } else {
    // CHECK-NEXT: }
    "epilogue"(%i) : (i64) -> ()

    // CHECK-NEXT: yield [[T]], [[JI]]#1, [[J_NEXT]] : i64, i64, i64
  } {"ttg.always-fuse"}
  tt.return
}

// CHECK-LABEL: @fuse_one_level_inouts
// CHECK-SAME: [[LBI:%.*]]: i64, [[UBI:%.*]]: i64, [[STEPI:%.*]]: i64, [[LBJ:%.*]]: i64, [[UBJ:%.*]]: i64, [[STEPJ:%.*]]: i64
// CHECK-SAME: [[INOUT:%.*]]: index
tt.func @fuse_one_level_inouts(%lbi: i64, %ubi: i64, %stepi: i64, %lbj: i64, %ubj: i64, %stepj: i64, %inout: index) -> index {
  // CHECK: [[I_INIT:%.*]] = arith.subi [[LBI]], [[STEPI]]
  // CHECK: [[OUTER_OUTS:%.*]]:6 = scf.for %{{.*}} = %c0_i64 to [[TOTAL_ITERS:%.*]] step %c1_i64 iter_args(
  // CHECK-SAME: [[T_ARG:%arg[0-9]+]] = %c-1_i64,
  // CHECK-SAME: [[I_ARG:%arg[0-9]+]] = [[I_INIT]]
  // CHECK-SAME: [[M:%arg[0-9]+]] = [[INOUT]]
  // CHECK-SAME: [[J_ARG:%arg[0-9]+]] = %c0_i64
  // CHECK-SAME: [[K_ARG:%arg[0-9]+]] = %c0
  // CHECK-SAME: [[PROLOGUE_OUT_ARG:%arg[0-9]+]] = %c0
  // CHECK-SAME: ) -> (i64, i64, index, i64, index, index) : i64 {
  %outer_out = scf.for %i = %lbi to %ubi step %stepi iter_args(%m = %inout) -> index : i64 {
    // if T == 0:
    //   i += stepi
    //   prologue(i)
    //   j = lbj
    //
    // CHECK:      [[PROLOGUE_OUTS:%.*]]:4 = scf.if %{{[0-9]+}} -> (i64, index, index, i64) {
    // CHECK-NEXT:   [[I:%.*]] = arith.addi [[I_ARG]], [[STEPI]]
    // CHECK-NEXT:   [[PROLOGUE_RES:%.*]] = "prologue"([[I]], [[INOUT]], [[M]]) : (i64, index, index) -> index
    // CHECK-NEXT:   yield [[LBJ]], [[PROLOGUE_RES]], [[M]], [[I]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[J_ARG]], [[PROLOGUE_OUT_ARG]], [[K_ARG]], [[I_ARG]]
    // CHECK-NEXT: }
    //
    // J := [[PROLOGUE_OUTS]]#0
    // PROLOGUE_OUT := [[PROLOGUE_OUTS]]#1
    // K := [[PROLOGUE_OUTS]]#2
    // I := [[PROLOGUE_OUTS]]#3
    %prologue_out = "prologue"(%i, %inout, %m) : (i64, index, index) -> index

    // if T >= 0 and T < len_j:
    //   body(i, j)
    //   j += stepj
    //
    // CHECK:      [[BODY_OUTS:%.*]]:2 = scf.if {{.*}} -> (i64, index) {
    // CHECK-NEXT:   [[BODY_OUT:%.*]] = "body"([[PROLOGUE_OUTS]]#3, [[PROLOGUE_OUTS]]#0, [[PROLOGUE_OUTS]]#2, [[PROLOGUE_OUTS]]#1, [[M]]) : (i64, i64, index, index, index) -> index
    // CHECK-NEXT:   [[J_INCR:%.*]] = arith.addi [[PROLOGUE_OUTS]]#0, [[STEPJ]]
    // CHECK-NEXT:   yield [[J_INCR]], [[BODY_OUT]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[PROLOGUE_OUTS]]#0, [[K_ARG]]
    // CHECK-NEXT: }
    %inner_out = scf.for %j = %lbj to %ubj step %stepj iter_args(%k = %m) -> index : i64 {
      %body_out = "body"(%i, %j, %k, %prologue_out, %m) : (i64, i64, index, index, index) -> index
      scf.yield %body_out : index
    }

    // if T == max(1, len_j) - 1:
    //   epilogue(i)
    //   i += stepi
    //
    // CHECK:      [[EPILOGUE_OUTS:%.*]] = scf.if {{.*}} -> (index) {
    // CHECK-NEXT:   [[EPILOGUE_OUT:%.*]] = "epilogue"([[PROLOGUE_OUTS]]#3, [[PROLOGUE_OUTS]]#1, [[BODY_OUTS]]#1, [[M]]) : (i64, index, index, index) -> index
    // CHECK-NEXT:   yield [[EPILOGUE_OUT]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[M]]
    // CHECK-NEXT: }
    %epilogue_out = "epilogue"(%i, %prologue_out, %inner_out, %m) : (i64, index, index, index) -> index

    // CHECK-NEXT: yield %{{.*}}, [[PROLOGUE_OUTS]]#3, [[EPILOGUE_OUTS]], [[BODY_OUTS]]#0, [[BODY_OUTS]]#1, [[PROLOGUE_OUTS]]#1 : i64, i64, index, i64, index, index
    scf.yield %epilogue_out : index
  } {"ttg.always-fuse"}
  // CHECK: return [[OUTER_OUTS]]#2
  tt.return %outer_out : index
}

// CHECK-LABEL: @multiple_loops
tt.func @multiple_loops(
    // CHECK-SAME: [[LBI:%arg[0-9]+]]: i64, [[UBI:%arg[0-9]+]]: i64, [[STEPI:%arg[0-9]+]]: i64,
    // CHECK-SAME: [[LBJ0:%arg[0-9]+]]: i64, [[UBJ0:%arg[0-9]+]]: i64, [[STEPJ0:%arg[0-9]+]]: i64,
    // CHECK-SAME: [[LBJ1:%arg[0-9]+]]: i64, [[UBJ1:%arg[0-9]+]]: i64, [[STEPJ1:%arg[0-9]+]]: i64,
    // CHECK-SAME: [[LBJ2:%arg[0-9]+]]: i64, [[UBJ2:%arg[0-9]+]]: i64, [[STEPJ2:%arg[0-9]+]]: i64,
    // CHECK-SAME: [[M0:%arg[0-9]+]]: f32
    %lbi: i64, %ubi: i64, %stepi: i64,
    %lbj0: i64, %ubj0: i64, %stepj0: i64,
    %lbj1: i64, %ubj1: i64, %stepj1: i64,
    %lbj2: i64, %ubj2: i64, %stepj2: i64,
    %m0: f32) -> f32 {
  // CHECK:      [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]]
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]]
  // CHECK-NEXT: [[DIFF_J0:%.*]] = arith.subi [[UBJ0]], [[LBJ0]]
  // CHECK-NEXT: [[LEN_J0:%.*]] = arith.ceildivsi [[DIFF_J0]], [[STEPJ0]]
  // CHECK-NEXT: [[DIFF_J1:%.*]] = arith.subi [[UBJ1]], [[LBJ1]]
  // CHECK-NEXT: [[LEN_J1:%.*]] = arith.ceildivsi [[DIFF_J1]], [[STEPJ1]]
  // CHECK-NEXT: [[DIFF_J2:%.*]] = arith.subi [[UBJ2]], [[LBJ2]]
  // CHECK-NEXT: [[LEN_J2:%.*]] = arith.ceildivsi [[DIFF_J2]], [[STEPJ2]]

  // CHECK:      [[PLEN0:%.*]] = arith.constant 0 : i64
  // CHECK:      [[LEN_J0_CLAMP:%.*]] = arith.maxsi %c1_i64, [[LEN_J0]]
  // CHECK-NEXT: [[PLEN1:%.*]] = arith.addi [[PLEN0]], [[LEN_J0_CLAMP]]
  // CHECK-NEXT: [[LEN_J1_CLAMP:%.*]] = arith.maxsi %c1_i64, [[LEN_J1]]
  // CHECK-NEXT: [[PLEN2:%.*]] = arith.addi [[PLEN1]], [[LEN_J1_CLAMP]]
  // CHECK-NEXT: [[LEN_J2_CLAMP:%.*]] = arith.maxsi %c1_i64, [[LEN_J2]]
  // CHECK-NEXT: [[PLEN3:%.*]] = arith.addi [[PLEN2]], [[LEN_J2_CLAMP]]
  // CHECK:      [[INNER_LEN:%.*]] = arith.subi [[PLEN3]], %c2_i64
  // CHECK-NEXT: [[TOTAL_ITERS:%.*]] = arith.muli [[LEN_I]], [[INNER_LEN]]

  // CHECK:      [[I_INIT:%.*]] = arith.subi [[LBI]], [[STEPI]]
  // CHECK:      [[OUTS:%.*]]:12 = scf.for %{{.*}} = %c0_i64 to [[TOTAL_ITERS]] step %c1_i64 iter_args(
  // CHECK-SAME: [[T_ARG:%arg[0-9]+]] = %c-1_i64,
  // CHECK-SAME: [[I_ARG:%arg[0-9]+]] = [[I_INIT]],
  // CHECK-SAME: [[M:%arg[0-9]+]] = [[M0]],
  // CHECK-SAME: [[J0_ARG:%arg[0-9]+]] = %c0_i64,
  // CHECK-SAME: [[J1_ARG:%arg[0-9]+]] = %c0_i64,
  // CHECK-SAME: [[J2_ARG:%arg[0-9]+]] = %c0_i64,
  // CHECK-SAME: [[BODY0_ARG:%arg[0-9]+]] = %cst,
  // CHECK-SAME: [[BODY1_ARG:%arg[0-9]+]] = %cst,
  // CHECK-SAME: [[BODY2_ARG:%arg[0-9]+]] = %cst,
  // CHECK-SAME: [[PROLOGUE0_ARG:%arg[0-9]+]] = %cst,
  // CHECK-SAME: [[PROLOGUE1_ARG:%arg[0-9]+]] = %cst,
  // CHECK-SAME: [[PROLOGUE2_ARG:%arg[0-9]+]] = %cst)
  %mN = scf.for %i = %lbi to %ubi step %stepi iter_args(%m = %m0) -> f32 : i64 {

    // CHECK:      [[T_PLUS_1:%.*]] = arith.addi [[T_ARG]], %c1_i64
    // CHECK-NEXT: [[T_END:%.*]] = arith.subi [[INNER_LEN]], %c1_i64
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[T_ARG]], [[T_END]]
    // CHECK-NEXT: [[T:%.*]] = arith.select [[ROLLOVER]], %c0_i64, [[T_PLUS_1]]

    // CHECK:      [[START0:%.*]] = arith.subi [[PLEN0]], %c0_i64
    // CHECK-NEXT: [[PROLOGUE_COND0:%.*]] = arith.cmpi eq, [[T]], [[START0]]
    // CHECK-NEXT: [[PROLOGUE0_OUTS:%.*]]:4 = scf.if [[PROLOGUE_COND0]]
    // CHECK-NEXT:   [[I:%.*]] = arith.addi [[I_ARG]], [[STEPI]]
    // CHECK-NEXT:   [[RES:%.*]] = "prologue0"([[I]], [[M]])
    // CHECK-NEXT:   yield [[LBJ0]], [[RES]], [[RES]], [[I]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[J0_ARG]], [[PROLOGUE0_ARG]], [[BODY0_ARG]], [[I_ARG]]
    %k00 = "prologue0"(%i, %m) : (i64, f32) -> f32

    // CHECK:      [[END0:%.*]] = arith.addi [[START0]], [[LEN_J0]]
    // CHECK-NEXT: [[GE0:%.*]] = arith.cmpi sge, [[T]], [[START0]]
    // CHECK-NEXT: [[LT0:%.*]] = arith.cmpi slt, [[T]], [[END0]]
    // CHECK-NEXT: [[BODY_COND0:%.*]] = arith.andi [[GE0]], [[LT0]]
    // CHECK-NEXT: [[BODY0_OUTS:%.*]]:2 = scf.if [[BODY_COND0]]
    // CHECK-NEXT:   [[RES:%.*]] = "body0"([[PROLOGUE0_OUTS]]#3, [[PROLOGUE0_OUTS]]#0, [[PROLOGUE0_OUTS]]#2)
    // CHECK-NEXT:   [[NEXT_J0:%.*]] = arith.addi [[PROLOGUE0_OUTS]]#0, [[STEPJ0]]
    // CHECK-NEXT:   yield [[NEXT_J0]], [[RES]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[PROLOGUE0_OUTS]]#0, [[BODY0_ARG]]
    %k0N = scf.for %j0 = %lbj0 to %ubj0 step %stepj0 iter_args(%k0 = %k00) -> f32 : i64 {
      %res = "body0"(%i, %j0, %k0) : (i64, i64, f32) -> f32
      scf.yield %res : f32
    }

    // CHECK:      [[START1:%.*]] = arith.subi [[PLEN1]], %c1_i64
    // CHECK-NEXT: [[PROLOGUE_COND1:%.*]] = arith.cmpi eq, [[T]], [[START1]]
    // CHECK-NEXT: [[PROLOGUE1_OUTS:%.*]]:3 = scf.if [[PROLOGUE_COND1]]
    // CHECK-NEXT:   [[RES:%.*]] = "prologue1"([[PROLOGUE0_OUTS]]#3, [[BODY0_OUTS]]#1)
    // CHECK-NEXT:   yield [[LBJ1]], [[RES]], [[RES]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[J1_ARG]], [[PROLOGUE1_ARG]], [[BODY1_ARG]]
    %k10 = "prologue1"(%i, %k0N) : (i64, f32) -> f32

    // CHECK:      [[END1:%.*]] = arith.addi [[START1]], [[LEN_J1]]
    // CHECK-NEXT: [[GE1:%.*]] = arith.cmpi sge, [[T]], [[START1]]
    // CHECK-NEXT: [[LT1:%.*]] = arith.cmpi slt, [[T]], [[END1]]
    // CHECK-NEXT: [[BODY_COND1:%.*]] = arith.andi [[GE1]], [[LT1]]
    // CHECK-NEXT: [[BODY1_OUTS:%.*]]:2 = scf.if [[BODY_COND1]]
    // CHECK-NEXT:   [[RES:%.*]] = "body1"([[PROLOGUE0_OUTS]]#3, [[PROLOGUE1_OUTS]]#0, [[PROLOGUE1_OUTS]]#2)
    // CHECK-NEXT:   [[NEXT_J1:%.*]] = arith.addi [[PROLOGUE1_OUTS]]#0, [[STEPJ1]]
    // CHECK-NEXT:   yield [[NEXT_J1]], [[RES]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[PROLOGUE1_OUTS]]#0, [[BODY1_ARG]]
    %k1N = scf.for %j1 = %lbj1 to %ubj1 step %stepj1 iter_args(%k1 = %k10) -> f32 : i64 {
      %res = "body1"(%i, %j1, %k1) : (i64, i64, f32) -> f32
      scf.yield %res : f32
    }

    // CHECK:      [[START2:%.*]] = arith.subi [[PLEN2]], %c2_i64
    // CHECK-NEXT: [[PROLOGUE_COND2:%.*]] = arith.cmpi eq, [[T]], [[START2]]
    // CHECK-NEXT: [[PROLOGUE2_OUTS:%.*]]:3 = scf.if [[PROLOGUE_COND2]]
    // CHECK-NEXT:   [[RES:%.*]] = "prologue2"([[PROLOGUE0_OUTS]]#3, [[BODY1_OUTS]]#1)
    // CHECK-NEXT:   yield [[LBJ2]], [[RES]], [[RES]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[J2_ARG]], [[PROLOGUE2_ARG]], [[BODY2_ARG]]
    %k20 = "prologue2"(%i, %k1N) : (i64, f32) -> f32

    // CHECK:      [[END2:%.*]] = arith.addi [[START2]], [[LEN_J2]]
    // CHECK-NEXT: [[GE2:%.*]] = arith.cmpi sge, [[T]], [[START2]]
    // CHECK-NEXT: [[LT2:%.*]] = arith.cmpi slt, [[T]], [[END2]]
    // CHECK-NEXT: [[BODY_COND2:%.*]] = arith.andi [[GE2]], [[LT2]]
    // CHECK-NEXT: [[BODY2_OUTS:%.*]]:2 = scf.if [[BODY_COND2]]
    // CHECK-NEXT:   [[RES:%.*]] = "body2"([[PROLOGUE0_OUTS]]#3, [[PROLOGUE2_OUTS]]#0, [[PROLOGUE2_OUTS]]#2)
    // CHECK-NEXT:   [[NEXT_J2:%.*]] = arith.addi [[PROLOGUE2_OUTS]]#0, [[STEPJ2]]
    // CHECK-NEXT:   yield [[NEXT_J2]], [[RES]]
    // CHECK-NEXT: else
    // CHECK-NEXT:   yield [[PROLOGUE2_OUTS]]#0, [[BODY2_ARG]]
    %k2N = scf.for %j2 = %lbj2 to %ubj2 step %stepj2 iter_args(%k2 = %k20) -> f32 : i64 {
      %res = "body2"(%i, %j2, %k2) : (i64, i64, f32) -> f32
      scf.yield %res : f32
    }

    // CHECK:      [[EPILOGUE_COND:%.*]] = arith.cmpi eq, [[T]], [[T_END]]
    // CHECK-NEXT: [[EPILOGUE_OUTS:%.*]] = scf.if [[EPILOGUE_COND]]
    // CHECK-NEXT:   [[RES:%.*]] = "epilogue"([[PROLOGUE0_OUTS]]#3, [[BODY2_OUTS]]#1)
    // CHECK-NEXT:   yield [[RES]]
    // CHECK-NEXT:  else
    // CHECK-NEXT:   yield [[M]]
    %out = "epilogue"(%i, %k2N) : (i64, f32) -> f32

    // CHECK:      scf.yield [[T]], [[PROLOGUE0_OUTS]]#3, [[EPILOGUE_OUTS]],
    // CHECK-SAME:           [[BODY0_OUTS]]#0, [[BODY1_OUTS]]#0, [[BODY2_OUTS]]#0,
    // CHECK-SAME:           [[PROLOGUE0_OUTS]]#1, [[PROLOGUE1_OUTS]]#1, [[PROLOGUE2_OUTS]]#1 :
    scf.yield %out : f32
  } {"ttg.always-fuse"}
  // CHECK: return [[OUTS]]#2
  tt.return %mN : f32
}

// CHECK-LABEL: @two_loop_nests
tt.func @two_loop_nests(%lbi: i64, %ubi: i64, %stepi: i64, %lbj: i64, %ubj: i64, %stepj: i64) {
  // CHECK-COUNT-2: scf.for
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }
  } {"ttg.always-fuse"}
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }
  } {"ttg.always-fuse"}
  // CHECK-NOT: scf.for
  // CHECK: tt.return
  tt.return
}

// CHECK-LABEL: @hoist_loop_bound_computations
// CHECK-SAME: [[LBI:%.*]]: i64, [[UBI:%.*]]: i64, [[STEPI:%.*]]: i64
tt.func @hoist_loop_bound_computations(%lbi: i64, %ubi: i64, %stepi: i64) {
  // CHECK-NEXT: [[LBJ:%.*]] = arith.addi [[LBI]], [[STEPI]]
  // CHECK-NEXT: [[UBJ:%.*]] = arith.addi [[UBI]], [[STEPI]]
  // CHECK-NEXT: [[STEPJ:%.*]] = arith.addi [[STEPI]], [[STEPI]]

  // CHECK-NEXT: [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]]
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]]
  // CHECK-NEXT: [[DIFF_J:%.*]] = arith.subi [[UBJ]], [[LBJ]]
  // CHECK-NEXT: [[LEN_J:%.*]] = arith.ceildivsi [[DIFF_J]], [[STEPJ]]

  // CHECK: scf.for
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    %lbj = arith.addi %lbi, %stepi : i64
    %ubj = arith.addi %ubi, %stepi : i64
    %stepj = arith.addi %stepi, %stepi : i64
    // CHECK: [[J:%.*]]:2 = scf.if
    // CHECK:   yield [[LBJ]]

    // CHECK: scf.if
    // CHECK-NEXT: "body"
    // CHECK-NEXT: arith.addi [[J]]#0, [[STEPJ]]
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }
  } {"ttg.always-fuse"}
  tt.return
}

// CHECK-LABEL: @cannot_fuse
tt.func @cannot_fuse(%lbi: i64, %ubi: i64, %stepi: i64) {
  // CHECK-COUNT-2: scf.for
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    %lbj = arith.addi %lbi, %stepi : i64
    %ubj = arith.addi %ubi, %i : i64
    %stepj = arith.addi %stepi, %stepi : i64
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }
  } {"ttg.always-fuse"}
  // CHECK-NOT: scf.for
  tt.return
}

// CHECK-LABEL: @upcast_i16_to_i32
// CHECK-SAME: [[LBI:%.*]]: i32, [[UBI:%.*]]: i32, [[STEPI:%.*]]: i32, [[LBJ:%.*]]: i16, [[UBJ:%.*]]: i16, [[STEPJ:%.*]]: i16
tt.func @upcast_i16_to_i32(%lbi: i32, %ubi: i32, %stepi: i32, %lbj: i16, %ubj: i16, %stepj: i16) {
  // CHECK-NEXT: [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]] : i32
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]] : i32
  // CHECK-NEXT: [[DIFF_J:%.*]] = arith.subi [[UBJ]], [[LBJ]] : i16
  // CHECK-NEXT: [[LEN_J:%.*]] = arith.ceildivsi [[DIFF_J]], [[STEPJ]] : i16

  // CHECK: arith.extsi [[LEN_J]] : i16 to i32
  scf.for %i = %lbi to %ubi step %stepi : i32 {
    scf.for %j = %lbj to %ubj step %stepj : i16 {
      "body"(%i, %j) : (i32, i16) -> ()
    }
  } {"ttg.always-fuse"}
  tt.return
}

// CHECK-LABEL: @upcast_index_to_i64
// CHECK-SAME: [[LBI:%.*]]: index, [[UBI:%.*]]: index, [[STEPI:%.*]]: index, [[LBJ:%.*]]: index, [[UBJ:%.*]]: index, [[STEPJ:%.*]]: index
tt.func @upcast_index_to_i64(%lbi: index, %ubi: index, %stepi: index, %lbj: index, %ubj: index, %stepj: index) {
  // CHECK-NEXT: [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]] : index
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]] : index
  // CHECK-NEXT: [[DIFF_J:%.*]] = arith.subi [[UBJ]], [[LBJ]] : index
  // CHECK-NEXT: [[LEN_J:%.*]] = arith.ceildivsi [[DIFF_J]], [[STEPJ]] : index

  // CHECK: arith.index_cast [[LEN_J]] : index to i64
  // CHECK: arith.index_cast [[LEN_I]] : index to i64
  scf.for %i = %lbi to %ubi step %stepi {
    scf.for %j = %lbj to %ubj step %stepj {
      "body"(%i, %j) : (index, index) -> ()
    }
  } {"ttg.always-fuse"}
  tt.return
}

// CHECK-LABEL: @triple_loop_nest
tt.func @triple_loop_nest(
    %lbi: i64, %ubi: i64, %stepi: i64,
    %lbj: i64, %ubj: i64, %stepj: i64,
    %lbk: i64, %ubk: i64, %stepk: i64) {
 // CHECK-COUNT-1: scf.for
 scf.for %i = %lbi to %ubi step %stepi : i64 {
   scf.for %j = %lbj to %ubj step %stepj : i64 {
      scf.for %k = %lbk to %ubk step %stepk : i64 {
        "body"(%i, %j, %k) : (i64, i64, i64) -> ()
      }
    }
  } {"ttg.always-fuse"}
  // CHECK-NOT: scf.for
  // CHECK: tt.return
  tt.return
}

// CHECK-LABEL: @preserve_stage_count
tt.func @preserve_stage_count(%lb: i32, %ub: i32) {
  %c1_i32 = arith.constant 1 : i32

  // CHECK-COUNT-1: scf.for
  scf.for %i = %lb to %ub step %c1_i32 : i32 {
    scf.for %j = %lb to %ub step %c1_i32 : i32 {
      "body"(%j) : (i32) -> ()
      scf.yield
    } {tt.num_stages = 4 : i32}
    scf.for %j = %lb to %ub step %c1_i32 : i32 {
      "body"(%j) : (i32) -> ()
      scf.yield
    } {tt.num_stages = 5 : i32}
  } {"ttg.always-fuse", "tt.disallow_acc_multi_buffer", tt.num_stages = 6 : i32}
  // CHECK: tt.disallow_acc_multi_buffer
  // CHECK: tt.num_stages = 6 : i32
  // CHECK-NOT: scf.for
  tt.return
}

// CHECK-LABEL: @fuse_attr_speculate
// CHECK-SAME: [[LB:%.*]]: i32, [[UB:%.*]]: i32
tt.func @fuse_attr_speculate(%lb: i32, %ub: i32) {
  %c1_i32 = arith.constant 1 : i32

  // CHECK: [[DIFF:%.*]] = arith.subi [[UB]], [[LB]]
  // CHECK: [[LEN:%.*]] = arith.ceildivsi [[DIFF]], %c1_i32
  // CHECK: [[IS_ZERO:%.*]] = arith.cmpi eq, [[LEN]], %c0_i32

  // CHECK: scf.if [[IS_ZERO]]
  // CHECK-NEXT: scf.for %{{.*}} = [[LB]] to [[UB]] step %c1_i32
  // CHECK-NEXT:   "prologue"
  // CHECK-NXET: }

  // CHECK: else
  // CHECK-COUNT-1: scf.for
  // CHECK-NOT: scf.for
  scf.for %i = %lb to %ub step %c1_i32 : i32 {
    // CHECK: "prologue"
    "prologue"(%i) : (i32) -> ()
    // CHECK: scf.if %true
    scf.for %j = %lb to %ub step %c1_i32 : i32 {
      // CHECK-NEXT: "body"
      "body"(%i, %j) : (i32, i32) -> ()
      scf.yield
    }
  } {tt.flatten}
  tt.return
}

// CHECK-LABEL: @speculate_hoist
// CHECK-SAME: [[LB:%.*]]: i32, [[UB:%.*]]: i32
tt.func @speculate_hoist(%lb: i32, %ub: i32) {
  %c1_i32 = arith.constant 1 : i32

  // CHECK: [[UBJ:%.*]] = arith.addi [[LB]], [[UB]]
  // CHECK: [[DIFF:%.*]] = arith.subi [[UBJ]], [[LB]]
  // CHECK: [[LEN:%.*]] = arith.ceildivsi [[DIFF]], %c1_i32
  // CHECK: [[IS_ZERO:%.*]] = arith.cmpi eq, [[LEN]], %c0_i32

  // CHECK: scf.if [[IS_ZERO]]
  scf.for %i = %lb to %ub step %c1_i32 : i32 {
    "prologue"(%i) : (i32) -> ()
    %ubj = arith.addi %lb, %ub : i32
    scf.for %j = %lb to %ubj step %c1_i32 : i32 {
      "body"(%i, %j) : (i32, i32) -> ()
      scf.yield
    }
  } {tt.flatten}
  tt.return
}

// CHECK-LABEL: @sink_prologue_to_epilogue
// CHECK-SAME: [[UB:%.*]]: i32
tt.func @sink_prologue_to_epilogue(%ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK: else
  // CHECK: scf.for
  %0 = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%k = %c0_i32) -> i32 : i32 {
    // CHECK: [[PROLOGUE_OUTS:%.*]]:2 = scf.if
    %0 = arith.addi %i, %ub : i32
    // CHECK: scf.if %true
    // CHECK-NEXT: "body"
    scf.for %j = %c0_i32 to %ub step %c1_i32 : i32 {
      "body"(%i, %j) : (i32, i32) -> ()
      scf.yield
    }
    // CHECK: scf.if
    // CHECK-NEXT: [[V0:%.*]] = arith.addi [[PROLOGUE_OUTS]]#1, [[UB]]
    // CHECK-NEXT: [[V1:%.*]] = arith.addi [[V0]], [[UB]]
    %1 = arith.addi %0, %ub : i32
    // CHECK-NEXT: "epilogue"([[V1]])
    "epilogue"(%1) : (i32) -> ()
    scf.yield %0 : i32
  } {tt.flatten}

  tt.return
}

// -----

// CHECK-LABEL: @prologue_output
tt.func @prologue_output(%ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK: scf.for
  %0 = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%k = %c0_i32) -> i32 : i32 {
    // CHECK: scf.if
    // CHECK: {increment}
    %next = arith.addi %k, %c1_i32 {increment} : i32
    // CHECK: scf.if
    scf.for %j = %c0_i32 to %ub step %c1_i32 : i32 {
      // CHECK-NEXT: "body"
      "body"(%i, %j) : (i32, i32) -> ()
    }
    // CHECK: scf.if {{%[0-9]+}} {
    // CHECK-NEXT: "epilogue"
    "epilogue"(%i) : (i32) -> ()
    // CHECK-NEXT: } else {
    scf.yield %next : i32
  } {"ttg.always-fuse"}

  tt.return
}
