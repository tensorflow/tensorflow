// GenericAtomicRMWOp should contain only ops with no side effects.
// Unfortunately, the legalization pattern for SelectAndScatterOp has to adapt
// to LMHLO dialect using allocs/deallocs inside of GenericAtomicRMWOp body.
// Lowering to STD dialect and store forwarding pass would be required to get
// rid of them. This is exactly what is done in the real MLIR GPU pipeline, but
// here we disable verification with `verify-each=0` to check the output IR.
// RUN: mlir-hlo-opt %s -lhlo-legalize-to-parallel-loops -canonicalize --verify-each=0 | FileCheck %s

func.func @select_and_scatter(%arg: memref<112x112xf32>,
                         %src: memref<56x56xf32>,
                         %init: memref<f32>,
                         %result: memref<112x112xf32>) {
  "lmhlo.select_and_scatter"(%arg, %src, %init, %result) ({
    // select
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %pred: memref<i1>):
      "lmhlo.compare"(%lhs, %rhs, %pred) {comparison_direction = #mhlo<"comparison_direction GE">} :
          (memref<f32>, memref<f32>, memref<i1>) -> ()
      "lmhlo.terminator"() : () -> ()
  }, {
    // scatter
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %out: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %out) :
          (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
  }) {
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    window_dimensions = dense<[3, 3]> : tensor<2xi64>,
    window_strides = dense<[2, 2]> : tensor<2xi64>
  } : (memref<112x112xf32>,
       memref<56x56xf32>,
       memref<f32>, memref<112x112xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}
// CHECK-LABEL: func.func
// CHECK: ^bb0(%[[ARG_BUF:.*]]: memref<112x112xf32>, %[[SRC_BUF:.*]]: memref<56x56xf32>, %[[INIT_BUF:.*]]: memref<f32>, %[[RESULT_BUF:.*]]: memref<112x112xf32>):

// Constants.
// CHECK-DAG: %[[C0_F32:.*]] = "arith.constant"() {value = 0.000000e+00 : f32}
// CHECK-DAG: %[[C0:.*]] = "arith.constant"() {value = 0 : index}
// CHECK-DAG: %[[C1:.*]] = "arith.constant"() {value = 1 : index}
// CHECK-DAG: %[[C2:.*]] = "arith.constant"() {value = 2 : index}
// CHECK-DAG: %[[C3:.*]] = "arith.constant"() {value = 3 : index}
// CHECK-DAG: %[[C56:.*]] = "arith.constant"() {value = 56 : index}
// CHECK-DAG: %[[C112:.*]] = "arith.constant"() {value = 112 : index}
// CHECK-DAG: %[[CFALSE:.*]] = "arith.constant"() {value = false}
// CHECK-DAG: %[[CTRUE:.*]] = "arith.constant"() {value = true}

// Parallel loop to initialize the output buffer.
// CHECK: %[[INIT:.*]] = "memref.load"(%[[INIT_BUF]]) : (memref<f32>) -> f32
// CHECK: "scf.parallel"(%[[C0]], %[[C0]], %[[C112]], %[[C112]], %[[C1]], %[[C1]]) ({
// CHECK: ^bb0(%[[I:.*]]: index, %[[J:.*]]: index):
// CHECK:   "memref.store"(%[[INIT]], %[[RESULT_BUF]], %[[I]], %[[J]])
// CHECK:   "scf.yield"() : () -> ()
// CHECK: })

// Parallel loop over source buffer to compute scattered values.
// CHECK: "scf.parallel"(%[[C0]], %[[C0]], %[[C56]], %[[C56]], %[[C1]], %[[C1]]) ({
// CHECK: ^bb0(%[[II:.*]]: index, %[[JJ:.*]]: index):

// Window loop w.r.t. first dim.
// CHECK:   %[[SEL_RES_I:.*]]:4 = "scf.for"(%[[C0]], %[[C3]], %[[C1]], %[[C0]], %[[C0]], %[[C0_F32]], %[[CFALSE]]) ({
// CHECK:      ^bb0(%[[WIN_I:.*]]: index, %[[SEL_I_0:.*]]: index, %[[SEL_J_0:.*]]: index, %[[SEL_VAL_0:.*]]: f32, %[[SEL_INIT_0:.*]]: i1):

// Window loop w.r.t. second dim.
// CHECK: %[[SEL_RES_J:.*]]:4 = "scf.for"(%[[C0]], %[[C3]], %[[C1]], %[[SEL_I_0]], %[[SEL_J_0]], %[[SEL_VAL_0]], %[[SEL_INIT_0]]) ({
// CHECK: ^bb0(%[[WIN_J:.*]]: index, %[[SEL_I:.*]]: index, %[[SEL_J:.*]]: index, %[[SEL_VAL:.*]]: f32, %[[SEL_INIT:.*]]: i1):

// Compute index I of the ARG buffer and check whether it is in padding area.
// CHECK: %[[START_I:.*]] = "arith.muli"(%[[II]], %[[C2]])
// CHECK: %[[ARG_I:.*]] = "arith.addi"(%[[START_I]], %[[WIN_I]])
// CHECK: %[[ARG_I_FITS:.*]] = "arith.cmpi"(%[[ARG_I]], %[[C112]])

// Compute index J of the ARG buffer and check whether it is in padding area.
// CHECK: %[[START_J:.*]] = "arith.muli"(%[[JJ]], %[[C2]])
// CHECK: %[[ARG_J:.*]] = "arith.addi"(%[[START_J]], %[[WIN_J]])
// CHECK: %[[ARG_J_FITS:.*]] = "arith.cmpi"(%[[ARG_J]], %[[C112]])

// Update `INBOUNDS`, i.e. whether or not ARG indices are inside the boundaries
// of the buffer or they are in the padding area.
// CHECK: %[[INBOUNDS_1:.*]] = "arith.andi"(%[[ARG_I_FITS]], %[[ARG_J_FITS]])

// If ARG ivs are in the padding area, then 'select' function does not have to
// be applied, current selected ivs (SEL_I, SEL_J) and value (SEL_VAL) are
// returned in that case.
// CHECK:  %[[IF_INBOUNDS_RES:.*]]:3
// CHECK-SAME:  = "scf.if"(%[[INBOUNDS_1]]) ({


  // INBOUNDS-THEN-BODY, i.e. if INBOUNDS == true

  // CHECK: %[[ARG_ELEM:.*]] = "memref.load"(%[[ARG_BUF]], %[[ARG_I]], %[[ARG_J]])
  // CHECK: %[[IF_INIT_RES:.*]]:3 = "scf.if"(%[[SEL_INIT]]) ({

    // INIT-THEN-BODY, i.e. INBOUNDS == true and INIT = true

    // The LHLO IR of the select block of the lhlo.select_and_scatter is applied
    // to the current selected value (SEL_VAL) and the element of the ARG buffer
    // to compute boolean PRED, whether the new value and ivs should replace the
    // current ones.

    // Allocate buffers for ARG element, current selected value to adapt LHLO
    // code.
    // CHECK: %[[ARG_ELEM_BUF:.*]] = "memref.alloc"() {{.*}} -> memref<f32>
    // CHECK: %[[SEL_VAL_BUF:.*]] = "memref.alloc"() {{.*}} -> memref<f32>
    // CHECK: %[[PRED_BUF:.*]] = "memref.alloc"() {{.*}} -> memref<i1>
    // CHECK: "memref.store"(%[[ARG_ELEM]], %[[ARG_ELEM_BUF]])
    // CHECK: "memref.store"(%[[SEL_VAL]], %[[SEL_VAL_BUF]])

    // Compute PRED.
    // CHECK:  "lmhlo.compare"(
    // CHECK-SAME:     %[[ARG_ELEM_BUF]], %[[SEL_VAL_BUF]], %[[PRED_BUF]])
    // CHECK:  %[[PRED:.*]] = "memref.load"(%[[PRED_BUF]]) : (memref<i1>) -> i1


    // Depending on PRED, return ARG ivs & elem or current select ivs and value.
    // CHECK: %[[IF_PRED_RES0:.*]] = "arith.select"(%[[PRED]], %[[ARG_I]], %[[SEL_I]])
    // CHECK: %[[IF_PRED_RES1:.*]] = "arith.select"(%[[PRED]], %[[ARG_J]], %[[SEL_J]])
    // CHECK: %[[IF_PRED_RES2:.*]] = "arith.select"(%[[PRED]], %[[ARG_ELEM]], %[[SEL_VAL]])

    // INIT-THEN-BODY yield.
    // CHECK:    "scf.yield"(%[[IF_PRED_RES0]], %[[IF_PRED_RES1]], %[[IF_PRED_RES2]])
    // CHECK: }

    // INIT-ELSE-BODY, i.e. if INBOUNDS == TRUE and INIT == FALSE, returns ARG
    // ivs and element without computing Select function.
    // CHECK:   "scf.yield"(%[[ARG_I]], %[[ARG_J]], %[[ARG_ELEM]])
    // CHECK: }

// Window loop w.r.t. first dim yield.
// CHECK:   "scf.yield"(%[[SEL_RES_J:.*]]#0, %[[SEL_RES_J]]#1, %[[SEL_RES_J]]#2, %[[SEL_RES_J]]#3)
// CHECK: }

// Use selected ivs to load element from the SRC buffer.
// CHECK: %[[SRC_ELEM:.*]] = "memref.load"(%[[SRC_BUF]], %[[II]], %[[JJ]]) : (memref<56x56xf32>, index, index) -> f32

// Update of RESULT[SELECTED_I, SELECTED_J] should be done atomically, because
// it may happen that several other threads select the same IVs if the windows
// overlap.
// CHECK: "memref.generic_atomic_rmw"(%[[RESULT_BUF]], %[[SEL_RES_I:.*]]#0, %[[SEL_RES_I]]#1) ({
// CHECK: ^bb0(%[[CUR_RES:.*]]: f32):

// Allocate buffers for ARG element, current selected value to adapt LHLO code.
// CHECK: %[[SRC_ELEM_BUF:.*]] = "memref.alloc"() {{.*}} : () -> memref<f32>
// CHECK: %[[CUR_RES_BUF:.*]] = "memref.alloc"() {{.*}} : () -> memref<f32>
// CHECK: %[[RES_BUF:.*]] = "memref.alloc"() {{.*}} : () -> memref<f32>
// CHECK: "memref.store"(%[[SRC_ELEM]], %[[SRC_ELEM_BUF]]) : (f32, memref<f32>) -> ()
// CHECK: "memref.store"(%[[CUR_RES]], %[[CUR_RES_BUF]]) : (f32, memref<f32>) -> ()

// Compute scatter value.
// CHECK: "lmhlo.add"(%[[SRC_ELEM_BUF]], %[[CUR_RES_BUF]], %[[RES_BUF]])
// CHECK: %[[RES:.*]] = "memref.load"(%[[RES_BUF]]) : (memref<f32>) -> f32


// Atomic RMW terminator that returns updated value.
// CHECK: "memref.atomic_yield"(%[[RES]]) : (f32) -> ()

// Parallel loop over source buffer yield
// CHECK: "scf.yield"() : () -> ()
