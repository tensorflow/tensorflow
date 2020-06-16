// GenericAtomicRMWOp should contain only ops with no side effects.
// Unfortunately, the legalization pattern for SelectAndScatterOp has to adapt
// to XLA LHLO dialect using allocs/deallocs inside of GenericAtomicRMWOp body.
// Lowering to STD dialect and store forwarding pass would be required to get
// rid of them. This is exactly what is done in the real MLIR GPU pipeline, but
// here we disable verification with `verify-each=0` to check the output IR.
// RUN: xla-opt %s -lhlo-legalize-to-parallel-loops -canonicalize --verify-each=0 | FileCheck %s

func @select_and_scatter(%arg: memref<112x112xf32>,
                         %src: memref<56x56xf32>,
                         %init: memref<f32>,
                         %result: memref<112x112xf32>) {
  "xla_lhlo.select_and_scatter"(%arg, %src, %init, %result) ( {
    // select
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %pred: memref<i1>):
      "xla_lhlo.compare"(%lhs, %rhs, %pred) {comparison_direction = "GE"} :
          (memref<f32>, memref<f32>, memref<i1>) -> ()
      "xla_lhlo.terminator"() : () -> ()
  }, {
    // scatter
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %out: memref<f32>):
      "xla_lhlo.add"(%lhs, %rhs, %out) :
          (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
  }) {
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    window_dimensions = dense<[3, 3]> : tensor<2xi64>,
    window_strides = dense<[2, 2]> : tensor<2xi64>
  } : (memref<112x112xf32>,
       memref<56x56xf32>,
       memref<f32>, memref<112x112xf32>) -> ()
  "xla_lhlo.terminator"() : () -> ()
}
// CHECK-LABEL: func @select_and_scatter(
// CHECK-SAME:   [[ARG_BUF:%.*]]: memref<112x112xf32>,
// CHECK-SAME:   [[SRC_BUF:%.*]]: memref<56x56xf32>,
// CHECK-SAME:   [[INIT_BUF:%.*]]: memref<f32>,
// CHECK-SAME:   [[RESULT_BUF:%.*]]: memref<112x112xf32>) {

// Constants.
// CHECK:  [[C56:%.*]] = constant 56 : index
// CHECK:  [[C1:%.*]] = constant 1 : index
// CHECK:  [[C0_F32:%.*]] = constant 0.000000e+00 : f32
// CHECK:  [[CFALSE:%.*]] = constant false
// CHECK:  [[C3:%.*]] = constant 3 : index
// CHECK:  [[C2:%.*]] = constant 2 : index
// CHECK:  [[C0:%.*]] = constant 0 : index
// CHECK:  [[C112:%.*]] = constant 112 : index
// CHECK:  [[CTRUE:%.*]] = constant true

// Parallel loop to initialize the output buffer.
// CHECK:    [[INIT:%.*]] = load [[INIT_BUF]][] : memref<f32>
// CHECK:    scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:          to ([[C112]], [[C112]]) step ([[C1]], [[C1]]) {
// CHECK:      store [[INIT]], [[RESULT_BUF]]{{\[}}[[I]], [[J]]]
// CHECK:      scf.yield
// CHECK:    }

// Parallel loop over source buffer to compute scattered values.
// CHECK:    scf.parallel ([[II:%.*]], [[JJ:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:          to ([[C56]], [[C56]]) step ([[C1]], [[C1]]) {

// Window loop w.r.t. first dim.
// CHECK:      [[SEL_RES_I:%.*]]:4
// CHECK-SAME:   = scf.for [[WIN_I:%.*]] = [[C0]] to [[C3]] step [[C1]]
// CHECK-SAME:     iter_args(
// CHECK-SAME:       [[SEL_I_0:%.*]] = [[C0]], [[SEL_J_0:%.*]] = [[C0]],
// CHECK-SAME:       [[SEL_VAL_0:%.*]] = [[C0_F32]],
// CHECK-SAME:       [[SEL_INIT_0:%.*]] = [[CFALSE]]
// CHECK-SAME:     ) -> (index, index, f32, i1) {

// Window loop w.r.t. second dim.
// CHECK:      [[SEL_RES_J:%.*]]:4
// CHECK-SAME:   = scf.for [[WIN_J:%.*]] = [[C0]] to [[C3]] step [[C1]]
// CHECK-SAME:     iter_args(
// CHECK-SAME:       [[SEL_I:%.*]] = [[SEL_I_0]], [[SEL_J:%.*]] = [[SEL_J_0]],
// CHECK-SAME:       [[SEL_VAL:%.*]] = [[SEL_VAL_0]],
// CHECK-SAME:       [[SEL_INIT:%.*]] = [[SEL_INIT_0]]
// CHECK-SAME:     ) -> (index, index, f32, i1) {

// Compute index I of the ARG buffer and check whether it is in padding area.
// CHECK:  [[START_I:%.*]] = muli [[II]], [[C2]] : index
// CHECK:  [[OFFSET_I:%.*]] = subi [[WIN_I]], [[C0]] : index
// CHECK:  [[ARG_I:%.*]] = addi [[START_I]], [[OFFSET_I]] : index
// CHECK:  [[ARG_I_FITS:%.*]] = cmpi "ult", [[ARG_I]], [[C112]] : index

// Update `INBOUNDS`, i.e. whether or not ARG indices are inside the boundaries
// of the buffer or they are in the padding area.
// CHECK:      [[INBOUNDS_0:%.*]] = and [[ARG_I_FITS]], [[CTRUE]] : i1

// Compute index J of the ARG buffer and check whether it is in padding area.
// CHECK:  [[START_J:%.*]] = muli [[JJ]], [[C2]] : index
// CHECK:  [[OFFSET_J:%.*]] = subi [[WIN_J]], [[C0]] : index
// CHECK:  [[ARG_J:%.*]] = addi [[START_J]], [[OFFSET_J]] : index
// CHECK:  [[ARG_J_FITS:%.*]] = cmpi "ult", [[ARG_J]], [[C112]] : index

// Update `INBOUNDS`, i.e. whether or not ARG indices are inside the boundaries
// of the buffer or they are in the padding area.
// CHECK:  [[INBOUNDS_1:%.*]] = and [[INBOUNDS_0]], [[ARG_J_FITS]] : i1

// If ARG ivs are in the padding area, then 'select' function does not have to
// be applied, current selected ivs (SEL_I, SEL_J) and value (SEL_VAL) are
// returned in that case.
// CHECK:  [[IF_INBOUNDS_RES:%.*]]:4
// CHECK-SAME:  = scf.if [[INBOUNDS_1]] -> (index, index, f32, i1) {


  // INBOUNDS-THEN-BODY, i.e. if INBOUNDS == true

  // CHECK: [[ARG_ELEM:%.*]] = load [[ARG_BUF]]{{\[}}[[ARG_I]], [[ARG_J]]]
  // CHECK: [[IF_INIT_RES:%.*]]:4
  // CHECK-SAME:  = scf.if [[SEL_INIT]] -> (index, index, f32, i1) {

    // INIT-THEN-BODY, i.e. INBOUNDS == true and INIT = true

    // The LHLO IR of the select block of the lhlo.select_and_scatter is applied
    // to the current selected value (SEL_VAL) and the element of the ARG buffer
    // to compute boolean PRED, whether the new value and ivs should replace the
    // current ones.

    // Allocate buffers for ARG element, current selected value to adapt LHLO
    // code.
    // CHECK:  [[ARG_ELEM_BUF:%.*]] = alloc() : memref<f32>
    // CHECK:  [[SEL_VAL_BUF:%.*]] = alloc() : memref<f32>
    // CHECK:  [[PRED_BUF:%.*]] = alloc() : memref<i1>
    // CHECK:  store [[ARG_ELEM]], [[ARG_ELEM_BUF]][] : memref<f32>
    // CHECK:  store [[SEL_VAL]], [[SEL_VAL_BUF]][] : memref<f32>

    // Compute PRED.
    // CHECK:  "xla_lhlo.compare"(
    // CHECK-SAME:     [[ARG_ELEM_BUF]], [[SEL_VAL_BUF]], [[PRED_BUF]])
    // CHECK:      [[PRED:%.*]] = load [[PRED_BUF]][] : memref<i1>


    // Depending on PRED, return ARG ivs & elem or current select ivs and value.
    // CHECK:  [[IF_PRED_RES:%.*]]:4 = scf.if [[PRED]]
    // CHECK:    scf.yield [[ARG_I]], [[ARG_J]], [[ARG_ELEM]], [[CTRUE]]
    // CHECK:  } else {
    // CHECK:    scf.yield [[SEL_I]], [[SEL_J]], [[SEL_VAL]], [[SEL_INIT]]
    // CHECK:  }

    // INIT-THEN-BODY yield.
    // CHECK:  scf.yield [[IF_PRED_RES]]#0, [[IF_PRED_RES]]#1,
    // CHECK-SAME:        [[IF_PRED_RES]]#2, [[IF_PRED_RES]]#3

    // INIT-ELSE-BODY, i.e. if INBOUNDS == TRUE and INIT == FALSE, returns ARG
    // ivs and element without computing Select function.
    // CHECK:  scf.yield [[ARG_I]], [[ARG_J]], [[ARG_ELEM]],
    // CHECK-SAME:        [[CTRUE]] : index, index, f32, i1
    // CHECK:  }

  // INBOUNDS-THEN-BODY yield.
  // CHECK:  scf.yield [[IF_INIT_RES]]#0, [[IF_INIT_RES]]#1, [[IF_INIT_RES]]#2,
  // CHECK-SAME:        [[IF_INIT_RES]]#3 : index, index, f32, i1
  // CHECK:  }

  // INBOUNDS-ELSE-REGION, i.e. if INBOUNDS == FALSE
  // We are in the pad area, return current iter_args.
  // CHECK:  scf.yield [[SEL_I]], [[SEL_J]], [[SEL_VAL]],
  // CHECK-SAME:  [[SEL_INIT]] : index, index, f32, i1
  // CHECK:  }

// Window loop w.r.t. second dim yield.
// CHECK:  scf.yield [[IF_INBOUNDS_RES]]#0, [[IF_INBOUNDS_RES]]#1,
// CHECK-SAME:        [[IF_INBOUNDS_RES]]#2, [[IF_INBOUNDS_RES]]#3
// CHECK:  }

// Window loop w.r.t. first dim yield.
// CHECK:    scf.yield [[SEL_RES_J]]#0, [[SEL_RES_J]]#1, [[SEL_RES_J]]#2,
// CHECK-SAME:          [[SEL_RES_J]]#3 : index, index, f32, i1
// CHECK:  }

// Use selected ivs to load element from the SRC buffer.
// CHECK: [[SRC_ELEM:%.*]] = load [[SRC_BUF]]{{\[}}[[II]], [[JJ]]]

// Update of RESULT[SELECTED_I, SELECTED_J] should be done atomically, because
// it may happen that several other threads select the same IVs if the windows
// overlap.
// CHECK: generic_atomic_rmw [[RESULT_BUF]]{{\[}}[[SEL_RES_I]]#0,
// CHECK-SAME:                 [[SEL_RES_I]]#1] : memref<112x112xf32>
// CHECK: ^bb0([[CUR_RES:%.*]]: f32):

// Allocate buffers for ARG element, current selected value to adapt LHLO code.
// CHECK:  [[SRC_ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:  [[CUR_RES_BUF:%.*]] = alloc() : memref<f32>
// CHECK:  [[RES_BUF:%.*]] = alloc() : memref<f32>
// CHECK:  store [[SRC_ELEM]], [[SRC_ELEM_BUF]][] : memref<f32>
// CHECK:  store [[CUR_RES]], [[CUR_RES_BUF]][] : memref<f32>

// Compute scatter value.
// CHECK:  "xla_lhlo.add"([[SRC_ELEM_BUF]], [[CUR_RES_BUF]], [[RES_BUF]]) :
// CHECK-SAME: (memref<f32>, memref<f32>, memref<f32>) -> ()
// CHECK:  [[RES:%.*]] = load [[RES_BUF]][] : memref<f32>

// Atomic RMW terminator that returns updated value.
// CHECK:  atomic_yield [[RES]] : f32

// Parallel loop over source buffer yield
// CHECK:  scf.yield
