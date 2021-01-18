// RUN: mlir-hlo-opt %s -lhlo-legalize-to-parallel-loops -canonicalize -split-input-file | FILECHECK_OPTS="" FileCheck %s

func @reduce(%arg: memref<100x10x5xf32>,
             %init: memref<f32>,
             %result: memref<100x5xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<100x10x5xf32>, memref<f32>, memref<100x5xf32>) -> ()
  return
}
// CHECK-LABEL: func @reduce(
// CHECK-SAME: [[ARG_BUF:%.*]]: memref<100x10x5xf32>,
// CHECK-SAME: [[INIT_BUF:%.*]]: memref<f32>,
// CHECK-SAME: [[RESULT_BUF:%.*]]: memref<100x5xf32>) {
// CHECK-DAG:  [[C0:%.*]] = constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = constant 1 : index
// CHECK-DAG:  [[C5:%.*]] = constant 5 : index
// CHECK-DAG:  [[C10:%.*]] = constant 10 : index
// CHECK-DAG:  [[C100:%.*]] = constant 100 : index
// CHECK:  [[INIT:%.*]] = load [[INIT_BUF]]
// CHECK:  scf.parallel ([[I:%.*]], [[K:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:                     to ([[C100]], [[C5]]) step ([[C1]], [[C1]]) {
// CHECK:    [[REDUCTION_RESULT:%.*]] = scf.parallel ([[J:%.*]]) =
// CHECK-SAME:      ([[C0]]) to ([[C10]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:      [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]
// CHECK-SAME:                 {{\[}}[[I]], [[J]], [[K]]] : memref<100x10x5xf32>
// CHECK:      scf.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:      ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:        [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        [[ACC_OUT_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:        store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:        "lmhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_OUT_BUF]])
// CHECK:        [[ACC_RESULT:%.*]] = load [[ACC_OUT_BUF]][] : memref<f32>
// CHECK:        scf.reduce.return [[ACC_RESULT]] : f32
// CHECK:      }
// CHECK:      scf.yield
// CHECK:    }
// CHECK:    store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[I]], [[K]]]
// CHECK:    scf.yield

// -----

func @reduce_no_outer_loop(%arg: memref<100xf32>,
                           %init: memref<f32>,
                           %result: memref<1xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[0]> : tensor<1xi64>}
      : (memref<100xf32>, memref<f32>, memref<1xf32>) -> ()
  return
}
// CHECK-LABEL: func @reduce_no_outer_loop(
// CHECK-SAME: [[ARG_BUF:%.*]]: memref<100xf32>,
// CHECK-SAME: [[ELEM_TO_REDUCE_BUF:%.*]]: memref<f32>,
// CHECK-SAME: [[RESULT_BUF:%.*]]: memref<1xf32>) {
// CHECK-DAG:  [[C0:%.*]] = constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = constant 1 : index
// CHECK-DAG:  [[C100:%.*]] = constant 100 : index
// CHECK:      [[INIT:%.*]] = load [[INIT_BUF]]
// CHECK:      [[REDUCTION_RESULT:%.*]] = scf.parallel ([[I:%.*]]) = ([[C0]])
// CHECK-SAME:     to ([[C100]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:        [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]{{\[}}[[I]]{{\]}}
// CHECK:        scf.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:        ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:          [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:          [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:          [[ACC_OUT_BUF:%.*]] = alloc() : memref<f32>
// CHECK:          store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:          store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:          "lmhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_OUT_BUF]])
// CHECK:          [[ACC_RESULT:%.*]] = load [[ACC_OUT_BUF]][] : memref<f32>
// CHECK:          scf.reduce.return [[ACC_RESULT]]
// CHECK:        }
// CHECK:        scf.yield
// CHECK:      store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[C0]]]

// -----

func @dynamic_reduce(%arg: memref<?x?x?xf32>,
                     %init: memref<f32>,
                     %result: memref<?x?xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<?x?x?xf32>, memref<f32>, memref<?x?xf32>) -> ()
  return
}
// CHECK-LABEL: func @dynamic_reduce(
// CHECK-SAME: [[ARG_BUF:%.*]]: memref<?x?x?xf32>,
// CHECK-SAME: [[INIT_BUF:%.*]]: memref<f32>,
// CHECK-SAME: [[RESULT_BUF:%.*]]: memref<?x?xf32>) {
// CHECK-DAG:  [[C0:%.*]] = constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = constant 1 : index
// CHECK-DAG:  [[C2:%.*]] = constant 2 : index
// CHECK:  [[DIM0:%.*]] = dim [[ARG_BUF]], [[C0]] : memref<?x?x?xf32>
// CHECK:  [[DIM1:%.*]] = dim [[ARG_BUF]], [[C1]] : memref<?x?x?xf32>
// CHECK:  [[DIM2:%.*]] = dim [[ARG_BUF]], [[C2]] : memref<?x?x?xf32>
// CHECK:  [[INIT:%.*]] = load [[INIT_BUF]]
// CHECK:  scf.parallel ([[I:%.*]], [[K:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:                     to ([[DIM0]], [[DIM2]]) step ([[C1]], [[C1]]) {
// CHECK:    [[REDUCTION_RESULT:%.*]] = scf.parallel ([[J:%.*]]) =
// CHECK-SAME:     ([[C0]]) to ([[DIM1]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:      [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]
// CHECK-SAME:                 {{\[}}[[I]], [[J]], [[K]]] : memref<?x?x?xf32>
// CHECK:      scf.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:      ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:        [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        [[ACC_OUT_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:        store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:        "lmhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_OUT_BUF]])
// CHECK:        [[ACC_RESULT:%.*]] = load [[ACC_OUT_BUF]][] : memref<f32>
// CHECK:        scf.reduce.return [[ACC_RESULT]] : f32
// CHECK:      }
// CHECK:      scf.yield
// CHECK:    }
// CHECK:    store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[I]], [[K]]]
// CHECK:    scf.yield

// -----

func @reduce_window(%arg: memref<112x112xf32>,
             %init: memref<f32>,
             %result: memref<56x56xf32>) {
  "lmhlo.reduce_window"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.maximum"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {
      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
      window_dimensions = dense<[3, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 2]> : tensor<2xi64>
    } : (memref<112x112xf32>, memref<f32>, memref<56x56xf32>) -> ()
  return
}
// CHECK-LABEL: func @reduce_window(
// CHECK-SAME:      [[OPERAND_BUF:%.*]]: memref<112x112xf32>,
// CHECK-SAME:      [[INIT_BUF:%.*]]: memref<f32>,
// CHECK-SAME:      [[RESULT_BUF:%.*]]: memref<56x56xf32>) {
// CHECK-DAG:  [[C0:%.*]] = constant 0 : index
// CHECK-DAG:  [[C1:%.*]] = constant 1 : index
// CHECK-DAG:  [[C2:%.*]] = constant 2 : index
// CHECK-DAG:  [[C3:%.*]] = constant 3 : index
// CHECK-DAG:  [[C56:%.*]] = constant 56 : index
// CHECK-DAG:  [[C112:%.*]] = constant 112 : index
// CHECK:      [[INIT:%.*]] = load [[INIT_BUF]][] : memref<f32>
// CHECK:      scf.parallel ([[I:%.*]], [[J:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:         to ([[C56]], [[C56]]) step ([[C1]], [[C1]]) {
// CHECK:        [[REDUCTION_RESULT:%.*]] = scf.parallel
// CHECK-SAME:       ([[IW:%.*]], [[JW:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:       to ([[C3]], [[C3]]) step ([[C1]], [[C1]])
// CHECK-SAME:       init ([[INIT]]) -> f32 {

// CHECK:          [[START_I:%.*]] = muli [[I]], [[C2]] : index
// CHECK:          [[INDEX_I:%.*]] = addi [[START_I]], [[IW]] : index
// CHECK:          [[INDEX_I_FITS:%.*]] = cmpi ult, [[INDEX_I]], [[C112]]

// CHECK:          [[START_J:%.*]] = muli [[J]], [[C2]] : index
// CHECK:          [[INDEX_J:%.*]] = addi [[START_J]], [[JW]] : index
// CHECK:          [[INDEX_J_FITS:%.*]] = cmpi ult, [[INDEX_J]], [[C112]]
// CHECK:          [[IN_BOUNDS_1:%.*]] = and [[INDEX_I_FITS]], [[INDEX_J_FITS]]

// CHECK:          [[ELEM_TO_REDUCE:%.*]] = scf.if [[IN_BOUNDS_1]] -> (f32) {
// CHECK:            [[OPERAND_ELEM:%.*]] =
// CHECK-SAME:         load [[OPERAND_BUF]]{{\[}}[[INDEX_I]], [[INDEX_J]]]
// CHECK:              scf.yield [[OPERAND_ELEM]] : f32
// CHECK:            } else {
// CHECK:              scf.yield [[INIT]] : f32
// CHECK:            }

// CHECK:          scf.reduce([[ELEM_TO_REDUCE]])  : f32 {
// CHECK:          ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:            [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:            [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:            [[ACC_OUT_BUF:%.*]] = alloc() : memref<f32>
// CHECK:            store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:            store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:            "lmhlo.maximum"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_OUT_BUF]])
// CHECK:            [[ACC_RESULT:%.*]] = load [[ACC_OUT_BUF]][] : memref<f32>
// CHECK:            scf.reduce.return [[ACC_RESULT]] : f32
// CHECK:          }
// CHECK:          scf.yield
// CHECK:        }
// CHECK:        store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[I]], [[J]]]
// CHECK:        scf.yield
// CHECK:      }
// CHECK:      return
// CHECK:    }
