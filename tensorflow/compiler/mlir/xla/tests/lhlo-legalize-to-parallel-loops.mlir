// RUN: tf-opt %s -lhlo-legalize-to-parallel-loops -canonicalize -split-input-file | FileCheck %s --dump-input-on-failure

func @reduce(%arg: memref<100x10x5xf32>,
             %init: memref<f32>,
             %result: memref<100x5xf32>) {
  "xla_lhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "xla_lhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
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
// CHECK:  loop.parallel ([[I:%.*]], [[K:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:                     to ([[C100]], [[C5]]) step ([[C1]], [[C1]]) {
// CHECK:    [[REDUCTION_RESULT:%.*]] = loop.parallel ([[J:%.*]]) =
// CHECK-SAME:      ([[C0]]) to ([[C10]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:      [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]
// CHECK-SAME:                 {{\[}}[[I]], [[J]], [[K]]] : memref<100x10x5xf32>
// CHECK:      loop.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:      ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:        [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:        [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:        "xla_lhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_BUF]])
// CHECK:        [[ACC_RESULT:%.*]] = load [[ACC_BUF]][] : memref<f32>
// CHECK:        loop.reduce.return [[ACC_RESULT]] : f32
// CHECK:      }
// CHECK:      loop.yield
// CHECK:    }
// CHECK:    store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[I]], [[K]]]
// CHECK:    loop.yield

// -----

func @reduce_no_outer_loop(%arg: memref<100xf32>,
                           %init: memref<f32>,
                           %result: memref<1xf32>) {
  "xla_lhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "xla_lhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
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
// CHECK:      [[REDUCTION_RESULT:%.*]] = loop.parallel ([[I:%.*]]) = ([[C0]])
// CHECK-SAME:     to ([[C100]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:        [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]{{\[}}[[I]]{{\]}}
// CHECK:        loop.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:        ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:          [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:          store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:          [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:          store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:          "xla_lhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_BUF]])
// CHECK:          [[ACC_RESULT:%.*]] = load [[ACC_BUF]][] : memref<f32>
// CHECK:          loop.reduce.return [[ACC_RESULT]]
// CHECK:        }
// CHECK:        loop.yield
// CHECK:      store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[C0]]]

// -----

func @dynamic_reduce(%arg: memref<?x?x?xf32>,
                     %init: memref<f32>,
                     %result: memref<?x?xf32>) {
  "xla_lhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "xla_lhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
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
// CHECK:  [[DIM0:%.*]] = dim [[ARG_BUF]], 0 : memref<?x?x?xf32>
// CHECK:  [[DIM1:%.*]] = dim [[ARG_BUF]], 1 : memref<?x?x?xf32>
// CHECK:  [[DIM2:%.*]] = dim [[ARG_BUF]], 2 : memref<?x?x?xf32>
// CHECK:  [[INIT:%.*]] = load [[INIT_BUF]]
// CHECK:  loop.parallel ([[I:%.*]], [[K:%.*]]) = ([[C0]], [[C0]])
// CHECK-SAME:                     to ([[DIM0]], [[DIM2]]) step ([[C1]], [[C1]]) {
// CHECK:    [[REDUCTION_RESULT:%.*]] = loop.parallel ([[J:%.*]]) =
// CHECK-SAME:     ([[C0]]) to ([[DIM1]]) step ([[C1]]) init ([[INIT]]) -> f32 {
// CHECK:      [[ELEM_TO_REDUCE:%.*]] = load [[ARG_BUF]]
// CHECK-SAME:                 {{\[}}[[I]], [[J]], [[K]]] : memref<?x?x?xf32>
// CHECK:      loop.reduce([[ELEM_TO_REDUCE]]) : f32 {
// CHECK:      ^bb0([[ELEM:%.*]]: f32, [[ACC:%.*]]: f32):
// CHECK:        [[ELEM_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ELEM]], [[ELEM_BUF]][] : memref<f32>
// CHECK:        [[ACC_BUF:%.*]] = alloc() : memref<f32>
// CHECK:        store [[ACC]], [[ACC_BUF]][] : memref<f32>
// CHECK:        "xla_lhlo.add"([[ELEM_BUF]], [[ACC_BUF]], [[ACC_BUF]])
// CHECK:        [[ACC_RESULT:%.*]] = load [[ACC_BUF]][] : memref<f32>
// CHECK:        loop.reduce.return [[ACC_RESULT]] : f32
// CHECK:      }
// CHECK:      loop.yield
// CHECK:    }
// CHECK:    store [[REDUCTION_RESULT]], [[RESULT_BUF]]{{\[}}[[I]], [[K]]]
// CHECK:    loop.yield
