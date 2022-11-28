// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=2 op-label=root" --test-gml-st-greedy-fusion | \
// RUN: FileCheck %s

// CHECK-LABEL: func @fuse_broadcast_map
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16x32xf32>)
func.func @fuse_broadcast_map(%arg0: tensor<16xf32>, %arg1: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %init = tensor.empty() : tensor<16x32xf32>
  %bcast = linalg.broadcast
    ins(%arg0 : tensor<16xf32>)
    outs(%init : tensor<16x32xf32>)
    dimensions = [1]

  %result = linalg.map
    ins(%bcast, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>)
    outs(%init : tensor<16x32xf32>)
    { op_label = "root" }
    (%lhs_elem: f32, %rhs_elem: f32) {
      %0 = arith.addf %lhs_elem, %rhs_elem: f32
      linalg.yield %0: f32
    }
  func.return %result : tensor<16x32xf32>
}

// CHECK:      %[[INIT:.*]] = tensor.empty()
// CHECK:      %[[RESULT:.*]] = gml_st.parallel
// CHECK-DAG:  gml_st.tile
// CHECK-DAG:  gml_st.tile
// CHECK-DAG:  %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK-DAG:  %[[ARG0_SLICE:.*]] = gml_st.materialize %[[ARG0]]
// CHECK:      %[[BCAST:.*]] = linalg.broadcast
// CHECK-NEXT:   ins(%[[ARG0_SLICE]]
// CHECK-NEXT:   outs(%[[INIT_SLICE]]
// CHECK:      %[[ARG1_SLICE:.*]] = gml_st.materialize %[[ARG1]]
// CHECK:      %[[MAPPED:.*]] = linalg.map
// CHECK-NEXT:   ins(%[[BCAST]], %[[ARG1_SLICE]]
// CHECK-NEXT:   outs(%[[INIT_SLICE]]
// CHECK:      gml_st.set_yield %[[MAPPED]]
// CHECK:      return %[[RESULT]]

// -----

// CHECK-LABEL: func @do_not_fuse_map_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32>, %[[ARG1:.*]]: tensor<16xf32>)
func.func @do_not_fuse_map_reduce(%arg0: tensor<16x32xf32>, %arg1: tensor<16xf32>)
    -> tensor<16xf32> {
  %init = tensor.empty() : tensor<16xf32>
  %reduce = linalg.reduce
    ins(%arg0 : tensor<16x32xf32>)
    outs(%init : tensor<16xf32>)
    dimensions = [1]
    (%lhs_elem: f32, %rhs_elem: f32) {
      %0 = arith.addf %lhs_elem, %rhs_elem: f32
      linalg.yield %0: f32
    }

  %result = linalg.map
    ins(%reduce, %arg1 : tensor<16xf32>, tensor<16xf32>)
    outs(%init : tensor<16xf32>)
    { op_label = "root" }
    (%lhs_elem: f32, %rhs_elem: f32) {
      %0 = arith.addf %lhs_elem, %rhs_elem: f32
      linalg.yield %0: f32
    }
  func.return %result : tensor<16xf32>
}

// CHECK:      %[[INIT:.*]] = tensor.empty()
// CHECK:      %[[REDUCE:.*]] = linalg.reduce
// CHECK:      %[[RESULT:.*]] = gml_st.parallel
// CHECK-DAG:  gml_st.tile
// CHECK-DAG:  %[[REDUCE_SLICE:.*]] = gml_st.materialize %[[REDUCE]]
// CHECK-DAG:  %[[ARG1_SLICE:.*]] = gml_st.materialize %[[ARG1]]
// CHECK-DAG:  %[[INIT_SLICE:.*]] = gml_st.materialize %[[INIT]]
// CHECK:      %[[MAPPED:.*]] = linalg.map
// CHECK-NEXT:   ins(%[[REDUCE_SLICE]], %[[ARG1_SLICE]]
// CHECK-NEXT:   outs(%[[INIT_SLICE]]
// CHECK:      gml_st.set_yield %[[MAPPED]]
// CHECK:      return %[[RESULT]]

// -----

// Only basic checks that all maps and fills were fused into gml_st.parallel.
// This test verified that ops are fused in correct order. If something is
// broken, the test will take exponential time and/or memory to finish.
// CHECK-LABEL:    func @fuse_fibonacci
// CHECK-NOT:      linalg.fill
// CHECK-NOT:      linalg.map
// CHECK:          gml_st.parallel
// CHECK-COUNT-2:    linalg.fill
// CHECK-COUNT-38:   linalg.map
// CHECK-NOT:        linalg.fill
// CHECK-NOT:        linalg.map
// CHECK:            gml_st.set_yield
// CHECK:          return
func.func @fuse_fibonacci(%init : tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64

  %0 = linalg.fill ins(%c0 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %1 = linalg.fill ins(%c1 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %2 = linalg.map ins(%0, %1 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %3 = linalg.map ins(%1, %2 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %4 = linalg.map ins(%2, %3 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %5 = linalg.map ins(%3, %4 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %6 = linalg.map ins(%4, %5 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %7 = linalg.map ins(%5, %6 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %8 = linalg.map ins(%6, %7 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %9 = linalg.map ins(%7, %8 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %10 = linalg.map ins(%8, %9 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %11 = linalg.map ins(%9, %10 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %12 = linalg.map ins(%10, %11 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %13 = linalg.map ins(%11, %12 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %14 = linalg.map ins(%12, %13 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %15 = linalg.map ins(%13, %14 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %16 = linalg.map ins(%14, %15 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %17 = linalg.map ins(%15, %16 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %18 = linalg.map ins(%16, %17 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %19 = linalg.map ins(%17, %18 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %20 = linalg.map ins(%18, %19 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %21 = linalg.map ins(%19, %20 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %22 = linalg.map ins(%20, %21 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %23 = linalg.map ins(%21, %22 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %24 = linalg.map ins(%22, %23 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %25 = linalg.map ins(%23, %24 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %26 = linalg.map ins(%24, %25 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %27 = linalg.map ins(%25, %26 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %28 = linalg.map ins(%26, %27 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %29 = linalg.map ins(%27, %28 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %30 = linalg.map ins(%28, %29 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %31 = linalg.map ins(%29, %30 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %32 = linalg.map ins(%30, %31 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %33 = linalg.map ins(%31, %32 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %34 = linalg.map ins(%32, %33 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %35 = linalg.map ins(%33, %34 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %36 = linalg.map ins(%34, %35 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %37 = linalg.map ins(%35, %36 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %38 = linalg.map ins(%36, %37 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  %39 = linalg.map ins(%37, %38 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    { op_label = "root" }
    (%l : i64, %r : i64) { %t = arith.addi %l, %r : i64 linalg.yield %t : i64 }
  func.return %39 : tensor<?xi64>
}
