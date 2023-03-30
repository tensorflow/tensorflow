// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --test-hlo-transform-dialect-interpreter --canonicalize -cse \
// RUN: --test-gml-st-greedy-fusion |  FileCheck %s

// CHECK-LABEL: func @fuse_broadcast_map
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xf32>, %[[ARG1:.*]]: tensor<16x32xf32>)
func.func @fuse_broadcast_map(%arg0: tensor<16xf32>, %arg1: tensor<16x32xf32>)
    -> tensor<16x32xf32> {
  %init = tensor.empty() : tensor<16x32xf32>
  %bcast = linalg.broadcast
    ins(%arg0 : tensor<16xf32>)
    outs(%init : tensor<16x32xf32>)
    dimensions = [1]

  %result = linalg.map { arith.addf }
    ins(%bcast, %arg1 : tensor<16x32xf32>, tensor<16x32xf32>)
    outs(%init : tensor<16x32xf32>)
  func.return %result : tensor<16x32xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %forall_op, %tiled_op = transform.structured.tile_to_forall_op %0 num_threads [10, 20]
}

// CHECK:      %[[INIT:.*]] = tensor.empty()
// CHECK:      %[[RESULT:.*]] = scf.forall
// CHECK-SAME:    shared_outs(%[[INIT_:.*]] = %[[INIT]])
// CHECK-DAG:  %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT]]
// CHECK-DAG:  %[[ARG0_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:      %[[BCAST:.*]] = linalg.broadcast
// CHECK-SAME:   ins(%[[ARG0_SLICE]]
// CHECK-SAME:   outs(%[[INIT_SLICE]]
// CHECK:      %[[ARG1_SLICE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK-DAG:  %[[INIT_SLICE_:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK:      %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:   ins(%[[BCAST]], %[[ARG1_SLICE]]
// CHECK-SAME:   outs(%[[INIT_SLICE_]]
// CHECK:      tensor.parallel_insert_slice %[[MAPPED]]
// CHECK:      return %[[RESULT]]

// -----

// CHECK-LABEL: func @do_not_fuse_multiple_uses
func.func @do_not_fuse_multiple_uses(%arg0: tensor<?xf32>,
    %arg1: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %bcast = linalg.broadcast
    ins(%arg0 : tensor<?xf32>)
    outs(%init : tensor<?x?xf32>)
    dimensions = [1]

  %result = linalg.map { arith.addf }
    ins(%bcast, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>)
    { op_label = "root" }
  func.return %result, %bcast : tensor<?x?xf32>, tensor<?x?xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [0, 2]
}

// CHECK: tensor.empty
// CHECK: %[[BCAST:.*]] = linalg.broadcast
// CHECK: %[[RESULT:.*]] = scf.forall
// CHECK:   linalg.map
// CHECK:   scf.forall.in_parallel
// CHECK: return %[[RESULT]], %[[BCAST]]

// -----

// CHECK-LABEL: func @do_not_fuse_map_reduce
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32>, %[[ARG1:.*]]: tensor<16xf32>)
func.func @do_not_fuse_map_reduce(%arg0: tensor<16x32xf32>, %arg1: tensor<16xf32>)
    -> tensor<16xf32> {
  %init = tensor.empty() : tensor<16xf32>
  %reduce = linalg.reduce { arith.addf }
    ins(%arg0 : tensor<16x32xf32>)
    outs(%init : tensor<16xf32>)
    dimensions = [1]

  %result = linalg.map { arith.addf }
    ins(%reduce, %arg1 : tensor<16xf32>, tensor<16xf32>)
    outs(%init : tensor<16xf32>)
  func.return %result : tensor<16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [2]
}

// CHECK:      %[[INIT:.*]] = tensor.empty()
// CHECK:      %[[REDUCE:.*]] = linalg.reduce
// CHECK:      %[[RESULT:.*]] = scf.forall
// CHECK-SAME:    shared_outs(%[[INIT_:.*]] = %[[INIT]])
// CHECK-DAG:  %[[REDUCE_SLICE:.*]] = tensor.extract_slice %[[REDUCE]]
// CHECK-DAG:  %[[ARG1_SLICE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK-DAG:  %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK:      %[[MAPPED:.*]] = linalg.map
// CHECK-SAME:   ins(%[[REDUCE_SLICE]], %[[ARG1_SLICE]]
// CHECK-SAME:   outs(%[[INIT_SLICE]]
// CHECK:      tensor.parallel_insert_slice %[[MAPPED]]
// CHECK:      return %[[RESULT]]

// -----

// Only basic checks that all maps and fills were fused into scf.forall.
// This test verified that ops are fused in correct order. If something is
// broken, the test will take exponential time and/or memory to finish.
// CHECK-LABEL:    func @fuse_fibonacci
// CHECK-NOT:      linalg.fill
// CHECK-NOT:      linalg.map
// CHECK:          scf.forall
// CHECK-COUNT-2:    linalg.fill
// CHECK-COUNT-38:   linalg.map
// CHECK-NOT:        linalg.fill
// CHECK-NOT:        linalg.map
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return
func.func @fuse_fibonacci(%init : tensor<?xi64>) -> tensor<?xi64> {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64

  %0 = linalg.fill ins(%c0 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %1 = linalg.fill ins(%c1 : i64) outs(%init : tensor<?xi64>) -> tensor<?xi64>
  %2 = linalg.map { arith.addi } ins(%0, %1 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %3 = linalg.map { arith.addi } ins(%1, %2 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %4 = linalg.map { arith.addi } ins(%2, %3 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %5 = linalg.map { arith.addi } ins(%3, %4 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %6 = linalg.map { arith.addi } ins(%4, %5 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %7 = linalg.map { arith.addi } ins(%5, %6 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %8 = linalg.map { arith.addi } ins(%6, %7 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %9 = linalg.map { arith.addi } ins(%7, %8 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %10 = linalg.map { arith.addi } ins(%8, %9 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %11 = linalg.map { arith.addi } ins(%9, %10 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %12 = linalg.map { arith.addi } ins(%10, %11 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %13 = linalg.map { arith.addi } ins(%11, %12 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %14 = linalg.map { arith.addi } ins(%12, %13 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %15 = linalg.map { arith.addi } ins(%13, %14 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %16 = linalg.map { arith.addi } ins(%14, %15 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %17 = linalg.map { arith.addi } ins(%15, %16 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %18 = linalg.map { arith.addi } ins(%16, %17 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %19 = linalg.map { arith.addi } ins(%17, %18 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %20 = linalg.map { arith.addi } ins(%18, %19 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %21 = linalg.map { arith.addi } ins(%19, %20 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %22 = linalg.map { arith.addi } ins(%20, %21 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %23 = linalg.map { arith.addi } ins(%21, %22 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %24 = linalg.map { arith.addi } ins(%22, %23 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %25 = linalg.map { arith.addi } ins(%23, %24 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %26 = linalg.map { arith.addi } ins(%24, %25 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %27 = linalg.map { arith.addi } ins(%25, %26 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %28 = linalg.map { arith.addi } ins(%26, %27 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %29 = linalg.map { arith.addi } ins(%27, %28 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %30 = linalg.map { arith.addi } ins(%28, %29 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %31 = linalg.map { arith.addi } ins(%29, %30 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %32 = linalg.map { arith.addi } ins(%30, %31 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %33 = linalg.map { arith.addi } ins(%31, %32 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %34 = linalg.map { arith.addi } ins(%32, %33 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %35 = linalg.map { arith.addi } ins(%33, %34 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %36 = linalg.map { arith.addi } ins(%34, %35 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %37 = linalg.map { arith.addi } ins(%35, %36 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %38 = linalg.map { arith.addi } ins(%36, %37 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
  %39 = linalg.map { arith.addi } ins(%37, %38 : tensor<?xi64>, tensor<?xi64>) outs(%init : tensor<?xi64>)
    {op_label="root"}
  func.return %39 : tensor<?xi64>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1]
}

// -----

func.func @fuse_reshape_middle_unit_dim_map(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_3D = tensor.empty() : tensor<10x1x16xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] :
              tensor<10x16xf32> into tensor<10x1x16xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<10x1x16xf32>)
         outs(%empty_3D : tensor<10x1x16xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %collapsed = tensor.collapse_shape %abs [[0], [1, 2]] :
               tensor<10x1x16xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @fuse_reshape_middle_unit_dim_map
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK-NOT:      tensor.expand_shape
// CHECK-NOT:      tensor.collapse_shape
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:            %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]]
// CHECK:            %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:            %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:            linalg.map { arith.addf } ins(%[[COLLAPSE]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

// -----

func.func @fuse_reshape_trailing_unit_dim_map(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_5D = tensor.empty() : tensor<10x16x1x1x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2, 3, 4]] :
              tensor<10x16xf32> into tensor<10x16x1x1x1xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<10x16x1x1x1xf32>)
         outs(%empty_5D : tensor<10x16x1x1x1xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %collapsed = tensor.collapse_shape %abs [[0], [1, 2, 3, 4]] :
               tensor<10x16x1x1x1xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @fuse_reshape_trailing_unit_dim_map
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK-NOT:      tensor.expand_shape
// CHECK-NOT:      tensor.collapse_shape
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:            %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]]
// CHECK:            %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:            %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:            linalg.map { arith.addf } ins(%[[COLLAPSE]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

// -----

func.func @fuse_reshape_leading_unit_dim_map(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_5D = tensor.empty() : tensor<1x1x1x10x16xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3], [4]] :
              tensor<10x16xf32> into tensor<1x1x1x10x16xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<1x1x1x10x16xf32>)
         outs(%empty_5D : tensor<1x1x1x10x16xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %collapsed = tensor.collapse_shape %abs [[0, 1, 2, 3], [4]] :
               tensor<1x1x1x10x16xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @fuse_reshape_leading_unit_dim_map
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK-NOT:      tensor.expand_shape
// CHECK-NOT:      tensor.collapse_shape
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:            %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]]
// CHECK:            %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:            %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:            linalg.map { arith.addf } ins(%[[COLLAPSE]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

// -----

func.func @fuse_reshape_multiple_unit_dims_map(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_4D = tensor.empty() : tensor<10x1x16x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3]] :
              tensor<10x16xf32> into tensor<10x1x16x1xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<10x1x16x1xf32>)
         outs(%empty_4D : tensor<10x1x16x1xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %collapsed = tensor.collapse_shape %abs [[0, 1], [2, 3]] :
               tensor<10x1x16x1xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @fuse_reshape_multiple_unit_dims_map
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK-NOT:      tensor.expand_shape
// CHECK-NOT:      tensor.collapse_shape
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:            %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]]
// CHECK:            %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:            %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:            linalg.map { arith.addf } ins(%[[COLLAPSE]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

// -----

func.func @fuse_reshape_reassoc_only_unit_dims_map(%arg0: tensor<10x16xf32>)
    -> tensor<10x16x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty_5D = tensor.empty() : tensor<10x1x16x1x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3, 4]] :
              tensor<10x16xf32> into tensor<10x1x16x1x1xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<10x1x16x1x1xf32>)
         outs(%empty_5D : tensor<10x1x16x1x1xf32>)

  %empty_3D = tensor.empty() : tensor<10x16x1xf32>
  %collapsed = tensor.collapse_shape %abs [[0, 1], [2], [3, 4]] :
               tensor<10x1x16x1x1xf32> into tensor<10x16x1xf32>
  %neg = linalg.map { arith.negf }
              ins(%collapsed : tensor<10x16x1xf32>)
              outs(%empty_3D : tensor<10x16x1xf32>)
    {op_label="root"}
  return %neg : tensor<10x16x1xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @fuse_reshape_reassoc_only_unit_dims_map
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>)
// CHECK-NOT:      tensor.expand_shape
// CHECK-NOT:      tensor.collapse_shape
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK:            %[[EXPAND:.*]] = tensor.expand_shape %[[EXTRACT]]
// CHECK:            %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:            %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:            linalg.map { arith.negf } ins(%[[COLLAPSE]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

// -----

func.func @do_not_fuse_collapse_shape(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty = tensor.empty() : tensor<10x1x4x4x1xf32>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2, 3, 4]] :
              tensor<10x16xf32> into tensor<10x1x4x4x1xf32>
  %abs = linalg.map { math.absf }
         ins(%expanded: tensor<10x1x4x4x1xf32>)
         outs(%empty: tensor<10x1x4x4x1xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %collapsed = tensor.collapse_shape %abs [[0, 1], [2, 3, 4]] :
              tensor<10x1x4x4x1xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%collapsed, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @do_not_fuse_collapse_shape
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK:          %[[EXPAND:.*]] = tensor.expand_shape %[[ARG0]]
// CHECK:          %[[ABS:.*]] = linalg.map { math.absf } ins(%[[EXPAND]]
// CHECK:          %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ABS]]
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[COLLAPSE]]
// CHECK:            linalg.map { arith.addf } ins(%[[EXTRACT]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return

//%test = tensor.collapse_shape %abs [[0, 1], [2]] :
//             tensor<10x16x1xf32> into tensor<160x1xf32>

// -----

func.func @do_not_fuse_expand_shape(%arg0: tensor<10x16xf32>,
    %arg1: tensor<10x16xf32>) -> tensor<10x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %empty = tensor.empty() : tensor<160xf32>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] :
               tensor<10x16xf32> into tensor<160xf32>
  %abs = linalg.map { math.absf }
         ins(%collapsed: tensor<160xf32>)
         outs(%empty: tensor<160xf32>)

  %empty_2D = tensor.empty() : tensor<10x16xf32>
  %expanded = tensor.expand_shape %abs [[0, 1]] :
              tensor<160xf32> into tensor<10x16xf32>
  %add = linalg.map { arith.addf }
              ins(%expanded, %arg1 : tensor<10x16xf32>, tensor<10x16xf32>)
              outs(%empty_2D : tensor<10x16xf32>)
    {op_label="root"}
  return %add : tensor<10x16xf32>
}
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.map"]}
                                    attributes{op_label="root"} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %loop, %1 = transform.structured.tile_to_forall_op %0 tile_sizes [1, 8]
}

// CHECK-LABEL: func @do_not_fuse_expand_shape
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<10x16xf32>, %[[ARG1:.*]]: tensor<10x16xf32>)
// CHECK:          %[[COLLAPSE:.*]] = tensor.collapse_shape %[[ARG0]]
// CHECK:          %[[ABS:.*]] = linalg.map { math.absf } ins(%[[COLLAPSE]]
// CHECK:          %[[EXPAND:.*]] = tensor.expand_shape %[[ABS]]
// CHECK:          scf.forall
// CHECK:            %[[EXTRACT:.*]] = tensor.extract_slice %[[EXPAND]]
// CHECK:            linalg.map { arith.addf } ins(%[[EXTRACT]]
// CHECK:            tensor.parallel_insert_slice
// CHECK:          return
