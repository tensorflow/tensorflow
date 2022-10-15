// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-2d" \
// RUN: --gml-tiling="tile-sizes=1,1 distribute=false op-label=tile-2d-point" \
// RUN: --gml-tiling="tile-sizes=1 distribute=false op-label=tile-1d-point" \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-3d" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-FOR

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=true op-label=tile-2d" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-PARALLEL

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add_static(%lhs: tensor<1024x1024xf32>, %rhs: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = tensor.empty() : tensor<1024x1024xf32>
  %add = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "tile-2d"}
      ins(%lhs, %rhs : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
      outs(%init : tensor<1024x1024xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<1024x1024xf32>
  func.return %add : tensor<1024x1024xf32>
}

// CHECK-FOR-LABEL: @add_static
// CHECK-FOR-SAME:  %[[ARG0:.*]]: tensor<1024x1024xf32>, %[[ARG1:.*]]: tensor<1024x1024xf32>

// CHECK-FOR-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[C256:.*]] = arith.constant 256
// CHECK-FOR-DAG:   %[[C512:.*]] = arith.constant 512
// CHECK-FOR-DAG:   %[[C1024:.*]] = arith.constant 1024
// CHECK-FOR:       %[[INIT:.*]] = tensor.empty()
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[C1024]], %[[C1024]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[ARG4:.*]] = %[[INIT]]: tensor<1024x1024xf32>)
// CHECK-FOR:         %[[SPACE:.*]] = gml_st.space [1024, 1024]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[I]], %[[J]]] [256, 512] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE]]]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG4]][%[[TILE]]]
// CHECK-FOR:         %[[GENERIC:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<256x512xf32>, tensor<256x512xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_1]] : tensor<256x512xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
// CHECK-FOR:           %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK-FOR:           linalg.yield %[[ADDF]]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC]] into %[[ARG4]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// -----

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %add = linalg.generic {
      indexing_maps = [#id_map, #id_map, #id_map],
      iterator_types = ["parallel", "parallel"],
      op_label = "tile-2d"}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
  ^bb0(%lhs_scalar: f32, %rhs_scalar: f32, %_: f32):
    %add_scalar = arith.addf %lhs_scalar, %rhs_scalar : f32
    linalg.yield %add_scalar : f32
  } -> tensor<?x?xf32>
  func.return %add : tensor<?x?xf32>
}


// CHECK-FOR-LABEL: @add(
// CHECK-FOR-SAME:  %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>

// CHECK-FOR:       %[[C0:.*]] = arith.constant 0
// CHECK-FOR:       %[[C1:.*]] = arith.constant 1
// CHECK-FOR:       %[[C256:.*]] = arith.constant 256
// CHECK-FOR:       %[[C512:.*]] = arith.constant 512
// CHECK-FOR:       %[[LHS_DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-FOR:       %[[LHS_DIM_1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-FOR:       %[[INIT:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<?x?xf32>)
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map0(%[[ARG2]])[%[[LHS_DIM_0]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG3]])[%[[LHS_DIM_1]]]
// CHECK-FOR:         %[[SPACE:.*]] = gml_st.space [%[[LHS_DIM_0]], %[[LHS_DIM_1]]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG0]][%[[TILE]]]
// CHECK-FOR:         %[[RHS_DIM_0:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK-FOR:         %[[RHS_DIM_1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK-FOR:         %[[SPACE_0:.*]] = gml_st.space [%[[RHS_DIM_0]], %[[RHS_DIM_1]]]
// CHECK-FOR:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
// CHECK-FOR:         %[[INIT_DIM_0:.*]] = tensor.dim %[[OUT]], %[[C0]]
// CHECK-FOR:         %[[INIT_DIM_1:.*]] = tensor.dim %[[OUT]], %[[C1]]
// CHECK-FOR:         %[[SPACE_1:.*]] = gml_st.space [%[[INIT_DIM_0]], %[[INIT_DIM_1]]]
// CHECK-FOR:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[OUT]][%[[TILE_1]]]
// CHECK-FOR:         %[[GENERIC:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32, %[[ARG7:.*]]: f32):
// CHECK-FOR:           %[[ADDF:.*]] = arith.addf %[[ARG5]], %[[ARG6]]
// CHECK-FOR:           linalg.yield %[[ADDF]]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC]] into %[[OUT]][%[[TILE_1]]]
// CHECK-FOR:       return %[[FOR]]


// CHECK-PARALLEL-LABEL: @add(
// CHECK-PARALLEL-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-PARALLEL:       %[[C0:.*]] = arith.constant 0
// CHECK-PARALLEL:       %[[C1:.*]] = arith.constant 1
// CHECK-PARALLEL:       %[[C256:.*]] = arith.constant 256
// CHECK-PARALLEL:       %[[C512:.*]] = arith.constant 512
// CHECK-PARALLEL:       %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0]]
// CHECK-PARALLEL:       %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-PARALLEL:       %[[INIT:.*]] = tensor.empty(%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-PARALLEL:       %[[PARALLEL:.*]] = gml_st.parallel (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-PARALLEL-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-PARALLEL-SAME:      step (%[[C256]], %[[C512]])
// CHECK-PARALLEL:         %[[MIN:.*]] = affine.min #map0(%[[ARG2]])[%[[LHS_DIM_0]]]
// CHECK-PARALLEL:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG3]])[%[[LHS_DIM_1]]]
// CHECK-PARALLEL:         %[[SPACE:.*]] = gml_st.space [%[[LHS_DIM_0]], %[[LHS_DIM_1]]]
// CHECK-PARALLEL:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[LHS]][%[[TILE]]]
// CHECK-PARALLEL:         %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0]]
// CHECK-PARALLEL:         %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1]]
// CHECK-PARALLEL:         %[[SPACE_0:.*]] = gml_st.space [%[[RHS_DIM_0]], %[[RHS_DIM_1]]]
// CHECK-PARALLEL:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[RHS]][%[[TILE_0]]]
// CHECK-PARALLEL:         %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-PARALLEL:         %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK-PARALLEL:         %[[SPACE_1:.*]] = gml_st.space [%[[INIT_DIM_0]], %[[INIT_DIM_1]]]
// CHECK-PARALLEL:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG2]], %[[ARG3]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[INIT]][%[[TILE_1]]]
// CHECK-PARALLEL:         %[[GENERIC:.*]] = linalg.generic
// CHECK-PARALLEL-SAME:        iterator_types = ["parallel", "parallel"]
// CHECK-PARALLEL-SAME:        ins(%[[MATERIALIZE]], %[[MATERIALIZE_0]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        outs(%[[MATERIALIZE_1]] : tensor<?x?xf32>)
// CHECK-PARALLEL-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-PARALLEL:         ^bb0(%[[OUT:.*]]: f32, %[[ARG5:.*]]: f32, %[[ARG6:.*]]: f32):
// CHECK-PARALLEL:           %[[ADDF:.*]] = arith.addf %[[OUT]], %[[ARG5]]
// CHECK-PARALLEL:           linalg.yield %[[ADDF]]
// CHECK-PARALLEL:         gml_st.set_yield %[[GENERIC]] into %[[INIT]][%[[TILE_1]]]
// CHECK-PARALLEL:       return %[[PARALLEL]]

// -----

func.func @reduce_row(%lhs: tensor<?x?xf32>,
                      %rhs: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>

  %init = tensor.empty(%0) : tensor<?xf32>
  %fill = linalg.fill ins(%cst : f32)
                      outs(%init : tensor<?xf32>) -> tensor<?xf32>
  %sum_of_prod = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"],
    op_label = "tile-2d"}
    ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%fill : tensor<?xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %prod = arith.mulf %l, %r : f32
    %add = arith.addf %prod, %o : f32
    linalg.yield %add : f32
  } -> tensor<?xf32>
  func.return %sum_of_prod : tensor<?xf32>
}


// CHECK-FOR-LABEL: @reduce_row
// CHECK-FOR-SAME:  %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>

// CHECK-FOR-DAG:   %[[C0_0:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[C1_0:.*]] = arith.constant 1
// CHECK-FOR-DAG:   %[[LHS_DIM_0:.*]] = tensor.dim %[[LHS]], %[[C0_0]]
// CHECK-FOR-DAG:   %[[LHS_DIM_1:.*]] = tensor.dim %[[LHS]], %[[C1_0]]
// CHECK-FOR-DAG:   %[[C256_0:.*]] = arith.constant 256
// CHECK-FOR-DAG:   %[[C512_0:.*]] = arith.constant 512
// CHECK-FOR-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00
// CHECK-FOR-DAG:   %[[INIT_0:.*]] = tensor.empty(%[[LHS_DIM_0]])
// CHECK-FOR-DAG:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT_0]] : tensor<?xf32>)
// CHECK-FOR:       %[[FOR_0:.*]] = gml_st.for (%[[ARG2_0:.*]], %[[ARG3_0:.*]]) = (%[[C0_0]], %[[C0_0]])
// CHECK-FOR-SAME:      to (%[[LHS_DIM_0]], %[[LHS_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256_0]], %[[C512_0]])
// CHECK-FOR-SAME:      outs (%[[OUT_0:.*]] = %[[FILL]]: tensor<?xf32>)
// CHECK-FOR:         %[[MIN_1:.*]] = affine.min #map0(%[[ARG2_0]])[%[[LHS_DIM_0]]]
// CHECK-FOR:         %[[MIN_2:.*]] = affine.min #map1(%[[ARG3_0]])[%[[LHS_DIM_1]]]
// CHECK-FOR:         %[[SPACE_2:.*]] = gml_st.space [%[[LHS_DIM_0]], %[[LHS_DIM_1]]]
// CHECK-FOR:         %[[TILE_2:.*]] = gml_st.tile %[[SPACE_2]] [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[LHS]][%[[TILE_2]]]
// CHECK-FOR:         %[[RHS_DIM_0:.*]] = tensor.dim %[[RHS]], %[[C0_0]]
// CHECK-FOR:         %[[RHS_DIM_1:.*]] = tensor.dim %[[RHS]], %[[C1_0]]
// CHECK-FOR:         %[[SPACE_3:.*]] = gml_st.space [%[[RHS_DIM_0]], %[[RHS_DIM_1]]]
// CHECK-FOR:         %[[TILE_3:.*]] = gml_st.tile %[[SPACE_3]] [%[[ARG2_0]], %[[ARG3_0]]] [%[[MIN_1]], %[[MIN_2]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_3:.*]] = gml_st.materialize %[[RHS]][%[[TILE_3]]]
// CHECK-FOR:         %[[DIM_16:.*]] = tensor.dim %[[OUT_0]], %[[C0_0]]
// CHECK-FOR:         %[[SPACE_4:.*]] = gml_st.space [%[[DIM_16]]]
// CHECK-FOR:         %[[TILE_4:.*]] = gml_st.tile %[[SPACE_4]] [%[[ARG2_0]]] [%[[MIN_1]]] [1]
// CHECK-FOR:         %[[MATERIALIZE_4:.*]] = gml_st.materialize %[[OUT_0]][%[[TILE_4]]]
// CHECK-FOR:         %[[GENERIC_0:.*]] = linalg.generic
// CHECK-FOR-SAME:        iterator_types = ["parallel", "reduction"]}
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE_2]], %[[MATERIALIZE_3]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_4]] : tensor<?xf32>)
// CHECK-FOR-SAME:        attrs =  {op_label = "tile-2d"}
// CHECK-FOR:         ^bb0(%[[ARG5_0:.*]]: f32, %[[ARG6_0:.*]]: f32, %[[ARG7_0:.*]]: f32):
// CHECK-FOR:           %[[MULF:.*]] = arith.mulf %[[ARG5_0]], %[[ARG6_0]]
// CHECK-FOR:           %[[ADDF_0:.*]] = arith.addf %[[MULF]], %[[ARG7_0]]
// CHECK-FOR:           linalg.yield %[[ADDF_0]]
// CHECK-FOR:         gml_st.set_yield %[[GENERIC_0]] into %[[OUT_0]][%[[TILE_4]]]
// CHECK-FOR:       return %[[FOR_0]]


// -----

func.func @thlo_reduction(
  %arg0: tensor<256x512xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> {
  %result = thlo.reduction ins(%arg0 : tensor<256x512xf32>)
      outs(%arg1 : tensor<256xf32>) dimensions = [1] { op_label = "tile-2d" }
      (%arg2: f32, %arg3: f32) {
    %add = arith.addf %arg3, %arg2 : f32
    thlo.yield %add : f32
  }
  return %result : tensor<256xf32>
}

// CHECK-FOR-LABEL: @thlo_reduction
// CHECK-FOR-SAME: %[[OPERAND:.*]]: tensor<256x512xf32>
// CHECK-FOR-SAME: %[[INIT:.*]]: tensor<256xf32>
// CHECK-FOR:      %[[RESULT:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
// CHECK-FOR:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<256xf32>)
// CHECK-FOR:      %[[TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]]]
// CHECK-FOR-SAME: [256, 512] [1, 1]
// CHECK-FOR:      %[[OP_SLICE:.*]] = gml_st.materialize %[[OPERAND]][%[[TILE]]]
// CHECK-FOR:      %[[TILE2:.*]] = gml_st.tile {{.*}} [%[[I]]] [256] [1]
// CHECK-FOR:      %[[INIT_SLICE:.*]] = gml_st.materialize %[[OUT]][%[[TILE2]]]
// CHECK-FOR:      %[[REDUCTION:.*]] = thlo.reduction ins(%[[OP_SLICE]] : tensor<256x512xf32>) outs(%[[INIT_SLICE]] : tensor<256xf32>)
// CHECK-FOR:      gml_st.set_yield %[[REDUCTION]] into %[[OUT]][%[[TILE2]]]
// CHECK-FOR:      return %[[RESULT:.*]] : tensor<256xf32>

// CHECK-PARALLEL-LABEL: @thlo_reduction

// -----

func.func @thlo_variadic_reduction(
    %inp0: tensor<256x512xf32>, %inp1: tensor<256x512xi64>,
    %out0: tensor<256xf32>, %out1: tensor<256xi64>)
    -> (tensor<256xf32>, tensor<256xi64>) {
  %result0, %result1 = thlo.reduction
      ins(%inp0: tensor<256x512xf32>, %inp1: tensor<256x512xi64>)
      outs(%out0: tensor<256xf32>, %out1: tensor<256xi64>)
      dimensions = [1] { op_label = "tile-2d" }
      (%in0: f32, %in1: i64, %init0: f32, %init1: i64) {
    %add = arith.addf %in0, %init0: f32
    %add2 = arith.addi %in1, %init1: i64
    thlo.yield %add, %add2: f32, i64
  }
  return %result0, %result1 : tensor<256xf32>, tensor<256xi64>
}

// CHECK-FOR-LABEL: @thlo_variadic_reduction
// CHECK-FOR-SAME: %[[OPERAND:.*]]: tensor<256x512xf32>
// CHECK-FOR-SAME: %[[OPERAND2:.*]]: tensor<256x512xi64>
// CHECK-FOR-SAME: %[[INIT:.*]]: tensor<256xf32>
// CHECK-FOR-SAME: %[[INIT2:.*]]: tensor<256xi64>
// CHECK-FOR:      %[[RESULT:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
// CHECK-FOR:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<256xf32>, %[[OUT2:.*]] = %[[INIT2]]: tensor<256xi64>)
// CHECK-FOR:      %[[TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]]]
// CHECK-FOR-SAME: [256, 512] [1, 1]
// CHECK-FOR:      %[[OP_SLICE:.*]] = gml_st.materialize %[[OPERAND]][%[[TILE]]]
// CHECK-FOR:      %[[OP2_SLICE:.*]] = gml_st.materialize %[[OPERAND2]][%[[TILE]]]
// CHECK-FOR:      %[[TILE2:.*]] = gml_st.tile {{.*}} [%[[I]]] [256] [1]
// CHECK-FOR:      %[[INIT_SLICE:.*]] = gml_st.materialize %[[OUT]][%[[TILE2]]]
// CHECK-FOR:      %[[INIT2_SLICE:.*]] = gml_st.materialize %[[OUT2]][%[[TILE2]]]
// CHECK-FOR:      %[[REDUCTION:.*]]:2 = thlo.reduction ins(%[[OP_SLICE]] : tensor<256x512xf32>, %[[OP2_SLICE]] : tensor<256x512xi64>) outs(%[[INIT_SLICE]] : tensor<256xf32>, %[[INIT2_SLICE]] : tensor<256xi64>)
// CHECK-FOR:      gml_st.set_yield %[[REDUCTION]]#0 into %[[OUT]][%[[TILE2]]]
// CHECK-FOR:      %[[REDUCTION]]#1 into %[[OUT2]][%[[TILE2]]]
// CHECK-FOR:      return %[[RESULT:.*]] : tensor<256xf32>, tensor<256xi64>

// CHECK-PARALLEL-LABEL: @thlo_variadic_reduction

// -----

func.func @thlo_map(%lhs: tensor<256x512xf32>, %rhs: tensor<256x512xf32>,
               %init: tensor<256x512xf32>) -> tensor<256x512xf32> {
   %add = thlo.map
          ins(%lhs:tensor<256x512xf32>, %rhs:tensor<256x512xf32>)
          outs(%init:tensor<256x512xf32>)
          { op_label = "tile-2d" }
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            thlo.yield %0: f32
          }
  func.return %add : tensor<256x512xf32>
}

// CHECK-FOR-LABEL: @thlo_map
// CHECK-FOR-SAME:    %[[ARG1:[a-zA-Z0-9]*]]: tensor<256x512xf32>
// CHECK-FOR-SAME:    %[[ARG2:[a-zA-Z0-9]*]]: tensor<256x512xf32>
// CHECK-FOR-SAME:    %[[INIT:.*]]: tensor<256x512xf32>
// CHECK-FOR-DAG:   %[[ZERO:.*]] = arith.constant 0
// CHECK-FOR:       %[[RESULT:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]]]
// CHECK-FOR:         %[[ARG1_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[TILE]]]
// CHECK-FOR:         %[[ARG2_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[TILE]]]
// CHECK-FOR:         %[[INIT_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[TILE]]]
// CHECK-FOR:         %[[MAP_SLICE:.*]] = thlo.map
// CHECK-FOR-SAME:       ins(%[[ARG1_SLICE]] :
// CHECK-FOR-SAME:         , %[[ARG2_SLICE]] :
// CHECK-FOR-SAME:       outs(%[[INIT_SLICE]]
// CHECK-FOR:         gml_st.set_yield %[[MAP_SLICE]]

// CHECK-PARALLEL-LABEL: @thlo_map
// CHECK-PARALLEL: gml_st.parallel
// -----

func.func @thlo_transpose(%input: tensor<256x512x64xf32>,
                          %init: tensor<512x64x256xf32>)
                          -> tensor<512x64x256xf32> {
  %transpose = thlo.transpose
      ins(%input:tensor<256x512x64xf32>)
      outs(%init:tensor<512x64x256xf32>)
      permutation = [1, 2, 0]
      { op_label = "tile-2d" }
  func.return %transpose : tensor<512x64x256xf32>
}
// CHECK-FOR-LABEL: func @thlo_transpose
// CHECK-FOR-SAME:    %[[INPUT:[a-zA-Z0-9]*]]: tensor<256x512x64xf32>
// CHECK-FOR-SAME:    %[[INIT:.*]]: tensor<512x64x256xf32>
// CHECK-FOR:       %[[ZERO:.*]] = arith.constant 0
// CHECK-FOR:       %[[RESULT:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
// CHECK-FOR:         %[[INPUT_TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]], 0]
// CHECK-FOR:         %[[INPUT_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[INPUT_TILE]]]
// CHECK-FOR:         %[[INIT_TILE:.*]] = gml_st.tile {{.*}} [%[[J]], 0, %[[I]]]
// CHECK-FOR:         %[[INIT_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[INIT_TILE]]]
// CHECK-FOR:         %[[RES_SLICE:.*]] = thlo.transpose
// CHECK-FOR-SAME:       ins(%[[INPUT_SLICE]] :
// CHECK-FOR-SAME:       outs(%[[INIT_SLICE]]
// CHECK-FOR:         gml_st.set_yield %[[RES_SLICE]]

// CHECK-PARALLEL-LABEL: func @thlo_transpose
// CHECK-PARALLEL: gml_st.parallel

// -----

func.func @dynamic_broadcast_in_dim_at_tile(%init : tensor<?x?x?xf32>,
    %arg : tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %bcast = thlo.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%init: tensor<?x?x?xf32>) broadcast_dimensions = [0, 2]
      { op_label = "tile-3d" }
  func.return %bcast : tensor<?x?x?xf32>
}


// CHECK-FOR-LABEL: @dynamic_broadcast_in_dim_at_tile
// CHECK-FOR-SAME:  %[[INIT:.*]]: tensor<?x?x?xf32>, %[[ARG:.*]]: tensor<?x?xf32>

// CHECK-FOR:       %[[C0:.*]] = arith.constant 0
// CHECK-FOR:       %[[C1:.*]] = arith.constant 1
// CHECK-FOR:       %[[C2:.*]] = arith.constant 2
// CHECK-FOR:       %[[C256:.*]] = arith.constant 256
// CHECK-FOR:       %[[C512:.*]] = arith.constant 512
// CHECK-FOR:       %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-FOR:       %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK-FOR:       %[[INIT_DIM_2:.*]] = tensor.dim %[[INIT]], %[[C2]]
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[ARG2:.*]], %[[ARG3:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[INIT_DIM_0]], %[[INIT_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<?x?x?xf32>)
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map0(%[[ARG2]])[%[[INIT_DIM_0]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG3]])[%[[INIT_DIM_1]]]
// CHECK-FOR:         %[[OUT_DIM_0:.*]] = tensor.dim %[[OUT]], %[[C0]]
// CHECK-FOR:         %[[OUT_DIM_1:.*]] = tensor.dim %[[OUT]], %[[C1]]
// CHECK-FOR:         %[[OUT_DIM_2:.*]] = tensor.dim %[[OUT]], %[[C2]]
// CHECK-FOR:         %[[SPACE:.*]] = gml_st.space [%[[OUT_DIM_0]], %[[OUT_DIM_1]], %[[OUT_DIM_2]]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG2]], %[[ARG3]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK-FOR:         %[[ARG_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK-FOR:         %[[ARG_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// CHECK-FOR:         %[[SPACE_0:.*]] = gml_st.space [%[[ARG_DIM_0]], %[[ARG_DIM_1]]]
// CHECK-FOR:         %[[CMPI:.*]] = arith.cmpi ne, %[[ARG_DIM_0]], %[[OUT_DIM_0]]
// CHECK-FOR:         %[[CMPI_0:.*]] = arith.cmpi ne, %[[ARG_DIM_1]], %[[OUT_DIM_2]]
// CHECK-FOR:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[ARG2]]
// CHECK-FOR:         %[[SELECT_0:.*]] = arith.select %[[CMPI]], %[[C1]], %[[MIN]]
// CHECK-FOR:         %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[INIT_DIM_2]]
// CHECK-FOR:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[SELECT]], %[[C0]]] [%[[SELECT_0]], %[[SELECT_1]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[OUT]][%[[TILE]]]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG]][%[[TILE_0]]]
// CHECK-FOR:         %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE_0]]
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE]]
// CHECK-FOR-SAME:        broadcast_dimensions = [0, 2]
// CHECK-FOR:         gml_st.set_yield %[[DYNAMIC]] into %[[OUT]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// CHECK-PARALLEL-LABEL: @dynamic_broadcast_in_dim_at_tile

// -----

func.func @concatenate_at_tile(%init : tensor<?x?xi32>, %a: tensor<?x?xi32>,
    %b: tensor<?x?xi32>, %c: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>) {
      dimension = 1 : i64,
      op_label = "tile-2d" }
  func.return %concat : tensor<?x?xi32>
}


// CHECK-FOR-LABEL: @concatenate_at_tile
// CHECK-FOR-SAME:  %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>, %[[ARG3:.*]]: tensor<?x?xi32>

// CHECK-FOR:       %[[C0:.*]] = arith.constant 0
// CHECK-FOR:       %[[C1:.*]] = arith.constant 1
// CHECK-FOR:       %[[C256:.*]] = arith.constant 256
// CHECK-FOR:       %[[C512:.*]] = arith.constant 512
// CHECK-FOR:       %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-FOR:       %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[ARG4:.*]], %[[ARG5:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[DIM]], %[[DIM_0]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[ARG6:.*]] = %[[ARG0]]: tensor<?x?xi32>)
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map0(%[[ARG4]])[%[[DIM]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map1(%[[ARG5]])[%[[DIM_0]]]
// CHECK-FOR:         %[[DIM_1:.*]] = tensor.dim %[[ARG6]], %[[C0]]
// CHECK-FOR:         %[[DIM_2:.*]] = tensor.dim %[[ARG6]], %[[C1]]
// CHECK-FOR:         %[[SPACE:.*]] = gml_st.space [%[[DIM_1]], %[[DIM_2]]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile %[[SPACE]] [%[[ARG4]], %[[ARG5]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[DIM_3:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK-FOR:         %[[DIM_4:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK-FOR:         %[[SPACE_0:.*]] = gml_st.space [%[[DIM_3]], %[[DIM_4]]]
// CHECK-FOR:         %[[MINUI:.*]] = arith.minui %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SUBI:.*]] = arith.subi %[[DIM_4]], %[[MINUI]]
// CHECK-FOR:         %[[MINUI_0:.*]] = arith.minui %[[SUBI]], %[[MIN_0]]
// CHECK-FOR:         %[[TILE_0:.*]] = gml_st.tile %[[SPACE_0]] [%[[ARG4]], %[[MINUI]]] [%[[MIN]], %[[MINUI_0]]] [%[[C1]], %[[C1]]]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = gml_st.materialize %[[ARG1]][%[[TILE_0]]]
// CHECK-FOR:         %[[CMPI:.*]] = arith.cmpi ule, %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SUBI_0:.*]] = arith.subi %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[SUBI_0]]
// CHECK-FOR:         %[[DIM_5:.*]] = tensor.dim %[[ARG2]], %[[C1]]
// CHECK-FOR:         %[[SPACE_1:.*]] = gml_st.space [%[[DIM_3]], %[[DIM_5]]]
// CHECK-FOR:         %[[MINUI_1:.*]] = arith.minui %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SUBI_1:.*]] = arith.subi %[[DIM_5]], %[[MINUI_1]]
// CHECK-FOR:         %[[MINUI_2:.*]] = arith.minui %[[SUBI_1]], %[[MIN_0]]
// CHECK-FOR:         %[[TILE_1:.*]] = gml_st.tile %[[SPACE_1]] [%[[ARG4]], %[[MINUI_1]]] [%[[MIN]], %[[MINUI_2]]] [%[[C1]], %[[C1]]]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = gml_st.materialize %[[ARG2]][%[[TILE_1]]]
// CHECK-FOR:         %[[CMPI_0:.*]] = arith.cmpi ule, %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SUBI_2:.*]] = arith.subi %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[SUBI_2]]
// CHECK-FOR:         %[[DIM_6:.*]] = tensor.dim %[[ARG3]], %[[C1]]
// CHECK-FOR:         %[[SPACE_2:.*]] = gml_st.space [%[[DIM_3]], %[[DIM_6]]]
// CHECK-FOR:         %[[MINUI_3:.*]] = arith.minui %[[SELECT_0]], %[[DIM_6]]
// CHECK-FOR:         %[[SUBI_3:.*]] = arith.subi %[[DIM_6]], %[[MINUI_3]]
// CHECK-FOR:         %[[MINUI_4:.*]] = arith.minui %[[SUBI_3]], %[[MIN_0]]
// CHECK-FOR:         %[[TILE_2:.*]] = gml_st.tile %[[SPACE_2]] [%[[ARG4]], %[[MINUI_3]]] [%[[MIN]], %[[MINUI_4]]] [%[[C1]], %[[C1]]]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = gml_st.materialize %[[ARG3]][%[[TILE_2]]]
// CHECK-FOR:         %[[MATERIALIZE_2:.*]] = gml_st.materialize %[[ARG6]][%[[TILE]]]
// CHECK-FOR:         %[[CONCATENATE:.*]] = thlo.concatenate
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]] : tensor<?x?xi32>, %[[MATERIALIZE_0]] : tensor<?x?xi32>, %[[MATERIALIZE_1]] : tensor<?x?xi32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_2]] : tensor<?x?xi32>)
// CHECK-FOR-SAME:        {dimension = 1 : i64}
// CHECK-FOR:         gml_st.set_yield %[[CONCATENATE]] into %[[ARG6]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// CHECK-PARALLEL-LABEL: @concatenate_at_tile

// -----

func.func @scatter_i32_i64(%indices: tensor<?x2xi32>,
    %updates: tensor<?x?x?xi64>, %init: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %result = thlo.scatter
    ins (%indices: tensor<?x2xi32>, %updates: tensor<?x?x?xi64>)
    outs (%init: tensor<?x?xi64>) { op_label = "tile-1d-point" }
    (%in: i64, %out: i64) {
      %0 = arith.addi %in, %out: i64
      thlo.yield %0: i64
    }
  return %result : tensor<?x?xi64>
}

// CHECK-FOR-LABEL: func.func @scatter_i32_i64(
// CHECK-FOR-SAME:    %[[INDICES:.*]]: tensor<?x2xi32>,
// CHECK-FOR-SAME:    %[[UPDATES:.*]]: tensor<?x?x?xi64>,
// CHECK-FOR-SAME:    %[[INIT:.*]]: tensor<?x?xi64>

// CHECK-FOR-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-FOR-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-FOR-DAG:   %[[C2:.*]] = arith.constant 2 : index

// CHECK-FOR:       %[[INDICES_COUNT:.*]] = tensor.dim %[[INDICES]], %c0
// CHECK-FOR:       gml_st.for (%{{.*}}) = (%[[C0]]) to (%[[INDICES_COUNT]])

// CHECK-FOR:       %[[UPDATE_SUB:.*]] = gml_st.materialize %[[UPDATES]]
// CHECK-FOR-SAME:    : tensor<?x?x?xi64>[!gml_st.tile<1x?x?>]
// CHECK-FOR:       %[[INDICES_SUB:.*]] = gml_st.materialize %[[INDICES]]
// CHECK-FOR-SAME:    : tensor<?x2xi32>[!gml_st.tile<1x2>]
// CHECK-FOR:       %[[INIT_SUB:.*]] = gml_st.materialize
// CHECK-FOR-SAME:    : tensor<?x?xi64>[!gml_st.tile<?x?>]

// CHECK-FOR:       %[[SCATTER:.*]] = thlo.scatter
// CHECK-FOR-SAME:    ins(%[[INDICES_SUB]] : tensor<1x2xi32>,
// CHECK-FOR-SAME:        %[[UPDATE_SUB]] : tensor<1x?x?xi64>)
// CHECK-FOR-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xi64>)
// CHECK-FOR:           arith.addi
// CHECK-FOR:           thlo.yield
// CHECK-FOR:       gml_st.set_yield %[[SCATTER:.*]]

// -----

func.func @gather(%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x?x4xi64>,
    %init: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %result = thlo.gather
    ins (%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x?x4xi64>)
    outs (%init: tensor<?x?xf64>) { op_label = "tile-2d" }
  return %result : tensor<?x?xf64>
}

// CHECK-FOR-LABEL: @gather
// CHECK-FOR-SAME:    %[[OPERAND:.*]]: tensor<?x?x?x?xf64>
// CHECK-FOR-SAME:    %[[INDICES:.*]]: tensor<?x?x4xi64>
// CHECK-FOR-SAME:    %[[INIT:.*]]:
// CHECK-FOR-DAG:   %[[ZERO:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[ONE:.*]] = arith.constant 1
// CHECK-FOR:       %[[RESULT:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
// CHECK-FOR:         %[[SIZE0:.*]] = affine.min {{.*}}%[[I]]
// CHECK-FOR:         %[[SIZE1:.*]] = affine.min {{.*}}%[[J]]
// CHECK-FOR:         %[[INDEX_SLICE:.*]] = tensor.extract_slice
// CHECK-FOR-SAME:       %[[INDICES]][%[[I]], %[[J]], 0]
// CHECK-FOR-SAME:       [%[[SIZE0]], %[[SIZE1]], 4]
// CHECK-FOR-SAME:       [1, 1, 1]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile {{.*}} [%[[I]], %[[J]]]
// CHECK-FOR-SAME:       [%[[SIZE0]], %[[SIZE1]]] [1, 1]
// CHECK-FOR:         %[[INIT_SLICE:.*]] = gml_st.materialize
// CHECK-FOR-SAME:       [%[[TILE]]]
// CHECK-FOR:         %[[GATHER_SLICE:.*]] = thlo.gather
// CHECK-FOR-SAME:       ins(%[[OPERAND]] :
// CHECK-FOR-SAME:         , %[[INDEX_SLICE]]
// CHECK-FOR-SAME:       outs(%[[INIT_SLICE]]
// CHECK-FOR:         gml_st.set_yield %[[GATHER_SLICE]]

// CHECK-PARALLEL-LABEL: @gather
// CHECK-PARALLEL: gml_st.parallel

// -----

func.func @sort(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>,
                %init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
    -> (tensor<?x?x?xf32>, tensor<?x?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>)
      outs(%init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
      { dimension = 1 : i64, is_stable = true, op_label = "tile-3d" }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?x?xf32>, tensor<?x?x?xi32>
}

// CHECK-FOR-LABEL: func.func @sort
// CHECK-FOR-SAME:    (%[[IN0:[a-zA-Z_0-9]*]]: tensor<?x?x?xf32>,
// CHECK-FOR-SAME:     %[[IN1:[a-zA-Z_0-9]*]]: tensor<?x?x?xi32>,
// CHECK-FOR-SAME:     %[[INIT0:[a-zA-Z_0-9]*]]: tensor<?x?x?xf32>,
// CHECK-FOR-SAME:     %[[INIT1:[a-zA-Z_0-9]*]]: tensor<?x?x?xi32>)
// CHECK-FOR-DAG:   %[[C0:[a-zA-Z_0-9]*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[C2:.*]] = arith.constant 2
// CHECK-FOR-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-FOR-DAG:   %[[DIM0:.*]] = tensor.dim %[[INIT0]], %[[C0]]
// CHECK-FOR-DAG:   %[[DIM2:.*]] = tensor.dim %[[INIT0]], %[[C2]]
// CHECK-FOR:       gml_st.for
// CHECK-FOR-SAME:      (%[[START0:.*]], %[[START2:.*]]) = (%[[C0]], %[[C0]]) to (%[[DIM0]], %[[DIM2]])
// CHECK-FOR-SAME:      outs (%[[INIT0_:.*]] = %[[INIT0]]: tensor<?x?x?xf32>,
// CHECK-FOR-SAME:            %[[INIT1_:.*]] = %[[INIT1]]: tensor<?x?x?xi32>) {
// CHECK-FOR-DAG:     %[[TILE_SIZE0:.*]] = affine.min #map0(%[[START0]])[%[[DIM0]]]
// CHECK-FOR-DAG:     %[[TILE_SIZE2:.*]] = affine.min #map1(%[[START2]])[%[[DIM2]]]
// CHECK-FOR-DAG:     %[[DIM1:.*]] = tensor.dim %[[IN0]], %[[C1]]
// CHECK-FOR-DAG:     %[[DIM0_:.*]] = tensor.dim %[[IN0]], %[[C0]]
// CHECK-FOR-DAG:     %[[DIM2_:.*]] = tensor.dim %[[IN0]], %[[C2]]
// CHECK-FOR:         %[[SPACE:.*]] = gml_st.space [%[[DIM0_]], %[[DIM1]], %[[DIM2_]]]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile
// CHECK-FOR-SAME:        %[[SPACE]]
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR-DAG:     %[[IN0_SUB:.*]] = gml_st.materialize %[[IN0]][%[[TILE]]]
// CHECK-FOR-DAG:     %[[IN1_SUB:.*]] = gml_st.materialize %[[IN1]][%[[TILE]]]
// CHECK-FOR-DAG:     %[[INIT0_SUB:.*]] = gml_st.materialize %[[INIT0_]][%[[TILE]]]
// CHECK-FOR-DAG:     %[[INIT1_SUB:.*]] = gml_st.materialize %[[INIT1_]][%[[TILE]]]
// CHECK-FOR:         thlo.sort
// CHECK-FOR-SAME:        ins(%[[IN0_SUB]] : tensor<?x?x?xf32>, %[[IN1_SUB]] : tensor<?x?x?xi32>)
// CHECK-FOR-SAME:        outs(%[[INIT0_SUB]] : tensor<?x?x?xf32>, %[[INIT1_SUB]] : tensor<?x?x?xi32>)
// CHECK-FOR:         gml_st.set_yield
// CHECK-FOR-SAME:        %[[RESULT_TILE:.*]]#0 into %[[INIT0_]][%[[TILE]]]
// CHECK-FOR:             %[[RESULT_TILE]]#1 into %[[INIT1_]][%[[TILE]]]
