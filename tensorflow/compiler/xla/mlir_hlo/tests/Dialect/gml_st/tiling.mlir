// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-2d" \
// RUN: --gml-tiling="tile-sizes=1,1 distribute=false op-label=tile-2d-point" \
// RUN: --gml-tiling="tile-sizes=1 distribute=false op-label=tile-1d-point" \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=false op-label=tile-3d" \
// RUN: --gml-tiling="tile-sizes=10 distribute=false op-label=tile-1d" \
// RUN: --gml-tiling="tile-sizes=2,4 distribute=false op-label=tile-pad" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-FOR

// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-tiling="tile-sizes=256,512 distribute=true op-label=tile-3d" \
// RUN: --cse | \
// RUN: FileCheck %s --check-prefix=CHECK-PARALLEL

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
// CHECK-FOR:       %[[FOR:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-FOR-SAME:      to (%[[INIT_DIM_0]], %[[INIT_DIM_1]])
// CHECK-FOR-SAME:      step (%[[C256]], %[[C512]])
// CHECK-FOR-SAME:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<?x?x?xf32>)
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[I]])[%[[INIT_DIM_0]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[J]])[%[[INIT_DIM_1]]]
// CHECK-FOR:         %[[ARG_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK-FOR:         %[[ARG_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// CHECK-FOR:         %[[CMPI:.*]] = arith.cmpi ne, %[[ARG_DIM_0]], %[[INIT_DIM_0]]
// CHECK-FOR:         %[[CMPI_0:.*]] = arith.cmpi ne, %[[ARG_DIM_1]], %[[INIT_DIM_2]]
// CHECK-FOR:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[I]]
// CHECK-FOR:         %[[SELECT_0:.*]] = arith.select %[[CMPI]], %[[C1]], %[[MIN]]
// CHECK-FOR:         %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[INIT_DIM_2]]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[OUT]]
// CHECK-FOR-SAME:      [%[[I]], %[[J]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG]]
// CHECK-FOR-SAME:      [%[[SELECT]], %[[C0]]] [%[[SELECT_0]], %[[SELECT_1]]] [1, 1]
// CHECK-FOR:         %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE_0]]
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE]]
// CHECK-FOR-SAME:        broadcast_dimensions = [0, 2]
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK-FOR:         gml_st.set_yield %[[DYNAMIC]] into %[[OUT]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// CHECK-PARALLEL-LABEL: @dynamic_broadcast_in_dim_at_tile
// CHECK-PARALLEL-SAME:  %[[INIT:.*]]: tensor<?x?x?xf32>, %[[ARG:.*]]: tensor<?x?xf32>

// CHECK-PARALLEL:       %[[C0:.*]] = arith.constant 0
// CHECK-PARALLEL:       %[[C1:.*]] = arith.constant 1
// CHECK-PARALLEL:       %[[C2:.*]] = arith.constant 2
// CHECK-PARALLEL:       %[[C256:.*]] = arith.constant 256
// CHECK-PARALLEL:       %[[C512:.*]] = arith.constant 512
// CHECK-PARALLEL:       %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-PARALLEL:       %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK-PARALLEL:       %[[INIT_DIM_2:.*]] = tensor.dim %[[INIT]], %[[C2]]
// CHECK-PARALLEL:       %[[PARALLEL:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) =
// CHECK-PARALLEL-SAME:      (%[[C0]], %[[C0]])
// CHECK-PARALLEL-SAME:      to (%[[INIT_DIM_0]], %[[INIT_DIM_1]])
// CHECK-PARALLEL-SAME:      step (%[[C256]], %[[C512]])
// CHECK-PARALLEL-SAME:      outs (%[[OUT:.*]] = %[[INIT]]: tensor<?x?x?xf32>)
// CHECK-PARALLEL:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[I]])[%[[INIT_DIM_0]]]
// CHECK-PARALLEL:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[J]])[%[[INIT_DIM_1]]]
// CHECK-PARALLEL:         %[[ARG_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK-PARALLEL:         %[[ARG_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// CHECK-PARALLEL:         %[[CMPI:.*]] = arith.cmpi ne, %[[ARG_DIM_0]], %[[INIT_DIM_0]]
// CHECK-PARALLEL:         %[[CMPI_0:.*]] = arith.cmpi ne, %[[ARG_DIM_1]], %[[INIT_DIM_2]]
// CHECK-PARALLEL:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[I]]
// CHECK-PARALLEL:         %[[SELECT_0:.*]] = arith.select %[[CMPI]], %[[C1]], %[[MIN]]
// CHECK-PARALLEL:         %[[SELECT_1:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[INIT_DIM_2]]
// CHECK-PARALLEL:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[OUT]]
// CHECK-PARALLEL-SAME:      [%[[I]], %[[J]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK-PARALLEL:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG]]
// CHECK-PARALLEL-SAME:      [%[[SELECT]], %[[C0]]] [%[[SELECT_0]], %[[SELECT_1]]] [1, 1]
// CHECK-PARALLEL:         %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
// CHECK-PARALLEL-SAME:        ins(%[[MATERIALIZE_0]]
// CHECK-PARALLEL-SAME:        outs(%[[MATERIALIZE]]
// CHECK-PARALLEL-SAME:        broadcast_dimensions = [0, 2]
// CHECK-PARALLEL:         %[[TILE:.*]] = gml_st.tile [%[[I]], %[[J]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK-PARALLEL:         gml_st.set_yield %[[DYNAMIC]] into %[[OUT]][%[[TILE]]]
// CHECK-PARALLEL:       return %[[PARALLEL]]

// -----

func.func @scatter_i64(%indices: tensor<?x2xindex>,
    %updates: tensor<?x?x?xi64>, %init: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %result = thlo.scatter
    ins (%indices: tensor<?x2xindex>, %updates: tensor<?x?x?xi64>)
    outs (%init: tensor<?x?xi64>) { op_label = "tile-1d-point" }
    (%in: i64, %out: i64) {
      %0 = arith.addi %in, %out: i64
      thlo.yield %0: i64
    }
  return %result : tensor<?x?xi64>
}

// CHECK-FOR-LABEL: func.func @scatter_i64(
// CHECK-FOR-SAME:    %[[INDICES:.*]]: tensor<?x2xindex>,
// CHECK-FOR-SAME:    %[[UPDATES:.*]]: tensor<?x?x?xi64>,
// CHECK-FOR-SAME:    %[[INIT:.*]]: tensor<?x?xi64>

// CHECK-FOR-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-FOR-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-FOR-DAG:   %[[C2:.*]] = arith.constant 2 : index

// CHECK-FOR:       %[[INDICES_COUNT:.*]] = tensor.dim %[[INDICES]], %c0
// CHECK-FOR:       gml_st.for (%{{.*}}) = (%[[C0]]) to (%[[INDICES_COUNT]])
// CHECK-FOR-SAME:    outs (%[[INIT_:.*]] = %[[INIT]]: tensor<?x?xi64>) {

// CHECK-FOR:       %[[UPDATE_SUB:.*]] = tensor.extract_slice %[[UPDATES]]
// CHECK-FOR-SAME:    : tensor<?x?x?xi64>
// CHECK-FOR:       %[[INDICES_SUB:.*]] = tensor.extract_slice %[[INDICES]]
// CHECK-FOR-SAME:    : tensor<?x2xindex>
// CHECK-FOR:       %[[INIT_SUB:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK-FOR-SAME:    : tensor<?x?xi64>

// CHECK-FOR:       %[[SCATTER:.*]] = thlo.scatter
// CHECK-FOR-SAME:    ins(%[[INDICES_SUB]] : tensor<1x2xindex>,
// CHECK-FOR-SAME:        %[[UPDATE_SUB]] : tensor<1x?x?xi64>)
// CHECK-FOR-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xi64>)
// CHECK-FOR:           arith.addi
// CHECK-FOR:           thlo.yield
// CHECK-FOR:       gml_st.set_yield %[[SCATTER:.*]]

// -----

func.func @gather(%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x4xindex>,
    %init: tensor<?x10xf64>) -> tensor<?x10xf64> {
  %result = thlo.gather
    ins (%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x4xindex>)
    outs (%init: tensor<?x10xf64>) { op_label = "tile-1d-point" }
  return %result : tensor<?x10xf64>
}

// CHECK-FOR-LABEL: @gather
// CHECK-FOR-SAME:    %[[OPERAND:.*]]: tensor<?x?x?x?xf64>
// CHECK-FOR-SAME:    %[[INDICES:.*]]: tensor<?x4xindex>
// CHECK-FOR-SAME:    %[[INIT:.*]]:
// CHECK-FOR-DAG:   %[[ZERO:.*]] = arith.constant 0
// CHECK-FOR-DAG:   %[[ONE:.*]] = arith.constant 1
// CHECK-FOR:       %[[RESULT:.*]] = gml_st.for (%[[I:.*]]) =
// CHECK-FOR-SAME:      (%[[INIT_:[a-z0-9]+]] = %[[INIT]]: tensor<?x10xf64>)

// CHECK-FOR:         %[[INDEX_SLICE:.*]] = tensor.extract_slice %[[INDICES]]
// CHECK-FOR-SAME:      [%[[I]], 0] [1, 4] [1, 1]

// CHECK-FOR:         %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK-FOR-SAME:      [%[[I]], 0] [1, 10] [1, 1]
// CHECK-FOR:         %[[GATHER_SLICE:.*]] = thlo.gather
// CHECK-FOR-SAME:       ins(%[[OPERAND]] : tensor<?x?x?x?xf64>,
// CHECK-FOR-SAME:           %[[INDEX_SLICE]] : tensor<1x4xindex>)
// CHECK-FOR-SAME:       outs(%[[INIT_SLICE]] : tensor<1x10xf64>)
// CHECK-FOR:         gml_st.set_yield %[[GATHER_SLICE]]

// -----

func.func @concatenate_at_tile(%init : tensor<?x?xi32>, %a: tensor<?x?xi32>,
    %b: tensor<?x?xi32>, %c: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>)
      dimension = 1
      { op_label = "tile-2d" }
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
// CHECK-FOR:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[ARG4]])[%[[DIM]]]
// CHECK-FOR:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[ARG5]])[%[[DIM_0]]]
// CHECK-FOR:         %[[DIM_4:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK-FOR:         %[[MINUI:.*]] = arith.minui %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SUBI:.*]] = arith.subi %[[DIM_4]], %[[MINUI]]
// CHECK-FOR:         %[[MINUI_0:.*]] = arith.minui %[[SUBI]], %[[MIN_0]]
// CHECK-FOR:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK-FOR-SAME:      [%[[ARG4]], %[[MINUI]]] [%[[MIN]], %[[MINUI_0]]] [1, 1]
// CHECK-FOR:         %[[CMPI:.*]] = arith.cmpi ule, %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SUBI_0:.*]] = arith.subi %[[ARG5]], %[[DIM_4]]
// CHECK-FOR:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[SUBI_0]]
// CHECK-FOR:         %[[DIM_5:.*]] = tensor.dim %[[ARG2]], %[[C1]]
// CHECK-FOR:         %[[MINUI_1:.*]] = arith.minui %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SUBI_1:.*]] = arith.subi %[[DIM_5]], %[[MINUI_1]]
// CHECK-FOR:         %[[MINUI_2:.*]] = arith.minui %[[SUBI_1]], %[[MIN_0]]
// CHECK-FOR:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG2]]
// CHECK-FOR-SAME:      [%[[ARG4]], %[[MINUI_1]]] [%[[MIN]], %[[MINUI_2]]] [1, 1]
// CHECK-FOR:         %[[CMPI_0:.*]] = arith.cmpi ule, %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SUBI_2:.*]] = arith.subi %[[SELECT]], %[[DIM_5]]
// CHECK-FOR:         %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[SUBI_2]]
// CHECK-FOR:         %[[DIM_6:.*]] = tensor.dim %[[ARG3]], %[[C1]]
// CHECK-FOR:         %[[MINUI_3:.*]] = arith.minui %[[SELECT_0]], %[[DIM_6]]
// CHECK-FOR:         %[[SUBI_3:.*]] = arith.subi %[[DIM_6]], %[[MINUI_3]]
// CHECK-FOR:         %[[MINUI_4:.*]] = arith.minui %[[SUBI_3]], %[[MIN_0]]
// CHECK-FOR:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[ARG3]]
// CHECK-FOR-SAME:      [%[[ARG4]], %[[MINUI_3]]] [%[[MIN]], %[[MINUI_4]]] [1, 1]
// CHECK-FOR:         %[[MATERIALIZE_2:.*]] = tensor.extract_slice %[[ARG6]]
// CHECK-FOR:         [%[[ARG4]], %[[ARG5]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         %[[CONCATENATE:.*]] = thlo.concatenate
// CHECK-FOR-SAME:        ins(%[[MATERIALIZE]] : tensor<?x?xi32>, %[[MATERIALIZE_0]] : tensor<?x?xi32>, %[[MATERIALIZE_1]] : tensor<?x?xi32>)
// CHECK-FOR-SAME:        outs(%[[MATERIALIZE_2]] : tensor<?x?xi32>)
// CHECK-FOR-SAME:        dimension = 1
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile [%[[ARG4]], %[[ARG5]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK-FOR:         gml_st.set_yield %[[CONCATENATE]] into %[[ARG6]][%[[TILE]]]
// CHECK-FOR:       return %[[FOR]]

// CHECK-PARALLEL-LABEL: @concatenate_at_tile

// -----

func.func @sort(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>,
                %init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
    -> (tensor<?x?x?xf32>, tensor<?x?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?x?xf32>, %input2: tensor<?x?x?xi32>)
      outs(%init1: tensor<?x?x?xf32>, %init2: tensor<?x?x?xi32>)
      dimension = 1
      is_stable = true
      {op_label = "tile-3d" }
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
// CHECK-FOR-DAG:     %[[TILE_SIZE0:.*]] = affine.min #map{{[0-9]*}}(%[[START0]])[%[[DIM0]]]
// CHECK-FOR-DAG:     %[[TILE_SIZE2:.*]] = affine.min #map{{[0-9]*}}(%[[START2]])[%[[DIM2]]]
// CHECK-FOR-DAG:     %[[DIM1:.*]] = tensor.dim %[[IN0]], %[[C1]]
// CHECK-FOR-DAG:     %[[IN0_SUB:.*]] = tensor.extract_slice %[[IN0]]
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR-DAG:     %[[IN1_SUB:.*]] = tensor.extract_slice %[[IN1]]
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR-DAG:     %[[INIT0_SUB:.*]] = tensor.extract_slice %[[INIT0_]]
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR-DAG:     %[[INIT1_SUB:.*]] = tensor.extract_slice %[[INIT1_]]
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR:         thlo.sort
// CHECK-FOR-SAME:        ins(%[[IN0_SUB]] : tensor<?x?x?xf32>, %[[IN1_SUB]] : tensor<?x?x?xi32>)
// CHECK-FOR-SAME:        outs(%[[INIT0_SUB]] : tensor<?x?x?xf32>, %[[INIT1_SUB]] : tensor<?x?x?xi32>)
// CHECK-FOR:         %[[TILE:.*]] = gml_st.tile
// CHECK-FOR-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-FOR-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-FOR-SAME:        [1, 1, 1]
// CHECK-FOR:         gml_st.set_yield
// CHECK-FOR-SAME:        %[[RESULT_TILE:.*]]0 into %[[INIT0_]][%[[TILE]]]
// CHECK-FOR:             %[[RESULT_TILE]]1 into %[[INIT1_]][%[[TILE]]]

// -----

func.func @sort2(%input1: tensor<1024x2048x4096xf32>,
                %input2: tensor<1024x2048x4096xi32>,
                %init1: tensor<1024x2048x4096xf32>,
                %init2: tensor<1024x2048x4096xi32>)
    -> (tensor<1024x2048x4096xf32>, tensor<1024x2048x4096xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<1024x2048x4096xf32>,
          %input2: tensor<1024x2048x4096xi32>)
      outs(%init1: tensor<1024x2048x4096xf32>,
           %init2: tensor<1024x2048x4096xi32>)
      dimension = 1
      is_stable = true
      { op_label = "tile-3d" }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
    %sorted1, %sorted2 : tensor<1024x2048x4096xf32>, tensor<1024x2048x4096xi32>
}

// CHECK-FOR-LABEL: func.func @sort2

// -----

func.func @reverse_static(%input: tensor<100xf32>, %init: tensor<100xf32>)
  -> tensor<100xf32> {
  %res = thlo.reverse
         ins(%input: tensor<100xf32>)
         outs(%init: tensor<100xf32>)
         reverse_dimensions = [0]
         { op_label = "tile-1d" }
  func.return %res : tensor<100xf32>
}

// CHECK-FOR-LABEL: func @reverse_static
//  CHECK-FOR-SAME: %[[ARG0:.*]]: tensor<100xf32>, %[[ARG1:.*]]: tensor<100xf32>
//   CHECK-FOR-DAG:   %[[C10:.*]] = arith.constant 10
//   CHECK-FOR-DAG:   %[[C100:.*]] = arith.constant 100
//       CHECK-FOR:   %[[FOR:.*]] = gml_st.for (%[[I:.*]]) =
//  CHECK-FOR-SAME:   outs (%[[ARG3:.*]] = %[[ARG1]]
//       CHECK-FOR:     %[[TEMP_SUB_RES:.*]] = arith.subi %[[C100]], %[[I]]
//       CHECK-FOR:     %[[IN_TILE_DIM:.*]] = arith.subi %[[TEMP_SUB_RES]], %[[C10]]
//   CHECK-FOR-DAG:     %[[IN_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[IN_TILE_DIM]]]
//   CHECK-FOR-DAG:     %[[INIT_SLICE:.*]] = tensor.extract_slice %[[ARG3]][%[[I]]]
//       CHECK-FOR:     %[[REVERSED:.*]] = thlo.reverse ins(%[[IN_SLICE]]
//       CHECK-FOR:       outs(%[[INIT_SLICE]]
//   CHECK-FOR-DAG:     %[[INIT_TILE:.*]] = gml_st.tile [%[[I]]]
//       CHECK-FOR:   gml_st.set_yield %[[REVERSED]] into %[[ARG3]][%[[INIT_TILE]]]
//       CHECK-FOR:   return %[[FOR]]

// -----

func.func @reverse_dynamic(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
  -> tensor<?x?xf32> {
  %res = thlo.reverse
         ins(%input: tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         reverse_dimensions = [0, 1]
         { op_label = "tile-2d" }
  func.return %res : tensor<?x?xf32>
}

// CHECK-FOR-LABEL: func @reverse_dynamic(
//  CHECK-FOR-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
//   CHECK-FOR-DAG:   %[[C0:.*]] = arith.constant 0
//   CHECK-FOR-DAG:   %[[C1:.*]] = arith.constant 1
//   CHECK-FOR-DAG:   %[[DIM:.*]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-FOR-DAG:   %[[DIM0:.*]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK-FOR:   %[[FOR:.*]] = gml_st.for (%[[I:.*]], %[[J:.*]]) =
//  CHECK-FOR-SAME:       (%[[C0]], %[[C0]]) to (%[[DIM]], %[[DIM0]])
//  CHECK-FOR-SAME:       outs (%[[ARG4:.*]] = %[[ARG1]]
//   CHECK-FOR-DAG:     %[[AFFINE_MIN1:.*]] = affine.min
//   CHECK-FOR-DAG:     %[[AFFINE_MIN2:.*]] = affine.min
//   CHECK-FOR-DAG:     %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-FOR-DAG:     %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-FOR-DAG:     %[[TEMP_SUB_RES0:.*]] = arith.subi %[[DIM1]], %[[I]]
//   CHECK-FOR-DAG:     %[[IN_TILE_DIM0:.*]] = arith.subi %[[TEMP_SUB_RES0]], %[[AFFINE_MIN1]]
//   CHECK-FOR-DAG:     %[[TEMP_SUB_RES1:.*]] = arith.subi %[[DIM2]], %[[J]]
//   CHECK-FOR-DAG:     %[[IN_TILE_DIM1:.*]] = arith.subi %[[TEMP_SUB_RES1]], %[[AFFINE_MIN2]]
//   CHECK-FOR-DAG:     %[[IN_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
//   CHECK-FOR-SAME:      [%[[IN_TILE_DIM0]], %[[IN_TILE_DIM1]]]
//   CHECK-FOR-DAG:     %[[INIT_SLICE:.*]] = tensor.extract_slice %[[ARG4]]
//   CHECK-FOR-SAME:      [%[[I]], %[[J]]]
//       CHECK-FOR:     %[[REVERSED:.*]] = thlo.reverse ins(%[[IN_SLICE]]
//  CHECK-FOR-SAME:     outs(%[[INIT_SLICE]]
//       CHECK-FOR:   %[[INIT_TILE:.*]] = gml_st.tile [%[[I]], %[[J]]]
//       CHECK-FOR:   gml_st.set_yield %[[REVERSED]] into %[[ARG4]][%[[INIT_TILE]]]
//       CHECK-FOR:   return %[[FOR]]
