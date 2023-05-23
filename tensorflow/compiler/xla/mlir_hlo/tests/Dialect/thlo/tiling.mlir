// RUN: mlir-hlo-opt %s -test-hlo-transform-dialect-interpreter -cse \
// RUN: -split-input-file | FileCheck %s

func.func @dynamic_broadcast_in_dim_at_tile(%init : tensor<?x?x?xf32>,
    %arg : tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %bcast = thlo.dynamic_broadcast_in_dim ins(%arg: tensor<?x?xf32>)
      outs(%init: tensor<?x?x?xf32>) broadcast_dimensions = [0, 2]
  func.return %bcast : tensor<?x?x?xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.dynamic_broadcast_in_dim"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile %0 [256, 512]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: @dynamic_broadcast_in_dim_at_tile
// CHECK-SAME:  %[[INIT:.*]]: tensor<?x?x?xf32>, %[[ARG:.*]]: tensor<?x?xf32>

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[C2:.*]] = arith.constant 2
// CHECK-DAG:  %[[C256:.*]] = arith.constant 256
// CHECK-DAG:  %[[C512:.*]] = arith.constant 512
// CHECK-DAG:  %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-DAG:  %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK-DAG:  %[[INIT_DIM_2:.*]] = tensor.dim %[[INIT]], %[[C2]]
// CHECK:      %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[INIT_DIM_0]]
// CHECK-SAME:      step %[[C256]]
// CHECK-SAME:      iter_args(%[[INIT_ARG0:.*]] = %[[INIT]])
// CHECK:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[I]])[%[[INIT_DIM_0]]]
// CHECK:         %[[INNER_FOR:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[INIT_DIM_1]]
// CHECK-SAME:       step %[[C512]]
// CHECK-SAME:       iter_args(%[[OUT:.*]] = %[[INIT_ARG0]])
// CHECK:          %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[J]])[%[[INIT_DIM_1]]]
// CHECK:          %[[ARG_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]]
// CHECK:          %[[ARG_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]]
// CHECK:          %[[CMPI:.*]] = arith.cmpi ne, %[[ARG_DIM_0]], %[[INIT_DIM_0]]
// CHECK:          %[[CMPI_0:.*]] = arith.cmpi ne, %[[ARG_DIM_1]], %[[INIT_DIM_2]]
// CHECK:          %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[I]]
// CHECK:          %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[C0]]
// CHECK:          %[[SELECT_1:.*]] = arith.select %[[CMPI]], %[[C1]], %[[MIN]]
// CHECK:          %[[SELECT_2:.*]] = arith.select %[[CMPI_0]], %[[C1]], %[[INIT_DIM_2]]
// CHECK:          %[[EXTRACT:.*]] = tensor.extract_slice %[[OUT]]
// CHECK-SAME:      [%[[I]], %[[J]], %[[C0]]] [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]] [1, 1, 1]
// CHECK:         %[[EXTRACT_0:.*]] = tensor.extract_slice %[[ARG]]
// CHECK-SAME:      [%[[SELECT]], %[[SELECT_0]]] [%[[SELECT_1]], %[[SELECT_2]]] [1, 1]
// CHECK:         %[[DYNAMIC:.*]] = thlo.dynamic_broadcast_in_dim
// CHECK-SAME:        ins(%[[EXTRACT_0]]
// CHECK-SAME:        outs(%[[EXTRACT]]
// CHECK-SAME:        broadcast_dimensions = [0, 2]
// CHECK:         %[[INSERTED:.*]] = tensor.insert_slice %[[DYNAMIC]]
// CHECK-SAME:       into %[[OUT]][%[[I]], %[[J]], %[[C0]]]
// CHECK-SAME:       [%[[MIN]], %[[MIN_0]], %[[INIT_DIM_2]]]
// CHECK-SAME:       [1, 1, 1]
// CHECK:         scf.yield %[[INSERTED]]
// CHECK:       scf.yield %[[INNER_FOR]]
// CHECK:     return %[[FOR]]

// -----

func.func @scatter_i64(%indices: tensor<?x2xindex>,
    %updates: tensor<?x?x?xi64>, %init: tensor<?x?xi64>) -> tensor<?x?xi64> {
  %result = thlo.scatter
    ins (%indices: tensor<?x2xindex>, %updates: tensor<?x?x?xi64>)
    outs (%init: tensor<?x?xi64>)
    (%in: i64, %out: i64) {
      %0 = arith.addi %in, %out: i64
      thlo.yield %0: i64
    }
  return %result : tensor<?x?xi64>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.scatter"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loop = transform.structured.tile %0 [1]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func.func @scatter_i64(
// CHECK-SAME:    %[[INDICES:.*]]: tensor<?x2xindex>,
// CHECK-SAME:    %[[UPDATES:.*]]: tensor<?x?x?xi64>,
// CHECK-SAME:    %[[INIT:.*]]: tensor<?x?xi64>

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[INDICES_COUNT:.*]] = tensor.dim %[[INDICES]], %c0

// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[INDICES_COUNT]] step %[[C1]]
// CHECK-SAME:    iter_args(%[[INIT_:.*]] = %[[INIT]])

// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[UPDATE_SUB:.*]] = tensor.extract_slice %[[UPDATES]][%[[I]]
// CHECK-SAME:    : tensor<?x?x?xi64>
// CHECK:       %[[INDICES_SUB:.*]] = tensor.extract_slice %[[INDICES]][%[[I]]
// CHECK-SAME:    : tensor<?x2xindex>
// CHECK-DAG:   %[[INIT_DIM_0:.*]] = tensor.dim %[[INIT]], %[[C0]]
// CHECK-DAG:   %[[INIT_DIM_1:.*]] = tensor.dim %[[INIT]], %[[C1]]
// CHECK:       %[[INIT_SUB:.*]] = tensor.extract_slice %[[INIT_]][0, 0]
// CHECK-SAME:     [%[[INIT_DIM_0]], %[[INIT_DIM_1]]] [1, 1]

// CHECK:       %[[SCATTER:.*]] = thlo.scatter
// CHECK-SAME:    ins(%[[INDICES_SUB]] : tensor<1x2xindex>,
// CHECK-SAME:        %[[UPDATE_SUB]] : tensor<1x?x?xi64>)
// CHECK-SAME:    outs(%[[INIT_SUB]] : tensor<?x?xi64>)
// CHECK:           arith.addi
// CHECK:           thlo.yield
// CHECK:       %[[INSERTED:.*]] = tensor.insert_slice %[[SCATTER]]
// CHECK-SAME:    into %[[INIT_]][0, 0]
// CHECK:       scf.yield %[[INSERTED:.*]]

// -----

func.func @gather(%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x4xindex>,
    %init: tensor<?x10xf64>) -> tensor<?x10xf64> {
  %result = thlo.gather
    ins (%operand: tensor<?x?x?x?xf64>, %indices: tensor<?x4xindex>)
    outs (%init: tensor<?x10xf64>)
  return %result : tensor<?x10xf64>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.gather"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loop = transform.structured.tile %0 [1]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// CHECK-LABEL: @gather
// CHECK-SAME:    %[[OPERAND:.*]]: tensor<?x?x?x?xf64>
// CHECK-SAME:    %[[INDICES:.*]]: tensor<?x4xindex>
// CHECK-SAME:    %[[INIT:.*]]:
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG:   %[[ONE:.*]] = arith.constant 1
// CHECK:       %[[RESULT:.*]] = scf.for %[[I:.*]] = %[[ZERO]] to
// CHECK-SAME:      (%[[INIT_:[a-z0-9]+]] = %[[INIT]])

// CHECK:         %[[INDEX_SLICE:.*]] = tensor.extract_slice %[[INDICES]]
// CHECK-SAME:      [%[[I]], 0] [1, 4] [1, 1]

// CHECK:         %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT_]]
// CHECK-SAME:      [%[[I]], 0] [1, 10] [1, 1]
// CHECK:         %[[GATHER_SLICE:.*]] = thlo.gather
// CHECK-SAME:       ins(%[[OPERAND]] : tensor<?x?x?x?xf64>,
// CHECK-SAME:           %[[INDEX_SLICE]] : tensor<1x4xindex>)
// CHECK-SAME:       outs(%[[INIT_SLICE]] : tensor<1x10xf64>)
// CHECK:         %[[INSERTED:.*]] = tensor.insert_slice %[[GATHER_SLICE]]
// CHECK-SAME:       into %[[INIT_]][%[[I]], 0] [1, 10]
// CHECK:         scf.yield %[[INSERTED]]

// -----

func.func @concatenate_at_tile(%init : tensor<?x?xi32>, %a: tensor<?x?xi32>,
    %b: tensor<?x?xi32>, %c: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
  %concat = thlo.concatenate
      ins(%a : tensor<?x?xi32>, %b : tensor<?x?xi32>, %c : tensor<?x?xi32>)
      outs(%init : tensor<?x?xi32>)
      dimension = 1
  func.return %concat : tensor<?x?xi32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.concatenate"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile %0 [256, 512]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: @concatenate_at_tile
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x?xi32>, %[[ARG1:.*]]: tensor<?x?xi32>, %[[ARG2:.*]]: tensor<?x?xi32>, %[[ARG3:.*]]: tensor<?x?xi32>

// CHECK-DAG:   %[[C0:.*]] = arith.constant 0
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1
// CHECK-DAG:   %[[C256:.*]] = arith.constant 256
// CHECK-DAG:   %[[C512:.*]] = arith.constant 512
// CHECK-DAG:   %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:   %[[DIM_0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK:       %[[FOR:.*]] = scf.for %[[ARG4:.*]] = %[[C0]] to %[[DIM]] step %[[C256]]
// CHECK-SAME:      iter_args(%[[INIT_:.*]] = %[[ARG0]])
// CHECK:         %[[MIN:.*]] = affine.min #map{{[0-9]*}}(%[[ARG4]])[%[[DIM]]]
// CHECK:         %[[INNER_FOR:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[DIM_0]] step %[[C512]]
// CHECK-SAME:      iter_args(%[[ARG6:.*]] = %[[INIT_]])
// CHECK:         %[[MIN_0:.*]] = affine.min #map{{[0-9]*}}(%[[ARG5]])[%[[DIM_0]]]
// CHECK:         %[[DIM_4:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK:         %[[MINUI:.*]] = arith.minui %[[ARG5]], %[[DIM_4]]
// CHECK:         %[[SUBI:.*]] = arith.subi %[[DIM_4]], %[[MINUI]]
// CHECK:         %[[MINUI_0:.*]] = arith.minui %[[SUBI]], %[[MIN_0]]
// CHECK:         %[[MATERIALIZE:.*]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:      [%[[ARG4]], %[[MINUI]]] [%[[MIN]], %[[MINUI_0]]] [1, 1]
// CHECK:         %[[CMPI:.*]] = arith.cmpi ule, %[[ARG5]], %[[DIM_4]]
// CHECK:         %[[SUBI_0:.*]] = arith.subi %[[ARG5]], %[[DIM_4]]
// CHECK:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[C0]], %[[SUBI_0]]
// CHECK:         %[[DIM_5:.*]] = tensor.dim %[[ARG2]], %[[C1]]
// CHECK:         %[[MINUI_1:.*]] = arith.minui %[[SELECT]], %[[DIM_5]]
// CHECK:         %[[SUBI_1:.*]] = arith.subi %[[DIM_5]], %[[MINUI_1]]
// CHECK:         %[[MINUI_2:.*]] = arith.minui %[[SUBI_1]], %[[MIN_0]]
// CHECK:         %[[MATERIALIZE_0:.*]] = tensor.extract_slice %[[ARG2]]
// CHECK-SAME:      [%[[ARG4]], %[[MINUI_1]]] [%[[MIN]], %[[MINUI_2]]] [1, 1]
// CHECK:         %[[CMPI_0:.*]] = arith.cmpi ule, %[[SELECT]], %[[DIM_5]]
// CHECK:         %[[SUBI_2:.*]] = arith.subi %[[SELECT]], %[[DIM_5]]
// CHECK:         %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[C0]], %[[SUBI_2]]
// CHECK:         %[[DIM_6:.*]] = tensor.dim %[[ARG3]], %[[C1]]
// CHECK:         %[[MINUI_3:.*]] = arith.minui %[[SELECT_0]], %[[DIM_6]]
// CHECK:         %[[SUBI_3:.*]] = arith.subi %[[DIM_6]], %[[MINUI_3]]
// CHECK:         %[[MINUI_4:.*]] = arith.minui %[[SUBI_3]], %[[MIN_0]]
// CHECK:         %[[MATERIALIZE_1:.*]] = tensor.extract_slice %[[ARG3]]
// CHECK-SAME:      [%[[ARG4]], %[[MINUI_3]]] [%[[MIN]], %[[MINUI_4]]] [1, 1]
// CHECK:         %[[MATERIALIZE_2:.*]] = tensor.extract_slice %[[ARG6]]
// CHECK:         [%[[ARG4]], %[[ARG5]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK:         %[[CONCATENATE:.*]] = thlo.concatenate
// CHECK-SAME:        ins(%[[MATERIALIZE]] : tensor<?x?xi32>, %[[MATERIALIZE_0]] : tensor<?x?xi32>, %[[MATERIALIZE_1]] : tensor<?x?xi32>)
// CHECK-SAME:        outs(%[[MATERIALIZE_2]] : tensor<?x?xi32>)
// CHECK-SAME:        dimension = 1
// CHECK:         %[[INSERTED:.*]] = tensor.insert_slice %[[CONCATENATE]]
// CHECK-SAME:        into %[[ARG6]][%[[ARG4]], %[[ARG5]]] [%[[MIN]], %[[MIN_0]]] [1, 1]
// CHECK:         scf.yield %[[INSERTED]]
// CHECK:       return %[[FOR]]

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
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?x?xf32>, tensor<?x?x?xi32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.sort"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile %0 [256, 512]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func.func @sort
// CHECK-SAME:    (%[[IN0:[a-zA-Z_0-9]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:     %[[IN1:[a-zA-Z_0-9]*]]: tensor<?x?x?xi32>,
// CHECK-SAME:     %[[INIT0:[a-zA-Z_0-9]*]]: tensor<?x?x?xf32>,
// CHECK-SAME:     %[[INIT1:[a-zA-Z_0-9]*]]: tensor<?x?x?xi32>)
// CHECK-DAG:   %[[C0:[a-zA-Z_0-9]*]] = arith.constant 0
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2
// CHECK-DAG:   %[[DIM0:.*]] = tensor.dim %[[INIT0]], %[[C0]]
// CHECK-DAG:   %[[DIM2:.*]] = tensor.dim %[[INIT0]], %[[C2]]
// CHECK:       scf.for
// CHECK-SAME:      %[[START0:.*]] = %[[C0]] to %[[DIM0]]
// CHECK-SAME:      iter_args(%[[INIT0_OUTER:.*]] = %[[INIT0]],
// CHECK-SAME:                %[[INIT1_OUTER:.*]] = %[[INIT1]])
// CHECK-DAG:     %[[TILE_SIZE0:.*]] = affine.min #map{{[0-9]*}}(%[[START0]])[%[[DIM0]]]
// CHECK:         scf.for
// CHECK-SAME:      %[[START2:.*]] = %[[C0]] to %[[DIM2]]
// CHECK-SAME:      iter_args(%[[INIT0_:.*]] = %[[INIT0_OUTER]],
// CHECK-SAME:                %[[INIT1_:.*]] = %[[INIT1_OUTER]])
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1
// CHECK-DAG:     %[[TILE_SIZE2:.*]] = affine.min #map{{[0-9]*}}(%[[START2]])[%[[DIM2]]]
// CHECK-DAG:     %[[DIM1:.*]] = tensor.dim %[[IN0]], %[[C1]]
// CHECK-DAG:     %[[IN0_SUB:.*]] = tensor.extract_slice %[[IN0]]
// CHECK-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK-DAG:     %[[IN1_SUB:.*]] = tensor.extract_slice %[[IN1]]
// CHECK-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK-DAG:     %[[INIT0_SUB:.*]] = tensor.extract_slice %[[INIT0_]]
// CHECK-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK-DAG:     %[[INIT1_SUB:.*]] = tensor.extract_slice %[[INIT1_]]
// CHECK-SAME:        [%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK:         %[[SORTED0:.*]], %[[SORTED1:.*]] = thlo.sort
// CHECK-SAME:        ins(%[[IN0_SUB]] : tensor<?x?x?xf32>, %[[IN1_SUB]] : tensor<?x?x?xi32>)
// CHECK-SAME:        outs(%[[INIT0_SUB]] : tensor<?x?x?xf32>, %[[INIT1_SUB]] : tensor<?x?x?xi32>)
// CHECK:         %[[INSERTED0:.*]] = tensor.insert_slice %[[SORTED0]]
// CHECK-SAME:        %[[INIT0_]][%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK:         %[[INSERTED1:.*]] = tensor.insert_slice %[[SORTED1]]
// CHECK-SAME:        %[[INIT1_]][%[[START0]], 0, %[[START2]]]
// CHECK-SAME:        [%[[TILE_SIZE0]], %[[DIM1]], %[[TILE_SIZE2]]]
// CHECK-SAME:        [1, 1, 1]
// CHECK:         scf.yield %[[INSERTED0]], %[[INSERTED1]]

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
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
    %sorted1, %sorted2 : tensor<1024x2048x4096xf32>, tensor<1024x2048x4096xi32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.sort"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile %0 [256, 512]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func.func @sort2

// -----

func.func @reverse_static(%input: tensor<100xf32>, %init: tensor<100xf32>)
  -> tensor<100xf32> {
  %res = thlo.reverse
         ins(%input: tensor<100xf32>)
         outs(%init: tensor<100xf32>)
         reverse_dimensions = [0]
  func.return %res : tensor<100xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.reverse"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loop = transform.structured.tile %0 [10]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func @reverse_static
//  CHECK-SAME: %[[ARG0:.*]]: tensor<100xf32>, %[[ARG1:.*]]: tensor<100xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//   CHECK-DAG:   %[[C10:.*]] = arith.constant 10
//   CHECK-DAG:   %[[C100:.*]] = arith.constant 100
//       CHECK:   %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]]
//  CHECK-SAME:   iter_args(%[[ARG3:.*]] = %[[ARG1]])
//       CHECK:     %[[TEMP_SUB_RES:.*]] = arith.subi %[[C100]], %[[I]]
//       CHECK:     %[[IN_TILE_DIM:.*]] = arith.subi %[[TEMP_SUB_RES]], %[[C10]]
//   CHECK-DAG:     %[[IN_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[IN_TILE_DIM]]]
//   CHECK-DAG:     %[[INIT_SLICE:.*]] = tensor.extract_slice %[[ARG3]][%[[I]]]
//       CHECK:     %[[REVERSED:.*]] = thlo.reverse ins(%[[IN_SLICE]]
//       CHECK:       outs(%[[INIT_SLICE]]
//       CHECK:     %[[INSERTED:.*]] = tensor.insert_slice %[[REVERSED]] into %[[ARG3]][%[[I]]
//       CHECK:     scf.yield %[[INSERTED]]
//       CHECK:   return %[[FOR]]

// -----

func.func @reverse_dynamic(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
  -> tensor<?x?xf32> {
  %res = thlo.reverse
         ins(%input: tensor<?x?xf32>)
         outs(%init: tensor<?x?xf32>)
         reverse_dimensions = [0, 1]
  func.return %res : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["thlo.reverse"]} in %arg1
      : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile %0 [256, 512]
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// CHECK-LABEL: func @reverse_dynamic(
//  CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1
//   CHECK-DAG:   %[[DIM:.*]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[DIM0:.*]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[DIM]]
//  CHECK-SAME:       iter_args(%[[ARG4:.*]] = %[[ARG1]])
//   CHECK-DAG:     %[[AFFINE_MIN1:.*]] = affine.min
//       CHECK:     %[[INNER_FOR:.*]] = scf.for  %[[J:.*]] = %[[C0]] to %[[DIM0]]
//  CHECK-SAME:       iter_args(%[[INIT_:.*]] = %[[ARG4]])
//   CHECK-DAG:     %[[AFFINE_MIN2:.*]] = affine.min
//   CHECK-DAG:     %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:     %[[DIM2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:     %[[TEMP_SUB_RES0:.*]] = arith.subi %[[DIM1]], %[[I]]
//   CHECK-DAG:     %[[IN_TILE_DIM0:.*]] = arith.subi %[[TEMP_SUB_RES0]], %[[AFFINE_MIN1]]
//   CHECK-DAG:     %[[TEMP_SUB_RES1:.*]] = arith.subi %[[DIM2]], %[[J]]
//   CHECK-DAG:     %[[IN_TILE_DIM1:.*]] = arith.subi %[[TEMP_SUB_RES1]], %[[AFFINE_MIN2]]
//   CHECK-DAG:     %[[IN_SLICE:.*]] = tensor.extract_slice %[[ARG0]]
//   CHECK-SAME:      [%[[IN_TILE_DIM0]], %[[IN_TILE_DIM1]]]
//   CHECK-DAG:     %[[INIT_SLICE:.*]] = tensor.extract_slice %[[INIT_]]
//   CHECK-SAME:      [%[[I]], %[[J]]]
//       CHECK:     %[[REVERSED:.*]] = thlo.reverse ins(%[[IN_SLICE]]
//  CHECK-SAME:     outs(%[[INIT_SLICE]]
//       CHECK:     %[[INSERTED:.*]] = tensor.insert_slice %[[REVERSED]]
//  CHECK-SAME:     into %[[INIT_]][%[[I]], %[[J]]
//       CHECK:     scf.yield %[[INSERTED]]
//       CHECK:   return %[[FOR]]
