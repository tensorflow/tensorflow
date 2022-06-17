// RUN: mlir-hlo-opt %s -gml-st-to-scf -split-input-file | FileCheck %s

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func.func @loop(%A: memref<192x192xf32>,
                 %B: memref<192x192xf32>,
                 %C: memref<192x192xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c24 = arith.constant 24 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index

  gml_st.loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B:  memref<192x192xf32>)
      outs (%C_ = %C: memref<192x192xf32>) {
    %0 = affine.min #map0(%i)
    %1 = memref.subview %A_[%i, 0] [%0, 192] [1, 1]
      : memref<192x192xf32> to memref<?x192xf32, #map1>
    %2 = affine.min #map2(%j)
    %3 = memref.subview %B_[0, %j] [192, %2] [1, 1]
      : memref<192x192xf32> to memref<192x?xf32, #map1>
    %4 = memref.subview %C_[%i, %j] [%0, %2] [1, 1]
      : memref<192x192xf32> to memref<?x?xf32, #map1>
    linalg.fill ins(%cst : f32) outs(%4 : memref<?x?xf32, #map1>)
    linalg.matmul ins(%1, %3 : memref<?x192xf32, #map1>,
                               memref<192x?xf32, #map1>)
                  outs(%4 : memref<?x?xf32, #map1>)
    gml_st.yield
  }
  func.return
}

// CHECK-LABEL: @loop
// CHECK-SAME:  %[[A:.*]]: memref<192x192xf32>, %[[B:.*]]: memref<192x192xf32>,
// CHECK-SAME:  %[[C:.*]]: memref<192x192xf32>) {
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:      to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]]) {
// CHECK:         %[[A_sub:.*]] = memref.subview %[[A]][%[[I]]
// CHECK:         %[[B_sub:.*]] = memref.subview %[[B]][0, %[[J]]]
// CHECK:         %[[C_sub:.*]] = memref.subview %[[C]][%[[I]]
// CHECK:         linalg.fill
// CHECK:         linalg.matmul

// -----


func.func @parallel(%A: memref<192x192xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c24 = arith.constant 24 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index

  gml_st.parallel (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16) {
    linalg.fill ins(%cst : f32) outs(%A : memref<192x192xf32>)
    gml_st.subset_yield
  }
  func.return
}

// CHECK-LABEL: @parallel
// CHECK-SAME:  %[[A:.*]]: memref<192x192xf32>
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:      to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]]) {
// CHECK:         linalg.fill

// -----

func.func @loop_reduction(%A: memref<192x192xf32>,
                           %B: memref<192x192xf32>,
                           %C: memref<f32>) {
   %c24 = arith.constant 24 : index
   %c16 = arith.constant 16 : index
   %c0 = arith.constant 0 : index
   %c192 = arith.constant 192 : index
   %cst = arith.constant 0.000000e+00 : f32

  gml_st.loop (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B:  memref<192x192xf32>)
      outs (%C_ = %C: memref<f32>)
      iterators["reduction", "reduction"] {
    linalg.fill ins(%cst : f32) outs(%A_ : memref<192x192xf32>)
    gml_st.yield
  }
  func.return
}

// CHECK-LABEL: @loop_reduction
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C24]]
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C16]]
// CHECK:           linalg.fill

// -----

func.func @for(%A: memref<192x192xf32>) {
   %c24 = arith.constant 24 : index
   %c16 = arith.constant 16 : index
   %c0 = arith.constant 0 : index
   %c192 = arith.constant 192 : index
   %cst = arith.constant 0.000000e+00 : f32

  gml_st.for (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16) {
    linalg.fill ins(%cst : f32) outs(%A : memref<192x192xf32>)
    gml_st.subset_yield
  }
  func.return
}

// CHECK-LABEL: @for
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C24]]
// CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C192]] step %[[C16]]
// CHECK:           linalg.fill

// -----

#strided_1d = affine_map<(d0)[s0] -> (d0 + s0)>
#strided_2d = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>

func.func @loop_row_reduction(%A: memref<10x8xf32>,
                               %B: memref<8xf32>) {
   %c0 = arith.constant 0 : index
   %c2 = arith.constant 2 : index
   %c4 = arith.constant 4 : index
   %c8 = arith.constant 8 : index
   %c10 = arith.constant 10 : index
   %cst = arith.constant 0.000000e+00 : f32

  gml_st.loop (%i, %j) = (%c0, %c0) to (%c10, %c8) step (%c2, %c4)
      ins (%A_ = %A: memref<10x8xf32>)
      outs (%B_ = %B: memref<8xf32>)
      iterators["reduction", "parallel"] {
    %A_sub = memref.subview %A_[%i, %j][2, 4][1, 1]
      : memref<10x8xf32> to memref<2x4xf32, #strided_2d>
    %B_sub = memref.subview %B_[%j][4][1]
      : memref<8xf32> to memref<4xf32, #strided_1d>
    linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (j)>],
        iterator_types = ["reduction", "parallel"]}
        ins(%A_sub : memref<2x4xf32, #strided_2d>)
        outs(%B_sub : memref<4xf32, #strided_1d>) {
      ^bb(%a: f32, %b: f32) :
        %0 = arith.addf %a, %b: f32
        linalg.yield %0 : f32
      }
    gml_st.yield
  }
  func.return
}

// CHECK-LABEL: @loop_row_reduction

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index

// CHECK:     scf.parallel (%[[J:.*]]) = (%[[C0]]) to (%[[C8]]) step (%[[C4]])
// CHECK-NEXT:  scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C2]]
// CHECK-NEXT:    memref.subview %arg{{[0-9]+}}[%[[I]], %[[J]]] [2, 4] [1, 1]
// CHECK-SAME:      : memref<10x8xf32> to memref<2x4xf32, #map{{[0-9]+}}>
// CHECK-NEXT:    memref.subview %arg{{[0-9]+}}[%[[J]]] [4] [1]
// CHECK-SAME:      : memref<8xf32> to memref<4xf32, #map{{[0-9]+}}>

// -----

#strided_1d = affine_map<(d0)[s0] -> (d0 + s0)>
#strided_2d = affine_map<(d0, d1)[s0] -> (d0 * 8 + s0 + d1)>

func.func @loop_col_reduction(%A: memref<10x8xf32>,
                               %B: memref<10xf32>) {
   %c0 = arith.constant 0 : index
   %c2 = arith.constant 2 : index
   %c4 = arith.constant 4 : index
   %c8 = arith.constant 8 : index
   %c10 = arith.constant 10 : index
   %cst = arith.constant 0.000000e+00 : f32

  gml_st.loop (%i, %j) = (%c0, %c0) to (%c10, %c8) step (%c2, %c4)
      ins (%A_ = %A: memref<10x8xf32>)
      outs (%B_ = %B: memref<10xf32>)
      iterators["parallel", "reduction"] {
    %A_sub = memref.subview %A_[%i, %j][2, 4][1, 1]
      : memref<10x8xf32> to memref<2x4xf32, #strided_2d>
    %B_sub = memref.subview %B_[%i][2][1]
      : memref<10xf32> to memref<2xf32, #strided_1d>
    linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>,
                         affine_map<(i, j) -> (i)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%A_sub : memref<2x4xf32, #strided_2d>)
        outs(%B_sub : memref<2xf32, #strided_1d>) {
      ^bb(%a: f32, %b: f32) :
        %0 = arith.addf %a, %b: f32
        linalg.yield %0 : f32
      }
    gml_st.yield
  }
  func.return
}

// CHECK-LABEL: @loop_col_reduction

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index

// CHECK:     scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[C10]]) step (%[[C2]])
// CHECK-NEXT:  scf.for %[[J:.*]] = %[[C0]] to %[[C8]] step %[[C4]]
// CHECK-NEXT:    memref.subview %arg{{[0-9]+}}[%[[I]], %[[J]]] [2, 4] [1, 1]
// CHECK-SAME:      : memref<10x8xf32> to memref<2x4xf32, #map{{[0-9]+}}>
// CHECK-NEXT:    memref.subview %arg{{[0-9]+}}[%[[I]]] [2] [1]
// CHECK-SAME:      : memref<10xf32> to memref<2xf32, #map{{[0-9]+}}>
