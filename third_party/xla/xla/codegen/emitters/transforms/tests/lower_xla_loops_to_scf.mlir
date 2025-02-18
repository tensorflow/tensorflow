// RUN: emitters_opt %s -xla-lower-xla-loops-to-scf  \
// RUN: --split-input-file | FileCheck %s

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0 + 1, s1 - 1),"
  "domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32], s0 + s1 in [0, 90]">

func.func @loop_op(%input: tensor<1024x32xf32>, %init: f32, %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i, %j] -> (%ra, %rb)
      in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%ra, %rb] : tensor<1024x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  } {xla.range = [0 : index, 42 : index]}
  func.return %sum : f32
}

// CHECK-DAG: #[[$MAP:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (s0 + s1),
// CHECK-DAG: #[[$MAPA:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (s0 + 1),
// CHECK-DAG: #[[$MAPB:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (s1 - 1),

// CHECK-LABEL: func.func @loop_op(
// CHECK-SAME:    %[[IN:.*]]: tensor<1024x32xf32>,
// CHECK-SAME:    %[[INIT:.*]]: f32, %[[DIM:.*]]: index) -> f32 {

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:     %[[C33:.*]] = arith.constant 33 : index
// CHECK-DAG:     %[[C90:.*]] = arith.constant 90 : index
// CHECK-DAG:     %[[C1025:.*]] = arith.constant 1025 : index

// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C1025]] step %[[C1]]
// CHECK-SAME:   iter_args(%[[INIT_:.*]] = %[[INIT]]) -> (f32) {

// CHECK:      %[[INNER_FOR:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C33]]
// CHECK-SAME:   step %[[C1]] iter_args(%[[INIT__:.*]] = %[[INIT_]]) -> (f32) {

// CHECK:        %[[INDEX:.*]] = xla.apply_indexing
// CHECK-SAME:     #[[$MAP]](%[[DIM]])[%[[I]], %[[J]]]
// CHECK:        %[[VAL1:.*]] = arith.cmpi sge, %[[INDEX]], %[[C0]] : index
// CHECK:        %[[VAL2:.*]] = arith.cmpi sle, %[[INDEX]], %[[C90]] : index
// CHECK:        %[[VAL3:.*]] = arith.andi %[[VAL1]], %[[VAL2]] : i1
// CHECK:        %[[VAL4:.*]] = arith.cmpi sge, %[[DIM]], %[[C0]] : index
// CHECK:        %[[VAL5:.*]] = arith.cmpi sle, %[[DIM]], %[[C3]] : index
// CHECK:        %[[VAL6:.*]] = arith.andi %[[VAL4]], %[[VAL5]] : i1
// CHECK:        %[[INBOUNDS:.*]] = arith.andi %[[VAL3]], %[[VAL6]] : i1
// CHECK:        %[[IF_RESULT:.*]] = scf.if %[[INBOUNDS]] -> (f32) {
// CHECK:          %[[RA:.*]] = xla.apply_indexing #[[$MAPA]](%[[DIM]])[%[[I]], %[[J]]]
// CHECK:          %[[RB:.*]] = xla.apply_indexing #[[$MAPB]](%[[DIM]])[%[[I]], %[[J]]]
// CHECK:          %[[ELEM:.*]] = tensor.extract %[[IN]][%[[RA]], %[[RB]]]
// CHECK:          %[[SUM:.*]] = arith.addf %[[INIT__]], %[[ELEM]] : f32
// CHECK:          scf.yield %[[SUM]] : f32
// CHECK:        } else {
// CHECK:          scf.yield %[[INIT__]] : f32
// CHECK:        }
// CHECK:        scf.yield %[[IF_RESULT]] : f32
// CHECK:      }
// CHECK:      scf.yield %[[INNER_FOR]] : f32

// -----

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0 + 1, s1 - 1),"
  "domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32], s0 + s1 in [0, 90]">

func.func @loop_yields_value_from_above(%input: tensor<1024x32xf32>, %init: f32,
    %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i, %j] -> (%ra, %rb)
      in #map iter_args(%sum_ = %init) -> (f32) {
    xla.yield %init : f32
  }
  func.return %sum : f32
}
// CHECK-LABEL: func.func @loop_yields_value_from_above(
// CHECK-SAME:    %[[IN:.*]]: tensor<1024x32xf32>,
// CHECK-SAME:    %[[INIT:.*]]: f32, %[[DIM:.*]]: index) -> f32 {
// CHECK:         scf.yield %[[INIT]] : f32
