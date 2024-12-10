// RUN: emitters_opt -split-input-file %s -xla-gpu-peel-loops \
// RUN: | FileCheck %s

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0, s1), domain:"
  "d0 in [0, 3], s0 in [0, 7], s1 in [0, 10], d0 + s0 in [0, 9],"
  "d0 + s1 in [0, 12]">
func.func @peel_both_loops(%input: tensor<16x32xf32>,
    %init: f32, %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i, %j] -> (%r0, %r1)
      in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i, %j] : tensor<16x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  }
  func.return %sum : f32
}
// CHECK: #[[$PEELED_MAP:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (s0, s1), domain: d0 in [0, 3], s0 in [0, 6], s1 in [0, 9]">
// CHECK: #[[$TAIL_MAP0:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (7, s1), domain: d0 in [0, 2], s0 in [7, 7], s1 in [0, 9]">
// CHECK: #[[$TAIL_MAP1:.*]] = #xla.indexing_map<"(d0)[s0, s1] -> (s0, 10), domain: d0 in [0, 2], s0 in [0, 7], s1 in [10, 10]">

// CHECK-LABEL: func.func @peel_both_loops(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<16x32xf32>,
// CHECK-SAME:      %[[INIT:.*]]: f32, %[[DIM:.*]]: index)

// CHECK:      %[[PEELED:.*]] = xla.loop (%[[DIM]])[%[[I:.*]], %[[J:.*]]] ->
// CHECK-SAME:     in #[[$PEELED_MAP]] iter_args(%[[INIT_:.*]] = %[[INIT]])
// CHECK:        tensor.extract %[[INPUT]][%[[I]], %[[J]]] : tensor<16x32xf32>
// CHECK:        arith.addf %[[INIT_]]

// CHECK:      %[[TAIL0:.*]] = xla.loop (%[[DIM]])[%[[I:.*]], %[[J:.*]]]
// CHECK-SAME:     in #[[$TAIL_MAP0]] iter_args(%[[INIT_:.*]] = %[[PEELED]])
// CHECK:        tensor.extract %[[INPUT]][%[[I]], %[[J]]]
// CHECK:        arith.addf %[[INIT_]]

// CHECK:      %[[TAIL1:.*]] = xla.loop (%[[DIM]])[%[[I:.*]], %[[J:.*]]]
// CHECK-SAME:     in #[[$TAIL_MAP1]] iter_args(%[[INIT_:.*]] = %[[TAIL0]])
// CHECK:        tensor.extract %[[INPUT]][%[[I]], %[[J]]]
// CHECK:        arith.addf %[[INIT_]]

// CHECK: return %[[TAIL1]] : f32

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (s0),"
  "domain: d0 in [0, 3], s0 in [0, 7]">
func.func @not_constrained_symbol(%input: tensor<16xf32>, %init: f32,
    %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i] -> (%r0)
      in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i] : tensor<16xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  }
  func.return %sum : f32
}
// CHECK-LABEL: func.func @not_constrained_symbol
// CHECK:     xla.loop
// CHECK-NOT: xla.loop

// -----

#map = #xla.indexing_map<
"  (d0)[s0] -> (s0),"
"  domain:"
"  d0 in [0, 3],"
"  s0 in [0, 7],"
"  s0 mod 5 in [0, 1]">
func.func @constraint_exists_after_peeling(%input: tensor<16xf32>, %init: f32,
    %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i] -> (%r0)
      in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i] : tensor<16xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  }
  func.return %sum : f32
}
// CHECK-LABEL: func.func @constraint_exists_after_peeling
// CHECK:     xla.loop
// CHECK-NOT: xla.loop
