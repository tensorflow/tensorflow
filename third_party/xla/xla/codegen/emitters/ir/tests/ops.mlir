// RUN: emitters_opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: emitters_opt %s --split-input-file | emitters_opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: emitters_opt %s --split-input-file --mlir-print-op-generic | emitters_opt --split-input-file | FileCheck %s

func.func @atomic_rmw(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 42.0 : f32
      %add = arith.addf %current, %c42 : f32
      xla.yield %add : f32
  }
  return %ret : tensor<2x3xf32>
}
// CHECK-LABEL: @atomic_rmw
// CHECK: xla.atomic_rmw

// -----

func.func @atomic_rmw(%in: tensor<16xf32>, %i: index) -> (tensor<16xf32>) {
  %ret = xla.atomic_rmw %in[%i] : tensor<16xf32> {
    ^bb0(%current : vector<2xf32>):
      %c42 = arith.constant dense<42.0> : vector<2xf32>
      %add = arith.addf %current, %c42 : vector<2xf32>
      xla.yield %add : vector<2xf32>
  }
  return %ret : tensor<16xf32>
}
// CHECK-LABEL: @atomic_rmw
// CHECK: xla.atomic_rmw

// -----

func.func private @add(%a: f32, %b: f32) -> f32 {
  %ret = arith.addf %a, %b : f32
  return %ret : f32
}

func.func @caller(%a: f32, %b: f32) -> f32 {
  %c = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %d = xla.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %ret = arith.addf %c, %d : f32
  return %ret : f32
}
// CHECK-LABEL: @caller
// CHECK: %[[C:.*]] = xla.pure_call @add
// CHECK: %[[D:.*]] = xla.pure_call @add
// CHECK: arith.addf %[[C]], %[[D]]

// CHECK-CSE: @caller
// CHECK-CSE: %[[C:.*]] = xla.pure_call @add
// CHECK-CSE: arith.addf %[[C]], %[[C]]

// -----

#map0 = #xla.indexing_map<"(d0, d1)[s0] -> (d0, d1 + s0),"
  "domain: d0 in [1, 2], d1 in [5, 8], s0 in [0, 32]">
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  %0:2 = xla.apply_indexing #map0 (%d0, %d1)[%s0]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla.indexing_map<"
// CHECK-SAME:              (d0, d1)[s0] -> (d0, d1 + s0)
// CHECK-SAME:              domain:
// CHECK-SAME:              d0 in [1, 2]
// CHECK-SAME:              d1 in [5, 8]
// CHECK-SAME:              s0 in [0, 32]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index, %[[s0:.*]]: index)
// CHECK: xla.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]], %[[d1]])[%[[s0]]]

// -----

#map0 = #xla.indexing_map<"(d0, d1) -> (d0, d1),"
  "domain: d0 in [0, 2], d1 in [1, 3]">
func.func @apply_indexing_no_symbols(%d0: index, %d1: index) -> (index, index) {
  %0:2 = xla.apply_indexing #map0 (%d0, %d1)
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla.indexing_map<"
// CHECK-SAME:              (d0, d1) -> (d0, d1)
// CHECK-SAME:              domain:
// CHECK-SAME:              d0 in [0, 2]
// CHECK-SAME:              d1 in [1, 3]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing_no_symbols
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index)
// CHECK: xla.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]], %[[d1]])

// -----

#map0 = #xla.indexing_map<"()[s0] -> (s0, s0),"
  "domain: s0 in [2, 4]">
func.func @apply_indexing_no_dims(%s0: index) -> (index, index) {
  %0:2 = xla.apply_indexing #map0 [%s0]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla.indexing_map<"
// CHECK-SAME:              ()[s0] -> (s0, s0)
// CHECK-SAME:              domain:
// CHECK-SAME:              s0 in [2, 4]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing_no_dims
// CHECK: (%[[s0:.*]]: index)
// CHECK: xla.apply_indexing #[[$MAP0]][%[[s0]]]

// -----

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0, s1), "
  "domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32]">
func.func @loop_op(%input: tensor<1024x32xf32>, %init: f32,
                   %dim: index) -> (f32) {
  %sum = xla.loop (%dim)[%i, %j] -> (%r0, %r1)
     in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  } {xla.range = [0 : index, 42 : index]}
  func.return %sum : f32
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map
// CHECK:       xla.loop (%{{.*}})[%[[I:.*]], %[[J:.*]]] ->
// CHECK-SAME:     (%[[R0:.*]], %[[R1:.*]]) in #[[$MAP]]
// CHECK-SAME:     iter_args(%[[SUM_ITER:.*]] = %{{.*}}) -> (f32) {
// CHECK:        %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]]
// CHECK:        %[[ADD:.*]] = arith.addf %{{.*}}, %[[EXTRACTED]] : f32
// CHECK:        xla.yield %[[ADD]] : f32
// CHECK:      } {xla.range = [0 : index, 42 : index]}

// -----

func.func @test_backend_kind(%arg0: f32) attributes { xla.backend_kind = #xla.backend_kind<cpu> } {
  func.return
}
// CHECK:      @test_backend_kind
// CHECK-SAME: #xla.backend_kind<cpu>

// -----

func.func @test_backend_kind(%arg0: f32) attributes { xla.backend_kind = #xla.backend_kind<gpu> } {
  func.return
}
// CHECK:      @test_backend_kind
// CHECK-SAME: #xla.backend_kind<gpu>

// -----

func.func @test_backend_kind(%arg0: f32) attributes { xla.backend_kind = #xla.backend_kind<tpu> } {
  func.return
}
// CHECK:      @test_backend_kind
// CHECK-SAME: #xla.backend_kind<tpu>

// -----

func.func @forall_op(%input: tensor<1024x32xf32>) -> (tensor<1024x32xf32>) {
  %double_elem = xla.forall (%i, %j) in (1024, 32) with (%captured_input = %input) -> tensor<1024x32xf32> {
    %t = tensor.extract %captured_input[%i, %j] : tensor<1024x32xf32>
    %add = arith.addf %t, %t : f32
    %result = tensor.insert %add into %captured_input[%i, %j] : tensor<1024x32xf32>
    xla.yield %result : tensor<1024x32xf32>
  }
  func.return %double_elem : tensor<1024x32xf32>
}
// CHECK: [[FORALL:.*]] = xla.forall ([[IDX_0:.*]], [[IDX_1:.*]]) in (1024, 32) with ([[CAPTURED_OUTPUT:.*]] = [[CAPTURED_ARG:.*]]) -> tensor<1024x32xf32>
