// RUN: mlir_fusions_opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir_fusions_opt %s --split-input-file | mlir_fusions_opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir_fusions_opt %s --split-input-file --mlir-print-op-generic | mlir_fusions_opt --split-input-file | FileCheck %s

func.func @shared_and_sync() -> (tensor<2xf32>, tensor<2xf32>) {
  %shared1 = xla_gpu.allocate_shared : tensor<2xf32>
  %shared2 = xla_gpu.allocate_shared : tensor<2xf32>
  %sync:2 = xla_gpu.sync_threads %shared1, %shared2
    : tensor<2xf32>, tensor<2xf32>
  return %sync#0, %sync#1 : tensor<2xf32>, tensor<2xf32>
}
// CHECK-LABEL: @shared_and_sync
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: sync_threads
// CHECK-NEXT: return

// -----

func.func @atomic_rmw(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  %ret = xla_gpu.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 42.0 : f32
      %add = arith.addf %current, %c42 : f32
      xla_gpu.yield %add : f32
  }
  return %ret : tensor<2x3xf32>
}
// CHECK-LABEL: @atomic_rmw
// CHECK: xla_gpu.atomic_rmw

// -----

func.func private @add(%a: f32, %b: f32) -> f32 {
  %ret = arith.addf %a, %b : f32
  return %ret : f32
}

func.func @caller(%a: f32, %b: f32) -> f32 {
  %c = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %d = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
  %ret = arith.addf %c, %d : f32
  return %ret : f32
}
// CHECK-LABEL: @caller
// CHECK: %[[C:.*]] = xla_gpu.pure_call @add
// CHECK: %[[D:.*]] = xla_gpu.pure_call @add
// CHECK: arith.addf %[[C]], %[[D]]

// CHECK-CSE: @caller
// CHECK-CSE: %[[C:.*]] = xla_gpu.pure_call @add
// CHECK-CSE: arith.addf %[[C]], %[[C]]

// -----

#map0 = #xla_gpu.indexing_map<"(d0, d1)[s0] -> (d0, d1 + s0),"
  "domain: d0 in [1, 2], d1 in [5, 8], s0 in [0, 32]">
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 (%d0, %d1)[%s0]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla_gpu.indexing_map<"
// CHECK-SAME:              (d0, d1)[s0] -> (d0, d1 + s0)
// CHECK-SAME:              domain:
// CHECK-SAME:              d0 in [1, 2]
// CHECK-SAME:              d1 in [5, 8]
// CHECK-SAME:              s0 in [0, 32]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index, %[[s0:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]], %[[d1]])[%[[s0]]]

// -----

#map0 = #xla_gpu.indexing_map<"(d0, d1) -> (d0, d1),"
  "domain: d0 in [0, 2], d1 in [1, 3]">
func.func @apply_indexing_no_symbols(%d0: index, %d1: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 (%d0, %d1)
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla_gpu.indexing_map<"
// CHECK-SAME:              (d0, d1) -> (d0, d1)
// CHECK-SAME:              domain:
// CHECK-SAME:              d0 in [0, 2]
// CHECK-SAME:              d1 in [1, 3]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing_no_symbols
// CHECK: (%[[d0:.*]]: index, %[[d1:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]]
// CHECK-SAME:  (%[[d0]], %[[d1]])

// -----

#map0 = #xla_gpu.indexing_map<"()[s0] -> (s0, s0),"
  "domain: s0 in [2, 4]">
func.func @apply_indexing_no_dims(%s0: index) -> (index, index) {
  %0:2 = xla_gpu.apply_indexing #map0 [%s0]
  func.return %0#0, %0#1 : index, index
}
// CHECK: #[[$MAP0:.*]] = #xla_gpu.indexing_map<"
// CHECK-SAME:              ()[s0] -> (s0, s0)
// CHECK-SAME:              domain:
// CHECK-SAME:              s0 in [2, 4]
// CHECK-SAME:             >

// CHECK-LABEL: @apply_indexing_no_dims
// CHECK: (%[[s0:.*]]: index)
// CHECK: xla_gpu.apply_indexing #[[$MAP0]][%[[s0]]]

// -----

#map = #xla_gpu.indexing_map<"(d0)[s0, s1] -> (s0, s1), "
  "domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32]">
func.func @loop_op(%input: tensor<1024x32xf32>, %init: f32,
                   %dim: index) -> (f32) {
  %sum = xla_gpu.loop (%dim)[%i, %j] -> (%r0, %r1)
     in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla_gpu.yield %add : f32
  } {xla.range = [0 : index, 42 : index]}
  func.return %sum : f32
}
// CHECK: #[[$MAP:.*]] = #xla_gpu.indexing_map
// CHECK:       xla_gpu.loop (%{{.*}})[%[[I:.*]], %[[J:.*]]] ->
// CHECK-SAME:     (%[[R0:.*]], %[[R1:.*]]) in #[[$MAP]]
// CHECK-SAME:     iter_args(%[[SUM_ITER:.*]] = %{{.*}}) -> (f32) {
// CHECK:        %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]]
// CHECK:        %[[ADD:.*]] = arith.addf %{{.*}}, %[[EXTRACTED]] : f32
// CHECK:        xla_gpu.yield %[[ADD]] : f32
// CHECK:      } {xla.range = [0 : index, 42 : index]}

// -----

func.func private @exp(%p0: tensor<32x64xf32>, %i: index, %j: index) -> f32

#map = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1),"
  "domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1024], s1 in [0, 32]">
#map1 = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] -> (s0, s1),"
  "domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1024], s1 in [0, 32]">
#map2 = #xla_gpu.indexing_map<"(d0, d1) -> (d0, d1),"
  "domain: d0 in [0, 32], d1 in [0, 2]">

func.func @materialize_and_insert(%input: tensor<32x64xf32>, %i: index,
    %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = xla_gpu.materialize @exp(%input) at #map(%i, %j)
    : (tensor<32x64xf32>) -> !xla_gpu.indexed_vector<32x64xf32, #map1>
  %1 = xla_gpu.insert %0(%i, %j) into %output at #map2
    : !xla_gpu.indexed_vector<32x64xf32, #map1> -> tensor<32x64xf32>
  func.return %1 : tensor<32x64xf32>
}

// CHECK: #[[$MAP:.*]] = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] -> (d0 + s0, d1 + s1)
// CHECK-SAME: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1024], s1 in [0, 32]
// CHECK: #[[$MAP1:.*]] = #xla_gpu.indexing_map<"(d0, d1)[s0, s1] -> (s0, s1)
// CHECK-SAME: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1024], s1 in [0, 32]
// CHECK: #[[$MAP2:.*]] = #xla_gpu.indexing_map<"(d0, d1) -> (d0, d1)
// CHECK-SAME: d0 in [0, 32], d1 in [0, 2]">
// CHECK-LABEL: @materialize_and_insert
// CHECK: %[[MATERIALIZED:.*]] = xla_gpu.materialize @exp(%{{.*}}) at
// CHECK-SAME: #[[$MAP]](%{{.*}}, %{{.*}})
// CHECK: xla_gpu.insert %[[MATERIALIZED]](%{{.*}}, %{{.*}}) into
// CHECK-SAME: at #[[$MAP2]] : <32x64xf32, #[[$MAP1]]>

// -----

func.func @add(%a_acc: f32, %b_acc: i32, %a: f32, %b: i32)
    -> (f32, i32) {
  %0 = arith.addf %a_acc, %a : f32
  %1 = arith.addi %b_acc, %b : i32
  func.return %0, %1 : f32, i32
}
func.func @reduce(%in0: tensor<16x8x4xf32>, %init0: f32,
    %in1: tensor<16x8x4xi32>, %init1: i32) -> (tensor<8xf32>, tensor<8xi32>) {
  %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
    combiner=@add {xla.range = [0 : index, 42 : index]}
    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>
  func.return %sum#0, %sum#1 : tensor<8xf32>, tensor<8xi32>
}
// CHECK-LABEL: func.func @reduce(
// CHECK-SAME:    %[[IN1:.*]]: tensor<16x8x4xf32>, %[[INIT1:.*]]: f32,
// CHECK-SAME:    %[[IN2:.*]]: tensor<16x8x4xi32>, %[[INIT2:.*]]: i32)

// CHECK:        xla_gpu.reduce(%[[IN1]], %[[IN2]])
// CHECK-SAME:    inits(%[[INIT1]], %[[INIT2]]) dimensions=[0, 2]
// CHECK-SAME:    combiner=@add {xla.range = [0 : index, 42 : index]}
// CHECK-SAME:    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>

// -----

func.func @add(%a_acc: f32, %a: f32) -> (f32) {
  %0 = arith.addf %a_acc, %a : f32
  func.return %0 : f32
}

func.func @reduce_middle_dim(%in: tensor<16x8x4xf32>, %init: f32)
    -> (tensor<16x4xf32>) {
  %sum = xla_gpu.reduce (%in) inits(%init) dimensions=[1]
    combiner=@add : tensor<16x8x4xf32> to tensor<16x4xf32>
  func.return %sum : tensor<16x4xf32>
}

// CHECK-LABEL: func.func @reduce_middle_dim(
// CHECK-SAME:    %[[IN:.*]]: tensor<16x8x4xf32>, %[[INIT:.*]]: f32)
// CHECK:        xla_gpu.reduce(%[[IN]])
// CHECK-SAME:    inits(%[[INIT]]) dimensions=[1]
// CHECK-SAME:    combiner=@add
// CHECK-SAME:    : tensor<16x8x4xf32> to tensor<16x4xf32>

// -----

func.func @do_nothing(%a: f32, %b: i32, %c: f32, %d: i32) -> (f32, i32) {
  return %a, %b : f32, i32
}
func.func @shuffler(%a: f32, %b: i32) -> (f32, i32) {
  %ret:2 = xla_gpu.shuffle_reduce(%a, %b) to 4 combiner=@do_nothing
    {xla.range = [0 : index, 42 : index]} : f32, i32
  return %ret#0, %ret#1 : f32, i32
}
// CHECK-LABEL: func.func @shuffler(
// CHECK-SAME:    %[[IN1:.*]]: f32, %[[IN2:.*]]: i32)

// CHECK:        xla_gpu.shuffle_reduce(%[[IN1]], %[[IN2]]) to 4
// CHECK-SAME:    combiner=@do_nothing {xla.range = [0 : index, 42 : index]}
// CHECK-SAME:    : f32, i32
