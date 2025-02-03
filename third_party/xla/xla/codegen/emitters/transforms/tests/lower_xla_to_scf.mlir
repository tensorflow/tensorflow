// RUN: emitters_opt %s -xla-lower-xla-to-scf --split-input-file  \
// RUN: | FileCheck %s

func.func @combiner(%a: f32, %b: i32, %c: f32, %d: i32) -> (f32, i32) {
  return %a, %b : f32, i32
}

func.func @shuffler(%a: f32, %b: i32) -> (f32, i32) {
  %ret:2 = xla_gpu.shuffle_reduce (%a, %b) to 4 combiner=@combiner: f32, i32
  return %ret#0, %ret#1 : f32, i32
}
// CHECK: @shuffler(%[[A:.*]]: f32, %[[B:.*]]: i32)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[C32:.*]] = arith.constant 32
// CHECK: %[[A4H:.*]], {{.*}} = gpu.shuffle down %[[A]], %[[C4]], %[[C32]]
// CHECK: %[[B4H:.*]], {{.*}} = gpu.shuffle down %[[B]], %[[C4]], %[[C32]]
// CHECK: %[[AB4_0:.*]], %[[AB4_1:.*]] = xla.pure_call @combiner(%[[A]], %[[B]], %[[A4H]], %[[B4H]])
// CHECK: %[[A2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_0]], %[[C2]], %[[C32]]
// CHECK: %[[B2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_1]], %[[C2]], %[[C32]]
// CHECK: %[[AB2_0:.*]], %[[AB2_1:.*]] = xla.pure_call @combiner(%[[AB4_0]], %[[AB4_1]], %[[A2H]], %[[B2H]])
// CHECK: %[[A1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_0]], %[[C1]], %[[C32]]
// CHECK: %[[B1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_1]], %[[C1]], %[[C32]]
// CHECK: %[[AB1_0:.*]], %[[AB1_1:.*]] = xla.pure_call @combiner(%[[AB2_0]], %[[AB2_1]], %[[A1H]], %[[B1H]])
// CHECK: return %[[AB1_0]], %[[AB1_1]]

// -----

func.func @combiner(%a: f64, %b: f64) -> f64 {
  return %a : f64
}

func.func @shuffler(%a: f64) -> f64 {
  %ret = xla_gpu.shuffle_reduce(%a) to 1 combiner=@combiner : f64
  return %ret : f64
}
// CHECK: @shuffler(%[[A:.*]]: f64
// CHECK: gpu.shuffle down {{.*}}, %[[C1]]
// CHECK: gpu.shuffle down {{.*}}, %[[C1]]

// -----

func.func @combiner(%a: complex<f64>, %b: complex<f64>) -> complex<f64> {
  return %a : complex<f64>
}

func.func @shuffler(%a: complex<f64>) -> complex<f64> {
  %ret = xla_gpu.shuffle_reduce(%a) to 1 combiner=@combiner : complex<f64>
  return %ret : complex<f64>
}
// CHECK: @shuffler
// CHECK-COUNT-4: gpu.shuffle down {{.*}}, %[[C1]]

// -----

func.func @combiner(%a: ui64, %b: ui64) -> ui64 {
  return %a : ui64
}

func.func @shuffler(%a: ui64) -> ui64 {
  %ret = xla_gpu.shuffle_reduce (%a) to 1 combiner=@combiner : ui64
  return %ret : ui64
}
// CHECK: @shuffler
// CHECK: unrealized_conversion_cast
// CHECK-COUNT-2: gpu.shuffle down {{.*}}, %[[C1]]

// -----

func.func @combiner(%a: i8, %b: i8) -> i8 {
  return %a : i8
}

func.func @shuffler_i8(%a: i8) -> i8 {
  %ret = xla_gpu.shuffle_reduce (%a) to 1 combiner=@combiner : i8
  return %ret : i8
}
// CHECK: @shuffler_i8(
// CHECK-NOT: vector
// CHECK-COUNT-1: gpu.shuffle down {{.*}}, %[[C1]]

// -----

func.func @predicated_insert(
    %v: i32, %tensor: tensor<2xi32>, %index: index,
    %cond: i1) -> tensor<2xi32> {
  %ret = xla.predicated_insert %v into %tensor[%index] if %cond
    : tensor<2xi32>
  return %ret : tensor<2xi32>
}
// CHECK: @predicated_insert
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[UPD:.*]] = tensor.insert %[[V]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[UPD]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[TENSOR]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]

// -----

func.func @predicated_extract(
    %v: i32, %tensor: tensor<2xi32>, %index: index,
    %cond: i1) -> i32 {
  %ret = xla.predicated_extract %tensor[%index] if %cond else %v
    : tensor<2xi32>
  return %ret : i32
}
// CHECK: @predicated_extract
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[VAL:.*]] = tensor.extract  %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[VAL]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[V]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]

// -----

func.func private @exp(%p0: tensor<32x64xf32>, %i: index, %j: index) -> f32

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1], s1 in [0, 1]">

func.func @materialize(%input: tensor<32x64xf32>, %i: index, %j: index)
    -> !xla_gpu.indexed_vector<32x2x2xf32, #map1> {
  %0 = xla_gpu.materialize @exp(%input) at #map(%i, %j)
    : (tensor<32x64xf32>) -> !xla_gpu.indexed_vector<32x2x2xf32, #map1>
  func.return %0 : !xla_gpu.indexed_vector<32x2x2xf32, #map1>
}
// CHECK-DAG: #[[$MAP:.*]] = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1 * 32 + d0 * 2 + s0, s1)
// CHECK-DAG: #[[$MAP1:.*]] = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d0 * 2 + s0, s1)

// CHECK: @materialize(%[[INPUT:.*]]: tensor<32x64xf32>, %[[INDEX1:.*]]: index, %[[INDEX2:.*]]: index)

// CHECK:      %[[INIT_VEC:.*]] = arith.constant {{.*}} : vector<2x2xf32>
// CHECK:      xla.loop (%[[INDEX1]], %[[INDEX2]])[%[[S0:.*]], %[[S1:.*]]]
// CHECK-SAME:   -> (%[[MAP_RESULT1:.*]], %[[MAP_RESULT2:.*]]) in
// CHECK-SAME:   #[[$MAP]] iter_args(%[[ITER_ARG:.*]] = %[[INIT_VEC]])

// CHECK: %[[PURE_CALL:.*]] = xla.pure_call @exp(%[[INPUT]], %[[MAP_RESULT1]], %[[MAP_RESULT2]])
// CHECK: vector.insert %[[PURE_CALL]], %[[ITER_ARG]] [%[[S0]], %[[S1]]]
// CHECK xla.yield %{{.*}} : vector<2x2xf32>

// -----

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #xla.indexing_map<"(d0, d1) -> (d0 mod 16, d1), domain: d0 in [0, 32], d1 in [0, 2]">

func.func @insert(%input: !xla_gpu.indexed_vector<32x64xf32, #map>,
    %i: index, %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = xla_gpu.insert %input(%i, %j) into %output at #map1
    : !xla_gpu.indexed_vector<32x64xf32, #map> -> tensor<32x64xf32>
  func.return %0 : tensor<32x64xf32>
}
// CHECK-DAG: #[[$MAP:.*]] = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1 * 32 + d0 * 2 + s0, s1)
// CHECK-DAG: #[[$MAP1:.*]] = #xla.indexing_map<"(d0, d1) -> (d0 mod 16, d1)

// CHECK:      @insert(%[[INPUT:.*]]: !xla_gpu.indexed_vector<32x64xf32, #[[$MAP]]>,
// CHECK-SAME:   %[[I:.*]]: index, %[[J:.*]]: index,
// CHECK-SAME:   %[[OUTPUT:.*]]: tensor<32x64xf32>)

// CHECK:      xla.loop (%[[I]], %[[J]])[%[[S0:.*]], %[[S1:.*]]] ->
// CHECK-SAME:   (%[[MAP_RESULT1:.*]], %[[MAP_RESULT2:.*]]) in #[[$MAP]]
// CHECK-SAME:   iter_args(%[[TENSOR:.*]] = %[[OUTPUT]])

// CHECK: %[[SCALAR:.*]] = vector.extract %{{.*}}[%[[S0]], %[[S1]]]
// CHECK-SAME: : f32 from vector<2x2xf32>
// CHECK: %[[MAP1_RESULT:.*]]:2 = xla.apply_indexing
// CHECK-SAME: #[[$MAP1]](%[[MAP_RESULT1]], %[[MAP_RESULT2]])
// CHECK: %[[NEW_TENSOR:.*]] = tensor.insert %[[SCALAR]]
// CHECK-SAME: into %[[TENSOR]][%[[MAP1_RESULT]]#0, %[[MAP1_RESULT]]#1]
// CHECK: xla.yield %[[NEW_TENSOR]]

// -----

func.func private @exp(%p0: tensor<32x64xf32>, %i: index, %j: index) -> f32

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 1], s1 in [0, 1]">
#map1 = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 1], s1 in [0, 1]">
#map2 = #xla.indexing_map<"(d0, d1) -> (d0, d1), domain: d0 in [0, 32], d1 in [0, 2]">

func.func @materialize_and_insert(%input: tensor<32x64xf32>, %i: index,
    %j: index, %output: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = xla_gpu.materialize @exp(%input) at #map(%i, %j)
    : (tensor<32x64xf32>) -> !xla_gpu.indexed_vector<32x2x2xf32, #map1>
  %1 = xla_gpu.insert %0(%i, %j) into %output at #map2
    : !xla_gpu.indexed_vector<32x2x2xf32, #map1> -> tensor<32x64xf32>
  func.return %1 : tensor<32x64xf32>
}
// CHECK-NOT: unrealized_conversion_cast

// -----

func.func private @exp(%p0: tensor<32x64xcomplex<f32>>, %i: index, %j: index) -> complex<f32>

#map = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d1*32+d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 8], s0 in [0, 2], s1 in [0, 3]">
#map1 = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 2], s1 in [0, 3]">
func.func @materialize_complex(
  %input: tensor<32x64xcomplex<f32>>,
  %output: tensor<32x64xcomplex<f32>>,
  %d0: index,
  %d1: index) -> !xla_gpu.indexed_vector<32x3x4xcomplex<f32>, #map1> {

  %0 = xla_gpu.materialize @exp(%input) at #map(%d0, %d1)
    : (tensor<32x64xcomplex<f32>>)
    -> !xla_gpu.indexed_vector<32x3x4xcomplex<f32>, #map1>
  func.return %0 : !xla_gpu.indexed_vector<32x3x4xcomplex<f32>, #map1>
}

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: xla.loop ({{.*}})[%[[I:.*]], %[[J:.*]]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = {{.*}})
// CHECK: %[[PURE_CALL:.*]] = xla.pure_call
// CHECK-SAME: complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[PURE_CALL]]
// CHECK: %[[IMAG:.*]] = complex.im %[[PURE_CALL]]
// CHECK: %[[TEMP:.*]] = vector.insert %[[REAL]], %[[ITER]] [%[[C0]], %[[I]], %[[J]]]
// CHECK: %[[FINAL:.*]] = vector.insert %[[IMAG]], %[[TEMP]] [%[[C1]], %[[I]], %[[J]]]
// CHECK: xla.yield %[[FINAL]] : vector<2x3x4xf32>

// -----

#map1 = #xla.indexing_map<"(d0, d1)[s0, s1] -> (d0*2+s0, s1), domain: d0 in [0, 32], d1 in [0, 2], s0 in [0, 2], s1 in [0, 3]">
#map2 = #xla.indexing_map<"(d0, d1) -> (d0, d1), domain: d0 in [0, 32], d1 in [0, 2]">
func.func @insert_complex(
  %input: !xla_gpu.indexed_vector<32x3x4xcomplex<f32>, #map1>,
  %output: tensor<32x64xcomplex<f32>>,
  %d0: index,
  %d1: index) -> tensor<32x64xcomplex<f32>> {

  %1 = xla_gpu.insert %input(%d0, %d1) into %output at #map2
    : !xla_gpu.indexed_vector<32x3x4xcomplex<f32>, #map1>
    -> tensor<32x64xcomplex<f32>>
  func.return %1 : tensor<32x64xcomplex<f32>>
}

// CHECK-LABEL: @insert_complex
// CHECK-SAME: %[[INPUT:.*]]: !xla_gpu.indexed_vector<32x3x4xcomplex<f32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[VECTOR:.*]] = builtin.unrealized_conversion_cast %[[INPUT]]
// CHECK-SAME: to vector<2x3x4xf32>
// CHECK: xla.loop ({{.*}})[%[[I:.*]], %[[J:.*]]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = {{.*}})
// CHECK: %[[REAL:.*]] = vector.extract %[[VECTOR]][%[[C0]], %[[I]], %[[J]]]
// CHECK: %[[IMAG:.*]] = vector.extract %[[VECTOR]][%[[C1]], %[[I]], %[[J]]]
// CHECK: %[[COMPLEX:.*]] = complex.create %[[REAL]], %[[IMAG]]
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[COMPLEX]] into %[[ITER]]
// CHECK: xla.yield %[[INSERTED]] : tensor<32x64xcomplex<f32>>
