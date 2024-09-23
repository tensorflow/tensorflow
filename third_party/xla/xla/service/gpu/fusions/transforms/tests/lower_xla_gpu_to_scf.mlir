// RUN: mlir_fusions_opt %s -xla-gpu-lower-xla-gpu-to-scf --split-input-file  \
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
// CHECK: %[[AB4_0:.*]], %[[AB4_1:.*]] = xla_gpu.pure_call @combiner(%[[A]], %[[B]], %[[A4H]], %[[B4H]])
// CHECK: %[[A2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_0]], %[[C2]], %[[C32]]
// CHECK: %[[B2H:.*]], {{.*}} = gpu.shuffle down %[[AB4_1]], %[[C2]], %[[C32]]
// CHECK: %[[AB2_0:.*]], %[[AB2_1:.*]] = xla_gpu.pure_call @combiner(%[[AB4_0]], %[[AB4_1]], %[[A2H]], %[[B2H]])
// CHECK: %[[A1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_0]], %[[C1]], %[[C32]]
// CHECK: %[[B1H:.*]], {{.*}} = gpu.shuffle down %[[AB2_1]], %[[C1]], %[[C32]]
// CHECK: %[[AB1_0:.*]], %[[AB1_1:.*]] = xla_gpu.pure_call @combiner(%[[AB2_0]], %[[AB2_1]], %[[A1H]], %[[B1H]])
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
  %ret = xla_gpu.predicated_insert %v into %tensor[%index] if %cond
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
  %ret = xla_gpu.predicated_extract %tensor[%index] if %cond else %v
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

func.func private @exp(%p0: tensor<128x64xf32>, %i: index, %j: index) -> f32

// The block ID (d1) defines the offset into the input tensor.
#map = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (
    (d1 floordiv 2) * 32 + s0,
    (d1 mod 2) * 32 + s1
  ),
  domain: d0 in [0, 255], d1 in [0, 7], s0 in [0, 31], s1 in [0, 31],
  is_simplified: false>

// Each thread holds a 2x2 block of the input tensor. d0 is the thread ID.
#map1 = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (
    (d0 mod 16) * 2 + s0,
    (d0 floordiv 16) * 2 + s1),
  domain: d0 in [0, 255], d1 in [0, 7], s0 in [0, 1], s1 in [0, 1],
  is_simplified: false>

func.func @materialize(%input: tensor<128x64xf32>, %th: index, %bl: index)
    -> !xla_gpu.indexed_vector<32x32xf32, #map1> {
  %0 = xla_gpu.materialize @exp(%input) at #map(%th, %bl)
    : (tensor<128x64xf32>) -> !xla_gpu.indexed_vector<32x32xf32, #map1>
  func.return %0 : !xla_gpu.indexed_vector<32x32xf32, #map1>
}
// CHECK-DAG: #[[$ENCODING:.*]] = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> ((d0 mod 16) * 2 + s0, (d0 floordiv 16) * 2 + s1)
// CHECK-DAG: #[[$X:.*]] = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> ((d1 floordiv 2) * 32 + s0)
// CHECK-DAG: #[[$Y:.*]] = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> ((d1 mod 2) * 32 + s1)

// CHECK: @materialize(%[[INPUT:.*]]: tensor<128x64xf32>, %[[TH:.*]]: index, %[[BL:.*]]: index)

// CHECK:      %[[INIT_VEC:.*]] = arith.constant {{.*}} : vector<2x2xf32>
// CHECK:      xla_gpu.loop (%[[TH]], %[[BL]])[%[[S0:.*]], %[[S1:.*]]]
// CHECK-SAME:   -> (%[[VX:.*]], %[[VY:.*]]) in
// CHECK-SAME:   #[[$ENCODING]] iter_args(%[[ITER_ARG:.*]] = %[[INIT_VEC]])

// CHECK: %[[TENSOR_X:.*]] = xla_gpu.apply_indexing #[[$X]](%[[TH]], %[[BL]])[%[[VX]], %[[VY]]]
// CHECK: %[[TENSOR_Y:.*]] = xla_gpu.apply_indexing #[[$Y]](%[[TH]], %[[BL]])[%[[VX]], %[[VY]]]

// CHECK: %[[PURE_CALL:.*]] = xla_gpu.pure_call @exp(%[[INPUT]], %[[TENSOR_X]], %[[TENSOR_Y]])
// CHECK: vector.insert %[[PURE_CALL]], %[[ITER_ARG]] [%[[S0]], %[[S1]]]
// CHECK xla_gpu.yield %{{.*}} : vector<2x2xf32>

// -----

#map = #xla_gpu.indexing_map<(d0)[s0] -> (d0 * 2 + s0),
  domain: d0 in [0, 127], s0 in [0, 1],
  is_simplified: false>
#map1 = #xla_gpu.indexing_map<(d0)[s0] -> (s0 floordiv 16, s0 mod 16),
  domain: d0 in [0, 127], s0 in [0, 255],
  is_simplified: false>

func.func @insert(%input: !xla_gpu.indexed_vector<256xf32, #map>,
    %th: index, %output: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = xla_gpu.insert %input(%th) into %output at #map1
    : !xla_gpu.indexed_vector<256xf32, #map> -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-DAG: #[[$ENCODING:.*]] = #xla_gpu.indexing_map<(d0)[s0] -> (d0 * 2 + s0),
// CHECK-DAG: #[[$X:.*]] = #xla_gpu.indexing_map<(d0)[s0] -> (s0 floordiv 16),
// CHECK-DAG: #[[$Y:.*]] = #xla_gpu.indexing_map<(d0)[s0] -> (s0 mod 16),

// CHECK:      @insert(%[[INPUT:.*]]: !xla_gpu.indexed_vector<256xf32, #[[$ENCODING]]>,
// CHECK-SAME:   %[[TH:.*]]: index,
// CHECK-SAME:   %[[OUTPUT:.*]]: tensor<16x16xf32>)

// CHECK:      xla_gpu.loop (%[[TH]])[%[[S0:.*]]] ->
// CHECK-SAME:   (%[[VECTOR_X:.*]]) in #[[$ENCODING]]
// CHECK-SAME:   iter_args(%[[TENSOR:.*]] = %[[OUTPUT]])

// CHECK:        %[[SCALAR:.*]] = vector.extract %{{.*}}[%[[S0]]]
// CHECK-SAME: :     f32 from vector<2xf32>
// CHECK:        %[[TENSOR_X:.*]] = xla_gpu.apply_indexing #[[$X]]
// CHECK-SAME:       (%[[TH]])[%[[VECTOR_X]]]
// CHECK:        %[[TENSOR_Y:.*]] = xla_gpu.apply_indexing #[[$Y]]
// CHECK-SAME:       (%[[TH]])[%[[VECTOR_X]]]
// CHECK:        %[[NEW_TENSOR:.*]] = tensor.insert %[[SCALAR]]
// CHECK-SAME:       into %[[TENSOR]][%[[TENSOR_X]], %[[TENSOR_Y]]]
// CHECK:        xla_gpu.yield %[[NEW_TENSOR]]

// -----

func.func private @exp(%p0: tensor<32x64xf32>, %i: index, %j: index) -> f32

#map = #xla_gpu.indexing_map<(d0, d1)[s0, s1, s2] -> (d1*32+s0, s1*2 + s2),
  domain: d0 in [0, 31], d1 in [0, 8], s0 in [0, 31], s1 in [0, 1], s2 in [0, 1],
  is_simplified: false>
#map1 = #xla_gpu.indexing_map<(d0, d1)[s0, s1, s2] -> (s0, s0, s1),
  domain: d0 in [0, 31], d1 in [0, 8], s0 in [0, 31], s1 in [0, 1], s2 in [0, 1],
  is_simplified: false>
#map2 = #xla_gpu.indexing_map<(d0, d1)[s0, s1, s2] -> (s0, s1 * 2 + s2),
  domain: d0 in [0, 31], d1 in [0, 8], s0 in [0, 31], s1 in [0, 1], s2 in [0, 1],
  is_simplified: false>

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

func.func private @exp(%p0: tensor<1024xcomplex<f32>>, %i: index) -> complex<f32>

// Each thread holds four consecutive complex values of the input.
#map = #xla_gpu.indexing_map<(d0, d1)[s0] -> (d1 * 256 + s0),
  domain: d0 in [0, 63], d1 in [0, 3],
  s0 in [0, 255], is_simplified: false>
#map1 = #xla_gpu.indexing_map<(d0, d1)[s0] -> (d0 * 4 + s0),
  domain: d0 in [0, 63], d1 in [0, 3],
  s0 in [0, 3], is_simplified: false>
func.func @materialize_complex(
  %input: tensor<1024xcomplex<f32>>,
  %d0: index,
  %d1: index) -> !xla_gpu.indexed_vector<256xcomplex<f32>, #map1> {

  %0 = xla_gpu.materialize @exp(%input) at #map(%d0, %d1)
    : (tensor<1024xcomplex<f32>>)
    -> !xla_gpu.indexed_vector<256xcomplex<f32>, #map1>
  func.return %0 : !xla_gpu.indexed_vector<256xcomplex<f32>, #map1>
}

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: xla_gpu.loop ({{.*}})[%[[I:.*]]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = {{.*}})
// CHECK: %[[PURE_CALL:.*]] = xla_gpu.pure_call
// CHECK-SAME: complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[PURE_CALL]]
// CHECK: %[[IMAG:.*]] = complex.im %[[PURE_CALL]]
// CHECK: %[[TEMP:.*]] = vector.insert %[[REAL]], %[[ITER]] [%[[C0]], %[[I]]]
// CHECK: %[[FINAL:.*]] = vector.insert %[[IMAG]], %[[TEMP]] [%[[C1]], %[[I]]]
// CHECK: xla_gpu.yield %[[FINAL]] : vector<2x4xf32>

// -----

#map1 = #xla_gpu.indexing_map<(d0, d1)[s0, s1] -> (d1 * 12 + s0 * 4 + s1),
  domain: d0 in [0, 31], d1 in [0, 2],
  s0 in [0, 2], s1 in [0, 3], is_simplified: false>
#map2 = #xla_gpu.indexing_map<(d0, d1)[s0] -> (d1 * 384 + s0),
  domain: d0 in [0, 31], d1 in [0, 2], s0 in [0, 383],
  is_simplified: false>
func.func @insert_complex(
  %input: !xla_gpu.indexed_vector<384xcomplex<f32>, #map1>,
  %output: tensor<1024xcomplex<f32>>,
  %d0: index,
  %d1: index) -> tensor<1024xcomplex<f32>> {

  %1 = xla_gpu.insert %input(%d0, %d1) into %output at #map2
    : !xla_gpu.indexed_vector<384xcomplex<f32>, #map1>
    -> tensor<1024xcomplex<f32>>
  func.return %1 : tensor<1024xcomplex<f32>>
}

// CHECK-LABEL: @insert_complex
// CHECK-SAME: %[[INPUT:.*]]: !xla_gpu.indexed_vector<384xcomplex<f32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[VECTOR:.*]] = builtin.unrealized_conversion_cast %[[INPUT]]
// CHECK-SAME: to vector<2x3x4xf32>
// CHECK: xla_gpu.loop ({{.*}})[%[[I:.*]], %[[J:.*]]]
// CHECK-SAME: iter_args(%[[ITER:.*]] = {{.*}})
// CHECK: %[[REAL:.*]] = vector.extract %[[VECTOR]][%[[C0]], %[[I]], %[[J]]]
// CHECK: %[[IMAG:.*]] = vector.extract %[[VECTOR]][%[[C1]], %[[I]], %[[J]]]
// CHECK: %[[COMPLEX:.*]] = complex.create %[[REAL]], %[[IMAG]]
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[COMPLEX]] into %[[ITER]]
// CHECK: xla_gpu.yield %[[INSERTED]] : tensor<1024xcomplex<f32>>