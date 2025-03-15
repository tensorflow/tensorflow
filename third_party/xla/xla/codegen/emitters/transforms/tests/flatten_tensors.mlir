// RUN: emitters_opt %s -split-input-file -xla-flatten-tensors \
// RUN: --verify-diagnostics | FileCheck %s

func.func @tensor_extract(
    %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
    %arg1: index, %arg2: index) -> f32 {
  %v = tensor.extract %arg0[%arg1, %arg2]
      : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = #xla.indexing_map<"(d0, d1) -> (d1 * 2 + d0), domain: d0 in [0, 1], d1 in [0, 2]">

// CHECK-LABEL: func.func @tensor_extract(
// CHECK-SAME:      %[[SRC:.*]]: tensor<6xf32>,
// CHECK-SAME:      %[[I:.*]]: index, %[[J:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = xla.apply_indexing #[[$MAP]](%[[I]], %[[J]])
// CHECK:        tensor.extract %[[SRC]][%[[INDEX]]] : tensor<6xf32>

// -----

func.func @tensor_insert(
    %arg0: tensor<10x24xcomplex<f32>>) -> tensor<10x24xcomplex<f32>> {
  %c1 = arith.constant 1 : index
  %real = arith.constant 3.0 : f32
  %imag = arith.constant 2.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %out = tensor.insert %complex into %arg0[%c1, %c1]
    : tensor<10x24xcomplex<f32>>
  func.return %out : tensor<10x24xcomplex<f32>>
}
// CHECK-LABEL: func.func @tensor_insert(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<240xcomplex<f32>>) -> tensor<240xcomplex<f32>> {
// CHECK:         %[[INDEX:.*]] = arith.constant 25
// CHECK:         %[[COMPLEX:.*]] = complex.create
// CHECK:         tensor.insert %[[COMPLEX]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-SAME:      : tensor<240xcomplex<f32>>

// -----

func.func @update(%arg0: tensor<10x24xf32>) -> tensor<10x24xf32> {
  %c1 = arith.constant 1 : index
  %c42_f32 = arith.constant 42.0 : f32
  %out = tensor.insert %c42_f32 into %arg0[%c1, %c1] : tensor<10x24xf32>
  func.return %out : tensor<10x24xf32>
}

func.func @pure_call(%arg0: tensor<10x24xf32>) -> tensor<10x24xf32> {
  %updated_tensor = xla.pure_call @update(%arg0)
    : (tensor<10x24xf32>) -> (tensor<10x24xf32>)
  func.return %updated_tensor : tensor<10x24xf32>
}
// CHECK-LABEL: func.func @pure_call(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<240xf32>) -> tensor<240xf32> {
// CHECK-NEXT:    xla.pure_call @update(%[[TENSOR]])
// CHECK-SAME:      : (tensor<240xf32>) -> tensor<240xf32>
// CHECK-NEXT:  return

// -----

func.func @atomic_rmw(%in: tensor<2x4xf32>, %i: index, %j: index)
    -> (tensor<2x4xf32>) {
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x4xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 1.0 : f32
      %add = arith.minimumf %current, %c42 : f32
      xla.yield %add : f32
  }
  return %ret : tensor<2x4xf32>
}
// CHECK: #[[$MAP:.+]] = #xla.indexing_map<"(d0, d1) -> (d0 * 4 + d1), domain: d0 in [0, 1], d1 in [0, 3]">
// CHECK-LABEL: func.func @atomic_rmw(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<8xf32>, %[[I:.*]]: index,
// CHECK-SAME:      %[[J:.*]]: index) -> tensor<8xf32> {
// CHECK:         %[[INDEX:.*]] = xla.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[I]], %[[J]])
// CHECK:         xla.atomic_rmw %[[TENSOR]][%[[INDEX]]] : tensor<8xf32>

// -----

func.func @for_loop(%t0: tensor<32x1024xf32>, %t1: tensor<64x8x4xf32>)
    -> (tensor<32x1024xf32>, tensor<64x8x4xf32>, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0_f32 = arith.constant 0.0 : f32
  %for:2 = scf.for %i = %c0 to %c64 step %c32 iter_args(%t0_ = %t0, %t1_ = %t1)
    -> (tensor<32x1024xf32>, tensor<64x8x4xf32>) {
    %update0 = tensor.insert %c0_f32 into %t0_[%c1, %i] : tensor<32x1024xf32>
    %update1 = tensor.insert %c0_f32 into %t1_[%i, %c1, %c1]
      : tensor<64x8x4xf32>
    scf.yield %update0, %update1 : tensor<32x1024xf32>, tensor<64x8x4xf32>
  } {some_attr}
    return %for#0, %for#1, %c0_f32 : tensor<32x1024xf32>, tensor<64x8x4xf32>, f32
}
// CHECK: #[[$MAP0:.+]] = #xla.indexing_map<"(d0) -> (d0 + 1024)
// CHECK: #[[$MAP1:.+]] = #xla.indexing_map<"(d0) -> (d0 * 32 + 5)
// CHECK-LABEL: func.func @for_loop(
// CHECK-SAME:      %[[T0:.*]]: tensor<32768xf32>,
// CHECK-SAME:      %[[T1:.*]]: tensor<2048xf32>) -> (tensor<32768xf32>, tensor<2048xf32>, f32) {

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:  %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:  %[[F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[FOR:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[C64]]
// CHECK-SAME:     step %[[C32]]
// CHECK-SAME:     iter_args(%[[T0_:.*]] = %[[T0]], %[[T1_:.*]] = %[[T1]])
// CHECK:        %[[IND0:.*]] = xla.apply_indexing #[[$MAP0]](%[[I]])
// CHECK:        %[[UPD0:.*]] = tensor.insert %[[F32]] into %[[T0_]][%[[IND0]]]
// CHECK:        %[[IND1:.*]] = xla.apply_indexing #[[$MAP1]](%[[I]])
// CHECK:        %[[UPD1:.*]] = tensor.insert %[[F32]] into %[[T1_]][%[[IND1]]]
// CHECK:        scf.yield %[[UPD0]], %[[UPD1]] : tensor<32768xf32>, tensor<2048xf32>

// -----

#map = #xla.indexing_map<"(d0, d1) -> ((d1 * 128 + d0) floordiv 36), domain: d0 in [0, 127], d1 in [0, 393749]">
#map1 = #xla.indexing_map<"(d0, d1) -> (((d1 * 128 + d0) floordiv 9) mod 4), domain: d0 in [0, 127], d1 in [0, 393749]">
#map2 = #xla.indexing_map<"(d0, d1) -> ((d1 * 128 + d0) mod 9), domain: d0 in [0, 127], d1 in [0, 393749]">
func.func @if_op(%arg0: tensor<4000x4x9xf32>, %arg1: tensor<1400x1xi32>,
    %arg2: tensor<1400x1x4x9xf32>, %arg3: tensor<4000x4x9xf32>)
     -> tensor<4000x4x9xf32> {
	%c0 = arith.constant 0 : index
	%c3999 = arith.constant 3999 : index
	%th_x = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
	%bl_x = gpu.block_id  x {xla.range = [0 : index, 393749 : index]}
	%0 = xla.apply_indexing #map(%th_x, %bl_x)
	%extracted = tensor.extract %arg1[%0, %c0] : tensor<1400x1xi32>
	%1 = arith.index_cast %extracted : i32 to index
	%2 = arith.cmpi ule, %1, %c3999 : index
	%3 = scf.if %2 -> (tensor<4000x4x9xf32>) {
		%4 = xla.apply_indexing #map1(%th_x, %bl_x)
		%5 = xla.apply_indexing #map2(%th_x, %bl_x)
		%elem = tensor.extract %arg2[%0, %c0, %4, %5] : tensor<1400x1x4x9xf32>
		%atomic_rmw = xla.atomic_rmw %arg3[%1, %4, %5] : tensor<4000x4x9xf32> {
		^bb0(%arg4: f32):
			%6 = arith.addf %arg4, %elem : f32
			xla.yield %6 : f32
		}
		scf.yield %atomic_rmw : tensor<4000x4x9xf32>
	} else {
		scf.yield %arg3 : tensor<4000x4x9xf32>
	}
	return %3 : tensor<4000x4x9xf32>
}
// CHECK-LABEL: func.func @if_op
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         scf.if %{{.*}} -> (tensor<144000xf32>) {
// CHECK-COUNT-2:   scf.yield %{{.*}} : tensor<144000xf32>
// CHECK:         return %{{.*}} : tensor<144000xf32>

// -----

func.func @allocate_shared() -> tensor<10x15xf32> {
  %shmem = xla_gpu.allocate_shared : tensor<10x15xf32>
  func.return %shmem : tensor<10x15xf32>
}
// CHECK-LABEL: func.func @allocate_shared() -> tensor<150xf32>
// CHECK:         xla_gpu.allocate_shared : tensor<150xf32>
// CHECK-NOT:     builtin.unrealized_conversion_cast

// -----

func.func @sync() -> (tensor<8x4xf32>, tensor<8x4xf32>) {
  %shared1 = xla_gpu.allocate_shared : tensor<8x4xf32>
  %shared2 = xla_gpu.allocate_shared : tensor<8x4xf32>
  %sync:2 = xla_gpu.sync_threads %shared1, %shared2
    : tensor<8x4xf32>, tensor<8x4xf32>
  return %sync#0, %sync#1 : tensor<8x4xf32>, tensor<8x4xf32>
}
// CHECK-LABEL: func.func @sync() -> (tensor<32xf32>, tensor<32xf32>) {
// CHECK:         %[[SHARED1:.*]] = xla_gpu.allocate_shared : tensor<32xf32>
// CHECK:         %[[SHARED2:.*]] = xla_gpu.allocate_shared : tensor<32xf32>
// CHECK:         %[[SYNC:.*]] = xla_gpu.sync_threads %[[SHARED1]], %[[SHARED2]]
// CHECK-SAME:      : tensor<32xf32>, tensor<32xf32>
// CHECK-NEXT:    return

// -----

func.func @index_switch(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>,
    %arg2: tensor<2x3xf32>, %arg3: tensor<2x3xf32>
    ) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %block_id_y = gpu.block_id  y {xla.range = [0 : index, 1 : index]}
  %0:2 = scf.index_switch %block_id_y -> tensor<2x3xf32>, tensor<2x3xf32>
  case 1 {
    scf.yield %arg0, %arg3 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  default {
    scf.yield %arg1, %arg2 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  return %0#0, %0#1: tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK-LABEL: func.func @index_switch
// CHECK-SAME: -> (tensor<6xf32>, tensor<6xf32>)
// CHECK-NOT:  builtin.unrealized_conversion_cast

// -----

func.func @cpu_load(%arg0: !xla_cpu.call_frame) -> tensor<10x10x4xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<10x10x4xf32>
  return %0 : tensor<10x10x4xf32>
}
// CHECK-LABEL: func.func @cpu_load
// CHECK-SAME: -> tensor<400xf32>
// CHECK-NOT:  builtin.unrealized_conversion_cast

// -----

func.func @constant() -> tensor<2x3xf32> {
   %cst = arith.constant dense<[
    [-3.000000e+00, 2.000000e+00, 1.000000e+00],
    [0.000000e+00, -3.000000e+00, 1.000000e+00]
   ]> : tensor<2x3xf32>
   return %cst : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @constant
// CHECK-SAME: -> tensor<6xf32>
// CHECK-NOT:  builtin.unrealized_conversion_cast

// -----

func.func @dangling_cast(%arg0: tensor<6xf32>, %arg1: index) -> i32 {
  %v = tensor.extract %arg0[%arg1] : tensor<6xf32>
	%cast = builtin.unrealized_conversion_cast %v : f32 to i32
  func.return %cast : i32
}
// CHECK: FlattenTensorsPass failed to converge

// -----

func.func @vector_extract(%arg0: vector<2x3xf32>, %arg1: index) -> f32 {
  %v = vector.extract %arg0[%arg1, 2] : f32 from vector<2x3xf32>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = #xla.indexing_map<"(d0) -> (d0 * 3 + 2),
// CHECK-SAME: domain: d0 in [0, 1]

// CHECK-LABEL: func.func @vector_extract(
// CHECK-SAME:      %[[SRC:.*]]: vector<6xf32>, %[[I:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = xla.apply_indexing #[[$MAP]](%[[I]])
// CHECK:        vector.extract %[[SRC]][%[[INDEX]]] : f32 from vector<6xf32>

// -----

func.func @vector_transfer_read(%arg0: tensor<64x66xbf16>, %i: index, %j: index)
    -> vector<2xbf16> {
  %cst = arith.constant 0.0 : bf16
  %v = vector.transfer_read %arg0[%i, %j], %cst {in_bounds = [true]}
    : tensor<64x66xbf16>, vector<2xbf16>
  func.return %v : vector<2xbf16>
}
// CHECK: #[[$MAP:.+]] = #xla.indexing_map<"(d0, d1) -> (d0 * 66 + d1)
// CHECK-SAME: domain: d0 in [0, 63], d1 in [0, 65]

// CHECK-LABEL: func.func @vector_transfer_read(
// CHECK-SAME:      %[[SRC:.*]]: tensor<4224xbf16>,
// CHECK-SAME:      %[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:        %[[INDEX:.*]] = xla.apply_indexing #[[$MAP]](%[[I]], %[[J]])
// CHECK:        vector.transfer_read %[[SRC]][%[[INDEX]]]

// -----

func.func @vector_insert(%arg0: vector<10x24xf32>, %i: index)
  -> vector<10x24xf32> {
  %scalar = arith.constant 3.0 : f32
  %out = vector.insert %scalar, %arg0 [1, %i] : f32 into vector<10x24xf32>
  func.return %out : vector<10x24xf32>
}
// CHECK: #[[$MAP:.+]] = #xla.indexing_map<"(d0) -> (d0 + 24),
// CHECK-SAME: domain: d0 in [0, 23]
// CHECK-LABEL: func.func @vector_insert(
// CHECK-SAME:      %[[VECTOR:.*]]: vector<240xf32>, %[[I:.*]]: index) ->
// CHECK-SAME:      vector<240xf32> {
// CHECK:         %[[INDEX:.*]] = xla.apply_indexing #[[$MAP]](%[[I]])
// CHECK:         vector.insert {{.*}}, %[[VECTOR]] [%[[INDEX]]]
// CHECK-SAME:      : f32 into vector<240xf32>

// -----

func.func @update(%arg0: vector<10x24xf32>) -> vector<10x24xf32> {
  %c1 = arith.constant 1 : index
  %c42_f32 = arith.constant 42.0 : f32
  %out = vector.insert %c42_f32, %arg0[%c1, %c1] : f32 into vector<10x24xf32>
  func.return %out : vector<10x24xf32>
}

func.func @pure_call_vector(%arg0: vector<10x24xf32>) -> vector<10x24xf32> {
  %updated_vector = xla.pure_call @update(%arg0)
    : (vector<10x24xf32>) -> (vector<10x24xf32>)
  func.return %updated_vector : vector<10x24xf32>
}
// CHECK-LABEL: func.func @pure_call_vector(
// CHECK-SAME:      %[[VECTOR:.*]]: vector<240xf32>) -> vector<240xf32> {
// CHECK-NEXT:    xla.pure_call @update(%[[VECTOR]])
// CHECK-SAME:      : (vector<240xf32>) -> vector<240xf32>
// CHECK-NEXT:  return

// -----

func.func @for_loop_vector(%t0: vector<32x1024xf32>, %t1: vector<64x8x4xf32>)
    -> (vector<32x1024xf32>, vector<64x8x4xf32>, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0_f32 = arith.constant 0.0 : f32
  %for:2 = scf.for %i = %c0 to %c64 step %c32 iter_args(%t0_ = %t0, %t1_ = %t1)
    -> (vector<32x1024xf32>, vector<64x8x4xf32>) {
    %update0 = vector.insert %c0_f32, %t0_ [%c1, %i] :
      f32 into vector<32x1024xf32>
    %update1 = vector.insert %c0_f32, %t1_ [%i, %c1, %c1] :
      f32 into vector<64x8x4xf32>
    scf.yield %update0, %update1 : vector<32x1024xf32>, vector<64x8x4xf32>
  } {some_attr}
    return %for#0, %for#1, %c0_f32 :
      vector<32x1024xf32>, vector<64x8x4xf32>, f32
}
// CHECK: #[[$MAP0:.+]] = #xla.indexing_map<"(d0) -> (d0 + 1024)
// CHECK: #[[$MAP1:.+]] = #xla.indexing_map<"(d0) -> (d0 * 32 + 5)
// CHECK-LABEL: func.func @for_loop_vector(
// CHECK-SAME:      %[[V0:.*]]: vector<32768xf32>,
// CHECK-SAME:      %[[V1:.*]]: vector<2048xf32>) ->
// CHECK-SAME:      (vector<32768xf32>, vector<2048xf32>, f32) {

// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:  %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:  %[[F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:      %[[FOR:.*]]:2 = scf.for %[[I:.*]] = %[[C0]] to %[[C64]]
// CHECK-SAME:     step %[[C32]]
// CHECK-SAME:     iter_args(%[[V0_:.*]] = %[[V0]], %[[V1_:.*]] = %[[V1]])
// CHECK:        %[[IND0:.*]] = xla.apply_indexing #[[$MAP0]](%[[I]])
// CHECK:        %[[UPD0:.*]] = vector.insert %[[F32]], %[[V0_]] [%[[IND0]]]
// CHECK:        %[[IND1:.*]] = xla.apply_indexing #[[$MAP1]](%[[I]])
// CHECK:        %[[UPD1:.*]] = vector.insert %[[F32]], %[[V1_]] [%[[IND1]]]
// CHECK:        scf.yield %[[UPD0]], %[[UPD1]] :
// CHECK-SAME:      vector<32768xf32>, vector<2048xf32>

// -----

func.func @if_op_vector(%arg0: vector<8x1xf32>, %value: f32)
  -> vector<8x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c599 = arith.constant 599 : index
  %c32 = arith.constant 32 : index
  %thread_id_x = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %block_id_x = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %0 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %cst)
    -> (vector<8x1xf32>) {
    %9 = arith.cmpi sle, %thread_id_x, %c32 : index
    %17 = arith.cmpi sle, %block_id_x, %c599 : index
    %23 = arith.andi %9, %17 : i1
    %24 = scf.if %23 -> (vector<8x1xf32>) {
      %29 = vector.insert %value, %arg3 [%arg2, %c0] : f32 into vector<8x1xf32>
      scf.yield %29 : vector<8x1xf32>
    } else {
      scf.yield %arg3 : vector<8x1xf32>
    }
    scf.yield %24 : vector<8x1xf32>
  }
  return %0 : vector<8x1xf32>
}

// CHECK-LABEL: func.func @if_op_vector
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK:         scf.if %{{.*}} -> (vector<8xf32>) {
// CHECK-COUNT-3:   scf.yield %{{.*}} : vector<8xf32>
// CHECK:         return %{{.*}} : vector<8xf32>

// -----

func.func @constant_vector() -> vector<2x3xf32> {
   %cst = arith.constant dense<[
    [-3.000000e+00, 2.000000e+00, 1.000000e+00],
    [0.000000e+00, -3.000000e+00, 1.000000e+00]
   ]> : vector<2x3xf32>
   return %cst : vector<2x3xf32>
}
// CHECK-LABEL: func.func @constant_vector
// CHECK-SAME: -> vector<6xf32>
// CHECK-NOT:  builtin.unrealized_conversion_cast
