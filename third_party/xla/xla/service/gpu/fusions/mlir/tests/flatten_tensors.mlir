// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-flatten-tensors \
// RUN: --verify-diagnostics | FileCheck %s

func.func @tensor_extract(
    %arg0: tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>,
    %arg1: index, %arg2: index) -> f32 {
  %v = tensor.extract %arg0[%arg1, %arg2]
      : tensor<2x3xf32, dense<[0, 1]> : tensor<2xi64>>
  func.return %v : f32
}
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d1 * 2 + d0)>

// CHECK-LABEL: func.func @tensor_extract(
// CHECK-SAME:      %[[SRC:.*]]: tensor<6xf32>,
// CHECK-SAME:      %[[I:.*]]: index, %[[J:.*]]: index) -> f32 {
// CHECK:        %[[INDEX:.*]] = xla_gpu.apply_indexing #[[$MAP]](%[[I]]
// CHECK-SAME:     in [0, 1], %[[J]] in [0, 2])
// CHECK:        tensor.extract %[[SRC]][%[[INDEX]]] : tensor<6xf32>

// -----

func.func @tensor_insert(
    %arg0: tensor<10x24xcomplex<f32>>) -> tensor<10x24xcomplex<f32>> {
  %c1 = arith.constant 1 : index
  %real = arith.constant 3.0 : f32
  %imag = arith.constant 2.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %out = tensor.insert %complex into %arg0[%c1, %c1] : tensor<10x24xcomplex<f32>>
  func.return %out : tensor<10x24xcomplex<f32>>
}
// CHECK-LABEL: func.func @tensor_insert(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<240xcomplex<f32>>) -> tensor<240xcomplex<f32>> {
// CHECK:         %[[INDEX:.*]] = arith.constant 25
// CHECK:         %[[COMPLEX:.*]] = complex.create
// CHECK:         tensor.insert %[[COMPLEX]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-SAME:      : tensor<240xcomplex<f32>>

// -----

func.func @atomic_rmw(%in: tensor<2x4xf32>, %i: index, %j: index)
    -> (tensor<2x4xf32>) {
  %ret = xla_gpu.atomic_rmw %in[%i, %j] : tensor<2x4xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 1.0 : f32
      %add = arith.minimumf %current, %c42 : f32
      xla_gpu.yield %add : f32
  }
  return %ret : tensor<2x4xf32>
}
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// CHECK-LABEL: func.func @atomic_rmw(
// CHECK-SAME:      %[[TENSOR:.*]]: tensor<8xf32>, %[[I:.*]]: index,
// CHECK-SAME:      %[[J:.*]]: index) -> tensor<8xf32> {
// CHECK:         %[[INDEX:.*]] = xla_gpu.apply_indexing #[[$MAP]]
// CHECK-SAME:      (%[[I]] in [0, 1], %[[J]] in [0, 3])
// CHECK:         xla_gpu.atomic_rmw %[[TENSOR]][%[[INDEX]]] : tensor<8xf32>

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
    %update1 = tensor.insert %c0_f32 into %t1_[%i, %c1, %c1] : tensor<64x8x4xf32>
    scf.yield %update0, %update1 : tensor<32x1024xf32>, tensor<64x8x4xf32>
  } {some_attr}
    return %for#0, %for#1, %c0_f32 : tensor<32x1024xf32>, tensor<64x8x4xf32>, f32
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (d0 + 1024)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0) -> (d0 * 32 + 5)>
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
// CHECK:        %[[IND0:.*]] = xla_gpu.apply_indexing #[[$MAP0]](%[[I]] in [0, 1023])
// CHECK:        %[[UPD0:.*]] = tensor.insert %[[F32]] into %[[T0_]][%[[IND0]]]
// CHECK:        %[[IND1:.*]] = xla_gpu.apply_indexing #[[$MAP1]](%[[I]] in [0, 63])
// CHECK:        %[[UPD1:.*]] = tensor.insert %[[F32]] into %[[T1_]][%[[IND1]]]
// CHECK:        scf.yield %[[UPD0]], %[[UPD1]] : tensor<32768xf32>, tensor<2048xf32>

// -----

#map = affine_map<(d0, d1) -> ((d1 * 128 + d0) floordiv 36)>
#map1 = affine_map<(d0, d1) -> (((d1 * 128 + d0) floordiv 9) mod 4)>
#map2 = affine_map<(d0, d1) -> ((d1 * 128 + d0) mod 9)>
func.func @if_op(%arg0: tensor<4000x4x9xf32>, %arg1: tensor<1400x1xi32>,
    %arg2: tensor<1400x1x4x9xf32>, %arg3: tensor<4000x4x9xf32>)
     -> tensor<4000x4x9xf32> {
	%c0 = arith.constant 0 : index
	%c3999 = arith.constant 3999 : index
	%th_x = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
	%bl_x = gpu.block_id  x {xla.range = [0 : index, 393749 : index]}
	%0 = xla_gpu.apply_indexing #map(%th_x in [0, 127], %bl_x in [0, 393749])
	%extracted = tensor.extract %arg1[%0, %c0] : tensor<1400x1xi32>
	%1 = arith.index_cast %extracted : i32 to index
	%2 = arith.cmpi ule, %1, %c3999 : index
	%3 = scf.if %2 -> (tensor<4000x4x9xf32>) {
		%4 = xla_gpu.apply_indexing #map1(%th_x in [0, 127], %bl_x in [0, 393749])
		%5 = xla_gpu.apply_indexing #map2(%th_x in [0, 127], %bl_x in [0, 393749])
		%elem = tensor.extract %arg2[%0, %c0, %4, %5] : tensor<1400x1x4x9xf32>
		%atomic_rmw = xla_gpu.atomic_rmw %arg3[%1, %4, %5] : tensor<4000x4x9xf32> {
		^bb0(%arg4: f32):
			%6 = arith.addf %arg4, %elem : f32
			xla_gpu.yield %6 : f32
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

func.func @dangling_cast(%arg0: tensor<6xf32>, %arg1: index) -> i32 {
  %v = tensor.extract %arg0[%arg1] : tensor<6xf32>
	%cast = builtin.unrealized_conversion_cast %v : f32 to i32
  func.return %cast : i32
}
// CHECK: FlattenTensorsPass failed to converge
