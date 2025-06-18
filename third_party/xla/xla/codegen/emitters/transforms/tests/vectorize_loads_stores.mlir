// RUN: emitters_opt -allow-unregistered-dialect %s -split-input-file \
// RUN: -xla-vectorize-loads-stores="gpu_device_info='cuda_compute_capability {major: 6}'" -cse -canonicalize \
// RUN: | FileCheck %s

// RUN: emitters_opt -allow-unregistered-dialect %s -split-input-file \
// RUN: -xla-vectorize-loads-stores="target_type=cpu" -cse -canonicalize \
// RUN: | FileCheck %s

// RUN: emitters_opt %s --allow-unregistered-dialect -split-input-file \
// RUN: -xla-vectorize-loads-stores="gpu_device_info='cuda_compute_capability {major: 9}'" -cse -canonicalize \
// RUN: | FileCheck %s --check-prefix=CHECK-HOPPER

#map = #xla.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @simple_read(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 2), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = xla.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addf

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (d0 * 4 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 3]">
func.func @simple_read(%arg0: tensor<256xf16>) -> (f16) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f16
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f16 {
    %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter1 = %iter) -> f16 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<256xf16>
      %added = arith.addf %iter1, %extracted : f16
      scf.yield %added : f16
    }
    scf.yield %inner : f16
  }
  return %outer : f16
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 4), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = xla.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addf

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (d0 * 8 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 7]">
func.func @simple_read(%arg0: tensor<512xi8>) -> (i8) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0 : i8
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> i8 {
    %inner = scf.for %j = %c0 to %c8 step %c1 iter_args(%iter1 = %iter) -> i8 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<512xi8>
      %added = arith.addi %iter1, %extracted : i8
      scf.yield %added : i8
    }
    scf.yield %inner : i8
  }
  return %outer : i8
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map<"(d0) -> (d0 * 8), domain: d0 in [0, 63]">
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = xla.apply_indexing #[[$MAP]](%[[I]])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addi

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (d0 * 2 + s0 + 1),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_indexing_map(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_indexing_map
// CHECK-NOT: vector.transfer_read

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (d0 * 3 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_indexing_map_2(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_indexing_map_2
// CHECK-NOT: vector.transfer_read

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (3 * d0 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @misaligned_shape(%arg0: tensor<192xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<192xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @misaligned_shape
// CHECK-NOT: vector.transfer_read

// -----

#map = #xla.indexing_map<"(d0)[s0] -> (d0 + s0 * 2),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @wrong_stride(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @wrong_stride
// CHECK-NOT: vector.transfer_read

// -----

// We could vectorize this as a float vector load of double the size, but we
// don't currently.
#map = #xla.indexing_map<"(d0)[s0] -> (2 * d0 + s0),"
  "domain: d0 in [0, 127], s0 in [0, 1]">
func.func @simple_read_complex(%arg0: tensor<128xcomplex<f32>>, %i: index) -> (complex<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
  %loop = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter = %cst) -> complex<f32> {
    %idx = xla.apply_indexing #map(%i)[%j]
    %extracted = tensor.extract %arg0[%idx] : tensor<128xcomplex<f32>>
    %added = complex.add %iter, %extracted : complex<f32>
    scf.yield %added : complex<f32>
  }
  return %loop : complex<f32>
}

// CHECK-LABEL: @simple_read_complex
// CHECK-NOT: vector.transfer_read

// -----

// This is vectorizable, but not currently supported.
func.func @layout(%arg0: tensor<2x64xf32, dense<[0, 1]> : tensor<2xi64>>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %extracted = tensor.extract %arg0[%j, %i]
        : tensor<2x64xf32, dense<[0, 1]> : tensor<2xi64>>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @layout
// CHECK-NOT: vector.transfer_read

// -----

func.func @simple_write(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %iter[%j] : tensor<64xf32>
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @simple_write
// CHECK-SAME:     (%[[ARG0:.*]]: tensor{{.*}})
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[V:.*]] = scf.for
// CHECK-SAME:      (vector<4xf32>)
// CHECK-NEXT:    vector.insert
// CHECK-NEXT:    scf.yield
// CHECK:       %[[WRITTEN:.*]] = vector.transfer_write %[[V]], %[[ARG0]][%[[C0]]]
// CHECK-NEXT:  return %[[WRITTEN]]

// -----

func.func @write_with_use(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %iter[%j] : tensor<64xf32>
    "dummy.op1"(%inserted) : (tensor<64xf32>) -> ()
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @write_with_use
// CHECK-NOT:   transfer_write

// -----

  func.func @write_not_to_iter_arg(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
      %inserted = tensor.insert %cst into %arg0[%j] : tensor<64xf32>
      scf.yield %inserted : tensor<64xf32>
    }
    return %loop : tensor<64xf32>
  }

// CHECK-LABEL: @write_not_to_iter_arg
// CHECK-NOT:   transfer_write

// -----

func.func @write_not_yielded(%arg0: tensor<64xf32>) -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<64xf32> {
    %inserted = tensor.insert %cst into %arg0[%j] : tensor<64xf32>
    scf.yield %arg0 : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @write_not_yielded
// CHECK-NOT:   transfer_write

// -----

#map = #xla.indexing_map<"(d0, d1)[s0] -> (d1 * 2 + d0 + s0 * 512),"
  "domain: d0 in [0, 7], d1 in [0, 255], s0 in [0, 7]">
#map1 = #xla.indexing_map<
  "(d0, d1, d2)[s0] -> (d0 * 32 + d2 * 2 + d1 + s0 * 512),"
  "domain: d0 in [0, 7], d1 in [0, 1], d2 in [0, 255], s0 in [0, 7]">
func.func @multiple(%arg0: tensor<131072xf32>, %arg1: tensor<4096xbf16>,
      %arg2: tensor<32xf32>, %arg3: tensor<131072xf32>,
      %arg4: index) -> (tensor<131072xf32>, f32) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %extracted1 = tensor.extract %arg2[%arg4] : tensor<32xf32>
  %0:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%iter0 = %arg3, %iter1 = %cst) -> (tensor<131072xf32>, f32) {
    %1:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter2 = %iter0, %iter3 = %iter1) -> (tensor<131072xf32>, f32) {
      %2 = xla.apply_indexing #map(%j, %arg4)[%i]
      %idx = xla.apply_indexing #map1(%i, %j, %arg4)[%i]
      %extracted2 = tensor.extract %arg0[%idx] : tensor<131072xf32>
      %extracted3 = tensor.extract %arg1[%2] : tensor<4096xbf16>
      %3 = arith.extf %extracted3 : bf16 to f32
      %4 = arith.addf %extracted2, %3 : f32
      %5 = arith.addf %extracted1, %4 : f32
      %6 = arith.addf %iter3, %5 : f32
      %inserted = tensor.insert %5 into %iter2[%idx] : tensor<131072xf32>
      scf.yield %inserted, %6 : tensor<131072xf32>, f32
    }
    scf.yield %1#0, %1#1 : tensor<131072xf32>, f32
  }
  return %0#0, %0#1 : tensor<131072xf32>, f32
}
// CHECK-DAG: #[[$MAP:.*]] = #xla.indexing_map<"(d0, d1) -> (d0 * 2 + d1 * 512), domain: d0 in [0, 255], d1 in [0, 7]">
// CHECK-DAG: #[[$MAP1:.*]] = #xla.indexing_map<"(d0, d1, d2) -> (d0 * 32 + d1 * 2 + d2 * 512), domain: d0 in [0, 7], d1 in [0, 255], d2 in [0, 7]">
// CHECK-LABEL: @multiple
// CHECK-SAME: (%[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: index)
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      scf.for %[[I:.*]] = %[[C0]]
// CHECK-DAG:  %[[BASE:.*]] = xla.apply_indexing #[[$MAP]](%[[ARG4]], %[[I]])
// CHECK-DAG:  %[[IDX:.*]] = xla.apply_indexing #[[$MAP1]](%[[I]], %[[ARG4]], %[[I]])
// CHECK:      %[[READ1:.*]] = vector.transfer_read %[[ARG1]][%[[BASE]]]
// CHECK:      %[[READ2:.*]] = vector.transfer_read %[[ARG0]][%[[IDX]]]
// CHECK:      %[[INNER:.*]]:2 = scf.for %[[J:.*]] = %[[C0]] {{.*}} iter_args(%[[F:.*]] = {{.*}}, %[[V:.*]] = {{.*}}) -> (f32, vector<2xf32>)
// CHECK-DAG:  vector.extract %[[READ1]][%[[J]]]
// CHECK-DAG:  vector.extract %[[READ2]][%[[J]]]
// CHECK:      extf
// CHECK-NEXT: addf
// CHECK-NEXT: %[[TO_INSERT:.*]] = arith.addf
// CHECK-NEXT: %[[TO_YIELD:.*]] = arith.addf
// CHECK-NEXT: %[[V_NEXT:.*]] = vector.insert %[[TO_INSERT]], %[[V]] [%[[J]]]
// CHECK-NEXT: scf.yield %[[TO_YIELD]], %[[V_NEXT]]
// CHECK:      %[[WRITTEN:.*]] = vector.transfer_write %[[INNER]]#1, %{{.*}}[%[[IDX]]]
// CHECK:      scf.yield %[[WRITTEN]], %[[INNER]]#0

// -----

#map = #xla.indexing_map<"(d0)[s0] -> ((d0 * 4) mod 64 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @remainder_with_modulo(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map<"(d0) -> ((d0 mod 16) * 4),
// CHECK-LABEL: @remainder_with_modulo
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[I:.*]] = %[[C0]]
// CHECK: %[[BASE:.*]] = xla.apply_indexing #[[$MAP]](%[[I]]
// CHECK: vector.transfer_read {{.*}}[%[[BASE]]]

// -----

#map = #xla.indexing_map<"(d0)[s0] -> ((d0 * 4) mod 65 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
func.func @remainder_with_modulo_misaligned(%arg0: tensor<128xf32>) -> (f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c63 = arith.constant 63 : index
  %cst = arith.constant 0.0 : f32
  %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
    %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
      %idx = xla.apply_indexing #map(%i)[%j]
      %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
      %added = arith.addf %iter1, %extracted : f32
      scf.yield %added : f32
    }
    scf.yield %inner : f32
  }
  return %outer : f32
}
// CHECK-LABEL: @remainder_with_modulo_misaligned
// CHECK-NOT: vector.transfer_read

// -----

#map0 = #xla.indexing_map<"(d0) -> (d0 + 5),"
  "domain: d0 in [0, 63]">
#map1 = #xla.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
module {
  func.func @apply_indexing_sequence(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %offset = xla.apply_indexing #map0(%i)
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla.apply_indexing #map1(%offset)[%j]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK: #[[$MAP0:.*]] = #xla.indexing_map<"(d0) -> (d0 * 2 + 10),
// CHECK-SAME:                                  domain: d0 in [0, 63]">
// CHECK-LABEL: @apply_indexing_sequence
// CHECK: %[[BASE:.*]] = xla.apply_indexing #[[$MAP0]]
// CHECK: vector.transfer_read {{.*}}[%[[BASE]]]

// -----


#map0 = #xla.indexing_map<"(d0) -> (d0 + 5),"
  "domain: d0 in [0, 63]">
#map1 = #xla.indexing_map<"(d0)[s0] -> (d0 * 2 + s0),"
  "domain: d0 in [0, 63], s0 in [0, 1]">
module {
  func.func @apply_indexing_sequence_same_block(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        // Usually, this will be hoisted by LICM or folded, so we do not detect
        // this pattern.
        %offset = xla.apply_indexing #map0(%i)
        %idx = xla.apply_indexing #map1(%offset)[%j]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @apply_indexing_sequence_same_block
// CHECK-NOT: vector.transfer_read

// -----

func.func @simple_atomic_rmw(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c42 = arith.constant 42.0 : f32
  %loop = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter = %arg0)
  -> tensor<2xf32> {
    %atomic = xla.atomic_rmw %iter[%j] : tensor<2xf32> {
      ^bb0(%current: f32):
        %add = arith.addf %current, %c42 : f32
        xla.yield %add : f32
      }
    scf.yield %atomic : tensor<2xf32>
  }
  return %loop : tensor<2xf32>
}

// CHECK-HOPPER-LABEL:       @simple_atomic_rmw
// CHECK-HOPPER-SAME:          (%[[ARG0:.*]]: tensor{{.*}})
// CHECK-HOPPER-DAG:           %[[C0:.*]] = arith.constant 0 : index
// CHECK-HOPPER:               %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<2xf32>
// CHECK-HOPPER:               %[[LOOP:.*]] = scf.for
// CHECK-HOPPER-SAME:          iter_args(%[[ITER:.*]] = %[[INIT]])
// CHECK-HOPPER-NEXT:            vector.insert
// CHECK-HOPPER-NEXT:            scf.yield
// CHECK-HOPPER:               xla.atomic_rmw %[[ARG0]]
// CHECK-HOPPER-NEXT:            ^bb0(%[[CURRENT:.*]]: vector<2xf32>):
// CHECK-HOPPER-NEXT:              arith.addf %[[CURRENT]], %[[LOOP]]

// -----

func.func @fold_insert_extract(%in: tensor<64xf32>, %out: tensor<64xf32>)
    -> tensor<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 2 : index
  %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%out_ = %out) -> tensor<64xf32> {
    %extracted = tensor.extract %in[%j] : tensor<64xf32>
    %inserted = tensor.insert %extracted into %out_[%j] : tensor<64xf32>
    scf.yield %inserted : tensor<64xf32>
  }
  return %loop : tensor<64xf32>
}
// CHECK-LABEL: @fold_insert_extract
// CHECK-NOT:   scf.for
// CHECK:         vector.transfer_read
// CHECK-NEXT:    vector.transfer_write


// -----

func.func @fold_insert_extract_two_results(
    %arg0: tensor<8xf64>, %arg1: tensor<8xf64>,
    %arg2: tensor<8xf64>, %arg3: tensor<8xf64>, %arg4: tensor<8xf64>)
      -> (tensor<8xf64>, tensor<8xf64>) {
  %cst = arith.constant 0.00e+00 : f64
  %cst_0 = arith.constant dense<0.00e+00> : vector<4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %18 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]}
    : tensor<8xf64>, vector<4xf64>
  %16 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]}
    : tensor<8xf64>, vector<4xf64>
  %20 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]}
    : tensor<8xf64>, vector<4xf64>
  %21:4 = scf.for %i = %c0 to %c4 step %c1
      iter_args(%arg6 = %arg3, %arg7 = %arg4, %arg8 = %cst_0, %arg9 = %cst_0)
      -> (tensor<8xf64>, tensor<8xf64>, vector<4xf64>, vector<4xf64>) {
    %24 = vector.extract %20[%i] : f64 from vector<4xf64>
    %25 = vector.extract %18[%i] : f64 from vector<4xf64>
    %26 = arith.addf %24, %25 : f64
    %27 = vector.extract %16[%i] : f64 from vector<4xf64>
    %28 = vector.insert %27, %arg8 [%i] : f64 into vector<4xf64>
    %29 = vector.insert %26, %arg9 [%i] : f64 into vector<4xf64>
    scf.yield %arg6, %arg7, %28, %29
      : tensor<8xf64>, tensor<8xf64>, vector<4xf64>, vector<4xf64>
  }
  %22 = vector.transfer_write %21#3, %arg3[%c0] {in_bounds = [true]}
    : vector<4xf64>, tensor<8xf64>
  %23 = vector.transfer_write %21#2, %arg4[%c0] {in_bounds = [true]}
    : vector<4xf64>, tensor<8xf64>
  return %22, %23 : tensor<8xf64>, tensor<8xf64>
}
// CHECK-LABEL:   func.func @fold_insert_extract_two_results(
// CHECK-SAME:      %[[VAL_0:[a-zA-Z0-9_]*]]: tensor<8xf64>,
// CHECK-SAME:      %[[VAL_1:[a-zA-Z0-9_]*]]: tensor<8xf64>,
// CHECK-SAME:      %[[VAL_2:[a-zA-Z0-9_]*]]: tensor<8xf64>,
// CHECK-SAME:      %[[VAL_3:[a-zA-Z0-9_]*]]: tensor<8xf64>,
// CHECK-SAME:      %[[VAL_4:[a-zA-Z0-9_]*]]: tensor<8xf64>) -> (tensor<8xf64>, tensor<8xf64>) {

// CHECK-DAG:   %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:   %[[C0_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<4xf64>
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[VAL_0_VEC:.*]] = vector.transfer_read %[[VAL_0]]
// CHECK-DAG:   %[[VAL_1_VEC:.*]] = vector.transfer_read %[[VAL_1]]
// CHECK-DAG:   %[[VAL_2_VEC:.*]] = vector.transfer_read %[[VAL_2]]

// CHECK:   %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:   iter_args(%[[INIT:.*]] = %[[C0_VEC]]) -> (vector<4xf64>) {

// CHECK:     %[[VEC_0_ELEM:.*]] = vector.extract %[[VAL_0_VEC]][%[[I]]]
// CHECK:     %[[VEC_1_ELEM:.*]] = vector.extract %[[VAL_1_VEC]][%[[I]]]
// CHECK:     %[[ADD:.*]] = arith.addf %[[VEC_0_ELEM]], %[[VEC_1_ELEM]] : f64
// CHECK:     %[[INSERT:.*]] = vector.insert %[[ADD]], %[[INIT]] [%[[I]]]
// CHECK:     scf.yield %[[INSERT]] : vector<4xf64>
// CHECK:   }
// CHECK:   %[[RES0:.*]] = vector.transfer_write %[[FOR]], %[[VAL_3]][%[[C0]]]
// CHECK:   %[[RES1:.*]] = vector.transfer_write %[[VAL_2_VEC]], %[[VAL_4]][%[[C0]]]
// CHECK:   return %[[RES0]], %[[RES1]] : tensor<8xf64>, tensor<8xf64>
// CHECK: }

// -----

func.func @avoid_folding_small_tensors(%arg0: tensor<2xi4>, %arg1: tensor<2xi4>)
    -> tensor<2xi4> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %0 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %arg1)
      -> (tensor<2xi4>) {
    %extracted = tensor.extract %arg0[%arg2] : tensor<2xi4>
    %inserted = tensor.insert %extracted into %arg3[%arg2] : tensor<2xi4>
    scf.yield %inserted : tensor<2xi4>
  }
  return %0 : tensor<2xi4>
}
// CHECK-LABEL: func.func @avoid_folding_small_tensors
// CHECK:         scf.for
