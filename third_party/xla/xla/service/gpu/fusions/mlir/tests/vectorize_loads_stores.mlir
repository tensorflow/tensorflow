// RUN: mlir_fusions_opt -allow-unregistered-dialect %s -split-input-file -xla-gpu-vectorize-loads-stores -canonicalize -cse | FileCheck %s

#map = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
module {
  func.func @simple_read(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: @simple_read
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] iter_args(%[[ITER:.*]] =
// CHECK:         %[[BASE:.*]] = xla_gpu.apply_indexing #map(%[[I]] in [0, 63])
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[BASE]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]
// CHECK-NEXT:      addf

// -----

module {
  func.func @simple_read_2d(%arg0: tensor<64x2xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %extracted = tensor.extract %arg0[%i, %j] : tensor<64x2xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @simple_read_2d
// CHECK-SAME:     (%[[ARG0:.*]]: tensor
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       scf.for %[[I:.*]] = %[[C0]]
// CHECK-NEXT:    %[[V:.*]] = vector.transfer_read %[[ARG0]][%[[I]], %[[C0]]]
// CHECK-NEXT:    scf.for %[[J:.*]] = %[[C0]]
// CHECK-NEXT:      vector.extract %[[V]][%[[J]]]

// -----

#map = affine_map<(d0)[s0] -> (d0 * 2 + s0 + 1)>
module {
  func.func @misaligned_indexing_map(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @misaligned_indexing_map
// CHECK-NOT: vector.transfer_read

// -----

#map = affine_map<(d0)[s0] -> (d0 * 3 + s0)>
module {
  func.func @misaligned_indexing_map_2(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @misaligned_indexing_map_2
// CHECK-NOT: vector.transfer_read

// -----

module {
  func.func @misaligned_shape(%arg0: tensor<64x3xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c64 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %extracted = tensor.extract %arg0[%i, %j] : tensor<64x3xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @misaligned_shape
// CHECK-NOT: vector.transfer_read

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0 * 2)>
module {
  func.func @wrong_stride(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @wrong_stride
// CHECK-NOT: vector.transfer_read

// -----

// We could vectorize this as a float vector load of double the size, but we
// don't currently.
module {
  func.func @simple_read_complex(%arg0: tensor<64x2xcomplex<f32>>, %i: index) -> (complex<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
    %loop = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter = %cst) -> complex<f32> {
      %extracted = tensor.extract %arg0[%i, %j] : tensor<64x2xcomplex<f32>>
      %added = complex.add %iter, %extracted : complex<f32>
      scf.yield %added : complex<f32>
    }
    return %loop : complex<f32>
  }
}

// CHECK-LABEL: @simple_read_complex
// CHECK-NOT: vector.transfer_read

// -----

// This is vectorizable, but not currently supported.
module {
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
}

// CHECK-LABEL: @layout
// CHECK-NOT: vector.transfer_read

// -----

module {
  func.func @simple_write(%arg0: tensor<16x4xf32>, %i: index) -> tensor<16x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<16x4xf32> {
      %inserted = tensor.insert %cst into %iter[%i, %j] : tensor<16x4xf32>
      scf.yield %inserted : tensor<16x4xf32>
    }
    return %loop : tensor<16x4xf32>
  }
}

// CHECK-LABEL: @simple_write
// CHECK-SAME:     (%[[ARG0:.*]]: tensor{{.*}}, %[[I:.*]]: index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[V:.*]] = scf.for
// CHECK-NEXT:    vector.insert
// CHECK-NEXT:    scf.yield
// CHECK:       %[[WRITTEN:.*]] = vector.transfer_write %[[V]], %[[ARG0]][%[[I]], %[[C0]]]
// CHECK-NEXT:  return %[[WRITTEN]]

// -----

module {
  func.func @write_with_use(%arg0: tensor<16x4xf32>, %i: index) -> tensor<16x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<16x4xf32> {
      %inserted = tensor.insert %cst into %iter[%i, %j] : tensor<16x4xf32>
      "dummy.op1"(%inserted) : (tensor<16x4xf32>) -> ()
      scf.yield %inserted : tensor<16x4xf32>
    }
    return %loop : tensor<16x4xf32>
  }
}

// CHECK-LABEL: @write_with_use
// CHECK-NOT:   transfer_write

// -----

module {
  func.func @write_not_to_iter_arg(%arg0: tensor<16x4xf32>, %i: index) -> tensor<16x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<16x4xf32> {
      %inserted = tensor.insert %cst into %arg0[%i, %j] : tensor<16x4xf32>
      scf.yield %inserted : tensor<16x4xf32>
    }
    return %loop : tensor<16x4xf32>
  }
}

// CHECK-LABEL: @write_not_to_iter_arg
// CHECK-NOT:   transfer_write

// -----

module {
  func.func @write_not_yielded(%arg0: tensor<16x4xf32>, %i: index) -> tensor<16x4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %cst = arith.constant 0.0 : f32
    %loop = scf.for %j = %c0 to %c4 step %c1 iter_args(%iter = %arg0) -> tensor<16x4xf32> {
      %inserted = tensor.insert %cst into %arg0[%i, %j] : tensor<16x4xf32>
      scf.yield %arg0 : tensor<16x4xf32>
    }
    return %loop : tensor<16x4xf32>
  }
}

// CHECK-LABEL: @write_not_yielded
// CHECK-NOT:   transfer_write

// -----

#map = affine_map<(d0, d1)[s0] -> (d1 * 2 + d0 + s0 * 512)>
module {
  func.func @multiple(%arg0: tensor<32x4096xf32>, %arg1: tensor<4096xbf16>,
        %arg2: tensor<32xf32>, %arg3: tensor<32x4096xf32>,
        %arg4: index) -> (tensor<32x4096xf32>, f32) {
    %cst = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %extracted1 = tensor.extract %arg2[%arg4] : tensor<32xf32>
    %0:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%iter0 = %arg3, %iter1 = %cst) -> (tensor<32x4096xf32>, f32) {
      %1:2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter2 = %iter0, %iter3 = %iter1) -> (tensor<32x4096xf32>, f32) {
        %2 = xla_gpu.apply_indexing #map(%j in [0, 1], %arg4 in [0, 255])[%i in [0, 7]]
        %extracted2 = tensor.extract %arg0[%i, %2] : tensor<32x4096xf32>
        %extracted3 = tensor.extract %arg1[%2] : tensor<4096xbf16>
        %3 = arith.extf %extracted3 : bf16 to f32
        %4 = arith.addf %extracted2, %3 : f32
        %5 = arith.addf %extracted1, %4 : f32
        %6 = arith.addf %iter3, %5 : f32
        %inserted = tensor.insert %5 into %iter2[%i, %2] : tensor<32x4096xf32>
        scf.yield %inserted, %6 : tensor<32x4096xf32>, f32
      }
      scf.yield %1#0, %1#1 : tensor<32x4096xf32>, f32
    }
    return %0#0, %0#1 : tensor<32x4096xf32>, f32
  }
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (d0 * 2 + s0 * 512)>
// CHECK-LABEL: @multiple
// CHECK-SAME: (%[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}, %[[ARG3:.*]]: tensor{{.*}}, %[[ARG4:.*]]: index)
// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      scf.for %[[I:.*]] = %[[C0]]
// CHECK:      %[[BASE:.*]] = xla_gpu.apply_indexing #[[$MAP]](%[[ARG4]] in [0, 255])[%[[I]] in [0, 7]]
// CHECK:      %[[READ1:.*]] = vector.transfer_read %[[ARG1]][%[[BASE]]]
// CHECK:      %[[READ2:.*]] = vector.transfer_read %[[ARG0]][%[[I]], %[[BASE]]]
// CHECK:      %[[INNER:.*]]:2 = scf.for %[[J:.*]] = %[[C0]] {{.*}} iter_args(%[[F:.*]] = {{.*}}, %[[V:.*]] = {{.*}}) -> (f32, vector<2xf32>)
// CHECK-DAG:  vector.extract %[[READ1]][%[[J]]]
// CHECK-DAG:  vector.extract %[[READ2]][%[[J]]]
// CHECK:      extf
// CHECK-NEXT: addf
// CHECK-NEXT: %[[TO_INSERT:.*]] = arith.addf
// CHECK-NEXT: %[[TO_YIELD:.*]] = arith.addf
// CHECK-NEXT: %[[V_NEXT:.*]] = vector.insert %[[TO_INSERT]], %[[V]] [%[[J]]]
// CHECK-NEXT: scf.yield %[[TO_YIELD]], %[[V_NEXT]]
// CHECK:      %[[WRITTEN:.*]] = vector.transfer_write %[[INNER]]#1, %{{.*}}[%[[I]], %[[BASE]]]
// CHECK:      scf.yield %[[WRITTEN]], %[[INNER]]#0

// -----

#map = affine_map<(d0)[s0] -> ((d0 * 4) mod 64 + s0)>
module {
  func.func @remainder_with_modulo(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0) -> ((d0 mod 16) * 4)>
// CHECK-LABEL: @remainder_with_modulo
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.for %[[I:.*]] = %[[C0]]
// CHECK: %[[BASE:.*]] = xla_gpu.apply_indexing #[[$MAP]](%[[I]]
// CHECK: vector.transfer_read {{.*}}[%[[BASE]]]

// -----

#map = affine_map<(d0)[s0] -> ((d0 * 4) mod 65 + s0)>
module {
  func.func @remainder_with_modulo_misaligned(%arg0: tensor<128xf32>) -> (f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c63 = arith.constant 63 : index
    %cst = arith.constant 0.0 : f32
    %outer = scf.for %i = %c0 to %c63 step %c1 iter_args(%iter = %cst) -> f32 {
      %inner = scf.for %j = %c0 to %c2 step %c1 iter_args(%iter1 = %iter) -> f32 {
        %idx = xla_gpu.apply_indexing #map(%i in [0, 63])[%j in [0, 1]]
        %extracted = tensor.extract %arg0[%idx] : tensor<128xf32>
        %added = arith.addf %iter1, %extracted : f32
        scf.yield %added : f32
      }
      scf.yield %inner : f32
    }
    return %outer : f32
  }
}

// CHECK-LABEL: @remainder_with_modulo_misaligned
// CHECK-NOT: vector.transfer_read
