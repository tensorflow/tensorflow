// RUN: mlir_fusions_opt  %s -split-input-file -verify-diagnostics

#map0 = #xla_gpu.indexing_map<
 (d0, d1)[s0] -> (d0, d1 + s0),
 domain:
 d0 in [1, 2],
 d1 in [5, 8],
 s0 in [0, 32]
>
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{operand count must match the number of dimensions and symbols in the affine map}}
  %0:2 = xla_gpu.apply_indexing #map0 (%d0)
  func.return %0#0, %0#1 : index, index
}

// -----

#map0 = #xla_gpu.indexing_map<
 (d0, d1)[s0] -> (d0, d1 + s0),
 domain:
 d0 in [1, 2],
 d1 in [5, 8],
 s0 in [0, 32],
 d0 mod 2 in [0, 1],
 d0 + s0 in [1, 10]
>
func.func @cannot_have_constraints(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{apply indexing op cannot have any constraints}}
  %0:2 = xla_gpu.apply_indexing #map0 (%d0, %d1)[%s0]
  func.return %0#0, %0#1 : index, index
}
