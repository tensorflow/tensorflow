// RUN: mlir_fusions_opt  %s -split-input-file -verify-diagnostics

#map0 = affine_map<(d0, d1)[s0] -> (d0, d1 + s0)>
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{operand, lower_bounds, upper_bounds count and affine map dimension and symbol count must match}}
  %0:2 = xla_gpu.apply_indexing #map0 (%d0 in [0, 3))
  func.return %0#0, %0#1 : index, index
}
