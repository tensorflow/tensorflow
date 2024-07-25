// RUN: mlir_fusions_opt %s -split-input-file | mlir_fusions_opt | FileCheck %s

// CHECK: #xla_gpu.indexing_map<(d0, d1, d2)[s0] -> (d0)
// CHECK-NEXT: domain:
// CHECK-NEXT: d0 in [1, 2]
// CHECK-NEXT: d1 in [5, 8]
// CHECK-NEXT: d2 in [10, 12]
// CHECK-NEXT: s0 in [0, 32]
// CHECK-NEXT: d0 mod 2 in [0, 1]
// CHECK-NEXT: d0 + s0 in [1, 10]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<(d0, d1, d2)[s0] -> (d0)
                            domain:
                            d0 in [1, 2]
                            d1 in [5, 8]
                            d2 in [10, 12]
                            s0 in [0, 32]
                            d0 mod 2 in [0, 1]
                            d0 + s0 in [1, 10]
                            >

func.func private @indexing_map_attr(tensor<32xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<(d0, d1)[s0, s1, s2] -> (d0 + s0, d1 + s1, d1 + s2)
// CHECK-NEXT: domain:
// CHECK-NEXT: d0 in [1, 2]
// CHECK-NEXT: d1 in [5, 8]
// CHECK-NEXT: s0 in [0, 10]
// CHECK-NEXT: s1 in [0, 5]
// CHECK-NEXT: s2 in [0, 32]
// CHECK-NEXT: d0 mod 2 in [0, 1]
// CHECK-NEXT: d0 + s0 in [1, 10]
// CHECK-NEXT: d1 + s1 + s2 in [1, 32]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<(d0, d1)[s0, s1, s2] -> (d0 + s0, d1 + s1, d1 + s2)
                            domain:
                            d0 in [1, 2]
                            d1 in [5, 8]
                            s0 in [0, 10]
                            s1 in [0, 5]
                            s2 in [0, 32]
                            d0 mod 2 in [0, 1]
                            d0 + s0 in [1, 10]
                            d1 + s1 + s2 in [1, 32]
                            >
func.func private @more_range_vars(tensor<32xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<(d0)[s0] -> (d0)
// CHECK-NEXT: domain:
// CHECK-NEXT: d0 in [0, 100]
// CHECK-NEXT: s0 in [-3, -1]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<(d0)[s0] -> (d0)
                            domain:
                            d0 in [0, 100]
                            s0 in [-3, -1]
                            >
func.func private @indexing_map_small(tensor<100xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<(d0, d1, d2)[s0] -> (d0)
// CHECK-NEXT: domain:
// CHECK-NEXT: d0 in [1, 2]
// CHECK-NEXT: d1 in [5, 8]
// CHECK-NEXT: d2 in [10, 12]
// CHECK-NEXT: s0 in [0, 32]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<(d0, d1, d2)[s0] -> (d0)
                            domain:
                            d0 in [1, 2]
                            d1 in [5, 8]
                            d2 in [10, 12]
                            s0 in [0, 32]
                            >
func.func private @no_constraints(tensor<32xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<()[s0] -> (s0)
// CHECK-NEXT: domain:
// CHECK-NEXT: s0 in [3, 5]
// CHECK-NEXT: s0 mod 2 in [0, 1]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<()[s0] -> (s0)
                            domain:
                            s0 in [3, 5]
                            s0 mod 2 in [0, 1]
                            >
func.func private @no_dimensions(tensor<100xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<(d0) -> (d0)
// CHECK-NEXT: domain:
// CHECK-NEXT: d0 in [3, 5]
// CHECK-NEXT: d0 mod 2 in [0, 1]
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<(d0) -> (d0)
                            domain:
                            d0 in [3, 5]
                            d0 mod 2 in [0, 1]
                            >
func.func private @no_symbols(tensor<100xf64, #map>)

// -----

// CHECK: #xla_gpu.indexing_map<() -> ()
// CHECK-NEXT: domain:
// CHECK-NEXT: >
#map = #xla_gpu.indexing_map<() -> ()
                            domain:
                            >
func.func private @empty(tensor<100xf64, #map>)