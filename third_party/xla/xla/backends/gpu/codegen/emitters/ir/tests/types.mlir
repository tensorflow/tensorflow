// RUN: emitters_opt %s -split-input-file | emitters_opt -split-input-file | FileCheck %s

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: (d0, d1, d2)[s0] -> (d0),
// CHECK-SAME: domain:
// CHECK-SAME: d0 in [1, 2],
// CHECK-SAME: d1 in [5, 8],
// CHECK-SAME: d2 in [10, 12],
// CHECK-SAME: s0 in [0, 32],
// CHECK-SAME: d0 + s0 in [1, 10],
// CHECK-SAME: d0 mod 2 in [0, 1]
// CHECK-SAME: >
#map = #xla.indexing_map<"(d0, d1, d2)[s0] -> (d0),"
                             "domain:"
                             "d0 in [1, 2],"
                             "d1 in [5, 8],"
                             "d2 in [10, 12],"
                             "s0 in [0, 32],"
                             "d0 mod 2 in [0, 1],"
                             "d0 + s0 in [1, 10]"
                            >

func.func private @indexing_map_attr(!xla_gpu.indexed_vector<64x64x32xf64, #map>)
// CHECK-LABEL: @indexing_map_attr
// CHECK: !xla_gpu.indexed_vector<64x64x32xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: (d0, d1)[s0, s1, s2] -> (d0 + s0, d1 + s1, d1 + s2)
// CHECK-SAME: domain:
// CHECK-SAME: d0 in [1, 2]
// CHECK-SAME: d1 in [5, 8]
// CHECK-SAME: s0 in [0, 10]
// CHECK-SAME: s1 in [0, 5]
// CHECK-SAME: s2 in [0, 32]
// CHECK-SAME: d0 + s0 in [1, 10]
// CHECK-SAME: d0 mod 2 in [0, 1]
// CHECK-SAME: d1 + s1 + s2 in [1, 32]
// CHECK-SAME: >
#map = #xla.indexing_map<
  "(d0, d1)[s0, s1, s2] -> (d0 + s0, d1 + s1, d1 + s2),"
  "domain:"
  "d0 in [1, 2],"
  "d1 in [5, 8],"
  "s0 in [0, 10],"
  "s1 in [0, 5],"
  "s2 in [0, 32],"
  "d0 mod 2 in [0, 1],"
  "d0 + s0 in [1, 10],"
  "d1 + s1 + s2 in [1, 32]"
  >
func.func private @more_range_vars(!xla_gpu.indexed_vector<100x32xf64, #map>)
// CHECK-LABEL: @more_range_vars
// CHECK: !xla_gpu.indexed_vector<100x32xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: (d0)[s0] -> (d0)
// CHECK-SAME: domain:
// CHECK-SAME: d0 in [0, 100]
// CHECK-SAME: s0 in [-3, -1]
// CHECK-SAME: >
#map = #xla.indexing_map<"(d0)[s0] -> (d0),"
                             "domain:"
                             "d0 in [0, 100],"
                             "s0 in [-3, -1]"
                            >
func.func private @indexing_map_small(!xla_gpu.indexed_vector<100xf64, #map>)
// CHECK-LABEL: @indexing_map_small
// CHECK: !xla_gpu.indexed_vector<100xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: (d0, d1, d2)[s0] -> (d0)
// CHECK-SAME: domain:
// CHECK-SAME: d0 in [1, 2]
// CHECK-SAME: d1 in [5, 8]
// CHECK-SAME: d2 in [10, 12]
// CHECK-SAME: s0 in [0, 32]
// CHECK-SAME: >
#map = #xla.indexing_map<"(d0, d1, d2)[s0] -> (d0),"
                             "domain:"
                             "d0 in [1, 2],"
                             "d1 in [5, 8],"
                             "d2 in [10, 12],"
                             "s0 in [0, 32]"
                            >
func.func private @no_constraints(!xla_gpu.indexed_vector<32xf64, #map>)
// CHECK-LABEL: @no_constraints
// CHECK: !xla_gpu.indexed_vector<32xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: ()[s0] -> (s0)
// CHECK-SAME: domain:
// CHECK-SAME: s0 in [3, 5]
// CHECK-SAME: s0 mod 2 in [0, 1]
// CHECK-SAME: >
#map = #xla.indexing_map<"()[s0] -> (s0),"
                            "domain:"
                            "s0 in [3, 5],"
                            "s0 mod 2 in [0, 1]"
                            >
func.func private @no_dimensions(!xla_gpu.indexed_vector<100xf64, #map>)
// CHECK-LABEL: @no_dimensions
// CHECK: !xla_gpu.indexed_vector<100xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: (d0) -> (d0)
// CHECK-SAME: domain:
// CHECK-SAME: d0 in [3, 5]
// CHECK-SAME: d0 mod 2 in [0, 1]
// CHECK-SAME: >
#map = #xla.indexing_map<"(d0) -> (d0),"
                            "domain:"
                            "d0 in [3, 5],"
                            "d0 mod 2 in [0, 1],"
                            >
func.func private @no_symbols(!xla_gpu.indexed_vector<100xf64, #map>)
// CHECK-LABEL: @no_symbols
// CHECK: !xla_gpu.indexed_vector<100xf64, #[[$INDEX_MAP]]>

// -----

// CHECK: #[[$INDEX_MAP:.*]] = #xla.indexing_map<
// CHECK-SAME: () -> ()
// CHECK-SAME: >
#map = #xla.indexing_map<"() -> ()">
func.func private @empty(!xla_gpu.indexed_vector<100xf64, #map>)
// CHECK-LABEL: @empty
// CHECK: !xla_gpu.indexed_vector<100xf64, #[[$INDEX_MAP]]>

// -----

func.func private @tensor_layout(
  %in0: tensor<42xf32, #xla_gpu.layout<"shmem",
     "(d0) -> ()," "domain: d0 in [0, 42]">>)
// CHECK:      #layout = #xla_gpu.layout<"shmem", "(d0) -> (), domain:
// CHECK: tensor<42xf32, #layout>
