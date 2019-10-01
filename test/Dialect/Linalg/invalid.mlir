// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @buffer_alloc_single_index() {
  // expected-error @+1 {{expected one index operand}}
  %0 = linalg.buffer_alloc : !linalg.buffer<?xf32>
}

// -----

func @buffer_alloc_unexpected_index(%s : index) {
  // expected-error @+1 {{expected zero operand}}
  %0 = linalg.buffer_alloc %s : !linalg.buffer<32xf32>
}

// -----

func @buffer_alloc_nonegative_size() {
  // expected-error @+1 {{expected nonnegative static buffer size}}
  %0 = linalg.buffer_alloc : !linalg.buffer<0xf32>
}

// -----

func @buffer_alloc_nonegative_alignment(%arg0: index) {
  // expected-error @+1 {{expected positive alignment}}
  %0 = linalg.buffer_alloc %arg0 {alignment = -123}: !linalg.buffer<?xf32>
}

// -----

func @buffer_alloc_powerof2_alignment(%arg0: index) {
  // expected-error @+1 {{expected power of 2 alignment}}
  %0 = linalg.buffer_alloc %arg0 {alignment = 123}: !linalg.buffer<?xf32>
}

// -----

func @buffer_valid_element_type() {
  // expected-error @+1 {{expected valid buffer element type}}
  %0 = linalg.buffer_alloc : !linalg.buffer<4xindex>
}

// -----

func @load_number_of_indices(%v : memref<f32>) {
  // expected-error @+2 {{incorrect number of indices for load}}
  %c0 = constant 0 : index
  load %v[%c0] : memref<f32>
}

// -----

func @slice_number_of_indexings(%arg0: memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+2 {{expected 2 indexings, got 1}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0] : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>, index, memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @slice_rank_vs_range_indices(%arg0: memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+2 {{op expected rank of the view(1) to be the number of ranges(0)}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0, %c0] : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>, index, index, memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @store_number_of_indices(%v : memref<f32>) {
  // expected-error @+3 {{store index operand count not equal to memref rank}}
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  store %f0, %v[%c0] : memref<f32>
}

// -----

func @subview_number_of_indices(%v : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+2 {{expected a strided memref followed by 6 indices specifying a range for each dimension}}
  %c0 = constant 0 : index
  linalg.subview %v[%c0, %c0] : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @transpose_not_permutation(%v : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+1 {{expected a permutation map}}
  linalg.transpose %v (i, j) -> (i, i) : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @transpose_bad_rank(%v : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>) {
  // expected-error @+1 {{expected a permutation map of same rank as the view}}
  linalg.transpose %v (i) -> (i) : memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @view_type(%buf: !linalg.buffer<?xf32>, %min: index, %max: index, %step: index) {
  // expected-error @+2 {{expected memref type}}
  %r = linalg.range %min:%max:%step : !linalg.range
  %0 = linalg.view %buf[%r]: !linalg.buffer<?xf32> -> index
}

// -----

func @view_num_ranges(%buf: !linalg.buffer<?xf32>, %min: index, %max: index, %step: index) {
  // expected-error @+2 {{expected 2 ranges}}
  %r = linalg.range %min:%max:%step : !linalg.range
  %0 = linalg.view %buf[%r]: !linalg.buffer<?xf32> -> memref<?x?xf32, (i, j)[off, M]->(off + M * i + j)>
}

// -----

func @yield_parent(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected 'linalg.generic' parent op}}
  linalg.yield %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @generic_at_least_2_operands(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected 2 or more operands}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @generic_exactly_2_views(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected exactly 2 view operands}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0, %arg0: memref<f32>, memref<f32>, memref<f32>
}

// -----

func @generic_undefined_fun(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun attribute to refer to a defined symbol}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0: memref<f32>, memref<f32>
}

// -----

func @foo() { return }

func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun arguments to match number of views}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) { return }

func @generic_mismatched_num_returns(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected fun results to match number of output views}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_symbol_in_map(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have no symbols}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ ()[N] -> (0) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_wrong_dim_in_map(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

func @generic_zero_d_view(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to be 0 to match 0-D view: 'memref<f32>'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (1) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<f32>
}

// -----

func @foo(%0: f32) -> f32 { return %0: f32 }

func @generic_one_d_view(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected indexing_map #0 results to match view rank: 'memref<?xf32, (d0)[s0] -> (d0 + s0)>'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0, 0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: i32) -> f32 {
  %1 = constant 0.0: f32
  return %1: f32
}

func @generic_fun_arg_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected fun argument 0 to match view element type: 'f32'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: f32) -> i4 {
  %1 = constant 1: i4
  return %1: i4
}

func @generic_fun_result_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected fun result 0 to match output view element type: 'f32'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: memref<?xf32, (i)[off]->(off + i)>
}

// -----

func @foo(%0: f32, %1: f32) -> f32 { return %1: f32 }

func @generic_singular_maps(%arg0: memref<?xf32, (i)[off]->(off + i)>, %arg1: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+1 {{op expected the concatenation of maps in indexing_map to be invertible}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [
      (i, j) -> (i + j) ,
      (i, j) -> (i + j)
    ],
    n_views = [1, 1],
    n_loop_types = [2, 0, 0]
  } %arg0, %arg1: memref<?xf32, (i)[off]->(off + i)>, memref<?xf32, (i)[off]->(off + i)>
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Region tests /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// -----

func @generic_empty_region(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected region with 1 block}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0 {
    ^bb1:
    ^bb2:
  }: memref<f32>, memref<f32>
}

// -----

func @generic_mismatched_num_arguments(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected number of block arguments to match number of views}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0 {
    ^bb:
  }: memref<f32>
}

// -----

func @generic_block_arg_type(%arg0: memref<f32>) {
  // expected-error @+1 {{op expected block argument 0 of the same type as elemental type of output view: 'memref<f32>'}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0 {
    ^bb(%i: i1):
  }: memref<f32>
}

// -----

func @generic_fun_result_0_element_type(%arg0: memref<?xf32, (i)[off]->(off + i)>) {
  // expected-error @+8 {{type of return operand 0 ('i1') doesn't match view element type ('f32')}}
  linalg.generic {
    indexing_maps =  [ (i) -> (i) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0 {
    ^bb(%i: f32):
      %0 = constant 0: i1
      linalg.yield %0: i1
  }: memref<?xf32, (i)[off]->(off + i)>
}
