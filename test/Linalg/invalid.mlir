// RUN: mlir-opt %s -split-input-file -verify-diagnostics

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Function Attribute tests /////////////////////////
////////////////////////////////////////////////////////////////////////////////

// -----

// CHECK-LABEL: slice_number_of_indexings
func @slice_number_of_indexings(%arg0: !linalg.view<?x?xf32>) {
  // expected-error @+2 {{expected number of indexings to match view rank}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0] : !linalg.view<?x?xf32>, index, !linalg.view<?x?xf32>
  return
}

// -----

// CHECK-LABEL: slice_rank_vs_range_indices
func @slice_rank_vs_range_indices(%arg0: !linalg.view<?x?xf32>) {
  // expected-error @+2 {{expected rank of the view(1) to be the number of its range indices(0)}}
  %c0 = constant 0: index
  %0 = linalg.slice %arg0[%c0, %c0] : !linalg.view<?x?xf32>, index, index, !linalg.view<?xf32>
  return
}

// -----

// CHECK-LABEL: generic_at_least_2_operands
func @generic_at_least_2_operands(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected 2 or more operands}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

// CHECK-LABEL: generic_exactly_2_views
func @generic_exactly_2_views(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected exactly 2 view operands}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0, %arg0: !linalg.view<f32>, !linalg.view<f32>, !linalg.view<f32>
  return
}

// -----

// CHECK-LABEL: generic_undefined_fun
func @generic_undefined_fun(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected fun attribute to refer to a defined symbol}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0: !linalg.view<f32>, !linalg.view<f32>
  return
}

// -----

func @foo() { return }

// CHECK-LABEL: generic_mismatched_num_arguments
func @generic_mismatched_num_arguments(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected fun arguments to match number of views}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

func @foo(%0: i32) { return }

// CHECK-LABEL: generic_mismatched_num_returns
func @generic_mismatched_num_returns(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected fun results to match number of output views}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

// CHECK-LABEL: generic_symbol_in_map
func @generic_symbol_in_map(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have no symbols}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ ()[N] -> (0) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

// CHECK-LABEL: generic_wrong_dim_in_map
func @generic_wrong_dim_in_map(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to have 1 dim(s) to match the number of loops}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

func @foo(%0: i32) -> i32 { return %0: i32 }

// CHECK-LABEL: generic_zero_d_view
func @generic_zero_d_view(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected indexing_map #0 to be 0 to match 0-D view: '!linalg.view<f32>'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (1) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<f32>
  return
}

// -----

func @foo(%0: f32) -> f32 { return %0: f32 }

// CHECK-LABEL: generic_one_d_view
func @generic_one_d_view(%arg0: !linalg.view<?xf32>) {
  // expected-error @+1 {{op expected indexing_map #0 results to match view rank: '!linalg.view<?xf32>'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0, 0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<?xf32>
  return
}

// -----

func @foo(%0: i32) -> f32 {
  %1 = constant 0.0: f32
  return %1: f32
}

// CHECK-LABEL: generic_fun_arg_0_element_type
func @generic_fun_arg_0_element_type(%arg0: !linalg.view<?xf32>) {
  // expected-error @+1 {{op expected fun argument 0 to match view element type: 'f32'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<?xf32>
  return
}

// -----

func @foo(%0: f32) -> i4 {
  %1 = constant 1: i4
  return %1: i4
}

// CHECK-LABEL: generic_fun_result_0_element_type
func @generic_fun_result_0_element_type(%arg0: !linalg.view<?xf32>) {
  // expected-error @+1 {{op expected fun result 0 to match output view element type: 'f32'}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0: !linalg.view<?xf32>
  return
}

// -----

func @foo(%0: f32, %1: f32) -> f32 { return %1: f32 }

// CHECK-LABEL: generic_singular_maps
func @generic_singular_maps(%arg0: !linalg.view<?xf32>, %arg1: !linalg.view<?xf32>) {
  // expected-error @+1 {{op expected the concatenation of maps in indexing_map to be invertible}}
  linalg.generic {
    fun = @foo,
    indexing_maps =  [
      (i, j) -> (i + j) ,
      (i, j) -> (i + j)
    ],
    n_views = [1, 1],
    n_loop_types = [2, 0, 0]
  } %arg0, %arg1: !linalg.view<?xf32>, !linalg.view<?xf32>
  return
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Region tests /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// -----

// CHECK-LABEL: generic_empty_region
func @generic_empty_region(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected region with 1 block}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [1, 1],
    n_loop_types = [0, 0, 0]
  } %arg0, %arg0 {
    ^bb1:
    ^bb2:
  }: !linalg.view<f32>, !linalg.view<f32>
  return
}

// -----

// CHECK-LABEL: generic_mismatched_num_arguments
func @generic_mismatched_num_arguments(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected number of block arguments to match number of views}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0 {
    ^bb:
  }: !linalg.view<f32>
  return
}

// -----

// CHECK-LABEL: generic_block_arg_type
func @generic_block_arg_type(%arg0: !linalg.view<f32>) {
  // expected-error @+1 {{op expected block argument 0 of the same type as elemental type of output view: '!linalg.view<f32>'}}
  linalg.generic {
    indexing_maps =  [ () -> (0) ],
    n_views = [0, 1],
    n_loop_types = [0, 0, 0]
  } %arg0 {
    ^bb(%i: i1):
  }: !linalg.view<f32>
  return
}

// -----

// CHECK-LABEL: generic_fun_result_0_element_type
func @generic_fun_result_0_element_type(%arg0: !linalg.view<?xf32>) {
  // expected-error @+8 {{type of return operand 0 ('i1') doesn't match view element type ('f32')}}
  linalg.generic {
    indexing_maps =  [ (i) -> (i) ],
    n_views = [0, 1],
    n_loop_types = [1, 0, 0]
  } %arg0 {
    ^bb(%i: f32):
      %0 = constant 0: i1
      linalg.yield %0: i1
  }: !linalg.view<?xf32>
  return
}

// -----

// CHECK-LABEL: generic_wrong_yield_parent
func @generic_fun_result_0_element_type(%arg0: !linalg.view<?xf32>) {
  // expected-error @+1 {{op expected 'linalg.generic' parent op}}
  linalg.yield %arg0: !linalg.view<?xf32>
}
