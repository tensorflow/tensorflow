// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func.func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> ()

func.func @loop_incorrent_num_yield_operands(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  %0 = gml_st.loop (%i, %j) = (%c0, %c0) to (%c192, %c192)
      step (%c24, %c24)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B: memref<192x192xf32>)
      outs (%CT_ = %C_tensor: tensor<192x192xf32>,
            %C_ = %C: memref<192x192xf32>) {
        func.call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> ()
    // expected-error @+1 {{expected number of tensor output args = 1 to match the number of yield operands = 0}}
    gml_st.yield
  }
  func.return
}

// -----

#map0 = affine_map<(d0) -> (24, -d0 + 192)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * 192 + s0 + d1)>
#map2 = affine_map<(d0) -> (16, -d0 + 192)>

func.func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> tensor<f32>

func.func @loop_incorrent_yield_operand_type(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  %0 = gml_st.loop (%i, %j) = (%c0, %c0) to (%c192, %c192)
      step (%c24, %c24)
      ins (%A_ = %A: memref<192x192xf32>, %B_ = %B: memref<192x192xf32>)
      outs (%CT_ = %C_tensor: tensor<192x192xf32>,
            %C_ = %C: memref<192x192xf32>) {
        %1 = func.call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> tensor<f32>
    // expected-error @+1 {{expected yield operand 0 with type = 'tensor<f32>' to match output arg type = 'tensor<192x192xf32>}}
    gml_st.yield %1 : tensor<f32>
  }
  func.return
}

// -----

func.func private @foo(%A: memref<192x192xf32>, %B: memref<192x192xf32>,
                  %C: memref<192x192xf32>) -> ()

func.func @loop_incorrent_iterator_types_count(%A: memref<192x192xf32>,
    %B: memref<192x192xf32>, %C: memref<192x192xf32>,
    %C_tensor: tensor<192x192xf32>) {
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  // expected-error @+1 {{expected iterator types array attribute size = 1 to match the number of loops = 2}}
  %0 = "gml_st.loop"(%c0, %c0, %c192, %c192, %c24, %c24, %A, %B, %C_tensor, %C) ({
    ^bb0(%arg4: index, %arg5: index, %A_: memref<192x192xf32>,
         %B_: memref<192x192xf32>, %CT_: tensor<192x192xf32>,
         %C_: memref<192x192xf32>):
      func.call @foo(%A_, %B_, %C_)
          : (memref<192x192xf32>, memref<192x192xf32>, memref<192x192xf32>)-> ()
      gml_st.yield %CT_ : tensor<192x192xf32>
    }) {
      iterator_types = ["parallel"],
      operand_segment_sizes = dense<2> : vector<5xi32>
    } : (index, index, index, index, index, index, memref<192x192xf32>,
      memref<192x192xf32>, tensor<192x192xf32>, memref<192x192xf32>
    ) -> tensor<192x192xf32>
  func.return
}

// -----

func.func private @foo(%A: memref<100xf32>) -> ()

func.func @loop_incorrent_block_arg_type(%A: memref<192xf32>) {
  %c0 = arith.constant 0 : index
  %c192 = arith.constant 192 : index
  %c24 = arith.constant 24 : index
  // expected-error @+1 {{expected output arg 0 with type = 'memref<192xf32>' to match region arg 1 type = 'memref<100xf32>'}}
  "gml_st.loop"(%c0, %c192, %c24, %A) ({
    ^bb0(%arg4: index, %A_: memref<100xf32>):
      func.call @foo(%A_) : (memref<100xf32>)-> ()
      gml_st.yield
    }) {
      iterator_types = ["parallel"],
      operand_segment_sizes = dense<[1, 1, 1, 0, 1]> : vector<5xi32>
    } : (index, index, index, memref<192xf32>) -> ()
  func.return
}

// -----

func.func @space_op_different_rank() {
  // expected-error@+1 {{expected 2 shapes values}}
  %0 = gml_st.space [64] : !gml_st.tile<32x32>
  func.return
}

// -----

func.func @space_op_dynamic_static_mismatch(%size : index) {
  // expected-error@+1 {{'gml_st.space' op inferred type(s) '!gml_st.tile<64x?>' are incompatible with return type(s) of operation '!gml_st.tile<64x32>'}}
  %0 = gml_st.space [64, %size] : !gml_st.tile<64x32>
  func.return
}

// -----

func.func @space_op_mismatch_shapes_and_static_shapes() {
  // expected-error@+1 {{expected 1 dynamic shapes values}}
  %0 = "gml_st.space"() {static_shapes = [5, -1]} : () -> !gml_st.tile<5x?>
  func.return
}

// -----

func.func @point_op_different_rank() {
  %0 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  // expected-error@+1 {{expected 2 indices values}}
  %1 = "gml_st.point"(%0) {static_indices = [0]} : (!gml_st.tile<64x32>) -> !gml_st.point
  func.return
}

// -----

func.func @point_op_of_point_op_expected_empty_static_indices() {
  %0 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  %1 = gml_st.point %0 [0, 0] : !gml_st.tile<64x32> to !gml_st.point
  // expected-error@+1 {{'gml_st.point' op expected empty indices and static_indices for a subset of type PointType}}
  %2 = gml_st.point %1 [0, 0] : !gml_st.point to !gml_st.point
  func.return
}

// -----

func.func @point_op_of_point_op_expected_empty_dynamic_indices(%i: index) {
  %0 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  %1 = gml_st.point %0 [%i, %i] : !gml_st.tile<64x32> to !gml_st.point
  // expected-error@+1 {{'gml_st.point' op expected empty indices and static_indices for a subset of type PointType}}
  %2 = gml_st.point %1 [%i, %i] : !gml_st.point to !gml_st.point
  func.return
}

// -----

func.func @point_op_mismatch_indices_and_static_indices(%i: index) {
  %0 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  // expected-error@+1 {{expected 0 dynamic indices values}}
  %1 = "gml_st.point"(%0, %i) {static_indices = [0, 0]} : (!gml_st.tile<64x32>, index) -> !gml_st.point
  func.return
}

// -----

func.func @point_op_static_indices_out_of_bounds() {
  %0 = gml_st.space [64, 32] : !gml_st.tile<64x32>
  // expected-error@+1 {{'gml_st.point' op expected index = 32 to be between 0 and 31}}
  %1 = gml_st.point %0 [5, 32] : !gml_st.tile<64x32> to !gml_st.point
  func.return
}

// -----

func.func @point_op_negative_static_indices(%size: index, %i: index) {
  %0 = gml_st.space [%size, 32] : !gml_st.tile<?x32>
  // expected-error@+1 {{'gml_st.point' op expected index = -2 to be non-negative}}
  %1 = gml_st.point %0 [-2, %i] : !gml_st.tile<?x32> to !gml_st.point
  func.return
}
