// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @extract_element_vector_type(%arg0: index) {
  // expected-error@+1 {{expected vector type}}
  %1 = vector.extractelement %arg0[] : index
}

// -----

func @extractelement_position_empty(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected non-empty position attribute}}
  %1 = vector.extractelement %arg0[] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_rank_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_rank_overflow_generic(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute of rank smaller than vector}}
  %1 = "vector.extractelement" (%arg0) { position = [0 : i32, 0 : i32, 0 : i32, 0 : i32] } : (vector<4x8x16xf32>) -> (vector<16xf32>)
}

// -----

func @extractelement_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #2 to be a positive integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 43 : i32, 0 : i32] : vector<4x8x16xf32>
}

// -----

func @extractelement_position_overflow(%arg0: vector<4x8x16xf32>) {
  // expected-error@+1 {{expected position attribute #3 to be a positive integer smaller than the corresponding vector dimension}}
  %1 = vector.extractelement %arg0[0 : i32, 0 : i32, -1 : i32] : vector<4x8x16xf32>
}

// -----

func @outerproduct_num_operands(%arg0: f32) {
  // expected-error@+1 {{expected at least 2 operands}}
  %1 = vector.outerproduct %arg0 : f32, f32
}
// -----

func @outerproduct_non_vector_operand(%arg0: f32) {
  // expected-error@+1 {{expected 2 vector types}}
  %1 = vector.outerproduct %arg0, %arg0 : f32, f32
}

// -----

func @outerproduct_operand_1(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #1}}
  %1 = vector.outerproduct %arg1, %arg1 : vector<4x8xf32>, vector<4x8xf32>
}

// -----

func @outerproduct_operand_2(%arg0: vector<4xf32>, %arg1: vector<4x8xf32>) {
  // expected-error@+1 {{expected 1-d vector for operand #2}}
  %1 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<4x8xf32>
}

// -----

func @outerproduct_result_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected 2-d vector result}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8xf32>)
}

// -----

func @outerproduct_operand_1_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #1 operand dim to match result dim #1}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<8x16xf32>)
}

// -----

func @outerproduct_operand_2_dim_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>) {
  // expected-error@+1 {{expected #2 operand dim to match result dim #2}}
  %1 = "vector.outerproduct" (%arg0, %arg1) : (vector<4xf32>, vector<8xf32>) -> (vector<4x16xf32>)
}

// -----

func @outerproduct_operand_3_result_type_generic(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x16xf32>) {
  // expected-error@+1 {{expected operand #3 of same type as result type}}
  %1 = "vector.outerproduct" (%arg0, %arg1, %arg2) : (vector<4xf32>, vector<8xf32>, vector<4x16xf32>) -> (vector<4x8xf32>)
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{two types required}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst { permutation_map = ()->(0) } : memref<?x?xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires 2 indices}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst { permutation_map = ()->(0) } : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires attribute 'permutation_map'}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {perm = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0)->(d0)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0, d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0 + d1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  %0 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = (d0, d1)->(d0 + 1)} : memref<?x?xf32>, vector<128xf32>
}

// -----

func @test_vector.transfer_read(%arg0: memref<?x?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant 3.0 : f32
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  %0 = vector.transfer_read %arg0[%c3, %c3, %c3], %cst {permutation_map = (d0, d1, d2)->(d0, d0)} : memref<?x?x?xf32>, vector<3x7xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{expected 5 operand types but had 4}}
  %0 = "vector.transfer_write"(%cst, %arg0, %c3, %c3, %c3) {permutation_map = ()->(0)} : (vector<128xf32>, memref<?x?xf32>, index, index) -> ()
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires 2 indices}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = ()->(0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires attribute 'permutation_map'}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {perm = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with input dims of the same rank as the memref type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0)->(d0)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a permutation_map with result dims of the same rank as the vector type}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0, d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + d1)} : vector<128xf32>, memref<?x?xf32>
}

// -----

func @test_vector.transfer_write(%arg0: memref<?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<128 x f32>
  // expected-error@+1 {{requires a projected permutation_map (at most one dim or the zero constant can appear in each result)}}
  vector.transfer_write %cst, %arg0[%c3, %c3] {permutation_map = (d0, d1)->(d0 + 1)} : vector<128xf32>, memref<?x?xf32>
}
// -----

func @test_vector.transfer_write(%arg0: memref<?x?x?xf32>) {
  %c3 = constant 3 : index
  %cst = constant dense<3.0> : vector<3 x 7 x f32>
  // expected-error@+1 {{requires a permutation_map that is a permutation (found one dim used more than once)}}
  vector.transfer_write %cst, %arg0[%c3, %c3, %c3] {permutation_map = (d0, d1, d2)->(d0, d0)} : vector<3x7xf32>, memref<?x?x?xf32>
}
