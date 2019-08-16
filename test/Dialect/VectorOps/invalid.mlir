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
