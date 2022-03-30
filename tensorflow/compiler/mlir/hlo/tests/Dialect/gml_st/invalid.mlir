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

