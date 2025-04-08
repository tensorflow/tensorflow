// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<128xi32>
  // expected-error @+1 {{rank of source must be same as rank of result}}
  %c = tt.broadcast %b : tensor<128xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

// Invalid bitcast between types of different bit width.
tt.func public @fn(%arg0: tensor<128xf32>) {
    // expected-error @+1 {{Cannot bitcast data-type of size}}
    %a = tt.bitcast %arg0 : tensor<128xf32> -> tensor<128xi16>
    tt.return
}
// -----

// Invalid bitcast between pointer and non-pointer type.
tt.func public @fn(%arg0: !tt.ptr<f32>) {
    // expected-error @+1 {{Cannot bitcast pointer to non-pointer type}}
    %a = tt.bitcast %arg0 : !tt.ptr<f32> -> i32
    tt.return
}
// -----

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<2x32xi32>
  // expected-error @+1 {{Different dimensions at index 0 between source and result.  Broadcast requires the source dimension to be 1.}}
  %c = tt.broadcast %b : tensor<2x32xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xf32>) {
    // expected-error @+1 {{packed_element}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r", packed_element=3:i32, pure=true} %arg0 : tensor<128xf32> -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xf32>, %arg1: tensor<64xf32>) {
    // expected-error @+1 {{same shape}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r,r", packed_element=1:i32, pure=true}
      %arg0, %arg1: tensor<128xf32>, tensor<64xf32> -> tensor<128xf32>
    tt.return
}
// -----

tt.func public @reshape_different_num_elements(%arg0: tensor<32x128xf16>) {
    // expected-error @+1 {{number of src and dst elements of reshape must be the same}}
    %a = tt.reshape %arg0 : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<33xf32>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<32x2xf32>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf16>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x32x2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) {
    // expected-error @+1 {{op result shape must be (32, 2), but got 64}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<64xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+1 {{result shape must be (32, 32, 2), but got 32, 64}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x64xf32>
    tt.return
}

// -----

// This one is OK
tt.func public @fn(%arg0: tensor<f32>, %arg1: tensor<f32>) {
    %a = tt.join %arg0, %arg1 : tensor<f32> -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{kind of type}}
    %a = tt.join %arg0, %arg1 : f32 -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%v: tensor<4x128xf64>) {
    // expected-error @+1 {{operand types and result types}}
    %a = "tt.reduce" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32}  : (tensor<4x128xf64>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func @reduce_different_input_shapes(%arg0: tensor<32x32x64xf32>, %arg1: tensor<16x32x64xf32>) -> (tensor<32x64xf32>, tensor<16x64xf32>) {
    // expected-error @below {{op requires the same shape for all operands}}
    %0:2 = "tt.reduce" (%arg0, %arg1) <{axis = 1 : i32}> ({
    ^bb0(%acc0: f32, %acc1: f32, %cur0: f32, %cur1: f32):
      %1 = arith.addf %acc0, %cur0 : f32
      %2 = arith.addf %acc1, %cur1 : f32
      tt.reduce.return %1, %2 : f32, f32
    }) : (tensor<32x32x64xf32>, tensor<16x32x64xf32>) -> (tensor<32x64xf32>, tensor<16x64xf32>)
    tt.return %0#0, %0#1 : tensor<32x64xf32>, tensor<16x64xf32>
}

// -----

tt.func public @fn(%v: tensor<4x128xf32>) {
    // expected-error @+1 {{requires the same shape}}
    %a = "tt.scan" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.scan.return %add : f32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.scan" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.scan.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<4x128xi64>, tensor<4x128xf32>)
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.reduce" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.reduce.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<128xi64>, tensor<128xf32>)
    tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+1 {{op result encoding must be specified}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32>
    tt.return
}
}  // end module

// -----

// Bad order; should be [1,0]
#blocked  = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [0,1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+1 {{op incompatible join layout}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

tt.func public @fn(%arg0: tensor<32xf32>) {
    // expected-error @+2 {{last dimension}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32xf32> -> tensor<16xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x2xf32>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32x2xf32> -> tensor<32xf16>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32) {
    // expected-error @+1 {{invalid kind of type}}
    %a, %b = tt.split %arg0 : f32 -> f16
    tt.return
}
// -----

tt.func public @fn(%arg0: tensor<2xf32>) {
    %a, %b = tt.split %arg0 : tensor<2xf32> -> tensor<f32> // OK
    tt.return
}

// -----

#blocked  = #ttg.blocked<{sizePerThread = [1,2,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// Bad order, should be [1,0].
#blocked1 = #ttg.blocked<{sizePerThread = [1,1], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [1,0]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked  = #ttg.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// bad sizePerThread; should be [1,1].
#blocked1 = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [0,1]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Valid ops.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1, 2>} : tensor<16x32x64xf32> -> tensor<16x32x64xf32>
    %b = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32> -> tensor<32x16x64xf32>
    tt.return
}
}  // end module

// -----

// Valid op with blocked encoding.
#blocked  = #ttg.blocked<{sizePerThread = [1,2,3,4], threadsPerWarp = [2,4,2,2], warpsPerCTA = [4,2,4,2], order = [3,2,1,0], CTAsPerCGA = [1,2,2,2], CTASplitNum = [1,2,4,8], CTAOrder = [3,2,1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2,4,3,1], threadsPerWarp = [4,2,2,2], warpsPerCTA = [2,2,4,4], order = [1,2,0,3], CTAsPerCGA = [2,2,2,1], CTASplitNum = [2,8,4,1], CTAOrder = [1,2,0,3]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2,1,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x4x8x16xf32, #blocked>, %arg1: tensor<16x32x64xf32, #blocked2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : tensor<2x4x8x16xf32, #blocked> -> tensor<4x16x8x2xf32, #blocked1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked2> -> tensor<32x16x64xf32, #blocked3>
    tt.return
}
}  // end module

// -----

// Valid op with shared encoding.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [3, 2, 1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 2, 0, 3]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CTAsPerCGA = [1, 2], CTASplitNum = [2, 4], CTAOrder = [0, 1]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32, CTAsPerCGA = [2, 1], CTASplitNum = [4, 2], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: !ttg.memdesc<2x4x8x16xf32, #shared, #smem>, %arg1: !ttg.memdesc<16x32xf32, #shared2, #smem>) {
    %a = ttg.memdesc_trans %arg0 {order = array<i32: 1, 3, 2, 0>} : !ttg.memdesc<2x4x8x16xf32, #shared, #smem> -> !ttg.memdesc<4x16x8x2xf32, #shared1, #smem>
    %b = ttg.memdesc_trans %arg1 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf32, #shared2, #smem> -> !ttg.memdesc<32x16xf32, #shared3, #smem>
    tt.return
}
}  // end module

// -----

// Invalid blocked encoding.
#blocked  = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #blocked>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked> -> tensor<32x16x64xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Invalid shared encoding.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 2, 1, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order must be a permutation}}
    %a = tt.trans %arg0 {order = array<i32: 0, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

// Invalid tensor with shared encoding.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{has an invalid layout: Shared layout is not allowed on tensor type.}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{indices and output shapes must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x4xi32>) -> tensor<512xf32>
  tt.return
}

// -----

#blocked  = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32, #blocked>) {
  // expected-error @below {{indices and output encodings must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x4xi32, #blocked>) -> tensor<512x4xf32, #blocked1>
  tt.return
}
}

// -----

tt.func @gather_op(%arg0: tensor<128x16xf16>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{input and output element types must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf16>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}

// -----

tt.func @gather_op(%arg0: tensor<128xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{input and indices ranks must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128xf32>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}

// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x32xi32>) {
  // expected-error @below {{indices dimension 1 must match the corresponding input dimension}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x32xi32>) -> tensor<512x32xf32>
  tt.return
}
// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{gather dimension must be less than the input rank}}
  %0 = tt.gather %arg0[%arg1] {axis = 3 : i32} : (tensor<128x16xf32>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}

// -----

tt.func @invalid_desc_load(%arg0: !tt.tensordesc<tensor<16x16xf32>>) {
  %c = arith.constant 0 : i32
  // expected-error @below {{ranked reduce load only allowed for unit dimension leading dim}}
  tt.descriptor_load %arg0[%c, %c] : !tt.tensordesc<tensor<16x16xf32>> -> tensor<16xf32>
  tt.return
}

// -----

tt.func @invalid_desc_load(%arg0: !tt.tensordesc<tensor<16x16xf32>>) {
  %c = arith.constant 0 : i32
  // expected-error @below {{tensor desciptor block and tensor types must match}}
  tt.descriptor_load %arg0[%c, %c] : !tt.tensordesc<tensor<16x16xf32>> -> tensor<16x16xf16>
  tt.return
}

// -----

tt.func @invalid_desc_store(%arg0: !tt.tensordesc<tensor<16x16xf32>>, %arg1: tensor<32x16xf32>) {
  %c = arith.constant 0 : i32
  // expected-error @below {{tensor desciptor block and tensor types must match}}
  tt.descriptor_store %arg0[%c, %c], %arg1 : !tt.tensordesc<tensor<16x16xf32>>, tensor<32x16xf32>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{block must be a 2D tensor}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<128xbf16>>, tensor<32xi32>, i32) -> tensor<32xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<2x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{block must have exactly 1 row}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<2x128xbf16>>, tensor<32xi32>, i32) -> tensor<32x128xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<1x32xi32>, %arg2: i32) {
  // expected-error @below {{x offsets must be a 1D tensor}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<1x32xi32>, i32) -> tensor<32x128xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{result must be a 2D tensor}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<128xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{result tensor number of columns must match block (128)}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{result tensor must have as many rows as indices (32)}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<64x128xbf16>
  tt.return
}

// -----

tt.func @invalid_tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16>>, %arg1: tensor<32xi32>, %arg2: i32) {
  // expected-error @below {{result tensor element type must match block ('bf16')}}
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16>>, tensor<32xi32>, i32) -> tensor<32x128xf32>
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @invalid_dot(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<16x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = tt.load %arg1 : tensor<16x32x!tt.ptr<f32>, #blocked>
    %11 = ttg.local_alloc %9 : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
    %12 = ttg.local_alloc %10 : (tensor<16x32xf32, #blocked>) -> !ttg.memdesc<16x32xf32, #shared, #smem>
    %13 = ttg.local_load %11 : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %14 = ttg.local_load %12 : !ttg.memdesc<16x32xf32, #shared, #smem> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %15 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>

    // expected-error @below {{'tt.dot' op expected the last dimension of the first operand to be equal to the second-to-last dimension of the second operand}}
    %16 = tt.dot %13, %14, %15 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %17 = ttg.convert_layout %16 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %17 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @dot_scaled_fp8(
    %a: tensor<128x32xi8, #blocked2>,
    %scale: tensor<128x2xi8, #blocked1>,
    %b_fp8: tensor<128x128xf8E4M3FN, #blocked>
    ) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // expected-error @below {{'tt.dot_scaled' op expected the last dimension of the first operand to be equal to the second-to-last dimension of the second operand}}
    %result = tt.dot_scaled %a scale %scale, %b_fp8, %cst lhs = e2m1 rhs = e4m3 {fastMath = true} : tensor<128x32xi8, #blocked2>, tensor<128x2xi8, #blocked1> * tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf32, #blocked>
    tt.return %result : tensor<128x128xf32, #blocked>
  }
}
