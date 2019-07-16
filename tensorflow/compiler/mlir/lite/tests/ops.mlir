// RUN: tf-opt -split-input-file -verify-diagnostics %s | FileCheck %s --dump-input-on-failure

// Unary math ops
// -----

// CHECK-LABEL: testCos
func @testCos(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.cos"(%arg0)
  %0 = "tfl.cos"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// -----

// test invalid Cos input
func @testCosWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{tfl.cos' op operand #0 must be tensor of floating-point values}}
  %0 = "tfl.cos"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testExp
func @testExp(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.exp"(%arg0)
  %0 = "tfl.exp"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testFloor
func @testFloor(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.floor"(%arg0)
  %0 = "tfl.floor"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testGather
func @testGather(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>) -> tensor<?xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<?xf32>,tensor<?xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testGather
func @testGather(%arg0 : tensor<2xf32>, %arg1 : tensor<2xi32>) -> tensor<2xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<2xf32>,tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}


// ----

// CHECK-LABEL: testGatherUnknownRank
func @testGatherUnknownRank(%arg0 : tensor<*xf32>, %arg1 : tensor<1xi32>) -> tensor<*xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<*xf32>,tensor<1xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @testGatherUnsupportedType(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{op failed to verify that params and output must have same element type}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<?xi32>,tensor<?xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func @testGatherUnsupportedRank(%arg0 : tensor<f32>, %arg1 : tensor<1xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{op failed to verify that operand 0 is 1-D}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<f32>,tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// -----

// CHECK-LABEL: testAbs
func @testAbs(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.abs"(%arg0)
  %0 = "tfl.abs"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testAddN
func @testAddN(tensor<? x f32>, tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>):
  // CHECK: "tfl.add_n"(%arg0, %arg1, %arg2)
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2): (tensor<? x f32>, tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// -----

// test invalid AddN
func @testAddNWrongOperandResultType(tensor<? x f16>, tensor<? x f16>, tensor<? x f16>) -> tensor<? x f16> {
^bb0(%arg0: tensor<? x f16>, %arg1: tensor<? x f16>, %arg2: tensor<? x f16>):
  // expected-error @+1 {{'tfl.add_n' op operand #0 must be tensor of 32-bit float or 32-bit integer values}}
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2): (tensor<? x f16>, tensor<? x f16>, tensor<? x f16>) -> tensor<? x f16>
  return %0 : tensor<? x f16>
}

// -----

// CHECK-LABEL: testLog
func @testLog(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.log"(%arg0)
  %0 = "tfl.log"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testNeg
func @testNeg(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.neg"(%arg0)
  %0 = "tfl.neg"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testRsqrt
func @testRsqrt(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.rsqrt"(%arg0)
  %0 = "tfl.rsqrt"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testSin
func @testSin(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.sin"(%arg0)
  %0 = "tfl.sin"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// -----

// test invalid Sin input
func @testSinWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{tfl.sin' op operand #0 must be tensor of floating-point values}}
  %0 = "tfl.sin"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testSqrt
func @testSqrt(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.sqrt"(%arg0)
  %0 = "tfl.sqrt"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testSquare
func @testSquare(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.square"(%arg0)
  %0 = "tfl.square"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testTanh
func @testTanh(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.tanh"(%arg0)
  %0 = "tfl.tanh"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testZerosLike
func @testZerosLike(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.zeros_like"(%arg0)
  %0 = "tfl.zeros_like"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testDequantize
func @testDequantize(tensor<? x i32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x i32>):
  // CHECK: "tfl.dequantize"(%arg0) : (tensor<?xi32>) -> tensor<?xf32>
  %0 = "tfl.dequantize"(%arg0): (tensor<? x i32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// CHECK-LABEL: testLogicalNot
func @testLogicalNot(tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>):
  // CHECK: "tfl.logical_not"(%arg0)
  %0 = "tfl.logical_not"(%arg0): (tensor<? x i1>) -> tensor<? x i1>
  return %0 : tensor<? x i1>
}

// -----

func @testLogicalNotWrongOperandType(tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_not' op operand #0 must be tensor of 1-bit integer values}}
  %0 = "tfl.logical_not"(%arg0) : (tensor<? x i32>) -> tensor<? x i32>
  return %0 : tensor<? x i32>
}

// Binary math ops
// -----

// CHECK-LABEL: testAdd
func @testAdd(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // TODO(jpienaar): Enable specifying label of enum for parsing.
  // CHECK: tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testSub
func @testSub(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testMul
func @testMul(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.mul %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testDiv
func @testDiv(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.div %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.div %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testLess
func @testLess(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: "tfl.less"(%arg0, %arg1)
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i1>
  return %0#0 : tensor<? x i1>
}

// -----

// CHECK-LABEL: testFloorDivI32
func @testFloorDivI32(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.floor_div %arg0, %arg1
  %0 = tfl.floor_div %arg0, %arg1 : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testFloorDivF32
func @testFloorDivF32(tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>):
  // CHECK: tfl.floor_div %arg0, %arg1
  %0 = tfl.floor_div %arg0, %arg1 : tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

func @testFloorDivF32(%arg0: tensor<2 x f32>, %arg1: tensor<2 x i32>) -> tensor<2 x f32> {
  // expected-error @+1 {{failed to verify that operands have same element type}}
  %0 = "tfl.floor_div"(%arg0, %arg1) : (tensor<2 x f32>, tensor<2 x i32>) -> tensor<2 x f32>
  return %0#0 : tensor<2 x f32>
}

// -----

// CHECK-LABEL: testFloorMod
func @testFloorMod(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.floor_mod %arg0, %arg1
  %0 = tfl.floor_mod %arg0, %arg1 : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testPow
func @testPow(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.pow %arg0, %arg1
  %0 = tfl.pow %arg0, %arg1 : tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testConv2D
func @testConv2D(tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>, %arg2: tensor<16xf32>):
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2)
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// CHECK-LABEL: testFakeQuant
func @testFakeQuant(tensor<? x f32>, f32, f32) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: f32, %arg2: f32):
  // CHECK: %0 = "tfl.fake_quant"(%arg0)  {minmax = [], narrow_range = true, num_bits = 2 : i32} : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.fake_quant"(%arg0) {minmax = [], num_bits = 2 : i32, narrow_range = true} : (tensor<? x f32>) -> tensor<? x f32>
  // CHECK: %1 = "tfl.fake_quant"(%0)  {minmax = [3.000000e-01, 1.400000e+00], narrow_range = false, num_bits = 6 : i32} : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tfl.fake_quant"(%0) {num_bits = 6 : i32, narrow_range = false, minmax = [0.3, 1.4]} : (tensor<? x f32>) -> tensor<? x f32>
  return %1 : tensor<? x f32>
}

// CHECK-LABEL: testQuantize
func @testQuantize(tensor<? x f32>) -> tensor<? x !quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<?x!quant.uniform<u8:f32, 1.000000e-01:128>>}
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<? x !quant.uniform<u8:f32, 0.1:128>>} : (tensor<? x f32>) -> tensor<? x !quant.uniform<u8:f32, 0.1:128>>
  return %0 : tensor<? x !quant.uniform<u8:f32, 0.1:128>>
}

// CHECK-LABEL: testLogicalAnd
func @testLogicalAnd(tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>, %arg1: tensor<? x i1>):
  // CHECK: tfl.logical_and %arg0, %arg1
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1>
  return %0#0 : tensor<? x i1>
}

// -----

func @testLogicalAndWrongOperandType(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_and' op operand #0 must be tensor of 1-bit integer values}}
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32>
  return %0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testLogicalOr
func @testLogicalOr(tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>, %arg1: tensor<? x i1>):
  // CHECK: tfl.logical_or %arg0, %arg1
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1>
  return %0#0 : tensor<? x i1>
}

// -----

func @testLogicalOrWrongOperandType(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_or' op operand #0 must be tensor of 1-bit integer values}}
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32>
  return %0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testEluF32
func @testEluF32(%arg0: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.elu"(%arg0)
  %0 = "tfl.elu"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testTileF32
func @testTileF32(%arg0: tensor<4 x 1 x f32>, %arg1: tensor<4 x i32>) -> tensor<? x f32> {
  // CHECK: "tfl.tile"(%arg0, %arg1)
  %0 = "tfl.tile"(%arg0, %arg1): (tensor<4 x 1 x f32>, tensor<4 x i32>) -> tensor<? x f32>
  return %0 : tensor<? x f32>
}

// -----

func @testEluI32(%arg0: tensor<? x i32>) -> tensor<? x i32> {
  // expected-error @+1 {{operand #0 must be tensor of floating-point values}}
  %0 = "tfl.elu"(%arg0): (tensor<? x i32>) -> tensor<? x i32>
  return %0#0 : tensor<? x i32>
}

// -----

func @testFusedActiviationFunction(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK: "NONE"
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xi32>
  // CHECK: "RELU"
  %1 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU"} : tensor<4xi32>
  // CHECK: "RELU_N1_TO_1"
  %2 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU_N1_TO_1"} : tensor<4xi32>
  // CHECK: "RELU6"
  %3 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<4xi32>
  // CHECK: "TANH"
  %4 = tfl.add %arg0, %arg1 {fused_activation_function = "TANH"} : tensor<4xi32>
  // CHECK: "SIGN_BIT"
  %5 = tfl.add %arg0, %arg1 {fused_activation_function = "SIGN_BIT"} : tensor<4xi32>
  return %0, %1, %2, %3, %4, %5: tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

func @testFusedActiviationFunction(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error @+1 {{attribute 'fused_activation_function' failed to satisfy constraint: fused activation enum}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "Relu6"} : tensor<4xi32>
  return %0: tensor<4xi32>
}

// -----

func @testPadding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>, %arg2: tensor<16xf32>) -> (tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>) {
  // CHECK: "SAME"
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  // CHECK: "VALID"
  %1 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0, %1 : tensor<256x30x30x16xf32>, tensor<256x30x30x16xf32>
}

// -----

func @testPadding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{attribute 'padding' failed to satisfy constraint: padding enum}}
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SOMETHING", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// -----

// CHECK-LABEL: testMaxPool2D
func @testMaxPool2D(tensor<256x32x32x3xf32>) -> tensor<?xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>):
  // CHECK: "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<?xf32>
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testMaxPool2DQuantized
func @testMaxPool2DQuantized(tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<?x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>):
  // CHECK: "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<?x!quant.uniform<i8:f32, 0.1:128>>
  return %0 : tensor<?x!quant.uniform<i8:f32, 0.1:128>>
}

// -----

// test invalid MaxPool2D
func @testMaxPool2DWrongOperandResultType(tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32> {
^bb0(%arg0: tensor<1x7x7x16xi32>):
  // expected-error @+1 {{failed to verify that MaxPool2D operand and result types match specified constraints}}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32>
  return %0 : tensor<1x7x7x16xi32>
}

// -----

// test invalid MaxPool2D
func @testMaxPool2DWrongOperandStorageType(tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>) -> tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>):
  // expected-error @+1 {{failed to verify that MaxPool2D operand and result types match specified constraints}}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>) -> tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>
  return %0 : tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>
}

// -----

// CHECK-LABEL: testLogistic
func @testLogistic(tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
^bb0(%arg0: tensor<1x2x3x4x5xbf16>):
  // CHECK: "tfl.logistic"(%arg0)
  %0 = "tfl.logistic"(%arg0): (tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
  return %0 : tensor<1x2x3x4x5xbf16>
}

// -----

// test invalid Logistic input
func @testLogisticWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{tfl.logistic' op operand #0 must be tensor of floating-point values}}
  %0 = "tfl.logistic"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstm
func @testUnidirectionalSequenceLstm(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstmWithNoneTypeAndOverrideAttr
func @testUnidirectionalSequenceLstmWithNoneTypeAndOverrideAttr(%arg0: tensor<? x f32>, %arg1: none, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// test invalid none type applied to a tensor type arg
func @testUnidirectionalSequenceLstmWithInvalidNoneType(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: none, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // expected-error @+1 {{'tfl.unidirectional_sequence_lstm' op operand #2 must be tensor of 32-bit float or 8-bit integer values}}
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<? x f32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// test violation of projection weight and projection bias pred op trait
func @testUnidirectionalSequenceLstmWithInvalidNoneType(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: none, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // expected-error @+1 {{'tfl.unidirectional_sequence_lstm' op failed to verify that either projection weight must be specified or both projection weight and projection bias must not be specified}}
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<? x f32>, tensor<? x f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testReverseV2
func @testReverseV2(%arg0: tensor<1x2x3x4xf32>, %arg1 : tensor<2xi32>) -> tensor<1x2x3x4xf32> {
  // CHECK: "tfl.reverse_v2"(%arg0, %arg1)
  %0 = "tfl.reverse_v2"(%arg0, %arg1): (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// test select
// CHECK-LABEL: testSelect
func @testSelect(%cond : tensor<?xi1>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi1>,tensor<?xi32>,tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func @testSelectWithUnsupportedType(%cond : tensor<?xi32>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{op operand #0 must be tensor of 1-bit integer values}}
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi32>,tensor<?xi32>,tensor<?xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func @testSelectWithUnsupportedType(%cond : tensor<?xi1>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xf32>) -> tensor<?xi32> {
  // expected-error @+1 {{failed to verify that operands have same element type}}
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi1>,tensor<?xi32>,tensor<?xf32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: topk
func @topk(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {
  %0, %1 = "tfl.topk_v2"(%arg0, %arg1) : (tensor<8xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
  return %0, %1: tensor<?xf32>, tensor<?xi32>
}

// -----

// CHECK-LABEL: topk
func @topk(%arg0: tensor<*xf32>, %arg1: tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0, %1 = "tfl.topk_v2"(%arg0, %arg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  return %0, %1: tensor<*xf32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: topk_2
func @topk_2(%arg0: tensor<3x4x8xf32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<3x4x8xf32>, tensor<i32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xi32>)
  return %1#0, %1#1: tensor<3x4x2xf32>, tensor<3x4x2xi32>
}

// -----

// CHECK-LABEL: topk_d
func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<?x2xf32>, tensor<?x2xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x2xf32>, tensor<?x2xi32>)
  return %1#0, %1#1: tensor<?x2xf32>, tensor<?x2xi32>
}

// -----

// CHECK-LABEL: topk_d
// TODO(jpienaar): This should fail but doesn't as the op definition does not
// include shape verification.
func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<?x3xf32>, tensor<?x3xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x3xf32>, tensor<?x3xi32>)
  return %1#0, %1#1: tensor<?x3xf32>, tensor<?x3xi32>
}

// -----

// CHECK-LABEL: topk_d
func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  return %1#0, %1#1: tensor<*xf32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: testEqual
func @testEqual(tensor<? x f32>, tensor<? x f32>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>):
  // CHECK: "tfl.equal"(%arg0, %arg1)
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<? x f32>, tensor<? x f32>) -> tensor<? x i1>
  return %0#0 : tensor<? x i1>
}

// -----

// CHECK-LABEL: testPad
func @testPad(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  // CHECK: "tfl.pad"(%arg0, %arg1)
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test Pad with invalid paddings size
func @testPadWithInvalidPaddingsDim(tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x2xi32>):
  // expected-error @+1 {{'tfl.pad' op failed to verify that operand 0's rank equals operand 1's size}}
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test Pad with invalid paddings rank
func @testPadWithInvalidPaddingsRank(tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<1x3x2xi32>):
  // expected-error @+1 {{'tfl.pad' op failed to verify that operand 1 is 2-D}}
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testPadV2
func @testPadV2(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = constant dense<2.0> : tensor<f32>
  // CHECK: "tfl.padv2"(%arg0, %arg1, %cst)
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid paddings size
func @testPadV2WithInvalidPaddingsDim(tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x2xi32>):
  %cst = constant dense<2.0> : tensor<f32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that operand 0's rank equals operand 1's size}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid paddings rank
func @testPadV2WithInvalidPaddingsRank(tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<1x3x2xi32>):
  %cst = constant dense<2.0> : tensor<f32>
  // expected-error @+1 {{'tfl.padv2' op failed to verify that operand 1 is 2-D}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<1x3x2xi32>, tensor<f32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid constant rank
func @testPadV2WithInvalidConstantScalar(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = constant dense<[2.0]> : tensor<1xf32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that operand 2 is 0-D}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<1xf32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid constant data type
func @testPadV2WithInvalidConstantScalar(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = constant dense<2> : tensor<i32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that input and constant value operands must have same element type}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<i32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>
}

// -----

func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{input count should match 'values_count' attribute}}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 1 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK: "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32}
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  return %0#0 : tensor<2xi32>

}

// -----

func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{output count should match 'num' attribute}}
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 2 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  return %0#0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: testMean
func @testMean(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // CHECK: "tfl.mean"(%arg0, %arg1) {keep_dims = false}
  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = false}: (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: testMean_true
func @testMean_true(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // CHECK: "tfl.mean"(%arg0, %arg1) {keep_dims = true}
  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = true}: (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// -----

func @testMean_missing_keep_dims(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // expected-error @+1 {{'tfl.mean' op requires attribute 'keep_dims'}}
  %0 = "tfl.mean"(%arg0, %arg1): (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: testBatchToSpaceND
func @testBatchToSpaceND(%arg0 : tensor<4x2x2x3xf32>, %arg1 : tensor<2xi32>, %arg2 : tensor<2x2xi32>) -> tensor<?xf32> {
  // CHECK:  "tfl.batch_to_space_nd"(%arg0, %arg1, %arg2)
  %0 = "tfl.batch_to_space_nd"(%arg0, %arg1, %arg2) : (tensor<4x2x2x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testSpaceToBatchND
func @testSpaceToBatchND(%arg0 : tensor<1x4x4x3xf32>, %arg1 : tensor<2xi32>, %arg2 : tensor<2x2xi32>) -> tensor<?xf32> {
  // CHECK: "tfl.space_to_batch_nd"(%arg0, %arg1, %arg2)
  %0 = "tfl.space_to_batch_nd"(%arg0, %arg1, %arg2) : (tensor<1x4x4x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func @testConcat(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

func @testConcatQuantized(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1:128>>, %arg1: tensor<2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
  // CHECK: "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2x!quant.uniform<i8:f32, 0.1:128>>, tensor<2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
}

// -----

func @testConcatInvalidOutputElementalType(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{'tfl.concatenation' op failed to verify that values and output must have same element type}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func @testConcatInvalidStorageType(%arg0: tensor<2x!quant.uniform<i9:f32, 0.1:128>>, %arg1: tensor<2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
  // expected-error @+1 {{'tfl.concatenation' op operand #0 must be tensor of 32-bit float or 64-bit integer or 32-bit integer or 16-bit integer or 8-bit integer or QI8 type or QUI8 type or TFLite uint8 type values}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2x!quant.uniform<i9:f32, 0.1:128>>, tensor<2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
  return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
}

// -----

// CHECK-LABEL: testResizeBilinear
func @testResizeBilinear(%arg0 : tensor<1x100x100x3xf32>, %arg1 : tensor<4xi32>) -> tensor<?xf32> {
  // CHECK: "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false}
  %0 = "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func @testResizeBilinearInvalidOutputType(%arg0 : tensor<1x100x100x3xf32>, %arg1 : tensor<4xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{'tfl.resize_bilinear' op result #0 must be tensor of 32-bit float values}}
  %0 = "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testStridedSlice
func @testStridedSlice(%arg0: tensor<12x2x2x5xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5xf32> {
  // CHECK: "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  return %0 : tensor<1x2x2x5xf32>
}

// -----

func @testStridedSliceWithInvalidOutputType(%arg0: tensor<12x2x2x5xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5xi32> {
  // expected-error @+1 {{op failed to verify that input and output must have same element type}}
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xi32>
  return %0 : tensor<1x2x2x5xi32>
}

