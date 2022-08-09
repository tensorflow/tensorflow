// RUN: tf-opt -split-input-file -verify-diagnostics -tfl-runtime-verify %s | FileCheck %s

// Unary math ops
// -----

// CHECK-LABEL: testCos
func.func @testCos(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.cos"(%arg0)
  %0 = "tfl.cos"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

// test invalid Cos input
func.func @testCosWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{tfl.cos' op operand #0 must be tensor of 32-bit float values}}
  %0 = "tfl.cos"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  func.return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testExp
func.func @testExp(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.exp"(%arg0)
  %0 = "tfl.exp"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testFloor
func.func @testFloor(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.floor"(%arg0)
  %0 = "tfl.floor"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testGather
func.func @testGather(%arg0 : tensor<?xf32>, %arg1 : tensor<?xi32>) -> tensor<?xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<?xf32>,tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testGather
func.func @testGather(%arg0 : tensor<2xf32>, %arg1 : tensor<2xi32>) -> tensor<2xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<2xf32>,tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: testGatherUnknownRank
func.func @testGatherUnknownRank(%arg0 : tensor<*xf32>, %arg1 : tensor<1xi32>) -> tensor<*xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<*xf32>,tensor<1xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testGatherUnsupportedType(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{op failed to verify that params and output must have same element type}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<?xi32>,tensor<?xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testGatherUnsupportedRank(%arg0 : tensor<f32>, %arg1 : tensor<1xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{op failed to verify that operand 0 is 1-D}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<f32>,tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testGatherUnsupported2DStringInput(%arg0 : tensor<?x?x!tf_type.string>, %arg1 : tensor<?xi32>) -> tensor<*x!tf_type.string> {
  // expected-error @+1 {{'tfl.gather' op expect 1d input when the given type is string, got 'tensor<?x?x!tf_type.string>'}}
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32}: (tensor<?x?x!tf_type.string>,tensor<?xi32>) -> tensor<*x!tf_type.string>
  func.return %0 : tensor<*x!tf_type.string>
}

// -----

// CHECK-LABEL: testGatherWithBatchDims
func.func @testGatherWithBatchDims(%arg0 : tensor<2xf32>, %arg1 : tensor<2xi32>) -> tensor<2xf32> {
  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32, batch_dims = 2 : i32}: (tensor<2xf32>,tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: testAbs
func.func @testAbs(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.abs"(%arg0)
  %0 = "tfl.abs"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testAddN
func.func @testAddN(tensor<? x f32>, tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>):
  // CHECK: "tfl.add_n"(%arg0, %arg1, %arg2)
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2): (tensor<? x f32>, tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

// test invalid AddN
func.func @testAddNWrongOperandResultType(tensor<? x f16>, tensor<? x f16>, tensor<? x f16>) -> tensor<? x f16> {
^bb0(%arg0: tensor<? x f16>, %arg1: tensor<? x f16>, %arg2: tensor<? x f16>):
  // expected-error @+1 {{'tfl.add_n' op operand #0 must be tensor of 32-bit float or 32-bit signless integer}}
  %0 = "tfl.add_n"(%arg0, %arg1, %arg2): (tensor<? x f16>, tensor<? x f16>, tensor<? x f16>) -> tensor<? x f16>
  func.return %0 : tensor<? x f16>
}

// -----

// CHECK-LABEL: testLog
func.func @testLog(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.log"(%arg0)
  %0 = "tfl.log"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testLogStaticShapeInputAndDynamicShapeOutput
func.func @testLogStaticShapeInputAndDynamicShapeOutput(tensor<8 x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<8 x f32>):
  // CHECK: "tfl.log"(%arg0)
  %0 = "tfl.log"(%arg0): (tensor<8 x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testNeg
func.func @testNeg(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.neg"(%arg0)
  %0 = "tfl.neg"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testRsqrt
func.func @testRsqrt(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.rsqrt"(%arg0)
  %0 = "tfl.rsqrt"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testRsqrtQuant
func.func @testRsqrtQuant(%arg0: tensor<1x80x1x!quant.uniform<i8:f32, 0.048358432948589325:-128>>) -> tensor<1x80x1x!quant.uniform<i8:f32, 0.0066055487841367722:-128>> {
  %0 = "tfl.rsqrt"(%arg0) : (tensor<1x80x1x!quant.uniform<i8:f32, 0.048358432948589325:-128>>) -> tensor<1x80x1x!quant.uniform<i8:f32, 0.0066055487841367722:-128>>
  func.return %0 : tensor<1x80x1x!quant.uniform<i8:f32, 0.0066055487841367722:-128>>
}

// CHECK-LABEL: testSin
func.func @testSin(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.sin"(%arg0)
  %0 = "tfl.sin"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

// test invalid Sin input
func.func @testSinWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{tfl.sin' op operand #0 must be tensor of 32-bit float values}}
  %0 = "tfl.sin"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  func.return %0#0 : tensor<?xi32>
}

// -----

// test invalid Sqrt input
func.func @testSqrtWithWrongInputType(tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>):
  // expected-error @+1 {{tfl.sqrt' op operand #0 must be tensor of 32-bit float values}}
  %0 = "tfl.sqrt"(%arg0): (tensor<? x i32>) -> tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// -----

// test invalid Square input
func.func @testSquareWithWrongInputType(tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>):
  // expected-error @+1 {{tfl.square' op operand #0 must be tensor of 32-bit float values}}
  %0 = "tfl.square"(%arg0): (tensor<? x i32>) -> tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testSquaredDifferenceQuant
func.func @testSquaredDifferenceQuant(%arg0: tensor<1x80x128x!quant.uniform<i8:f32, 0.089839041233062744:10>>, %arg1: tensor<1x80x128x!quant.uniform<i8:f32, 0.0019308560295030475:-6>>) -> tensor<1x80x128x!quant.uniform<i8:f32, 0.60070550441741943:-128>> {
  %0 = "tfl.squared_difference"(%arg0, %arg1) : (tensor<1x80x128x!quant.uniform<i8:f32, 0.089839041233062744:10>>, tensor<1x80x128x!quant.uniform<i8:f32, 0.0019308560295030475:-6>>) -> tensor<1x80x128x!quant.uniform<i8:f32, 0.60070550441741943:-128>>
  func.return %0 : tensor<1x80x128x!quant.uniform<i8:f32, 0.60070550441741943:-128>>
}

// -----

// CHECK-LABEL: testSqrt
func.func @testSqrt(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.sqrt"(%arg0)
  %0 = "tfl.sqrt"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testSquare
func.func @testSquare(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.square"(%arg0)
  %0 = "tfl.square"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

func.func @testQuantizedResizeNearestNeighbor(tensor<? x ? x ? x ? x !quant.uniform<u8:f32, 0.1>>, tensor<? x i32>) -> tensor<? x !quant.uniform<u8:f32, 0.1>> {
^bb0(%arg0: tensor<? x ? x ? x ? x !quant.uniform<u8:f32, 0.1>>, %arg1: tensor<? x i32>):
  %0 = "tfl.resize_nearest_neighbor"(%arg0, %arg1) { align_corners = false, half_pixel_centers = false } : (tensor<? x ? x ? x ? x !quant.uniform<u8:f32, 0.1>>, tensor<? x i32>) -> tensor<? x !quant.uniform<u8:f32, 0.1>>
  func.return %0 : tensor<? x !quant.uniform<u8:f32, 0.1>>
}

// CHECK-LABEL: testTanh
func.func @testTanh(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.tanh"(%arg0)
  %0 = "tfl.tanh"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testTanhWithQI8
func.func @testTanhWithQI8(%arg0: tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x !quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.tanh"(%arg0): (tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x !quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<? x !quant.uniform<i8:f32, 0.1>>
}

// CHECK-LABEL: testTanhWithQUI8
func.func @testTanhWithQUI8(%arg0: tensor<? x !quant.uniform<u8:f32, 0.1>>) -> tensor<? x !quant.uniform<u8:f32, 0.1>> {
  %0 = "tfl.tanh"(%arg0): (tensor<? x !quant.uniform<u8:f32, 0.1>>) -> tensor<? x !quant.uniform<u8:f32, 0.1>>
  func.return %0 : tensor<? x !quant.uniform<u8:f32, 0.1>>
}

// CHECK-LABEL: testZerosLike
func.func @testZerosLike(tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: "tfl.zeros_like"(%arg0)
  %0 = "tfl.zeros_like"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testDequantize
func.func @testDequantize(tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x !quant.uniform<i8:f32, 0.1>>):
  // CHECK: "tfl.dequantize"(%arg0) : (tensor<?x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<?xf32>
  %0 = "tfl.dequantize"(%arg0): (tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// CHECK-LABEL: testLogicalNot
func.func @testLogicalNot(tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>):
  // CHECK: "tfl.logical_not"(%arg0)
  %0 = "tfl.logical_not"(%arg0): (tensor<? x i1>) -> tensor<? x i1>
  func.return %0 : tensor<? x i1>
}

// -----

func.func @testLogicalNotWrongOperandType(tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_not' op operand #0 must be tensor of 1-bit signless integer values}}
  %0 = "tfl.logical_not"(%arg0) : (tensor<? x i32>) -> tensor<? x i32>
  func.return %0 : tensor<? x i32>
}

// Binary math ops
// -----

// CHECK-LABEL: testAdd
func.func @testAdd(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // TODO(jpienaar): Enable specifying label of enum for parsing.
  // CHECK: tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testSub
func.func @testSub(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testSubInt64
func.func @testSubInt64(tensor<? x i64>, tensor<? x i64>) -> tensor<? x i64> {
^bb0(%arg0: tensor<? x i64>, %arg1: tensor<? x i64>):
  // CHECK: tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i64>
  func.return %0#0 : tensor<? x i64>
}

// CHECK-LABEL: testMul
func.func @testMul(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.mul %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testMulComplex
func.func @testMulComplex(tensor<? x complex<f32>>, tensor<? x complex<f32>>) -> tensor<? x complex<f32>> {
^bb0(%arg0: tensor<? x complex<f32>>, %arg1: tensor<? x complex<f32>>):
  // CHECK: tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"}
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"}: tensor<? x complex<f32>>
  func.return %0#0 : tensor<? x complex<f32>>
}

// CHECK-LABEL: testAddWithI64Broadcasting
func.func @testAddWithI64Broadcasting(tensor< 2x3xi64>, tensor<3xi64>) -> tensor<2x3xi64> {
^bb0(%arg0: tensor<2x3xi64>, %arg1: tensor<3xi64>):
  // CHECK: tfl.add(%arg0, %arg1)
  %0 = tfl.add(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor< 2x3xi64>, tensor<3xi64>) -> tensor<2x3xi64>
  func.return %0#0 : tensor<2x3xi64>
}

// -----

func.func @add_with_quantized_i16_broadcasting(tensor<2x2xf32>, tensor<1xf32>) -> tensor<2x2x!quant.any<i16:f32>> {
^bb0(%arg0: tensor<2x2xf32>, %arg1: tensor<1xf32>):
  // expected-error @+1 {{Operands do not have valid shapes}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<2x2xf32>, tensor<1xf32>) -> tensor<2x2x!quant.any<i16:f32>>
  func.return %0#0 : tensor<2x2x!quant.any<i16:f32>>
}

// -----

func.func @sub_with_quantized_i8_five_dim_broadcasting(tensor<1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x1x1x1x!quant.any<i8:f32>> {
^bb0(%arg0: tensor<1x1x1x1x1xf32>, %arg1: tensor<1xf32>):
  // expected-error @+1 {{Operands do not have valid shapes}}
  %0 = "tfl.sub"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x1x1x1x1x!quant.any<i8:f32>>
  func.return %0#0 : tensor<1x1x1x1x1x!quant.any<i8:f32>>
}

// -----

func.func @mul_with_i32_five_dim_broadcasting(tensor<1x1x1x1x1xi32>, tensor<1xi32>) -> tensor<1x1x1x1x1xi32> {
^bb0(%arg0: tensor<1x1x1x1x1xi32>, %arg1: tensor<1xi32>):
  // expected-error @+1 {{Operands do not have valid shapes}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1x1x1x1x1xi32>, tensor<1xi32>) -> tensor<1x1x1x1x1xi32>
  func.return %0#0 : tensor<1x1x1x1x1xi32>
}

// -----

func.func @mul_with_quantized_i16_five_dim_broadcasting(tensor<1x1x1x1x1x!quant.any<i16:f32>>, tensor<1x!quant.any<i16:f32>>) -> tensor<1x1x1x1x1x!quant.any<i16:f32>> {
^bb0(%arg0: tensor<1x1x1x1x1x!quant.any<i16:f32>>, %arg1: tensor<1x!quant.any<i16:f32>>):
  // expected-error @+1 {{Operands do not have valid shapes}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1x1x1x1x1x!quant.any<i16:f32>>, tensor<1x!quant.any<i16:f32>>) -> tensor<1x1x1x1x1x!quant.any<i16:f32>>
  func.return %0#0 : tensor<1x1x1x1x1x!quant.any<i16:f32>>
}

// -----

func.func @mul_with_quantized_i16_to_uint8_broadcasting(tensor<1x1x!quant.any<i16:f32>>, tensor<1x!quant.any<i16:f32>>) -> tensor<1x1x!quant.any<ui8:f32>> {
^bb0(%arg0: tensor<1x1x!quant.any<i16:f32>>, %arg1: tensor<1x!quant.any<i16:f32>>):
  // expected-error @+1 {{Operands do not have valid shapes}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<1x1x!quant.any<i16:f32>>, tensor<1x!quant.any<i16:f32>>) -> tensor<1x1x!quant.any<ui8:f32>>
  func.return %0#0 : tensor<1x1x!quant.any<ui8:f32>>
}

// -----

// CHECK-LABEL: testMulNonQuantizedOperandsandQuantizedResult
func.func @testMulNonQuantizedOperandsandQuantizedResult(tensor<? x f32>, tensor<? x f32>) -> tensor<? x !quant.any<i16:f32>> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>):
  // CHECK: tfl.mul(%arg0, %arg1) {fused_activation_function = "RELU6"}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"}: (tensor<? x f32>, tensor<? x f32>) -> tensor<? x !quant.any<i16:f32>>
  func.return %0#0 : tensor<? x !quant.any<i16:f32>>
}

// -----

func.func @testMulInvalidOperands(tensor<? x f32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x i32>):
  // expected-error @+1 {{failed to verify that binary op operands have same element type}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"}: (tensor<? x f32>, tensor<? x i32>) -> tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// -----

func.func @testMulInvalidQuantizedOperands(tensor<* x !quant.any<i16:f32>>, tensor<* x !quant.any<i8:f32>>) -> tensor<* x !quant.any<i16:f32>> {
^bb0(%arg0: tensor<* x !quant.any<i16:f32>>, %arg1: tensor<* x !quant.any<i8:f32>>):
  // expected-error @+1 {{failed to verify that binary op operands have same element type}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "RELU6"}: (tensor<* x !quant.any<i16:f32>>, tensor<* x !quant.any<i8:f32>>) -> tensor<* x !quant.any<i16:f32>>
  func.return %0#0 : tensor<* x !quant.any<i16:f32>>
}

// -----

// CHECK-LABEL: testDiv
func.func @testDiv(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.div %arg0, %arg1 {fused_activation_function = "RELU6"}
  %0 = tfl.div %arg0, %arg1 {fused_activation_function = "RELU6"} : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testLess
func.func @testLess(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.less(%arg0, %arg1)
  %0 = "tfl.less"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i1>
  func.return %0#0 : tensor<? x i1>
}

// -----

// CHECK-LABEL: testFloorDivI32
func.func @testFloorDivI32(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.floor_div %arg0, %arg1
  %0 = tfl.floor_div %arg0, %arg1 : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testFloorDivF32
func.func @testFloorDivF32(tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>):
  // CHECK: tfl.floor_div %arg0, %arg1
  %0 = tfl.floor_div %arg0, %arg1 : tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

func.func @testFloorDivF32(%arg0: tensor<2 x f32>, %arg1: tensor<2 x i32>) -> tensor<2 x f32> {
  // expected-error @+1 {{failed to verify that binary op operands have same element type}}
  %0 = "tfl.floor_div"(%arg0, %arg1) : (tensor<2 x f32>, tensor<2 x i32>) -> tensor<2 x f32>
  func.return %0#0 : tensor<2 x f32>
}

// -----

// CHECK-LABEL: testFloorMod
func.func @testFloorMod(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>) -> tensor<? x i32> {
  %0 = "tfl.floor_mod"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32>
  func.return %0 : tensor<? x i32>
}

// CHECK-LABEL: testPow
func.func @testPow(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // CHECK: tfl.pow %arg0, %arg1
  %0 = tfl.pow %arg0, %arg1 : tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// CHECK-LABEL: testConv2D
func.func @testConv2D(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>):
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2)
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// CHECK-LABEL: testConv2DNoBias
func.func @testConv2DNoBias(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: none) -> tensor<256x32x32x16xf32> {
  // CHECK: "tfl.conv_2d"(%arg0, %arg1, %arg2)
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, none) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2D4DBias(tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>) -> tensor<256x32x32x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<1x1x1x16xf32>):
  // expected-error @+1 {{'tfl.conv_2d' op operand #2 must be 1D tensor of any type values or none type, but got 'tensor<1x1x1x16xf32>'}}
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "RELU6"} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

// CHECK-LABEL: testFakeQuant
func.func @testFakeQuant(tensor<? x f32>, f32, f32) -> tensor<? x f32> {
^bb0(%arg0: tensor<? x f32>, %arg1: f32, %arg2: f32):
  // CHECK: "tfl.fake_quant"(%arg0)  {max = 1.400000e+00 : f32, min = 3.000000e-01 : f32, narrow_range = false, num_bits = 6 : i32} : (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tfl.fake_quant"(%arg0) {num_bits = 6 : i32, narrow_range = false, min = 0.3:f32, max = 1.4:f32} : (tensor<? x f32>) -> tensor<? x f32>
  func.return %1 : tensor<? x f32>
}

// CHECK-LABEL: testQuantize
func.func @testQuantize(tensor<? x f32>) -> tensor<? x !quant.uniform<u8:f32, 0.1:128>> {
^bb0(%arg0: tensor<? x f32>):
  // CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<?x!quant.uniform<u8:f32, 1.000000e-01:128>>}
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<? x !quant.uniform<u8:f32, 0.1:128>>} : (tensor<? x f32>) -> tensor<? x !quant.uniform<u8:f32, 0.1:128>>
  func.return %0 : tensor<? x !quant.uniform<u8:f32, 0.1:128>>
}

// CHECK-LABEL: testLogicalAnd
func.func @testLogicalAnd(tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>, %arg1: tensor<? x i1>):
  // CHECK: tfl.logical_and %arg0, %arg1
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1>
  func.return %0#0 : tensor<? x i1>
}

// -----

func.func @testLogicalAndWrongOperandType(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_and' op operand #0 must be tensor of 1-bit signless integer values}}
  %0 = "tfl.logical_and"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32>
  func.return %0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testLogicalOr
func.func @testLogicalOr(tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x i1>, %arg1: tensor<? x i1>):
  // CHECK: tfl.logical_or %arg0, %arg1
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<? x i1>, tensor<? x i1>) -> tensor<? x i1>
  func.return %0#0 : tensor<? x i1>
}

// -----

func.func @testLogicalOrWrongOperandType(tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32> {
^bb0(%arg0: tensor<? x i32>, %arg1: tensor<? x i32>):
  // expected-error @+1 {{'tfl.logical_or' op operand #0 must be tensor of 1-bit signless integer values}}
  %0 = "tfl.logical_or"(%arg0, %arg1) : (tensor<? x i32>, tensor<? x i32>) -> tensor<? x i32>
  func.return %0 : tensor<? x i32>
}

// -----

// CHECK-LABEL: testEluF32
func.func @testEluF32(%arg0: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.elu"(%arg0)
  %0 = "tfl.elu"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testTileF32
func.func @testTileF32(%arg0: tensor<4 x 1 x f32>, %arg1: tensor<4 x i32>) -> tensor<? x f32> {
  // CHECK: "tfl.tile"(%arg0, %arg1)
  %0 = "tfl.tile"(%arg0, %arg1): (tensor<4 x 1 x f32>, tensor<4 x i32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

func.func @testEluI32(%arg0: tensor<? x i32>) -> tensor<? x i32> {
  // expected-error @+1 {{operand #0 must be tensor of 32-bit float or 8-bit signless integer values}}
  %0 = "tfl.elu"(%arg0): (tensor<? x i32>) -> tensor<? x i32>
  func.return %0#0 : tensor<? x i32>
}

// -----

func.func @testFusedActivationFunction(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
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
  func.return %0, %1, %2, %3, %4, %5: tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

func.func @testFusedActivationFunction(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error @+1 {{attribute 'fused_activation_function' failed to satisfy constraint: string attribute whose value is NONE, or RELU, or RELU_N1_TO_1, or RELU6, or TANH, or SIGN_BIT}}
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "Relu6"} : tensor<4xi32>
  func.return %0: tensor<4xi32>
}

// -----

func.func @testPadding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> (tensor<256x32x32x16xf32>, tensor<256x30x30x16xf32>) {
  // CHECK: "SAME"
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x32x32x16xf32>
  // CHECK: "VALID"
  %1 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0, %1 : tensor<256x32x32x16xf32>, tensor<256x30x30x16xf32>
}

// -----

func.func @testPadding(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{op attribute 'padding' failed to satisfy constraint: string attribute whose value is SAME, or VALID}}
  %0 = "tfl.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SOMETHING", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

// CHECK-LABEL: testMaxPool2D
func.func @testMaxPool2D(tensor<256x32x32x3xf32>) -> tensor<?xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>):
  // CHECK: "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<?xf32>
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testMaxPool2DQuantized
func.func @testMaxPool2DQuantized(tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<?x!quant.uniform<i8:f32, 0.1:128>> {
^bb0(%arg0: tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>):
  // CHECK: "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<256x32x32x3x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<?x!quant.uniform<i8:f32, 0.1:128>>
  func.return %0 : tensor<?x!quant.uniform<i8:f32, 0.1:128>>
}

// -----

// test invalid MaxPool2D
func.func @testMaxPool2DWrongOperandResultType(tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32> {
^bb0(%arg0: tensor<1x7x7x16xi32>):
  // expected-error @+1 {{failed to verify that MaxPool2D operand and result types match specified constraints}}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x7x7x16xi32>) -> tensor<1x7x7x16xi32>
  func.return %0 : tensor<1x7x7x16xi32>
}

// -----

// test invalid MaxPool2D
func.func @testMaxPool2DWrongOperandStorageType(tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>) -> tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>> {
^bb0(%arg0: tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>):
  // expected-error @+1 {{failed to verify that MaxPool2D operand and result types match specified constraints}}
  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 1 : i32, filter_width = 1 : i32, fused_activation_function = "RELU6", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>) -> tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>
  func.return %0 : tensor<1x7x7x16x!quant.uniform<i9:f32, 0.1:128>>
}

// -----

func.func @testTFLiteDetectionPostProcess(%arg0: tensor<1x64x64x32xf32>, %arg1: tensor<1x64x64x32xf32>, %arg2: tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) {
  %0, %1, %2, %3 = "tfl.custom_tf"(%arg0, %arg1, %arg2) ({
  ^bb0(%a0: tensor<1x64x64x32xf32>, %a1: tensor<1x64x64x32xf32>, %a2: tensor<1x64x64x32xf32>):
    %4, %5, %6, %7 = "tf.TFLite_Detection_PostProcess"(%a0, %a1, %a2) {_output_quantized = true, _output_types = [f32, f32, f32, f32], _support_output_type_float_in_quantized_op = true, detections_per_class = 100 : i64, device = "", h_scale = 5.000000e+00 : f32, max_classes_per_detection = 1 : i64, max_detections = 20 : i64, nms_iou_threshold = 6.000000e-01 : f32, nms_score_threshold = 3.000000e-01 : f32, num_classes = 90 : i64, use_regular_nms = false, w_scale = 5.000000e+00 : f32, x_scale = 1.000000e+01 : f32, y_scale = 1.000000e+01 : f32} : (tensor<1x64x64x32xf32>, tensor<1x64x64x32xf32>, tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
    "tfl.yield"(%4, %5, %6, %7) : (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) -> ()
  }) : (tensor<1x64x64x32xf32>, tensor<1x64x64x32xf32>, tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  func.return %0, %1 : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>
}

// -----

func.func @testMaxPoolingWithArgMax2D(%arg0: tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>) {
  // custom op for "tfl.max_pooling_with_argmax_2d"(%arg0) {filter_h = 2 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  %0, %1 = "tfl.custom"(%arg0) {custom_option = #tfl<const_bytes : "0x01000000020000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "MaxPoolingWithArgmax2D"} : (tensor<1x64x64x32xf32>) -> (tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>)
  func.return %0, %1 : tensor<1x32x32x32xf32>, tensor<1x32x32x32xf32>
}

// -----

func.func @testMaxUnpooling2D(%arg0: tensor<1x8x8x128xf32>, %arg1: tensor<1x8x8x128xf32>) -> tensor<1x8x8x128xf32> {
  // custom op for "tfl.max_unpooling_2d"(%arg0, %arg1) {filter_h = 2 : i32, filter_w = 2 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> (tensor<1x8x8x128xf32>)
  %0 = "tfl.custom"(%arg0, %arg1) {custom_option = #tfl<const_bytes : "0x01000000020000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "MaxUnpooling2D"} : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xf32>) -> (tensor<1x8x8x128xf32>)
  func.return %0 : tensor<1x8x8x128xf32>
}

// -----

// CHECK-LABEL: testLogistic
func.func @testLogistic(tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
^bb0(%arg0: tensor<1x2x3x4x5xf32>):
  // CHECK: "tfl.logistic"(%arg0)
  %0 = "tfl.logistic"(%arg0): (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
  func.return %0 : tensor<1x2x3x4x5xf32>
}

// -----

// test invalid Logistic input
func.func @testLogisticWithWrongInputType(tensor<?xi32>) -> tensor<?xi32> {
^bb0(%arg0: tensor<?xi32>):
  // expected-error @+1 {{'tfl.logistic' op operand #0 must be tensor of 32-bit float or QI8 type or QUI8 type or QI16 type or TFLite quint8 type values, but got 'tensor<?xi32>'}}
  %0 = "tfl.logistic"(%arg0): (tensor<?xi32>) -> tensor<?xi32>
  func.return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceRnn
func.func @testUnidirectionalSequenceRnn(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x ? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.unidirectional_sequence_rnn"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>) -> tensor<?xf32>
  %0 = "tfl.unidirectional_sequence_rnn"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "NONE", time_major = false} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstmWithoutProjection
func.func @testUnidirectionalSequenceLstmWithoutProjection(%arg0: tensor<? x ? x ? x f32>, %arg1: tensor<? x ? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: none, %arg17: none, %arg18: tensor<? x ? x f32>, %arg19: tensor<? x ? xf32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? xf32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstm
func.func @testUnidirectionalSequenceLstm(%arg0: tensor<? x ? x ? x f32>, %arg1: tensor<? x ? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x ? x f32>, %arg19: tensor<? x ? xf32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? xf32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstmWithNoneTypeAndOverrideAttr
func.func @testUnidirectionalSequenceLstmWithNoneTypeAndOverrideAttr(%arg0: tensor<? x ? x ? x f32>,
  %arg1: none, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>,
  %arg5: none, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>,
  %arg9: none, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>,
  %arg12: none, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>,
  %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>,
  %arg18: tensor<? x ? x f32>, %arg19: tensor<? x ? xf32>,
  %arg20: none, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? xf32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0,
  // CHECK-SAME: %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15,
  // CHECK-SAME: %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", time_major = false} :
  // CHECK-SAME: (tensor<?x?x?xf32>,
  // CHECK-SAME: none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
  // CHECK-SAME: none, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
  // CHECK-SAME: none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0,
    %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
    %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15,
    %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>,
      none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
      none, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
      tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testUnidirectionalSequenceLstmWithIntermediates
func.func @testUnidirectionalSequenceLstmWithIntermediates(%arg0: tensor<? x ? x ? x f32>, %arg1: tensor<? x ? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x ? x f32>, %arg19: tensor<? x ? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? xf32> {
  // CHECK: "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+01 : f32, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0077881771139800549>>, fused_activation_function = "TANH", input_to_cell_intermediate = tensor<0xf32>, input_to_forget_intermediate = tensor<0xf32>, input_to_input_intermediate = tensor<0xf32>, input_to_output_intermediate = tensor<0xf32>, proj_clip = 0.000000e+00 : f32, time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {cell_clip = 1.000000e+01 : f32, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0077881771139800549>>, fused_activation_function = "TANH", input_to_cell_intermediate = tensor<0xf32>, input_to_forget_intermediate = tensor<0xf32>, input_to_input_intermediate = tensor<0xf32>, input_to_output_intermediate = tensor<0xf32>, proj_clip = 0.000000e+00 : f32, time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// test violation of projection weight and projection bias pred op trait
func.func @testUnidirectionalSequenceLstmWithInvalidNoneType(%arg0: tensor<? x ? x ? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: none, %arg17: tensor<? x f32>, %arg18: tensor<? x ? x f32>, %arg19: tensor<? x ? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x ? x ? xf32> {
  // expected-error @+1 {{'tfl.unidirectional_sequence_lstm' op failed to verify that either projection weight must be specified or both projection weight and projection bias must not be specified}}
  %0 = "tfl.unidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) {fused_activation_function = "NONE", time_major = false} : (tensor<?x?x?xf32>, tensor<? x f32>, tensor<? x f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testLstmIntermediates
func.func @testLstmIntermediates(%arg0: tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, %arg1: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg2: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.031925998628139496>>, %arg3: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.056272000074386597>>, %arg4: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.063763998448848724>>, %arg5: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.013358999975025654>>, %arg6: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.022830000147223473>>, %arg7: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.032276000827550888>>, %arg8: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.035427000373601913>>, %arg9: tensor<2048x!quant.uniform<i32:f32, 4.2675782196965883E-7>>, %arg10: tensor<2048x!quant.uniform<i32:f32, 1.0742187583900886E-7>>, %arg11: tensor<2048x!quant.uniform<i32:f32, 1.6406249869760359E-7>>, %arg12: tensor<2048x!quant.uniform<i32:f32, 1.523437447303877E-7>>, %arg13: tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, %arg14: tensor<640x!quant.uniform<i32:f32, 1.601389680352559E-4>>, %arg15: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg16: tensor<2048x!quant.uniform<i16:f32, 1.1000000085914508E-4>>, %arg17: tensor<2048x!quant.uniform<i16:f32, 1.6799999866634607E-4>>, %arg18: tensor<2048x!quant.uniform<i16:f32, 1.55999994603917E-4>>, %arg19: tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, %arg20: tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>> {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %cst, %cst, %cst, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({}) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", input_to_input_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0049890000373125076>>, input_to_forget_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0078849997371435165>>, input_to_cell_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0087630003690719604>>, input_to_output_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0057529998011887074>>, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0075630000792443752:2>>, kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.01 : f32} : (tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.031925998628139496>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.056272000074386597>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.063763998448848724>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.013358999975025654>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.022830000147223473>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.032276000827550888>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.035427000373601913>>, none, none, none, tensor<2048x!quant.uniform<i32:f32, 4.2675782196965883E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.0742187583900886E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.6406249869760359E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.523437447303877E-7>>, tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, tensor<640x!quant.uniform<i32:f32, 1.601389680352559E-4>>, tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.1000000085914508E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.6799999866634607E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.55999994603917E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
  func.return %0 : tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
// CHECK: %[[RES0:.*]] = "tfl.no_value"() {value} : () -> none
// CHECK: %[[RES1:.*]] = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[RES0]], %[[RES0]], %[[RES0]], %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({
// CHECK: }) {cell_clip = 1.000000e+01 : f32, effective_hidden_scale_intermediate = tensor<0x!quant.uniform<i8<-127:127>:f32, 0.0075630000792443752:2>>, fused_activation_function = "TANH", input_to_cell_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0087630003690719604>>, input_to_forget_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0078849997371435165>>, input_to_input_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0049890000373125076>>, input_to_output_intermediate = tensor<0x!quant.uniform<i16:f32, 0.0057529998011887074>>, kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.00999999977 : f32} : (tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.031925998628139496>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.056272000074386597>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.063763998448848724>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.013358999975025654>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.022830000147223473>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.032276000827550888>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.035427000373601913>>, none, none, none, tensor<2048x!quant.uniform<i32:f32, 4.2675782196965883E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.0742187583900886E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.6406249869760359E-7>>, tensor<2048x!quant.uniform<i32:f32, 1.523437447303877E-7>>, tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, tensor<640x!quant.uniform<i32:f32, 1.601389680352559E-4>>, tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.1000000085914508E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.6799999866634607E-4>>, tensor<2048x!quant.uniform<i16:f32, 1.55999994603917E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
}

// -----

// CHECK-LABEL: testBidirectionalSequenceLstm
func.func @testBidirectionalSequenceLstm(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>, %arg5: tensor<?x?xf32>, %arg6: tensor<?x?xf32>, %arg7: tensor<?x?xf32>, %arg8: tensor<?x?xf32>, %arg9: tensor<?xf32>, %arg10: tensor<?xf32>, %arg11: tensor<?xf32>, %arg12: tensor<?xf32>, %arg13: tensor<?xf32>, %arg14: tensor<?xf32>, %arg15: tensor<?xf32>, %arg16: tensor<?x?xf32>, %arg17: tensor<?xf32>, %arg18: tensor<?x?xf32>, %arg19: tensor<?x?xf32>, %arg20: tensor<?x?xf32>, %arg21: tensor<?x?xf32>, %arg22: tensor<?x?xf32>, %arg23: tensor<?x?xf32>, %arg24: tensor<?x?xf32>, %arg25: tensor<?x?xf32>, %arg26: tensor<?xf32>, %arg27: tensor<?xf32>, %arg28: tensor<?xf32>, %arg29: tensor<?xf32>, %arg30: tensor<?xf32>, %arg31: tensor<?xf32>, %arg32: tensor<?xf32>, %arg33: tensor<?x?xf32>, %arg34: tensor<?xf32>, %arg35: tensor<?xf32>, %arg36: tensor<?xf32>, %arg37: tensor<?xf32>, %arg38: tensor<?xf32>, %arg39: tensor<?xf32>, %arg40: tensor<?xf32>, %arg41: tensor<?xf32>, %arg42: tensor<?xf32>, %arg43: tensor<?xf32>, %arg44: tensor<?xf32>, %arg45: tensor<?xf32>, %arg46: tensor<?xf32>, %arg47: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "tfl.bidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", merge_outputs = true, time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
  %0:2 = "tfl.bidirectional_sequence_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", merge_outputs = true, time_major = false} : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
  func.return %0#0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testLstmQuantizedType
func.func @testLstmQuantizedType(%arg0: tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, %arg1: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg2: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg3: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg4: tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg5: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg6: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg7: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg8: tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, %arg9: tensor<2048x!quant.uniform<i32:f32, 0.01>>, %arg10: tensor<2048x!quant.uniform<i32:f32, 0.01>>, %arg11: tensor<2048x!quant.uniform<i32:f32, 0.01>>, %arg12: tensor<2048x!quant.uniform<i32:f32, 0.01>>, %arg13: tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, %arg14: tensor<640x!quant.uniform<i32:f32, 9.9999999747524271E-7>>, %arg15: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg16: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg17: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg18: tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, %arg19: tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, %arg20: tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>> {
    %cst = "tfl.no_value"() {value = unit} : () -> none
    %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %cst, %cst, %cst, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({
    }) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.01 : f32} : (tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, none, none, none, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, tensor<640x!quant.uniform<i32:f32, 9.9999999747524271E-7>>, tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
    func.return %0 : tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
  // CHECK: %[[RES0:.*]] = "tfl.no_value"() {value} : () -> none
  // CHECK: %[[RES1:.*]] = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[RES0]], %[[RES0]], %[[RES0]], %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({
  // CHECK-NEXT: }) {cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.00999999977 : f32} : (tensor<1x528x!quant.uniform<i8:f32, 0.037248000502586365:-19>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x528x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, tensor<2048x640x!quant.uniform<i8<-127:127>:f32, 0.059801999479532242>>, none, none, none, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<2048x!quant.uniform<i32:f32, 1.000000e-02>>, tensor<640x2048x!quant.uniform<i8<-127:127>:f32, 0.021174000576138496>>, tensor<640x!quant.uniform<i32:f32, 9.9999999747524271E-7>>, tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>, tensor<1x2048x!quant.uniform<i16:f32, 4.8799999058246613E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>, tensor<2048x!quant.uniform<i16:f32, 4.3700000969693065E-4>>) -> tensor<1x640x!quant.uniform<i8:f32, 0.09671100229024887:10>>
  // CHECK: return %[[RES1]]
}

// -----

// CHECK-LABEL: testLstm
func.func @testLstm(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x ? x f32>, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23)
  // CHECK-NEXT: {fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>} : (tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) ({}) {fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>} : (tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testBasicLstm
func.func @testBasicLstm(%arg0: tensor<1x384xf32>, %arg1: tensor<1x96xf32>, %arg2: tensor<384x480xf32>, %arg3: tensor<384xf32>, %arg4: tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>) {
  %0:4 = "tfl.basic_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "TANH", kernel_type = #tfl<lstm_kernel_type_attr BASIC>} : (tensor<1x384xf32>, tensor<1x96xf32>, tensor<384x480xf32>, tensor<384xf32>, tensor<1x96xf32>) -> (tensor<1x96xf32>, tensor<1x96xf32>, tensor<1x480xf32>, tensor<1x384xf32>)
  func.return %0#0, %0#1 : tensor<1x96xf32>, tensor<1x96xf32>
}

// -----

// CHECK-LABEL: testQuantizedBasicLstm
func.func @testQuantizedBasicLstm(%arg0: tensor<1x384x!quant.uniform<u8:f32, 7.812500e-03:128>>, %arg1: tensor<1x96x!quant.uniform<u8:f32, 7.812500e-03:128>>, %arg2: tensor<384x480x!quant.uniform<u8<1:255>:f32, 0.070853792130947113:163>>, %arg3: tensor<384x!quant.uniform<i32:f32, 5.5354525102302432E-4>>, %arg4: tensor<1x96x!quant.uniform<i16:f32, 4.8828125E-4>>) -> (tensor<1x96x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<1x96x!quant.uniform<i16:f32, 4.8828125E-4>>) {
  %0:4 = "tfl.basic_lstm"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x384x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<1x96x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<384x480x!quant.uniform<u8<1:255>:f32, 0.070853792130947113:163>>, tensor<384x!quant.uniform<i32:f32, 5.5354525102302432E-4>>, tensor<1x96x!quant.uniform<i16:f32, 4.8828125E-4>>) -> (tensor<1x96x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<1x96x!quant.uniform<i16:f32, 4.8828125E-4>>, tensor<1x480x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<1x384x!quant.uniform<i16:f32, 2.44140625E-4>>)
  func.return %0#0, %0#1 : tensor<1x96x!quant.uniform<u8:f32, 7.812500e-03:128>>, tensor<1x96x!quant.uniform<i16:f32, 4.8828125E-4>>
}

// -----

// CHECK-LABEL: testLstmWithNoneTypeAndOverrideAttr
func.func @testLstmWithNoneTypeAndOverrideAttr(%arg0: tensor<? x f32>,
  %arg1: none, %arg2: tensor<? x ? x f32>, %arg3: tensor<? x ? x f32>, %arg4: tensor<? x ? x f32>,
  %arg5: none, %arg6: tensor<? x ? x f32>, %arg7: tensor<? x ? x f32>, %arg8: tensor<? x ? x f32>,
  %arg9: none, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>,
  %arg12: none, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>,
  %arg16: tensor<? x ? x f32>, %arg17: tensor<? x f32>,
  %arg18: tensor<? x f32>, %arg19: tensor<? x f32>,
  %arg20: none, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23)
  // CHECK-NEXT: {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>} :
  // CHECK-SAME: (tensor<?xf32>,
  // CHECK-SAME: none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
  // CHECK-SAME: none, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
  // CHECK-SAME: none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.lstm"(%arg0,
    %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8,
    %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15,
    %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) ({}) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>} : (tensor<?xf32>,
      none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, none, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
      none, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
      tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// test violation of projection weight and projection bias pred op trait
func.func @testLstmWithInvalidNoneType(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: none, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // expected-error @+1 {{'tfl.lstm' op failed to verify that either projection weight must be specified or both projection weight and projection bias must not be specified}}
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) ({}) {fused_activation_function = "NONE"} : (tensor<?xf32>, tensor<? x f32>, tensor<? x f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, none, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// test invalid input dimension, the third input operand for lstm op should be 2-D tensor.
func.func @testLstmWithInvalidInputDimension(%arg0: tensor<4 x f32>, %arg1: tensor<4 x f32>, %arg2: tensor<4 x f32>, %arg3: tensor<4 x f32>, %arg4: tensor<4 x f32>, %arg5: tensor<4 x f32>, %arg6: tensor<4 x f32>, %arg7: tensor<4 x f32>, %arg8: tensor<4 x f32>, %arg9: tensor<4 x f32>, %arg10: tensor<4 x f32>, %arg11: tensor<4 x f32>, %arg12: tensor<4 x f32>, %arg13: tensor<4 x f32>, %arg14: tensor<4 x f32>, %arg15: tensor<4 x f32>, %arg16: tensor<4 x f32>, %arg17: tensor<4 x f32>, %arg18: tensor<4 x f32>, %arg19: tensor<4 x f32>, %arg20: tensor<4 x f32>, %arg21: tensor<4 x f32>) -> tensor<4 x f32> {
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32> loc("Const")
  // expected-error @+1 {{'tfl.lstm' op failed to verify that operand 2 is 2-D}}
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %24 : tensor<4xf32>

}

// -----

// 'input_to_output_weights' input for lstm op has unmatched rank with `input`.
func.func @testLstmWithInvalidInputsRankMatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4x2xf32>, %arg2: tensor<4x2xf32>, %arg3: tensor<4x2xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x4xf32>, %arg6: tensor<4x4xf32>, %arg7: tensor<4x4xf32>, %arg8: tensor<4x4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4xf32>, %arg11: tensor<4xf32>, %arg12: tensor<1x4xf32>, %arg13: tensor<4xf32>, %arg14: tensor<4xf32>, %arg15: tensor<4xf32>, %arg16: tensor<4x4xf32>, %arg17: tensor<4xf32>, %arg18: tensor<4xf32>, %arg19: tensor<4xf32>, %arg20: tensor<4xf32>, %arg21: tensor<4xf32>) -> tensor<1x4xf32> {
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  // expected-error @+1 {{'tfl.lstm' op inputs don't match with the dimensions.}}
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {cell_clip = 0.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.000000e+00 : f32} : (tensor<1x4xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  func.return %24 : tensor<1x4xf32>
}

// -----

// Coefficient inputs of LSTM op don't match the dimension with input operand `input_to_output_weights`.
func.func @testLstmWithInvalidInputsRankMatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>, %arg6: tensor<4x4xf32>, %arg7: tensor<4x4xf32>, %arg8: tensor<4x4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4xf32>, %arg11: tensor<4xf32>, %arg12: tensor<1x4xf32>, %arg13: tensor<4xf32>, %arg14: tensor<4xf32>, %arg15: tensor<4xf32>, %arg16: tensor<4x4xf32>, %arg17: tensor<4xf32>, %arg18: tensor<3xf32>, %arg19: tensor<3xf32>, %arg20: tensor<3xf32>, %arg21: tensor<3xf32>) -> tensor<1x4xf32> {
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  // expected-error @+1 {{'tfl.lstm' op coefficient inputs have more than 2 dimensions or don't match the dimension with input operand `input_to_output_weights`.}}
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {cell_clip = 0.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.000000e+00 : f32} : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<1x4xf32>
  func.return %24 : tensor<1x4xf32>
}

// -----

// Coefficient inputs of LSTM op have unknown rank.
func.func @testLstmWithInvalidInputsRankMatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>, %arg6: tensor<4x4xf32>, %arg7: tensor<4x4xf32>, %arg8: tensor<4x4xf32>, %arg9: tensor<4xf32>, %arg10: tensor<4xf32>, %arg11: tensor<4xf32>, %arg12: tensor<1x4xf32>, %arg13: tensor<4xf32>, %arg14: tensor<4xf32>, %arg15: tensor<4xf32>, %arg16: tensor<4x4xf32>, %arg17: tensor<4xf32>, %arg18: tensor<3xf32>, %arg19: tensor<3xf32>, %arg20: tensor<3xf32>, %arg21: tensor<*xf32>) -> tensor<1x4xf32> {
  %cst0 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  %cst1 = "tfl.pseudo_const" () {value = dense<0.0> : tensor<1x4xf32>} : () -> tensor<1x4xf32> loc("Const")
  // expected-error @+1 {{'tfl.lstm' op coefficient inputs have more than 2 dimensions or don't match the dimension with input operand `input_to_output_weights`.}}
  %24 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %cst0, %cst1, %arg18, %arg19, %arg20, %arg21) ({}) {cell_clip = 0.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.000000e+00 : f32} : (tensor<1x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4x4xf32>, tensor<4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<*xf32>) -> tensor<1x4xf32>
  func.return %24 : tensor<1x4xf32>
}

// -----

// test invalid kernel type
func.func @testLstmWithInvalidKernelType(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>, %arg5: tensor<? x f32>, %arg6: tensor<? x f32>, %arg7: tensor<? x f32>, %arg8: tensor<? x f32>, %arg9: tensor<? x f32>, %arg10: tensor<? x f32>, %arg11: tensor<? x f32>, %arg12: tensor<? x f32>, %arg13: tensor<? x f32>, %arg14: tensor<? x f32>, %arg15: tensor<? x f32>, %arg16: tensor<? x f32>, %arg17: tensor<? x f32>, %arg18: tensor<? x f32>, %arg19: tensor<? x f32>, %arg20: tensor<? x f32>, %arg21: tensor<? x f32>, %arg22: tensor<? x f32>, %arg23: tensor<? x f32>) -> tensor<? x f32> {
  // expected-error @+1 {{'tfl.lstm' op attribute 'kernel_type' failed to satisfy constraint: lstm_kernel_type whose value is mlir::TFL::LSTMKernelType::FULL}}
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23) ({}) {cell_clip = 1.000000e+00 : f32, fused_activation_function = "NONE", kernel_type = #tfl<lstm_kernel_type_attr BASIC>} : (tensor<?xf32>, tensor<? x f32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testReverseV2
func.func @testReverseV2(%arg0: tensor<1x2x3x4xf32>, %arg1 : tensor<2xi32>) -> tensor<1x2x3x4xf32> {
  // CHECK: "tfl.reverse_v2"(%arg0, %arg1)
  %0 = "tfl.reverse_v2"(%arg0, %arg1): (tensor<1x2x3x4xf32>, tensor<2xi32>) -> tensor<1x2x3x4xf32>
  func.return %0 : tensor<1x2x3x4xf32>
}

// -----

// test select
// CHECK-LABEL: testSelect
func.func @testSelect(%cond : tensor<?xi1>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi1>,tensor<?xi32>,tensor<?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// test select with multi-dim inputs
// CHECK-LABEL: testSelectMultiDim
func.func @testSelectMultiDim(%cond : tensor<?xi1>, %arg0 : tensor<?x4xi32>, %arg1 : tensor<?x4xi32>) -> tensor<?x4xi32> {
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi1>,tensor<?x4xi32>,tensor<?x4xi32>) -> tensor<?x4xi32>
  func.return %0 : tensor<?x4xi32>
}

// -----

func.func @testSelectWithUnsupportedType(%cond : tensor<?xi32>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{op operand #0 must be tensor of 1-bit signless integer values}}
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi32>,tensor<?xi32>,tensor<?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

func.func @testSelectWithUnsupportedType(%cond : tensor<?xi1>, %arg0 : tensor<?xi32>, %arg1 : tensor<?xf32>) -> tensor<?xi32> {
  // expected-error @+1 {{failed to verify that operands have same element type}}
  %0 = "tfl.select"(%cond, %arg0, %arg1): (tensor<?xi1>,tensor<?xi32>,tensor<?xf32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testSelectV2
func.func @testSelectV2(%cond : tensor<*xi1>, %arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tfl.select_v2"(%arg0, %arg1, %arg2)
  %0 = "tfl.select_v2"(%cond, %arg0, %arg1): (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: testSelectV2WithHighDimInputs
func.func @testSelectV2WithHighDimInputs(%cond : tensor<1x2x3x4x5x6xi1>, %arg0 : tensor<1x2x3x4x5x6xf32>, %arg1 : tensor<1x2x3x4x5x6xf32>) -> tensor<1x2x3x4x5x6xf32> {
  // CHECK: "tfl.select_v2"(%arg0, %arg1, %arg2)
  %0 = "tfl.select_v2"(%cond, %arg0, %arg1): (tensor<1x2x3x4x5x6xi1>, tensor<1x2x3x4x5x6xf32>, tensor<1x2x3x4x5x6xf32>) -> tensor<1x2x3x4x5x6xf32>
  func.return %0 : tensor<1x2x3x4x5x6xf32>
}

// -----

// CHECK-LABEL: testSelectV2With4DBroadcasting
func.func @testSelectV2With4DBroadcasting(%cond : tensor<1x1x3x1xi1>, %arg0 : tensor<1x1x1x4xf32>, %arg1 : tensor<1x2x1x1xf32>) -> tensor<1x2x3x4xf32> {
  // CHECK: "tfl.select_v2"(%arg0, %arg1, %arg2)
  %0 = "tfl.select_v2"(%cond, %arg0, %arg1): (tensor<1x1x3x1xi1>, tensor<1x1x1x4xf32>, tensor<1x2x1x1xf32>) -> tensor<1x2x3x4xf32>
  func.return %0 : tensor<1x2x3x4xf32>
}

// -----

func.func @testSelectV2WithWrongBroadcastableArguments(%cond : tensor<3x4xi1>, %arg0 : tensor<2x3x4xf32>, %arg1 : tensor<4x3xf32>) -> tensor<2x3x4xf32> {
  // expected-error @+1 {{'tfl.select_v2' op operands don't have broadcast-compatible shapes}}
  %0 = "tfl.select_v2"(%cond, %arg0, %arg1): (tensor<3x4xi1>, tensor<2x3x4xf32>, tensor<4x3xf32>) -> tensor<2x3x4xf32>
  func.return %0 : tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: topk
func.func @topk(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {
  %0, %1 = "tfl.topk_v2"(%arg0, %arg1) : (tensor<8xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
  func.return %0, %1: tensor<?xf32>, tensor<?xi32>
}

// -----

// CHECK-LABEL: topk
func.func @topk(%arg0: tensor<*xf32>, %arg1: tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0, %1 = "tfl.topk_v2"(%arg0, %arg1) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  func.return %0, %1: tensor<*xf32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: topk_2
func.func @topk_2(%arg0: tensor<3x4x8xf32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xi32>) {
  %0 = arith.constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<3x4x8xf32>, tensor<i32>) -> (tensor<3x4x2xf32>, tensor<3x4x2xi32>)
  func.return %1#0, %1#1: tensor<3x4x2xf32>, tensor<3x4x2xi32>
}

// -----

// CHECK-LABEL: topk_d
func.func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<?x2xf32>, tensor<?x2xi32>) {
  %0 = arith.constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x2xf32>, tensor<?x2xi32>)
  func.return %1#0, %1#1: tensor<?x2xf32>, tensor<?x2xi32>
}

// -----

// CHECK-LABEL: topk_d
// TODO(jpienaar): This should fail but doesn't as the op definition does not
// include shape verification.
func.func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<?x3xf32>, tensor<?x3xi32>) {
  %0 = arith.constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x3xf32>, tensor<?x3xi32>)
  func.return %1#0, %1#1: tensor<?x3xf32>, tensor<?x3xi32>
}

// -----

// CHECK-LABEL: topk_d
func.func @topk_d(%arg0: tensor<?x8xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = arith.constant dense<2> : tensor<i32>
  %1:2 = "tfl.topk_v2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  func.return %1#0, %1#1: tensor<*xf32>, tensor<*xi32>
}

// -----

// CHECK-LABEL: testEqual
func.func @testEqual(tensor<? x f32>, tensor<? x f32>) -> tensor<? x i1> {
^bb0(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>):
  // CHECK: "tfl.equal"(%arg0, %arg1)
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<? x f32>, tensor<? x f32>) -> tensor<? x i1>
  func.return %0#0 : tensor<? x i1>
}

// -----

// CHECK-LABEL: testPad
func.func @testPad(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  // CHECK: "tfl.pad"(%arg0, %arg1)
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testPad5D
func.func @testPad5D(tensor<*xf32>, tensor<5x3xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<5x3xi32>):
  // CHECK: "tfl.pad"(%arg0, %arg1)
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<*xf32>, tensor<5x3xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test Pad with invalid paddings size
func.func @testPadWithInvalidPaddingsDim(tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x2xi32>):
  // expected-error @+1 {{'tfl.pad' op failed to verify that operand 0's rank equals operand 1's size}}
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test Pad with invalid paddings rank
func.func @testPadWithInvalidPaddingsRank(tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<1x3x2xi32>):
  // expected-error @+1 {{'tfl.pad' op failed to verify that operand 1 is 2-D}}
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

func.func @testPadUnknownPaddings(tensor<2x1x3xf32>, tensor<*xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<*xi32>):
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<*xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>

  // CHECK-LABEL: testPadUnknownPaddings
  // CHECK:  "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<*xi32>) -> tensor<?xf32>
  // CHECK:  return
}

// -----

func.func @testPadUnsupportedPaddings(tensor<*xf32>, tensor<6x3xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<6x3xi32>):
  // expected-error @+1 {{'tfl.pad' op failed to verify that the first dim size of the padding argument must be at most 5}}
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<*xf32>, tensor<6x3xi32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testPadQuantizedU8
func.func @testPadQuantizedU8(%arg0: tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<3x2xi32>) -> tensor<? x !quant.uniform<u8:f32, 0.1>> {
  // CHECK: "tfl.pad"(%arg0, %arg1)
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3x!quant.uniform<u8:f32, 0.1>>, tensor<3x2xi32>) -> tensor<? x !quant.uniform<u8:f32, 0.1>>
  func.return %0#0 : tensor<? x !quant.uniform<u8:f32, 0.1>>
}

// CHECK-LABEL: testPadQuantizedI8
func.func @testPadQuantizedI8(%arg0: tensor<2x1x3x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<3x2xi32>) -> tensor<? x !quant.uniform<i8:f32, 0.1>> {
  // CHECK: "tfl.pad"(%arg0, %arg1)
  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3x!quant.uniform<i8:f32, 0.1>>, tensor<3x2xi32>) -> tensor<? x !quant.uniform<i8:f32, 0.1>>
  func.return %0#0 : tensor<? x !quant.uniform<i8:f32, 0.1>>
}
// -----

// CHECK-LABEL: testPadV2
func.func @testPadV2(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  // CHECK: "tfl.padv2"(%arg0, %arg1, %cst)
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// CHECK-LABEL: testPadV25D
func.func @testPadV25D(tensor<*xf32>, tensor<5x3xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<5x3xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  // CHECK: "tfl.padv2"(%arg0, %arg1, %cst)
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<*xf32>, tensor<5x3xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid paddings size
func.func @testPadV2WithInvalidPaddingsDim(tensor<2x1x3xf32>, tensor<2x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x2xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that operand 0's rank equals operand 1's size}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<2x2xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid paddings rank
func.func @testPadV2WithInvalidPaddingsRank(tensor<2x1x3xf32>, tensor<1x3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<1x3x2xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  // expected-error @+1 {{'tfl.padv2' op failed to verify that operand 1 is 2-D}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<1x3x2xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid constant rank
func.func @testPadV2WithInvalidConstantScalar(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = arith.constant dense<[2.0]> : tensor<1xf32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that operand 2 is 0-D}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<1xf32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

// test PadV2 with invalid constant data type
func.func @testPadV2WithInvalidConstantScalar(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = arith.constant dense<2> : tensor<i32>
  //// expected-error @+1 {{'tfl.padv2' op failed to verify that input and constant value operands must have same element type}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<i32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

func.func @testPadV2UnknownPaddings(tensor<2x1x3xf32>, tensor<*xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<*xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<*xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>

  // CHECK-LABEL: testPadV2UnknownPaddings
  // CHECK:  "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<*xi32>, tensor<f32>) -> tensor<?xf32>
  // CHECK:  return
}

// -----

func.func @testPadV2UnsupportedPaddings(tensor<*xf32>, tensor<6x3xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<6x3xi32>):
  %cst = arith.constant dense<2.0> : tensor<f32>
  // expected-error @+1 {{'tfl.padv2' op failed to verify that the first dim size of the padding argument must be at most 5}}
  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<*xf32>, tensor<6x3xi32>, tensor<f32>) -> tensor<? x f32>
  func.return %0#0 : tensor<? x f32>
}

// -----

func.func @packQuantizedU8(%arg0: tensor<2x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<2x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1>> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2x!quant.uniform<u8:f32, 0.1>>, tensor<2x!quant.uniform<u8:f32, 0.1>>) -> tensor<2x2x!quant.uniform<u8:f32, 0.1>>
  func.return %0 : tensor<2x2x!quant.uniform<u8:f32, 0.1>>
}

func.func @packQuantizedI8(%arg0: tensor<2x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<2x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1>> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2x!quant.uniform<i8:f32, 0.1>>, tensor<2x!quant.uniform<i8:f32, 0.1>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.1>>
}

// -----

func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @packUnranked(%arg0: tensor<2xi32>, %arg1: tensor<*xi32>) -> tensor<2x2xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2xi32>, tensor<*xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @packInputRank(%arg0: tensor<1x4xi32>, %arg1: tensor<1x4xi32>) -> tensor<1x4x2xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = 2 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 2 : i32, values_count = 2 : i32} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x4x2xi32>
  func.return %0 : tensor<1x4x2xi32>
}

// -----

func.func @packNegInputAxis2(%arg0: tensor<1x4xi32>, %arg1: tensor<1x4xi32>) -> tensor<1x2x4xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = -2 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = -2 : i32, values_count = 2 : i32} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x2x4xi32>
  func.return %0 : tensor<1x2x4xi32>
}

func.func @packNegInputAxis3(%arg0: tensor<1x4xi32>, %arg1: tensor<1x4xi32>) -> tensor<2x1x4xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = -3 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = -3 : i32, values_count = 2 : i32} : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<2x1x4xi32>
  func.return %0 : tensor<2x1x4xi32>
}

// -----

func.func @packInputUnranked(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: "tfl.pack"(%arg0, %arg1) {axis = -2 : i32, values_count = 2 : i32}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = -2 : i32, values_count = 2 : i32} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{input count should match 'values_count' attribute}}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 1 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @pack(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{operands should be of the same type}}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<1xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @pack(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{op attribute 'axis' should be in range [-rank - 1, rank + 1), got rank = 1, and axis = 3}}
  %0 = "tfl.pack"(%arg0, %arg1) {axis = 3 : i32, values_count = 2 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK: "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32}
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK: "tfl.unpack"(%arg0) {axis = -1 : i32, num = 3 : i32}
  %0:3 = "tfl.unpack"(%arg0) {axis = -1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<3xi32> {
  // CHECK: "tfl.unpack"(%arg0) {axis = -2 : i32, num = 2 : i32}
  %0:2 = "tfl.unpack"(%arg0) {axis = -2 : i32, num = 2 : i32} : (tensor<2x3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0#0 : tensor<3xi32>
}

// -----

func.func @unpackQuantized(%arg0: tensor<2x3x!quant.uniform<u8:f32, 0.02>>) -> tensor<2x!quant.uniform<u8:f32, 0.02>> {
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<2x3x!quant.uniform<u8:f32, 0.02>>) -> (tensor<2x!quant.uniform<u8:f32, 0.02>>, tensor<2x!quant.uniform<u8:f32, 0.02>>, tensor<2x!quant.uniform<u8:f32, 0.02>>)
  func.return %0#0 : tensor<2x!quant.uniform<u8:f32, 0.02>>

}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{output count should match 'num' attribute}}
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 2 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{attribute 'axis' should be in range [-rank, rank), got axis = 2, and rank = 2}}
  %0:3 = "tfl.unpack"(%arg0) {axis = 2 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{attribute 'axis' should be in range [-rank, rank), got axis = -3, and rank = 2}}
  %0:3 = "tfl.unpack"(%arg0) {axis = -3 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<i32>) -> tensor<2xi32> {
  // expected-error @+1 {{input should be of rank larger than 0}}
  %0:3 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 3 : i32} : (tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<2xi32>', 'tensor<2xi32>', 'tensor<2xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>', 'tensor<2x1xi32>', 'tensor<2xi32>'}}
  %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2x1xi32>, tensor<2xi32>)
  func.return %0#0 : tensor<2xi32>
}

// -----

func.func @unpack(%arg0: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>) {
  %0:2 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 2 : i32} : (tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  func.return %0#0, %0#1 : tensor<*xi32>, tensor<*xi32>
}

// -----

func.func @unpack(%arg0: tensor<?x2x5x3xf32>) -> () {
  %0:2 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 2 : i32} : (tensor<?x2x5x3xf32>) -> (tensor<5x5x3xf32>, tensor<5x5x3xf32>)
  func.return
}

// -----

// CHECK-LABEL: testMean
func.func @testMean(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // CHECK: "tfl.mean"(%arg0, %arg1) {keep_dims = false}
  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = false}: (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: testMean_true
func.func @testMean_true(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // CHECK: "tfl.mean"(%arg0, %arg1) {keep_dims = true}
  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = true}: (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

func.func @testMean_missing_keep_dims(%arg0: tensor<2x2xf32>, %arg1 : tensor<1xi32>) -> tensor<1x2xf32> {
  // expected-error @+1 {{'tfl.mean' op requires attribute 'keep_dims'}}
  %0 = "tfl.mean"(%arg0, %arg1): (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: testBatchToSpaceND
func.func @testBatchToSpaceND(%arg0 : tensor<4x2x2x3xf32>, %arg1 : tensor<2xi32>, %arg2 : tensor<2x2xi32>) -> tensor<?xf32> {
  // CHECK:  "tfl.batch_to_space_nd"(%arg0, %arg1, %arg2)
  %0 = "tfl.batch_to_space_nd"(%arg0, %arg1, %arg2) : (tensor<4x2x2x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testSpaceToBatchND
func.func @testSpaceToBatchND(%arg0 : tensor<1x4x4x3xf32>, %arg1 : tensor<2xi32>, %arg2 : tensor<2x2xi32>) -> tensor<?xf32> {
  // CHECK: "tfl.space_to_batch_nd"(%arg0, %arg1, %arg2)
  %0 = "tfl.space_to_batch_nd"(%arg0, %arg1, %arg2) : (tensor<1x4x4x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testBatchMatmulQuant(%arg0 : tensor<1x4x384x32x!quant.uniform<i8:f32, 0.06:-2>>, %arg1 : tensor<1x4x384x32x!quant.uniform<i8:f32, 0.11:-16>>) -> tensor<1x4x384x384x!quant.uniform<i8:f32, 1.02:-73>> {
  // CHECK: "tfl.batch_matmul"(%arg0, %arg1)
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = true} : (tensor<1x4x384x32x!quant.uniform<i8:f32, 0.06:-2>>, tensor<1x4x384x32x!quant.uniform<i8:f32, 0.11:-16>>) -> tensor<1x4x384x384x!quant.uniform<i8:f32, 1.02:-73>>
  func.return %0 : tensor<1x4x384x384x!quant.uniform<i8:f32, 1.02:-73>>
}

// -----

func.func @testBatchMatmulHybridQuant(%arg0 : tensor<1x4x384x32xf32>, %arg1 : tensor<1x4x384x32x!quant.uniform<i8:f32, 0.11:-16>>) -> tensor<1x4x384x384xf32> {
  // CHECK: "tfl.batch_matmul"(%arg0, %arg1)
  %0 = "tfl.batch_matmul"(%arg0, %arg1) {adj_x = false, adj_y = true} : (tensor<1x4x384x32xf32>, tensor<1x4x384x32x!quant.uniform<i8:f32, 0.11:-16>>) -> tensor<1x4x384x384xf32>
  func.return %0 : tensor<1x4x384x384xf32>
}

// -----

func.func @testConcat(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>) -> tensor<2x2xi32> {
  // CHECK: "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatQuantized(%arg0: tensor<1x2x!quant.uniform<i8:f32, 0.1:128>>, %arg1: tensor<1x2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>> {
  // CHECK: "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x!quant.uniform<i8:f32, 0.1:128>>, tensor<1x2x!quant.uniform<i8:f32, 0.1:128>>) -> tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
  func.return %0 : tensor<2x2x!quant.uniform<i8:f32, 0.1:128>>
}

// -----

func.func @testConcatInvalidOutputElementalType(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{'tfl.concatenation' op failed to verify that values and output must have same element type}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testConcatInvalidAxis(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{'tfl.concatenation' op concatenation dimension must be in [-rank, rank)}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatInvalidAxisUnderflow(%arg0: tensor<1x2xi32>, %arg1: tensor<1x2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{'tfl.concatenation' op concatenation dimension must be in [-rank, rank)}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = -4 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatInvalidOperandRankLess(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{'tfl.concatenation' op rank of operand #0 must be equal to rank of output, expected 2, got 1}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatInvalidOperandRankGreater(%arg0: tensor<1x1x2xi32>, %arg1: tensor<1x1x2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{'tfl.concatenation' op rank of operand #0 must be equal to rank of output, expected 2, got 3}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x1x2xi32>, tensor<1x1x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatInvalidOperandDimSize(%arg0: tensor<1x2xi32>, %arg1: tensor<1x3xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{'tfl.concatenation' op dimension size of dimension #1 of operand #1 must be equal to dimension size of dimension #1 of output, expected 2, got 3}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<1x3xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatInvalidOperandDimSizeComparedToPrevInput(%arg0: tensor<1x2xi32>, %arg1: tensor<1x3xi32>) -> tensor<?x?xi32> {
  // expected-error @+1 {{'tfl.concatenation' op dimension size of dimension #1 of operand #1 must be equal to dimension size of dimension #1 of operand #0, expected 2, got 3}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<1x3xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

func.func @testConcatInvalidScales(%arg0: tensor<*x!quant.uniform<i8:f32, 1.0>>, %arg1: tensor<*x!quant.uniform<i8:f32, 2.0>>) -> tensor<*x!quant.uniform<i8:f32, 1.0>> {
  // expected-error @+1 {{'tfl.concatenation' op quantization parameters violate the same scale constraint: !quant.uniform<i8:f32, 1.000000e+00> vs. !quant.uniform<i8:f32, 2.000000e+00>}}
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 3 : i32, fused_activation_function = "NONE"} : (tensor<*x!quant.uniform<i8:f32, 1.0>>, tensor<*x!quant.uniform<i8:f32, 2.0>>) -> tensor<*x!quant.uniform<i8:f32, 1.0>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 1.0>>
}

// -----

func.func @testConcatBenignUnrankedOperand(%arg0: tensor<*xi32>, %arg1: tensor<1x2xi32>) -> tensor<2x2xi32> {
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @testConcatBenignDynamicDimSizeOperand(%arg0: tensor<1x?xi32>, %arg1: tensor<?x2xi32>) -> tensor<2x2xi32> {
  %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x?xi32>, tensor<?x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: testResizeBilinear
func.func @testResizeBilinear(%arg0 : tensor<1x100x100x3xf32>, %arg1 : tensor<4xi32>) -> tensor<?xf32> {
  // CHECK: "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false, half_pixel_centers = false}
  %0 = "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false, half_pixel_centers = false} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testResizeBilinearInvalidOutputType(%arg0 : tensor<1x100x100x3xf32>, %arg1 : tensor<4xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{'tfl.resize_bilinear' op failed to verify that input and output must have same element type}}
  %0 = "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = false} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: testStridedSlice
func.func @testStridedSlice(%arg0: tensor<12x2x2x5xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5xf32> {
  // CHECK: "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  func.return %0 : tensor<1x2x2x5xf32>
}

// CHECK-LABEL: testStridedSliceWithQI8
func.func @testStridedSliceWithQI8(%arg0: tensor<12x2x2x5x!quant.uniform<i8:f32, 0.1>>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5x!quant.uniform<i8:f32, 0.1>>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<1x2x2x5x!quant.uniform<i8:f32, 0.1>>
}

// CHECK-LABEL: testStridedSliceWithQUI8
func.func @testStridedSliceWithQUI8(%arg0: tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!quant.uniform<u8:f32, 0.1>> {
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5x!quant.uniform<u8:f32, 0.1>>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!quant.uniform<u8:f32, 0.1>>
  func.return %0 : tensor<1x2x2x5x!quant.uniform<u8:f32, 0.1>>
}

// CHECK-LABEL: testStridedSliceTFType
func.func @testStridedSliceTFType(%arg0: tensor<12x2x2x5xui8>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.quint8> {
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xui8>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.quint8>
  func.return %0 : tensor<1x2x2x5x!tf_type.quint8>
}

// CHECK-LABEL: testStridedSliceWithString
func.func @testStridedSliceWithString(%arg0: tensor<12x2x2x5x!tf_type.string>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.string> {
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5x!tf_type.string>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5x!tf_type.string>
  func.return %0 : tensor<1x2x2x5x!tf_type.string>
}

// -----

func.func @testStridedSliceWithInvalidOutputType(%arg0: tensor<12x2x2x5xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5xi32> {
  // expected-error @+1 {{op failed to verify that input and output must have same element type}}
  %0 = "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xi32>
  func.return %0 : tensor<1x2x2x5xi32>
}

// -----

// CHECK-LABEL: testOneHot
func.func @testOneHot(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<*xf32> {
  // CHECK: "tfl.one_hot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i32} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  %0 = "tfl.one_hot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i32} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testOneHotWithInvalidOutputType(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<*xi16> {
  // expected-error @+1 {{'tfl.one_hot' op result #0 must be tensor of 32-bit float or 32-bit signless integer or 64-bit signless integer or 1-bit signless integer or 8-bit signless integer or 8-bit unsigned integer values, but got 'tensor<*xi16>'}}
  %0 = "tfl.one_hot"(%arg0, %arg1, %arg2, %arg3) {axis = -1 : i32} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<*xi16>
  func.return %0 : tensor<*xi16>
}

// -----

func.func @testArgMax(%arg0: tensor<3xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "tfl.arg_max"(%arg0, %arg1) {output_type = 2 : i32} : (tensor<3xi32>, tensor<i32>) -> tensor<i32>
  %0 = "tfl.arg_max"(%arg0, %arg1) {output_type = 2 : i32} : (tensor<3xi32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @testArgMin(%arg0: tensor<3xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "tfl.arg_min"(%arg0, %arg1) {output_type = 2 : i32} : (tensor<3xi32>, tensor<i32>) -> tensor<i32>
  %0 = "tfl.arg_min"(%arg0, %arg1) {output_type = 2 : i32} : (tensor<3xi32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: testSpaceToDepth
func.func @testSpaceToDepthF32(%arg0: tensor<1x2x2x1xf32>) -> tensor<1x1x1x4xf32> {
  // CHECK: %[[ARG:.*]]: tensor<1x2x2x1xf32>
  // CHECK: "tfl.space_to_depth"(%[[ARG]]) {block_size = 2 : i32} : (tensor<1x2x2x1xf32>) -> tensor<1x1x1x4xf32>
  %0 = "tfl.space_to_depth"(%arg0) {block_size = 2: i32} : (tensor<1x2x2x1xf32>) -> tensor<1x1x1x4xf32>
  func.return %0 : tensor<1x1x1x4xf32>
}

// -----

func.func @testSpaceToDepthInvalidOutputType(%arg0: tensor<1x2x2x1xf32>) -> tensor<1x1x1x4xi32> {
  // expected-error @+1 {{'tfl.space_to_depth' op failed to verify that input and output must have same element type}}
  %0 = "tfl.space_to_depth"(%arg0) {block_size = 2: i32} : (tensor<1x2x2x1xf32>) -> tensor<1x1x1x4xi32>
  func.return %0 : tensor<1x1x1x4xi32>
}

// -----

func.func @testRange(%arg0 : tensor<i32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<?xi32> {
  %0 = "tfl.range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

func.func @testRangeNonScalarTensorInput(%arg0 : tensor<1xi32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<?xi32> {
  // expected-error @+1 {{op failed to verify that operand 0 is 0-D}}
  %0 = "tfl.range"(%arg0, %arg1, %arg2) : (tensor<1xi32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

func.func @testRangeOutputTypeMismatch(%arg0 : tensor<i32>, %arg1 : tensor<i32>, %arg2 : tensor<i32>) -> tensor<?xf32> {
  // expected-error @+1 {{op failed to verify that operands and output must have same element type}}
  %0 = "tfl.range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @transpose(%arg0 : tensor<2x2xi32>, %arg1 : tensor<2xi32>) -> tensor<2x2xi32> {
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: transpose_with_output_that_has_dynamic_sizes
func.func @transpose_with_output_that_has_dynamic_sizes(%arg0 : tensor<2x2xi32>, %arg1 : tensor<2xi32>) -> tensor<?x?xi32> {
  // CHECK: "tfl.transpose"(%arg0, %arg1)
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

func.func @transpose_perm_not_i32(%arg0 : tensor<2x2xi32>, %arg1 : tensor<2xf32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{op operand #1 must be tensor of 32-bit signless integer values}}
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2xf32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_perm_size(%arg0 : tensor<2x2xi32>, %arg1 : tensor<3xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{perm tensor elements size is not equal to input tensor rank}}
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<3xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_unranked_shape(%arg0 : tensor<*xi32>) -> tensor<2x2xi32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<*xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_dynamic_shape(%arg0 : tensor<2x?xi32>) -> tensor<?x2xi32> {
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<2x?xi32>, tensor<2xi32>) -> tensor<?x2xi32>
  func.return %0 : tensor<?x2xi32>
}


// -----

func.func @transpose_perm_axis_invalid(%arg0 : tensor<2x2xi32>) -> tensor<2x2xi32> {
  %cst = arith.constant dense<[1, -1]> : tensor<2xi32>
  // expected-error @+1 {{perm[1] must be in [0, rank)}}
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_perm_axis_duplicated(%arg0 : tensor<2x2xi32>) -> tensor<2x2xi32> {
  %cst = arith.constant dense<[1, 1]> : tensor<2xi32>
  // expected-error @+1 {{perm[1] cannot have duplicated axis}}
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_output_type_bad(%arg0 : tensor<3x4x5x6xi32>) -> tensor<3x4x5x6xi32> {
  %cst = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  // expected-error @+1 {{expect output type tensor<3x6x4x5xi32>, got tensor<3x4x5x6xi32>}}
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<3x4x5x6xi32>, tensor<4xi32>) -> tensor<3x4x5x6xi32>
  func.return %0 : tensor<3x4x5x6xi32>
}


// -----

func.func @transpose_element_type(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{input and output must have same element type}}
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// -----

func.func @transpose_1d_perm(%arg0 : tensor<2x2xi32>, %arg1 : tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error @+1 {{op failed to verify that operand 1 is 1-D}}
  %0 = "tfl.transpose"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

func.func @transpose_uniform_qtype(%arg0 : tensor<1x3x4x3xf32>) -> tensor<3x4x3x1x!quant.uniform<i8:f32, 0.0078356266021728515:-1>> {
  %cst = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi32>
  %0 = "tfl.quantize"(%arg0) {qtype = tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0078356266021728515:-1>>} : (tensor<1x3x4x3xf32>) -> tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0078356266021728515:-1>>
  // CHECK: "tfl.transpose"
  %1 = "tfl.transpose"(%0, %cst) : (tensor<1x3x4x3x!quant.uniform<i8:f32, 0.0078356266021728515:-1>>, tensor<4xi32>) -> tensor<3x4x3x1x!quant.uniform<i8:f32, 0.0078356266021728515:-1>>
  func.return %1 : tensor<3x4x3x1x!quant.uniform<i8:f32, 0.0078356266021728515:-1>>
}

// -----

func.func @transpose_uniform_per_axis_qtype(%arg0 : tensor<2x1x1x3x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.072314441204071045,0.050758145749568939}>> {
  %cst = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi32>
  // CHECK: "tfl.transpose"
  %0  = "tfl.transpose"(%arg0, %cst) : (tensor<2x1x1x3x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>, tensor<4xi32>) -> tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.072314441204071045,0.050758145749568939}>>
  func.return %0 : tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:3, {0.072314441204071045,0.050758145749568939}>>
}

// -----

func.func @transpose_uniform_per_axis_qtype_mismatch_axis(%arg0 : tensor<2x1x1x3x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>> {
  %cst = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi32>
  // expected-error @+1 {{op has mismatched quantized axes of input and output}}
  %0  = "tfl.transpose"(%arg0, %cst) : (tensor<2x1x1x3x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>, tensor<4xi32>) -> tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>
  func.return %0 : tensor<1x1x3x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>
}

// -----

func.func @anyWithI64Axis(%arg0: tensor<2x2xi1>, %arg1: tensor<i64>) -> tensor<i1> {
  // expected-error @+1 {{tfl.reduce_any' op operand #1 must be tensor of 32-bit signless integer values}}
  %0 = "tfl.reduce_any"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi1>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @testRoundInvalidInputType(%arg: tensor<?xi32>) -> tensor<?xi32> {
  // expected-error @+1 {{'tfl.round' op operand #0 must be tensor of 32-bit float values}}
  %0 = "tfl.round"(%arg) : (tensor<?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// -----

func.func @testSplitWithQuantizedTypes(%arg0 : tensor<i32>, %arg1 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.split"(%arg0, %arg1) {num_splits = 1 : i32} : (tensor<i32>, tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @testSplitVWithQuantizedTypes(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>, %arg1 : tensor<1xi32>, %arg2 : tensor<i32>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.split_v"(%arg0, %arg1, %arg2) {num_splits = 1 : i32} : (tensor<10x!quant.uniform<u8:f32, 1.0>>, tensor<1xi32>, tensor<i32>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @whereWithI32Input(%arg0: tensor<3x5xf64>) -> tensor<?x2xi64> {
  // expected-error @+1 {{'tfl.where' op operand #0 must be tensor of 1-bit signless integer}}
  %0 = "tfl.where"(%arg0) : (tensor<3x5xf64>) -> tensor<?x2xi64>
  func.return %0 : tensor<?x2xi64>
}

// -----

func.func @testMinimumWithQuantizedTypes(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>, %arg1 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<10x!quant.uniform<u8:f32, 1.0>>, tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @testMaximumWithQuantizedTypes(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>, %arg1 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<10x!quant.uniform<u8:f32, 1.0>>, tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @testReluWithQuantizedTypes(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.relu"(%arg0) : (tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @testRelu6WithQuantizedTypes(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>> {
  %0 = "tfl.relu6"(%arg0) : (tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 1.0>>
  func.return %0 : tensor<10x!quant.uniform<u8:f32, 1.0>>
}

// -----

func.func @testReluWithDifferentScales(%arg0 : tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 4.0>> {
  %0 = "tfl.relu"(%arg0) : (tensor<10x!quant.uniform<u8:f32, 1.0>>) -> tensor<10x!quant.uniform<u8:f32, 2.0>>
  %1 = "tfl.relu_n1_to_1"(%0) : (tensor<10x!quant.uniform<u8:f32, 2.0>>) -> tensor<10x!quant.uniform<u8:f32, 3.0>>
  %2 = "tfl.relu6"(%1) : (tensor<10x!quant.uniform<u8:f32, 3.0>>) -> tensor<10x!quant.uniform<u8:f32, 4.0>>
  func.return %2 : tensor<10x!quant.uniform<u8:f32, 4.0>>
}

// -----

func.func @testEmbeddingLookup(%arg0 : tensor<?xi32>, %arg1 : tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = "tfl.embedding_lookup"(%arg0, %arg1) : (tensor<?xi32>,tensor<?x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testEmbeddingLookupValueAndResultElementTypeTraitFailed(%arg0 : tensor<?xi32>, %arg1 : tensor<?x?xi8>) -> tensor<?xf32> {
  // expected-error @+1 {{'tfl.embedding_lookup' op failed to verify that value and output must have same element type}}
  %0 = "tfl.embedding_lookup"(%arg0, %arg1) : (tensor<?xi32>,tensor<?x?xi8>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testWrongQuantizedLocalResponseNormalization(%arg0 : tensor<1x56x56x192x!quant.uniform<u8:f32, 0.02>>) -> tensor<1x56x56x192x!quant.uniform<u8:f32, 0.02>> {
  // expected-error @+1 {{'tfl.local_response_normalization' op operand #0 must be tensor of 32-bit float values, but got 'tensor<1x56x56x192x!quant.uniform<u8:f32, 2.000000e-02>>'}}
  %0 = "tfl.local_response_normalization"(%arg0) {alpha = 9.99999974E-5 : f32, beta = 5.000000e-01 : f32, bias = 2.000000e+00 : f32, radius = 5 : i32} : (tensor<1x56x56x192x!quant.uniform<u8:f32, 0.02>>) -> tensor<1x56x56x192x!quant.uniform<u8:f32, 0.02>>
  func.return %0 : tensor<1x56x56x192x!quant.uniform<u8:f32, 0.02>>
}

// -----

// CHECK-LABEL: testSvdf
func.func @testSvdf(%arg0: tensor<? x f32>, %arg1: tensor<? x f32>, %arg2: tensor<? x f32>, %arg3: tensor<? x f32>, %arg4: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.svdf"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "RELU", rank = 2 : i32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.svdf"(%arg0, %arg1, %arg2, %arg3, %arg4) {fused_activation_function = "RELU", rank = 2 : i32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testDepthToSpace
func.func @testDepthToSpaceF32(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32> {
  // CHECK: %[[ARG:.*]]: tensor<1x1x1x4xf32>
  // CHECK: "tfl.depth_to_space"(%[[ARG]]) {block_size = 2 : i32} : (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
  %0 = "tfl.depth_to_space"(%arg0) {block_size = 2: i32} : (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
  func.return %0 : tensor<1x2x2x1xf32>
}

// -----

func.func @testDepthToSpaceInvalidOutputType(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xi32> {
  // expected-error @+1 {{'tfl.depth_to_space' op failed to verify that input and output must have same element type}}
  %0 = "tfl.depth_to_space"(%arg0) {block_size = 2: i32} : (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xi32>
  func.return %0 : tensor<1x2x2x1xi32>
}

// -----

func.func @testPReluWrongOutputRank(%arg0: tensor<10x10x10x10xf32>, %arg1: tensor<10x10x10x10xf32>) -> tensor<10x10xf32> {
  // expected-error @+1 {{'tfl.prelu' op result type '10x10' not broadcast compatible with broadcasted operands's shapes '10x10x10x10'}}
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<10x10x10x10xf32>, tensor<10x10x10x10xf32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}

// -----

func.func @testPReluWrongOutputShape(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<1x2x3x5xf32> {
  // expected-error @+1 {{'tfl.prelu' op result type '1x2x3x5' not broadcast compatible with broadcasted operands's shapes '1x2x3x4'}}
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<1x2x3x5xf32>
  func.return %0 : tensor<1x2x3x5xf32>
}

// -----

func.func @testPReluWrongAlphaRank(%arg0: tensor<7x3x2x14xf32>, %arg1: tensor<7x3x2x14xf32>) -> tensor<7x3x2x14xf32> {
  // expected-error @+1 {{'alpha' should have one less rank than 'input'.}}
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<7x3x2x14xf32>, tensor<7x3x2x14xf32>) -> tensor<7x3x2x14xf32>
  func.return %0 : tensor<7x3x2x14xf32>
}

// -----

func.func @testPReluInvalidBroadcast(%arg0: tensor<15x14x2x14xf32>, %arg1: tensor<1x1x3xf32>) -> tensor<15x14x2x14xf32> {
  // expected-error @+1 {{'tfl.prelu' op operands don't have broadcast-compatible shapes}}
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<15x14x2x14xf32>, tensor<1x1x3xf32>) -> tensor<15x14x2x14xf32>
  func.return %0 : tensor<15x14x2x14xf32>
}

// -----

func.func @testPReluValidSameSize(%arg0: tensor<16x20x20x13xf32>, %arg1: tensor<20x20x13xf32>) -> tensor<16x20x20x13xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<16x20x20x13xf32>, tensor<20x20x13xf32>) -> tensor<16x20x20x13xf32>
  func.return %0 : tensor<16x20x20x13xf32>
}

// -----

func.func @testPReluValidBroadcast(%arg0: tensor<19x7x12x14xf32>, %arg1: tensor<1x1x14xf32>) -> tensor<19x7x12x14xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<19x7x12x14xf32>, tensor<1x1x14xf32>) -> tensor<19x7x12x14xf32>
  func.return %0 : tensor<19x7x12x14xf32>
}

// -----

func.func @testPReluValidFullBroadcast(%arg0: tensor<7x8x9x10xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<7x8x9x10xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<7x8x9x10xf32>, tensor<1x1x1xf32>) -> tensor<7x8x9x10xf32>
  func.return %0 : tensor<7x8x9x10xf32>
}

// -----

func.func @testPReluValidQuantized(%arg0: tensor<1x96x96x16x!quant.uniform<u8:f32, 0.00784:128>>, %arg1: tensor<1x1x16x!quant.uniform<u8<1:255>:f32, 0.004846:14>>) -> tensor<1x96x96x16x!quant.uniform<u8:f32, 0.00784:128>> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<1x96x96x16x!quant.uniform<u8:f32, 0.00784:128>>, tensor<1x1x16x!quant.uniform<u8<1:255>:f32, 0.004846:14>>) -> tensor<1x96x96x16x!quant.uniform<u8:f32, 0.00784:128>>
  func.return %0 : tensor<1x96x96x16x!quant.uniform<u8:f32, 0.00784:128>>
}

// -----

func.func @testSlice(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceBadBeginDimension(%arg0: tensor<2x3x5xf32>, %arg1: tensor<2xi32>, %arg2: tensor<3xi32>) -> tensor<?x3x5xf32> {
  // expected-error @+1 {{begin tensor elements size is not equal to input tensor rank}}
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<2xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceBadSizeDimension(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>, %arg2: tensor<2xi32>) -> tensor<?x3x5xf32> {
  // expected-error @+1 {{size tensor elements size is not equal to input tensor rank}}
  %0 = "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<2xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceBadBegin(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %cst = arith.constant dense<[2, -1, 5]> : tensor<3xi32>
  // expected-error @+1 {{begin[1] cannot be negative}}
  %0 = "tfl.slice"(%arg0, %cst, %arg1) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceNegativeSize(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %cst = arith.constant dense<[-2, -1, 5]> : tensor<3xi32>
  // expected-error @+1 {{size[0] cannot be negative other than -1}}
  %0 = "tfl.slice"(%arg0, %arg1, %cst) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceSizeOutOfRange(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %cst = arith.constant dense<[2, 1, 5]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[0, 1, 1]> : tensor<3xi32>
  // expected-error @+1 {{begin[2] + size[2] cannot exceed dimension length: 5}}
  %0 = "tfl.slice"(%arg0, %cst_1, %cst) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSliceBeginOutOfRange(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %cst = arith.constant dense<[1, 1, 1]> : tensor<3xi32>
  %cst_1 = arith.constant dense<[3, 1, 3]> : tensor<3xi32>
  // expected-error @+1 {{begin[0] cannot exceed dimension length: 2}}
  %0 = "tfl.slice"(%arg0, %cst_1, %cst) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  func.return %0 : tensor<?x3x5xf32>
}

// -----

func.func @testSplitOpWithBadNumSplits(%arg0 : tensor<16xf32>) -> () {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op attribute 'num_splits' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  "tfl.split"(%split_dim, %arg0) {num_splits = 0 : i32} : (tensor<i32>, tensor<16xf32>) -> ()
  func.return
}

// -----

func.func @testSplitOpWithMismatchedNumResults(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op output count should match 'num_splits' attribute}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 4 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitOpWithBadSplitDimTensorType(%arg0: tensor<16x4x4xf32>) -> tensor<16x4x4xf32> {
  %split_dim = arith.constant dense<0> : tensor<2x2xi32>
  // expected-error @+1 {{'tfl.split' op failed to verify that operand #0 is an 0-d tensor or 1-d tensor w/ 1 element}}
  %0 = "tfl.split"(%split_dim, %arg0) {num_splits = 1 : i32} : (tensor<2x2xi32>, tensor<16x4x4xf32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitOpWithBadSplitDimUnrankedTensorType(%arg0: tensor<16x4x4xf32>, %split_dim : tensor<*xi32>) -> tensor<16x4x4xf32> {
  // expected-error @+1 {{'tfl.split' op failed to verify that operand #0 is an 0-d tensor or 1-d tensor w/ 1 element}}
  %0 = "tfl.split"(%split_dim, %arg0) {num_splits = 1 : i32} : (tensor<*xi32>, tensor<16x4x4xf32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitOpWithOutOfRangeSplitDim(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %split_dim = arith.constant dense<1> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitOpWithOutOfRangeSplitDimTFLConst(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %split_dim = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{'tfl.split' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitOpWithOutOfRangeSplitDimNegative(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %split_dim = arith.constant dense<-2> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitOpWithUnevenDivision(%arg0 : tensor<16xf32>) -> (tensor<6xf32>, tensor<5xf32>, tensor<5xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op 'num_splits' should evenly divide 'split_dim' axis}}
  %0, %1, %2 = "tfl.split"(%split_dim, %arg0) {num_splits = 3 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<6xf32>, tensor<5xf32>, tensor<5xf32>)
  func.return %0, %1, %2 : tensor<6xf32>, tensor<5xf32>, tensor<5xf32>
}

// -----

func.func @testSplitOpWithMismatchTensorTypeSplitDimOut0(%arg0 : tensor<16xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op output #0 should be 'tensor<8xf32>'}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @testSplitOpWithMismatchTensorTypeSplitDimOut1(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op output #1 should be 'tensor<8xf32>'}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16xf32>) -> (tensor<8xf32>, tensor<4xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<4xf32>
}

// -----

func.func @testSplitOpWithMismatchTensorTypeNonSplitDim(%arg0 : tensor<16x4xf32>) -> (tensor<8x2xf32>, tensor<8x2xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split' op output #0 should be 'tensor<8x4xf32>'}}
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16x4xf32>) -> (tensor<8x2xf32>, tensor<8x2xf32>)
  func.return %0, %1 : tensor<8x2xf32>, tensor<8x2xf32>
}

// -----

// CHECK-LABEL:testSplitOpWithValidTensorType
func.func @testSplitOpWithValidTensorType(%arg0 : tensor<16x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<16x2xf32>, tensor<16x2xf32>, tensor<16x2xf32>) {
  %split_dim_0 = arith.constant dense<0> : tensor<i32>
  %0, %1 = "tfl.split"(%split_dim_0, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>)
  %split_dim_1 = arith.constant dense<1> : tensor<i32>
  %2, %3 = "tfl.split"(%split_dim_1, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16x4xf32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  %split_dim_2 = arith.constant dense<1> : tensor<1xi32>
  %4, %5 = "tfl.split"(%split_dim_2, %arg0) {num_splits = 2 : i32} : (tensor<1xi32>, tensor<16x4xf32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  %6:2 = "tfl.split"(%split_dim_2, %arg0) {num_splits = 2 : i32} : (tensor<1xi32>, tensor<16x4xf32>) -> (tensor<16x2xf32>, tensor<16x?xf32>)
  %7:2 = "tfl.split"(%split_dim_2, %arg0) {num_splits = 2 : i32} : (tensor<1xi32>, tensor<16x4xf32>) -> (tensor<?x2xf32>, tensor<16x?xf32>)
  %8:2 = "tfl.split"(%split_dim_2, %arg0) {num_splits = 2 : i32} : (tensor<1xi32>, tensor<16x4xf32>) -> (tensor<16x2xf32>, tensor<*xf32>)
  func.return %0, %1, %2, %3, %4 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<16x2xf32>, tensor<16x2xf32>, tensor<16x2xf32>
}

// -----

func.func @testSplitOpWithValidTensorTypeDynamic(%arg0 : tensor<16x?xf32>) -> (tensor<8x?xf32>, tensor<8x?xf32>) {
  %split_dim = arith.constant dense<0> : tensor<i32>
  %0, %1 = "tfl.split"(%split_dim, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<16x?xf32>) -> (tensor<8x?xf32>, tensor<8x?xf32>)
  func.return %0, %1 : tensor<8x?xf32>, tensor<8x?xf32>
}

// -----

func.func @testSplitVOpWithBadNumSplits(%arg0 : tensor<16xf32>) -> () {
  %size_splits = arith.constant dense<[]> : tensor<0xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op attribute 'num_splits' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 0 : i32} : (tensor<16xf32>, tensor<0xi32>, tensor<i32>) -> ()
  func.return
}

// -----

func.func @testSplitVOpWithMismatchedNumResults(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %size_splits = arith.constant dense<[4, 4, 4, 4]> : tensor<4xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op output count should match 'num_splits' attribute}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 4 : i32} : (tensor<16xf32>, tensor<4xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsTensorType(%arg0: tensor<16x4x4xf32>) -> tensor<16x4x4xf32> {
  %size_splits = arith.constant dense<[[8, 8], [2, 2]]> : tensor<2x2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op operand #1 must be 1D tensor of 32-bit signless integer values}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<2x2xi32>, tensor<i32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsUnrankedTensorType(%arg0: tensor<16x4x4xf32>, %size_splits: tensor<*xi32>) -> tensor<16x4x4xf32> {
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op operand #1 must be 1D tensor of 32-bit signless integer values}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<*xi32>, tensor<i32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsConstant(%arg0: tensor<16x4x4xf32>) -> tensor<16x4x4xf32> {
  %size_splits = arith.constant dense<[-2]> : tensor<1xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op elements of 'size_splits' should be greater than or equal to -1}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<1xi32>, tensor<i32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsConstantMultipleNegativeOne(%arg0: tensor<16x4x4xf32>) -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<14x4x4xf32>) {
  %size_splits = arith.constant dense<[-1, -1, 14]> : tensor<3xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op 'size_splits' can only have one -1}}
  %0, %1, %2 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 3 : i32} : (tensor<16x4x4xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<14x4x4xf32>)
  func.return %0, %1, %2 : tensor<1x4x4xf32>, tensor<1x4x4xf32>, tensor<14x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsConstantSum(%arg0: tensor<16x4x4xf32>) -> (tensor<0x4x4xf32>, tensor<16x4x4xf32>) {
  %size_splits = arith.constant dense<[-1, 17]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op sum of non-negative elements of 'size_splits' is greater than the dimension size of 'split_dim' axis}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16x4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<0x4x4xf32>, tensor<16x4x4xf32>)
  func.return %0, %1 : tensor<0x4x4xf32>, tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSizeSplitsSize(%arg0: tensor<16x4x4xf32>) -> tensor<15x4x4xf32> {
  %size_splits = arith.constant dense<[15, 1]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op 'size_splits' should be 'tensor<1xi32>'}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<2xi32>, tensor<i32>) -> tensor<15x4x4xf32>
  func.return %0 : tensor<15x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSplitDimTensorType(%arg0: tensor<16x4x4xf32>) -> tensor<16x4x4xf32> {
  %size_splits = arith.constant dense<[16]> : tensor<1xi32>
  %split_dim = arith.constant dense<0> : tensor<2x2xi32>
  // expected-error @+1 {{'tfl.split_v' op operand #2 must be 0D tensor of 32-bit signless integer values}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<1xi32>, tensor<2x2xi32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithBadSplitDimUnrankedTensorType(%arg0: tensor<16x4x4xf32>, %split_dim : tensor<*xi32>) -> tensor<16x4x4xf32> {
  %size_splits = arith.constant dense<[16]> : tensor<1xi32>
  // expected-error @+1 {{'tfl.split_v' op operand #2 must be 0D tensor of 32-bit signless integer values}}
  %0 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 1 : i32} : (tensor<16x4x4xf32>, tensor<1xi32>, tensor<*xi32>) -> tensor<16x4x4xf32>
  func.return %0 : tensor<16x4x4xf32>
}

// -----

func.func @testSplitVOpWithOutOfRangeSplitDim(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<1> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitVOpWithOutOfRangeSplitDimTFLConst(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitVOpWithOutOfRangeSplitDimNegative(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<-2> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op 'split_dim' should be in [-rank, rank)}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @testSplitVOpWithMismatchSizeSplitsSum(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
  %size_splits = arith.constant dense<[8, 4]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op sum of 'size_splits' should match the dimension size of 'split_dim' axis}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<4xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<4xf32>
}

// -----

func.func @testSplitVOpWithMismatchTensorTypeSplitDimOut0(%arg0 : tensor<16xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op output #0 should be 'tensor<8xf32>'}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<4xf32>, tensor<4xf32>)
  func.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @testSplitVOpWithMismatchTensorTypeSplitDimOut1(%arg0 : tensor<16xf32>) -> (tensor<8xf32>, tensor<4xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op output #1 should be 'tensor<8xf32>'}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8xf32>, tensor<4xf32>)
  func.return %0, %1 : tensor<8xf32>, tensor<4xf32>
}

// -----

func.func @testSplitVOpWithMismatchTensorTypeNonSplitDim(%arg0 : tensor<16x4xf32>) -> (tensor<8x2xf32>, tensor<8x2xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  // expected-error @+1 {{'tfl.split_v' op output #0 should be 'tensor<8x4xf32>'}}
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8x2xf32>, tensor<8x2xf32>)
  func.return %0, %1 : tensor<8x2xf32>, tensor<8x2xf32>
}

// -----

func.func @testSplitVOpWithValidTensorType(%arg0 : tensor<16x4xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<16x2xf32>, tensor<16x2xf32>) {
  %size_splits_0 = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim_0 = arith.constant dense<0> : tensor<i32>
  %0, %1 = "tfl.split_v"(%arg0, %size_splits_0, %split_dim_0) {num_splits = 2 : i32} : (tensor<16x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8x4xf32>, tensor<8x4xf32>)

  %size_splits_1 = arith.constant dense<[2, 2]> : tensor<2xi32>
  %split_dim_1 = arith.constant dense<1> : tensor<i32>
  %2, %3 = "tfl.split_v"(%arg0, %size_splits_1, %split_dim_1) {num_splits = 2 : i32} : (tensor<16x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)

  func.return %0, %1, %2, %3 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<16x2xf32>, tensor<16x2xf32>
}

// -----

func.func @testSplitVOpWithValidTensorTypeDynamic(%arg0 : tensor<16x?xf32>) -> (tensor<8x?xf32>, tensor<8x?xf32>) {
  %size_splits = arith.constant dense<[8, 8]> : tensor<2xi32>
  %split_dim = arith.constant dense<0> : tensor<i32>
  %0, %1 = "tfl.split_v"(%arg0, %size_splits, %split_dim) {num_splits = 2 : i32} : (tensor<16x?xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<8x?xf32>, tensor<8x?xf32>)
  func.return %0, %1 : tensor<8x?xf32>, tensor<8x?xf32>
}

// -----

func.func @testSplitVOpWithValidSizeSplitsUneven(%arg0 : tensor<16x4xf32>) -> (tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>, tensor<16x1xf32>, tensor<16x3xf32>) {
  %size_splits_0 = arith.constant dense<[7, 3, 6]> : tensor<3xi32>
  %split_dim_0 = arith.constant dense<0> : tensor<i32>
  %0, %1, %2 = "tfl.split_v"(%arg0, %size_splits_0, %split_dim_0) {num_splits = 3 : i32} : (tensor<16x4xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>)

  %size_splits_1 = arith.constant dense<[1, 3]> : tensor<2xi32>
  %split_dim_1 = arith.constant dense<1> : tensor<i32>
  %3, %4 = "tfl.split_v"(%arg0, %size_splits_1, %split_dim_1) {num_splits = 2 : i32} : (tensor<16x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<16x1xf32>, tensor<16x3xf32>)

  func.return %0, %1, %2, %3, %4 : tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>, tensor<16x1xf32>, tensor<16x3xf32>
}

// -----

func.func @testSplitVOpWithValidSizeSplitsNegative(%arg0 : tensor<16x4xf32>) -> (tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>, tensor<16x0xf32>, tensor<16x4xf32>) {
  %size_splits_0 = arith.constant dense<[7, -1, 6]> : tensor<3xi32>
  %split_dim_0 = arith.constant dense<0> : tensor<i32>
  %0, %1, %2 = "tfl.split_v"(%arg0, %size_splits_0, %split_dim_0) {num_splits = 3 : i32} : (tensor<16x4xf32>, tensor<3xi32>, tensor<i32>) -> (tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>)

  %size_splits_1 = arith.constant dense<[-1, 4]> : tensor<2xi32>
  %split_dim_1 = arith.constant dense<1> : tensor<i32>
  %3, %4 = "tfl.split_v"(%arg0, %size_splits_1, %split_dim_1) {num_splits = 2 : i32} : (tensor<16x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<16x0xf32>, tensor<16x4xf32>)

  func.return %0, %1, %2, %3, %4 : tensor<7x4xf32>, tensor<3x4xf32>, tensor<6x4xf32>, tensor<16x0xf32>, tensor<16x4xf32>
}

// -----

func.func @testNonMaxSuppressionV4WithCorrectBoxShape(%arg0: tensor<3x4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> (tensor<2xi32>, tensor<i32>) {
  %0, %1 = "tfl.non_max_suppression_v4"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<3x4xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>, tensor<i32>)
  func.return %0, %1 : tensor<2xi32>, tensor<i32>
}

// -----

func.func @testNonMaxSuppressionV4WithWrongBoxShape(%arg0: tensor<3x2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>) -> (tensor<2xi32>, tensor<i32>) {
  // expected-error @+1 {{'tfl.non_max_suppression_v4' op failed to verify that boxes should have dim[1] == 4}}
  %0, %1 = "tfl.non_max_suppression_v4"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<3x2xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>, tensor<i32>)
  func.return %0, %1 : tensor<2xi32>, tensor<i32>
}

// -----

func.func @testNonMaxSuppressionV5WithCorrectBoxShape(%arg0: tensor<3x4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>) -> (tensor<2xi32>, tensor<2xf32>, tensor<i32>) {
  %0, %1, %2 = "tfl.non_max_suppression_v5"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<3x4xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>, tensor<2xf32>, tensor<i32>)
  func.return %0, %1, %2 : tensor<2xi32>, tensor<2xf32>, tensor<i32>
}

// -----

func.func @testNonMaxSuppressionV5WithWrongBoxShape(%arg0: tensor<3x2xf32>, %arg1: tensor<3xf32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>) -> (tensor<2xi32>, tensor<2xf32>, tensor<i32>) {
  // expected-error @+1 {{'tfl.non_max_suppression_v5' op failed to verify that boxes should have dim[1] == 4}}
  %0, %1, %2 = "tfl.non_max_suppression_v5"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (tensor<3x2xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>, tensor<2xf32>, tensor<i32>)
  func.return %0, %1, %2 : tensor<2xi32>, tensor<2xf32>, tensor<i32>
}

// -----

func.func @fully_connected(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>, %arg2: tensor<40xf32>) -> tensor<1x40xf32> {
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<1x40xf32>
  func.return %0 : tensor<1x40xf32>
}

// -----

func.func @fully_connected_no_bias(%arg0: tensor<2x2x10xf32>, %arg1: tensor<40x40xf32>, %arg2: none) -> tensor<1x40xf32> {
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x2x10xf32>, tensor<40x40xf32>, none) -> tensor<1x40xf32>
  func.return %0 : tensor<1x40xf32>
}

// -----

func.func @testFullyConnectedWith3DFilter(%arg0: tensor<1x37xf32>, %arg1: tensor<40x2x37xf32>, %arg2: tensor<40xf32>) -> tensor<1x40xf32> {
  // expected-error @+1 {{expect 2d filter, got 'tensor<40x2x37xf32>'}}
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x2x37xf32>, tensor<40xf32>) -> tensor<1x40xf32>
  func.return %0 : tensor<1x40xf32>
}

// -----

func.func @testFullyConnectedWithBadInputShape(%arg0: tensor<2x2x11xf32>, %arg1: tensor<40x40xf32>, %arg2: none) -> tensor<40xf32> {
  // expected-error @+1 {{expect 'input' num_elements % 40 == 0, got input type 'tensor<2x2x11xf32>'}}
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x2x11xf32>, tensor<40x40xf32>, none) -> tensor<1x40xf32>
  func.return %0 : tensor<1x40xf32>
}

// -----

func.func @testFullyConnectedWithBadBatch(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>, %arg2: tensor<40xf32>) -> tensor<2x40xf32> {
  // expected-error @+1 {{num_input_elements / z_in != num_output_elements / z_out}}
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<2x40xf32>
  func.return %0 : tensor<2x40xf32>
}

// -----

func.func @testFullyConnectedWithBadOutputShape(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>, %arg2: tensor<40xf32>) -> tensor<1x41xf32> {
  // expected-error @+1 {{expect 'output' num_elements % 40 == 0, got 'tensor<1x41xf32>'}}
  %0 = "tfl.fully_connected"(%arg0, %arg1, %arg2) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<1x41xf32>
  func.return %0 : tensor<1x41xf32>
}

// -----

func.func @testTransposeConv(%arg0: tensor<4xi32>, %arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  func.return %0 : tensor<1x64x84x32xf32>
}

// -----

// CHECK-LABEL: testTransposeConvWithOutputThatHasDynamicSizes
func.func @testTransposeConvWithOutputThatHasDynamicSizes(%arg0: tensor<4xi32>, %arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[NONE:.*]] = "tfl.no_value"() {value} : () -> none
  // CHECK: "tfl.transpose_conv"(%arg0, %arg1, %arg2, %[[NONE]])
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @testConvolution2DTransposeBias(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<1x32x42x128xf32>, %arg2: tensor<4xi32>) -> tensor<1x64x84x32xf32> {
  // custom op for "tfl.convolution_2d_transpose_bias"(%arg0, %arg1, %arg2) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
  %0 = "tfl.custom"(%arg0, %arg1, %arg2) {custom_option = #tfl<const_bytes : "0x010000000200000002000000">, custom_code = "Convolution2DTransposeBias"} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, tensor<4xi32>) -> tensor<1x64x84x32xf32>
  func.return %0 : tensor<1x64x84x32xf32>
}

// -----

func.func @testConvolution2DTransposeNoBias(%arg0: tensor<32x4x4x128xf32>, %arg1: tensor<1x32x42x128xf32>) -> tensor<1x64x84x32xf32> {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  // custom op for "tfl.convolution_2d_transpose_bias"(%arg0, %arg1, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  %0 = "tfl.custom"(%arg0, %arg1, %cst) {custom_option = #tfl<const_bytes : "0x010000000200000002000000">, custom_code = "Convolution2DTransposeBias"} : (tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x32xf32>
  func.return %0 : tensor<1x64x84x32xf32>
}

// -----

func.func @testTransposeConvBadOutputRank(%arg0: tensor<4xi32>, %arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<64x84x32xf32> {
  %cst = "tfl.no_value"() {value = unit} : () -> none
  // expected-error @+1 {{expect output type has rank = 4, got output type tensor<64x84x32xf32>}}
  %0 = "tfl.transpose_conv"(%arg0, %arg1, %arg2, %cst) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<64x84x32xf32>
  func.return %0 : tensor<64x84x32xf32>
}

// -----

func.func @testTransposeConvBadOutputShape(%arg1: tensor<32x4x4x128xf32>, %arg2: tensor<1x32x42x128xf32>) -> tensor<1x64x84x31xf32> {
  %cst = arith.constant dense<[1, 64, 84, 32]> : tensor<4xi32>
  %cst_1 = "tfl.no_value"() {value = unit} : () -> none
  // expected-error @+1 {{expect output type tensor<1x64x84x32xf32>, got tensor<1x64x84x31xf32>}}
  %0 = "tfl.transpose_conv"(%cst, %arg1, %arg2, %cst_1) {padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<4xi32>, tensor<32x4x4x128xf32>, tensor<1x32x42x128xf32>, none) -> tensor<1x64x84x31xf32>
  func.return %0 : tensor<1x64x84x31xf32>
}

// -----

// CHECK-LABEL: testDensify
func.func @testDensify(%arg0: tensor<? x f32>) -> tensor<? x f32> {
  // CHECK: "tfl.densify"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "tfl.densify"(%arg0): (tensor<? x f32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}

// -----

func.func @WhileOp_cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32> loc("Const")
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @WhileOp_body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %cst = arith.constant dense<1> : tensor<i32> loc("Const1")
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0, %1 : tensor<*xi32>, tensor<*xf32>
}

func.func @main(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<i32> {
  // expected-error @+1 {{number of operands does not match number of results}}
  %0:1 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
    %1 = func.call @WhileOp_cond(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
    %1:2 = func.call @WhileOp_body(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>)
  func.return %0#0 : tensor<i32>
}

// -----

func.func @WhileOp_cond(%arg0: tensor<*xi32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32> loc("Const")
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @WhileOp_body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>) {
  %cst = arith.constant dense<1> : tensor<i32> loc("Const1")
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  func.return %0, %1 : tensor<*xi32>, tensor<*xf32>
}

func.func @main(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<i32> {
  // expected-error @+1 {{number of arguments in condition function does not match number of arguments in body function}}
  %0:2 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xi32>):
    %1 = func.call @WhileOp_cond(%arg2) : (tensor<*xi32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
    %1:2 = func.call @WhileOp_body(%arg2, %arg3) : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>, tensor<*xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>)
  func.return %0#0 : tensor<i32>
}

// -----

func.func @WhileOp_cond(%arg0: tensor<*xi32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32> loc("Const")
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @WhileOp_body(%arg0: tensor<*xi32>, %arg1: tensor<2xf32>, %arg2: tensor<1xf32>) -> (tensor<*xi32>, tensor<2xf32>, tensor<1xf32>) {
  %cst = arith.constant dense<1> : tensor<i32> loc("Const1")
  %0 = "tfl.sub"(%arg0, %cst) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<2xf32>
  func.return %0, %1, %arg2 : tensor<*xi32>, tensor<2xf32>, tensor<1xf32>
}

func.func @main(%arg0: tensor<i32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>) -> tensor<i32> {
  // expected-error @+1 {{condition function's argument type does not match body function's argument type}}
  %0:3 = "tfl.while"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<*xi32>, %arg4: tensor<1xf32>, %arg5: tensor<2xf32>):
    %1 = func.call @WhileOp_cond(%arg3, %arg4, %arg5) : (tensor<*xi32>, tensor<1xf32>, tensor<2xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<*xi32>, %arg4: tensor<2xf32>, %arg5: tensor<1xf32>):
    %1:3 = func.call @WhileOp_body(%arg3, %arg4, %arg5) : (tensor<*xi32>, tensor<2xf32>, tensor<1xf32>) -> (tensor<*xi32>, tensor<2xf32>, tensor<1xf32>)
    "tfl.yield"(%1#0, %1#1, %1#2) : (tensor<*xi32>, tensor<2xf32>, tensor<1xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>, tensor<2xf32>) -> (tensor<i32>, tensor<2xf32>, tensor<1xf32>)
  func.return %0#0 : tensor<i32>
}

// -----

func.func @if_then_else(%arg0: tensor<i1>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tfl.if"(%arg0) ({
    "tfl.yield"(%arg1) : (tensor<1xf32>) -> ()
  },  {
    %1 = "tfl.sub"(%arg1, %arg1) {fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "tfl.yield"(%1) : (tensor<1xf32>) -> ()
  }) : (tensor<i1>) -> (tensor<1xf32>)
  func.return %0 : tensor<1xf32>
}

func.func @if_then(%arg0: tensor<i1>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error @+1 {{'tfl.if' op expected 2 regions}}
  %0 = "tfl.if"(%arg0) ({
    %1 = "tfl.sub"(%arg1, %arg1) {fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "tfl.yield"(%1) : (tensor<1xf32>) -> ()
  }) : (tensor<i1>) -> (tensor<1xf32>)
  func.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: valid_unranked_inputs_on_reshape
func.func @valid_unranked_inputs_on_reshape(%arg0: tensor<3x4xi32>, %arg1: tensor<*xi32>) -> tensor<3x4xi32> {
  // CHECK: "tfl.reshape"(%arg0, %arg1)
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<3x4xi32>, tensor<*xi32>) -> tensor<3x4xi32>
  func.return %0 : tensor<3x4xi32>
}

// -----

// CHECK-LABEL: valid_one_dynamic_dim_on_reshape
func.func @valid_one_dynamic_dim_on_reshape(%arg0: tensor<3x4xi32>) -> tensor<1x3x4xi32> {
  %cst = arith.constant dense<[1, -1, 4]> : tensor<3xi32>
  // CHECK: "tfl.reshape"(%arg0, %cst)
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<1x3x4xi32>
  func.return %0 : tensor<1x3x4xi32>
}

// -----

func.func @invalid_two_dynamic_dims_on_reshape(%arg0: tensor<3x4xi32>) -> tensor<1x3x4xi32> {
  %cst = arith.constant dense<[-1, -1, 4]> : tensor<3xi32>
  // expected-error @+1 {{tfl.reshape' op requires 'shape' to have at most one dynamic dimension, but got multiple dynamic dimensions at indices 0 and 1. You need to set up the unspecified size(s) to avoid this problem, for example,setting batch size in keras model or setting unspecified input size(s) with fixed ones.}}
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<3x4xi32>, tensor<3xi32>) -> tensor<1x3x4xi32>
  func.return %0 : tensor<1x3x4xi32>
}

// -----

// CHECK-LABEL: testBroadcastToWithI32ShapeTensor
func.func @testBroadcastToWithI32ShapeTensor(tensor<?x?x?x?x?x?xf32>, tensor<8xi32>) -> tensor<?x?x?x?x?x?x?x?xf32> {
^bb0(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<8xi32>):
  // CHECK: "tfl.broadcast_to"(%arg0, %arg1)
  %0 = "tfl.broadcast_to"(%arg0, %arg1): (tensor<?x?x?x?x?x?xf32>, tensor<8xi32>) -> tensor<?x?x?x?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?x?x?x?xf32>
}

// CHECK-LABEL: testBroadcastToWithI64ShapeTensor
func.func @testBroadcastToWithI64ShapeTensor(tensor<?x?x?x?x?x?xf32>, tensor<8xi64>) -> tensor<?x?x?x?x?x?x?x?xf32> {
^bb0(%arg0: tensor<?x?x?x?x?x?xf32>, %arg1: tensor<8xi64>):
  // CHECK: "tfl.broadcast_to"(%arg0, %arg1)
  %0 = "tfl.broadcast_to"(%arg0, %arg1): (tensor<?x?x?x?x?x?xf32>, tensor<8xi64>) -> tensor<?x?x?x?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: testFillWithQI8
func.func @testFillWithQI8(%arg0: tensor<1x4xi32>, %arg1: tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x !quant.uniform<i8:f32, 0.1>> {
  %0 = "tfl.fill"(%arg0, %arg1): (tensor<1x4xi32>, tensor<? x !quant.uniform<i8:f32, 0.1>>) -> tensor<? x !quant.uniform<i8:f32, 0.1>>
  func.return %0 : tensor<? x !quant.uniform<i8:f32, 0.1>>
}

// -----

// CHECK-LABEL: testConv3dWithFloatInput
func.func @testConv3dWithFloatInput(%arg0: tensor<?x?x?x?x?xf32>,%arg1:  tensor<?x?x?x?x?xf32>,%arg2: tensor<?xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK: "tfl.conv_3d"(%arg0, %arg1, %arg2)
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// CHECK-LABEL: testConv3dNoBiasInput
func.func @testConv3dNoBiasInput(%arg0: tensor<?x?x?x?x?xf32>,%arg1:  tensor<?x?x?x?x?xf32>,%arg2: none) -> tensor<?x?x?x?x?xf32> {
  // CHECK: "tfl.conv_3d"(%arg0, %arg1, %arg2)
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>, none) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @testConv3dInvalidFilterShape(%arg0: tensor<2x3x4x5x2xf32>,%arg1:  tensor<2x2x2x3x3xf32>,%arg2: tensor<?xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{failed to verify that dim 4 of operand 0 equals to dim 3 of operand 1}}
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<2x3x4x5x2xf32>, tensor<2x2x2x3x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @testConv3dInvalidBiasShape(%arg0: tensor<2x3x4x5x2xf32>,%arg1:  tensor<2x2x2x2x3xf32>,%arg2: tensor<4xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{failed to verify that bias must has num of elements equals to 4th dim of filter}}
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<2x3x4x5x2xf32>, tensor<2x2x2x2x3xf32>, tensor<4xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @testConv3dMisMatchInputType(%arg0: tensor<2x3x4x5x2xi32>,%arg1:  tensor<2x2x2x2x3xf32>,%arg2: tensor<3xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{op failed to verify that input and output must have same element type}}
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<2x3x4x5x2xi32>, tensor<2x2x2x2x3xf32>, tensor<3xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @testConv3dMisMatchBiasType(%arg0: tensor<2x3x4x5x2xf32>,%arg1:  tensor<2x2x2x2x3xf32>,%arg2: tensor<3xi32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{failed to verify that bias and output must have same element type}}
  %0 = "tfl.conv_3d"(%arg0, %arg1, %arg2) {dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"}: (tensor<2x3x4x5x2xf32>, tensor<2x2x2x2x3xf32>, tensor<3xi32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: testComplexAbs
func.func @testComplexAbs(%arg0: tensor<? x complex<f32>>) -> tensor<?xf32> {
  // CHECK: "tfl.complex_abs"(%arg0)
  %0 = "tfl.complex_abs"(%arg0): (tensor<? x complex<f32>>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testComplexAbsUnsupportedType(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{operand #0 must be tensor of complex type with 32-bit float elements or complex type with 64-bit float elements values}}
  %0 = "tfl.complex_abs"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testComplexAbsWrongShape(%arg0: tensor<2 x complex<f32>>) -> tensor<3xf32> {
  // expected-error @+1 {{requires the same shape for all operands and results}}
  %0 = "tfl.complex_abs"(%arg0): (tensor<2 x complex<f32>>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: testReal
func.func @testReal(%arg0: tensor<? x complex<f64>>) -> tensor<?xf64> {
  // CHECK: "tfl.real"(%arg0)
  %0 = "tfl.real"(%arg0): (tensor<? x complex<f64>>) -> tensor<?xf64>
  func.return %0 : tensor<?xf64>
}

// -----

func.func @testRealWrongShape(%arg0: tensor<3 x complex<f64>>) -> tensor<4xf32> {
  // expected-error @+1 {{requires the same shape for all operands and results}}
  %0 = "tfl.real"(%arg0): (tensor<3 x complex<f64>>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: testImag
func.func @testImag(%arg0: tensor<? x complex<f64>>) -> tensor<?xf64> {
  // CHECK: "tfl.imag"(%arg0)
  %0 = "tfl.imag"(%arg0): (tensor<? x complex<f64>>) -> tensor<?xf64>
  func.return %0 : tensor<?xf64>
}

// -----

func.func @testImagWrongType(%arg0: tensor<3 x complex<f64>>) -> tensor<4xi32> {
  // expected-error @+1 {{requires the same shape for all operands and results}}
  %0 = "tfl.imag"(%arg0): (tensor<3 x complex<f64>>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @all(%arg0: tensor<2x2xi1>, %arg1: tensor<i32>) -> tensor<i1> {
  // CHECK: "tfl.reduce_all"(%arg0, %arg1)
  %0 = "tfl.reduce_all"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi1>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @conv3d_transpose(%arg0: tensor<5xi32>, %arg1: tensor<?x?x?x?x2xf32>, %arg2: tensor<?x?x?x?x2xf32>, %arg3: none) -> tensor<?x?x?x?x?xf32> {
  // CHECK: "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3)
  %0 = "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3) {data_format = "NDHWC", dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"} : (tensor<5xi32>, tensor<?x?x?x?x2xf32>, tensor<?x?x?x?x2xf32>, none) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @conv3d_transpose_with_bias(%arg0: tensor<5xi32>, %arg1: tensor<?x?x?x?x2xf32>, %arg2: tensor<?x?x?x?x2xf32>, %arg3: tensor<2xf32>) -> tensor<?x?x?x?x?xf32> {
  // CHECK: "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3)
  %0 = "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3) {data_format = "NDHWC", dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"} : (tensor<5xi32>, tensor<?x?x?x?x2xf32>, tensor<?x?x?x?x2xf32>, tensor<2xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @conv3d_transpose_channel_dim_not_match(%arg0: tensor<5xi32>, %arg1: tensor<1x2x2x3x2xf32>, %arg2: tensor<2x5x6x8x3xf32>, %arg3: tensor<2xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{op failed to verify that dim 4 of operand 2 equals to dim 4 of operand 1}}
  %0 = "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3) {data_format = "NDHWC", dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"} : (tensor<5xi32>, tensor<1x2x2x3x2xf32>, tensor<2x5x6x8x3xf32>, tensor<2xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @conv3d_transpose_bias_size_not_match(%arg0: tensor<5xi32>, %arg1: tensor<1x2x2x3x2xf32>, %arg2: tensor<2x5x6x8x2xf32>, %arg3: tensor<3xf32>) -> tensor<?x?x?x?x?xf32> {
  // expected-error @+1 {{bias must has num of elements equals to 4th dim of filter}}
  %0 = "tfl.conv_3d_transpose"(%arg0, %arg1, %arg2, %arg3) {data_format = "NDHWC", dilation_d_factor = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, fused_activation_function = "NONE"} : (tensor<5xi32>, tensor<1x2x2x3x2xf32>, tensor<2x5x6x8x2xf32>, tensor<3xf32>) -> tensor<?x?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?x?xf32>
}

// -----

func.func @broadcast_args(%arg0: tensor<5xi32>, %arg1: tensor<2xi32>) -> tensor<5xi32> {
  // CHECK: "tfl.broadcast_args"(%arg0, %arg1)
  %0 = "tfl.broadcast_args"(%arg0, %arg1) {} : (tensor<5xi32>, tensor<2xi32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}

// -----

func.func @broadcast_args_unmatch_types(%arg0: tensor<5xi32>, %arg1: tensor<2xi64>) -> tensor<5xi32> {
  // expected-error @+1 {{failed to verify that BroadcastArgs op operands have same element type}}
  %0 = "tfl.broadcast_args"(%arg0, %arg1) {} : (tensor<5xi32>, tensor<2xi64>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}

// -----

func.func @mul_with_dynamic_shape_from_broadcast_args(%arg0: tensor<8x7x6x5x?x3x2x1xf32>, %arg1: tensor<?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32> {
  %0 = "tfl.shape"(%arg0) : (tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8xi64>
  %1 = "tfl.shape"(%arg1) : (tensor<?x3x2x1xf32>) -> tensor<4xi64>
  %2 = "tfl.broadcast_args"(%0, %1) : (tensor<8xi64>, tensor<4xi64>) -> tensor<8xi64>
  %3 = "tfl.broadcast_to"(%arg0, %2) : (tensor<8x7x6x5x?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  %4 = "tfl.broadcast_to"(%arg1, %2) : (tensor<?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  %5 = "tfl.mul"(%3, %4) {fused_activation_function = "NONE"} : (tensor<8x7x6x5x?x3x2x1xf32>, tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32>
  func.return %5 : tensor<8x7x6x5x?x3x2x1xf32>
  // CHECK: tfl.mul
}

// -----

func.func @mul_with_dynamic_shape_not_from_broadcast_args(%arg0: tensor<8x7x6x5x?x3x2x1xf32>, %arg1: tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32> {
  // expected-error @+1 {{'tfl.mul' op failed to verify that Operands do not have valid shapes}}
  %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<8x7x6x5x?x3x2x1xf32>, tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32>
  func.return %0 : tensor<8x7x6x5x?x3x2x1xf32>
}

// -----

func.func @select_v2_with_dynamic_shape_from_broadcast_args(%arg0: tensor<8x7x6x5x?x3x2x1xi1>, %arg1: tensor<8x7x6x5x?x3x2x1xf32>, %arg2: tensor<?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32> {
  %0 = "tfl.shape"(%arg1) : (tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8xi64>
  %1 = "tfl.shape"(%arg2) : (tensor<?x3x2x1xf32>) -> tensor<4xi64>
  %2 = "tfl.broadcast_args"(%0, %1) : (tensor<8xi64>, tensor<4xi64>) -> tensor<8xi64>
  %3 = "tfl.shape"(%arg0) : (tensor<8x7x6x5x?x3x2x1xi1>) -> tensor<8xi64>
  %4 = "tfl.broadcast_args"(%2, %3) : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi64>
  %5 = "tfl.broadcast_to"(%arg0, %4) : (tensor<8x7x6x5x?x3x2x1xi1>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xi1>
  %6 = "tfl.broadcast_to"(%arg1, %4) : (tensor<8x7x6x5x?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  %7 = "tfl.broadcast_to"(%arg2, %4) : (tensor<?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  %8 = "tfl.select_v2"(%5, %6, %7) : (tensor<8x7x6x5x?x3x2x1xi1>, tensor<8x7x6x5x?x3x2x1xf32>, tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32>
  func.return %8 : tensor<8x7x6x5x?x3x2x1xf32>
}

// -----

func.func @select_v2_with_dynamic_shape_not_from_broadcast_args(%arg0: tensor<8x7x6x5x?x3x2x1xi1>, %arg1: tensor<8x7x6x5x?x3x2x1xf32>, %arg2: tensor<?x3x2x1xf32>, %arg3: tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32> {
  %0 = "tfl.broadcast_to"(%arg1, %arg3) : (tensor<8x7x6x5x?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  %1 = "tfl.broadcast_to"(%arg2, %arg3) : (tensor<?x3x2x1xf32>, tensor<8xi64>) -> tensor<8x7x6x5x?x3x2x1xf32>
  // expected-error @+1 {{'tfl.select_v2' op failed to verify that operands do not have the same shape or broadcastable shapes within the rank 5}}
  %2 = "tfl.select_v2"(%arg0, %0, %1) : (tensor<8x7x6x5x?x3x2x1xi1>, tensor<8x7x6x5x?x3x2x1xf32>, tensor<8x7x6x5x?x3x2x1xf32>) -> tensor<8x7x6x5x?x3x2x1xf32>
  func.return %2 : tensor<8x7x6x5x?x3x2x1xf32>
}

// -----

func.func @depthwise_conv_2D_invalid_bias(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x112x112x32xf32> {
  %w = arith.constant dense<127.0> : tensor<32x3x3x3xf32>
  %b = arith.constant dense<0.0> : tensor<32x1xf32>
  // expected-error @+1 {{'tfl.depthwise_conv_2d' op operand #2 must be 1D tensor of any type values or none type, but got 'tensor<32x1xf32>'}}
  %dc = "tfl.depthwise_conv_2d"(%arg0, %w, %b) {depth_multiplier = 4 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 4 : i32, stride_w = 5 : i32} : (tensor<1x224x224x3xf32>, tensor<32x3x3x3xf32>, tensor<32x1xf32>) -> tensor<1x112x112x32xf32>
  func.return %dc : tensor<1x112x112x32xf32>
}


// -----

// CHECK-LABEL: testRandomUniform
func.func @testRandomUniform(%arg0: tensor<3xi32>) -> tensor<?x?x?xf32> {
  // CHECK: "tfl.random_uniform"(%arg0)
  %0 = "tfl.random_uniform"(%arg0) {seed = 0 : i64, seed2 = 0 : i64} : (tensor<3xi32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testRandomStandardNormal
func.func @testRandomStandardNormal(%arg0: tensor<3xi32>) -> tensor<?x?x?xf32> {
  // CHECK: "tfl.random_standard_normal"(%arg0)
  %0 = "tfl.random_standard_normal"(%arg0) {seed = 0 : i64, seed2 = 0 : i64} : (tensor<3xi32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: testMultinomial
func.func @testMultinomial(%arg0: tensor<2xf32>, %arg1: tensor<1xi32>) -> tensor<10xi64> {
  // CHECK: "tfl.multinomial"(%arg0, %arg1)
  %0 = "tfl.multinomial"(%arg0, %arg1) {seed = 0 : i64, seed2 = 0: i64} : (tensor<2xf32>, tensor<1xi32>) -> tensor<10xi64>
  func.return %0 :  tensor<10xi64>
}

// -----

// CHECK-LABEL: testMultinomialInt32
func.func @testMultinomialInt32(%arg0: tensor<2xf32>, %arg1: tensor<1xi32>) -> tensor<10xi32> {
  // CHECK: "tfl.multinomial"(%arg0, %arg1)
  %0 = "tfl.multinomial"(%arg0, %arg1) {seed = 0 : i64, seed2 = 0: i64} : (tensor<2xf32>, tensor<1xi32>) -> tensor<10xi32>
  func.return %0 :  tensor<10xi32>
}

// -----

// CHECK-LABEL: testGelu
func.func @testGelu(%arg0: tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32> {
  // CHECK: "tfl.gelu"(%arg0)
  %0 = "tfl.gelu"(%arg0) {approximate = false}: (tensor<1x2x3x4x5xf32>) -> tensor<1x2x3x4x5xf32>
  func.return %0 : tensor<1x2x3x4x5xf32>
}

// -----

// test invalid GELU input
func.func @testGeluWithWrongInputType(%arg0: tensor<?xf32>) -> tensor<?xi32> {
  // expected-error @+1 {{'tfl.gelu' op failed to verify that input and output must have same element type}}
  %0 = "tfl.gelu"(%arg0) {approximate = false}: (tensor<?xf32>) -> tensor<?xi32>
  func.return %0#0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: scatter_nd_i1
func.func @scatter_nd_i1(%arg0: tensor<?xi32>, %arg1: tensor<?xi1>, %arg2: tensor<?xi32>) -> tensor<?xi1> {
  // CHECK: "tfl.scatter_nd"(%arg0, %arg1, %arg2)
  %0 = "tfl.scatter_nd"(%arg0, %arg1, %arg2): (tensor<?xi32>, tensor<?xi1>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: testUnsortedSegmentSum
func.func @testUnsortedSegmentSum(%arg0: tensor<8xf32>, %arg1: tensor<8xi32>,  %arg2: tensor<i32>) -> tensor<8xf32> {
  // CHECK: "tfl.unsorted_segment_sum"(%arg0, %arg1, %arg2)
  %0 = "tfl.unsorted_segment_sum"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<8xi32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
  // CHECK: return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: testUnsortedSegmentProd
func.func @testUnsortedSegmentProd(%arg0: tensor<8xf32>, %arg1: tensor<8xi32>,  %arg2: tensor<i32>) -> tensor<8xf32> {
  // CHECK: "tfl.unsorted_segment_prod"(%arg0, %arg1, %arg2)
  %0 = "tfl.unsorted_segment_prod"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<8xi32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
  // CHECK: return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: testUnsortedSegmentMax
func.func @testUnsortedSegmentMax(%arg0: tensor<8xf32>, %arg1: tensor<8xi32>,  %arg2: tensor<i32>) -> tensor<8xf32> {
  // CHECK: "tfl.unsorted_segment_max"(%arg0, %arg1, %arg2)
  %0 = "tfl.unsorted_segment_max"(%arg0, %arg1, %arg2) : (tensor<8xf32>, tensor<8xi32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
  // CHECK: return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: testControlNodeLongForm
func.func @testControlNodeLongForm(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)->tensor<8xf32> {
  %0, %c0 = "tfl.control_node"() ({
    %1 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<8xf32>, tensor<8xf32>)->tensor<8xf32>
    "tfl.yield"(%1) : (tensor<8xf32>) -> ()
    }) : () -> (tensor<8xf32>, !tfl.control)
  func.return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: testControlNodeShortForm
func.func @testControlNodeShortForm(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)->tensor<8xf32> {
  %0, %c0 = tfl.control_node() controls "tfl.add"(%arg0, %arg1) {fused_activation_function = "RELU6"} : (tensor<8xf32>, tensor<8xf32>)->tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

