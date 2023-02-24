// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// Tests for TensorFlow ops with custom verifiers.

// TODO(hinsu): Remove tests for ops without custom verifiers. These tests were
// added along with manual op definition and are obsolete now that the op
// definitions are auto-generated.

// TODO(hinsu): Move attribute and type tests to types.mlir file.
//===--------------------------------------------------------------------===//
//  Test TF TensorProto attributes
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func @tensorProtoAttr
func.func @tensorProtoAttr() -> () {
^bb0:
// CHECK: "tf.TensorProtoIntTensor"() {bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4xi32>} : () -> ()
  "tf.TensorProtoIntTensor"(){bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4xi32>} : () -> ()
// CHECK: "tf.TensorProtoFloatTensor"() {bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4xf32>} : () -> ()
  "tf.TensorProtoFloatTensor"(){bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4xf32>} : () -> ()
// CHECK: "tf.TensorProtoStringTensor"() {bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4x!tf_type.string>} : () -> ()
  "tf.TensorProtoStringTensor"(){bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4x!tf_type.string>} : () -> ()
// CHECK: "tf.TensorProtoResourceTensor"() {bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4x!tf_type.resource>} : () -> ()
  "tf.TensorProtoResourceTensor"(){bar = #tf_type<tensor_proto : "0x68656C6C6F"> : tensor<2x1x4x!tf_type.resource>} : () -> ()
  func.return
}

//===--------------------------------------------------------------------===//
//  Test TF placeholder attribute
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func @placeholderattr
func.func @placeholderattr() -> ()
// CHECK:    attributes {some_placeholder = #tf_type.placeholder<"foo">} {
    attributes {some_placeholder = #tf_type.placeholder<"foo">} {
  func.return
}
//===--------------------------------------------------------------------===//
//  Test TF operations (tf.*)
//===--------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @testIdentity
func.func @testIdentity(%arg0: tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Identity"(%arg0) : (tensor<4x?x!tf_type.stringref>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

// -----

// CHECK-LABEL: func @testBitcast
func.func @testBitcast(%arg0: tensor<3x4xui16>) -> tensor<3x4x!tf_type.quint16> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x4xui16>) -> tensor<3x4x!tf_type.quint16>
  func.return %0 : tensor<3x4x!tf_type.quint16>
}

// -----

// CHECK-LABEL: func @testBitcast_v2
func.func @testBitcast_v2(%arg0: tensor<3x2xi16>) -> tensor<3xi32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi16>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testBitcast_v3(%arg0: tensor<3x5xi16>) -> tensor<3xi32> {
  // expected-error @+1 {{input rightmost dimension size is not equal to the divisor. the last dimension of input is expected to be 2}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x5xi16>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testBitcast_v4(%arg0: tensor<3x2xi16>) -> tensor<3x2xi32> {
  // expected-error @+1 {{rank of input tensor is 2. rank of output tensor is expected to be 1, instead of 2.}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi16>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @testBitcast_v5(%arg0: tensor<3x2xi16>) -> tensor<2xi32> {
  // expected-error @+1 {{the 0th dim of output tensor is 2. It is not equal to the one in input tensor, which is 3}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi16>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @testBitcast_v6
func.func @testBitcast_v6(%arg0: tensor<3x2xi32>) -> tensor<3x2x2xi16> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2x2xi16>
  func.return %0 : tensor<3x2x2xi16>
}

// -----

func.func @testBitcast_v7(%arg0: tensor<3x2xi32>) -> tensor<3x2x3xi16> {
  // expected-error @+1 {{output rightmost dimension size is not equal to the divisor. the last dimension of output is expected to be 2}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2x3xi16>
  func.return %0 : tensor<3x2x3xi16>
}

// -----

func.func @testBitcast_v8(%arg0: tensor<3x2xi32>) -> tensor<3x2xi16> {
  // expected-error @+1 {{rank of input tensor is 2. rank of output tensor is expected to be 3, instead of 2.}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2xi16>
  func.return %0 : tensor<3x2xi16>
}

// -----

// CHECK-LABEL: func @testBitcast_v9
func.func @testBitcast_v9(%arg0: tensor<3x2xi32>) -> tensor<3x2xf32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

func.func @testBitcast_v10(%arg0: tensor<3x2xi32>) -> tensor<3x4xf32> {
  // expected-error @+1 {{output tensor shape shall be equal to input tensor shape}}
  %0 = "tf.Bitcast"(%arg0) : (tensor<3x2xi32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @testBitcast_v11
func.func @testBitcast_v11(%arg0: tensor<10x10x2xf16>) -> tensor<*xf32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<10x10x2xf16>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @testReverseV2
func.func @testReverseV2(%arg0: tensor<2x4x3xui8>, %arg1: tensor<1xi32>) -> tensor<2x4x3xui8> {
  %0 = "tf.ReverseV2"(%arg0, %arg1) : (tensor<2x4x3xui8>, tensor<1xi32>) -> tensor<2x4x3xui8>
  func.return %0 :  tensor<2x4x3xui8>
}

// -----

func.func @testIdentityWrongType(%arg0: tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref> {
  // expected-error @+1 {{all operands and results to have compatible element}}
  %0 = "tf.Identity"(%arg0) : (tensor<4x2x!tf_type.string>) -> tensor<4x2x!tf_type.stringref>
  func.return %0 : tensor<4x2x!tf_type.stringref>
}

// -----

// TODO(hinsu): Move this to MLIR core once the test dialect have a custom type.

// Check that broadcastable trait accepts TF specific element type
// CHECK-LABEL: func @testAdd
func.func @testAdd(%arg0: tensor<4x2x!tf_type.string>, %arg1: tensor<2x!tf_type.string>) -> tensor<4x2x!tf_type.string> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x2x!tf_type.string>, tensor<2x!tf_type.string>) -> tensor<4x2x!tf_type.string>
  func.return %0 : tensor<4x2x!tf_type.string>
}

// -----

// Valid BiasAdd operation.
func.func @testBiasAdd(%arg0: tensor<2x3x5x7xf32>, %arg1: tensor<7xf32>) -> tensor<2x3x5x7xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<2x3x5x7xf32>, tensor<7xf32>) -> tensor<2x3x5x7xf32>
  func.return %0 : tensor<2x3x5x7xf32>
}

// -----

func.func @testBiasAddNoDataFormatOk(tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32> {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<16xf32>):
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT"}: (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @testBiasAddWrongDataFormat(tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32> {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<16xf32>):
  // expected-error @+1 {{attribute 'data_format' failed to satisfy constraint: 'NHWC' or 'NCHW' convnet data format}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "HWCN"} : (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  func.return %0 : tensor<1x32x32x16xf32>
}

// -----

func.func @testBiasAdd(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{requires value operand to have rank at least two with `NHWC` data format}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

func.func @testBiasAdd(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2x3xf32> {
  // expected-error @+1 {{requires value operand to have rank at least three with `NCHW` data format}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NCHW"} : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @testBiasAdd(%arg0: tensor<2x3x5x7xf32>, %arg1: tensor<5x7xf32>) -> tensor<2x3x5x7xf32> {
  // expected-error @+1 {{requires bias operand to have rank exactly one}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<2x3x5x7xf32>, tensor<5x7xf32>) -> tensor<2x3x5x7xf32>
  func.return %0 : tensor<2x3x5x7xf32>
}

// -----

func.func @testBiasAdd(%arg0: tensor<2x3x5x7xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x5x7xf32> {
  // expected-error @+1 {{requires channel dimension and feature dimension to match; found 7 and 5, respectively}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<2x3x5x7xf32>, tensor<5xf32>) -> tensor<2x3x5x7xf32>
  func.return %0 : tensor<2x3x5x7xf32>
}

// -----

// Valid BiasAddGrad operation.
func.func @testBiasAddGrad(%arg0: tensor<2x3x5x7xf32>) -> tensor<7xf32> {
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NHWC"} : (tensor<2x3x5x7xf32>) -> (tensor<7xf32>)
  func.return %0 : tensor<7xf32>
}

// -----

func.func @testBiasAddGrad(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{requires out_backprop operand to have rank at least two with `NHWC` data format}}
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NHWC"} : (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

func.func @testBiasAddGrad(%arg0: tensor<2x3xf32>) -> tensor<3xf32> {
  // expected-error @+1 {{requires out_backprop operand to have rank at least three with `NCHW` data format}}
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<2x3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// -----

// Test valid tf.BroadcastGradientArgs
// CHECK-LABEL: func @testBroadcastGradientArgs
func.func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsIncompatibleInputType(%s0: tensor<4xi32>, %s1: tensor<4xi64>) -> (tensor<1xi32>, tensor<0xi32>) {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi64>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsIncompatibleBroadcastShape() -> (tensor<1xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[4, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires broadcast compatible shape tensors for 's0' and 's1', but got dense<[4, 1]> : tensor<2xi32> and dense<[2, 4]> : tensor<2xi32>}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsInvalidS0Rank() -> (tensor<2x2xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[[4, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %s1 = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{failed to verify that operand 0 is 1-D}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2x2xi32>, tensor<2xi32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsInvalidS1Rank() -> (tensor<2xi32>, tensor<i32>) {
  %s0 = "tf.Const"() {value = dense<[4, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{failed to verify that operand 1 is 1-D}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2xi32>, tensor<i32>) -> (tensor<1xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsInvalidR0Rank() -> (tensor<2x2xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[4, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<[4, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{failed to verify that result 0 is 1-D}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<2x2xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsInvalidR1Rank(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<i32>) {
  // expected-error @+1 {{failed to verify that result 1 is 1-D}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<i32>)
  func.return %r0, %r1 : tensor<1xi32>, tensor<i32>
}

// -----

func.func @testBroadcastGradientArgsInvalidR0Size() -> (tensor<0xi32>, tensor<0xi32>) {
  %s0 = "tf.Const"() {value = dense<[4, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<[4, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires dimension 0 size of 'r0' to be 1 but got 0}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<0xi32>, tensor<0xi32>)
  func.return %r0, %r1 : tensor<0xi32>, tensor<0xi32>
}

// -----

func.func @testBroadcastGradientArgsInvalidR1Size() -> (tensor<0xi32>, tensor<3xi32>) {
  %s0 = "tf.Const"() {value = dense<[4, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %s1 = "tf.Const"() {value = dense<[1, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires dimension 0 size of 'r1' to be 2 but got 3}}
  %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<2xi32>, tensor<2xi32>) -> (tensor<0xi32>, tensor<3xi32>)
  func.return %r0, %r1 : tensor<0xi32>, tensor<3xi32>
}

// -----

// Test valid tf.BroadcastTo
// CHECK-LABEL: func @testBroadcastTo(%arg0: tensor<16xf32>)
func.func @testBroadcastTo(%arg0: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %cst = arith.constant dense<16> : tensor<4xi32>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}

// -----

// Test valid tf.LeakyRelu
// CHECK-LABEL: func @testLeakyRelu(%arg0: tensor<16xf32>)
func.func @testLeakyRelu(tensor<16xf32>) -> tensor<16xf32> {
^bb0(%arg0: tensor<16xf32>):
  %0 = "tf.LeakyRelu"(%arg0) {alpha = 0.2 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----
func.func @testLeakyWrongAlphaType(tensor<16xf32>) -> tensor<16xf32> {
^bb0(%arg0: tensor<16xf32>):
  // expected-error @+1 {{attribute 'alpha' failed to satisfy constraint: 32-bit float}}
  %0 = "tf.LeakyRelu"(%arg0) {alpha = 1: i32}: (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// Test tf.Min with complex numbers.
// Previous versions of tensorflow said complex numbers were allowed with
// tf.Min even though it doesn't make sense. The legalization of tf to xla
// requires that complex types are not allowed in tf.Min, so we have an
// explicit unit here to make sure that invariant is enforced.
func.func @testMinComplex(%arg0: tensor<4x8xcomplex<f32>>) -> tensor<4x1xcomplex<f32>> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<1xi64> } : () -> tensor<1xi64>
  // expected-error@below {{'tf.Min' op operand #0 must be tensor of}}
  %0 = "tf.Min"(%arg0, %dimension) { keep_dims = true }: (tensor<4x8xcomplex<f32>>, tensor<1xi64>) -> tensor<4x1xcomplex<f32>>
  func.return %0 : tensor<4x1xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @testMul
func.func @testMul(%arg0: tensor<2xui16>) -> (tensor<2xui16>) {
  %0 = "tf.Mul"(%arg0, %arg0) {T = "tfdtype$DT_UINT16", device = "/device:CPU:0", name = "Mul"} : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
  func.return %0 : tensor<2xui16>
}

// -----

// Test error message for incompatible element types.
func.func @testIncompatibleElementTypes(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf64>) -> (tensor<3x2xf32>) {
    // expected-error @+1 {{'tf.Mul' op requires compatible element types for all operands and results}}
  %0 = "tf.Mul"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3x2xf64>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// Test error message for incompatible element types.
func.func @testIncompatibleElementTypes(%arg0: tensor<3x2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<3x2xf64>) {
    // expected-error @+1 {{'tf.Mul' op requires compatible element types for all operands and results}}
  %0 = "tf.Mul"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf64>
  func.return %0 : tensor<3x2xf64>
}

// -----

func.func @testPadRank1Paddings(%input: tensor<2xi64>) -> tensor<3xi64> {
  %paddings = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{failed to verify that operand 1 is 2-D}}
  %0 = "tf.Pad"(%input, %paddings) : (tensor<2xi64>, tensor<2xi64>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

func.func @testPadV2Rank1Paddings(%input: tensor<2xi64>) -> tensor<3xi64> {
  %constant = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %paddings = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{failed to verify that operand 1 is 2-D}}
  %0 = "tf.PadV2"(%input, %paddings, %constant) : (tensor<2xi64>, tensor<2xi64>, tensor<i64>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

func.func @testMirrorPadRank1Paddings(%input: tensor<2xi64>) -> tensor<3xi64> {
  %paddings = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{failed to verify that operand 1 is 2-D}}
  %0 = "tf.MirrorPad"(%input, %paddings) { mode = "SYMMETRIC" }: (tensor<2xi64>, tensor<2xi64>) -> tensor<3xi64>
  func.return %0 : tensor<3xi64>
}

// -----

// CHECK-LABEL: func @testReshape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<10000xf32>, %arg3: tensor<*xi32>)
func.func @testReshape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<10000xf32>, %arg3: tensor<*xi32>) -> (tensor<100x100xf32>, tensor<*xf32>, tensor<100x100xf32>, tensor<100x100xf32>, tensor<*xf32>, tensor<*xf32>) {
  %shape1 = arith.constant dense<100> : tensor<2xi32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  %shape2 = "tf.Shape"(%arg0) : (tensor<*xf32>) -> tensor<?xi32>
  %r2 = "tf.Reshape"(%arg1, %shape2) : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
  %r3 = "tf.Reshape"(%arg2, %shape1) : (tensor<10000xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  %shape3 = arith.constant dense<[-1, 100]> : tensor<2xi32>
  %r4 = "tf.Reshape"(%arg2, %shape3) : (tensor<10000xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  %r5 = "tf.Reshape"(%arg0, %arg3) : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
  %r6 = "tf.Reshape"(%arg2, %arg3) : (tensor<10000xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return %r1, %r2, %r3, %r4, %r5, %r6: tensor<100x100xf32>, tensor<*xf32>, tensor<100x100xf32>, tensor<100x100xf32>, tensor<*xf32>, tensor<*xf32>
}

// -----
// tf.Reshape with incorrect type.
func.func @testReshape(tensor<*xf32>, tensor<*xf32>) -> (tensor<100x100xf32>) {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>):
  %shape1 = arith.constant dense<100.> : tensor<2xf32>
  // expected-error @+1 {{must be tensor of 32/64-bit signed integer values}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<*xf32>, tensor<2xf32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with incorrect element number.
func.func @testReshape(%arg0: tensor<10x10x10xf32>, %shape1: tensor<2xi32>) -> tensor<100x100xf32> {
  // expected-error @+1 {{requires 'output' number of elements to match 'tensor' number of elements, but got 10000 and 1000}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with incorrect shape operand rank.
func.func @testReshape(%arg0: tensor<10x10x10xf32>, %shape1: tensor<2x2xi32>) -> tensor<*xf32> {
  // expected-error @+1 {{requires 'shape' to be rank 1, but got 2}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return %r1 : tensor<*xf32>
}

// -----
// tf.Reshape with more than one -1 in the shape.
func.func @testReshape(%arg0: tensor<10x10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = arith.constant dense<-1> : tensor<2xi32>
  // expected-error @+1 {{requires 'shape' to have at most one dynamic dimension, but got multiple dynamic dimensions at indices 0 and 1}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10x10xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with shape operand element < -1.
func.func @testReshape(%arg0: tensor<10x10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = arith.constant dense<[100, -2]> : tensor<2xi32>
  // expected-error @+1 {{requires 'shape' to have dimensions greater than -1, but got -2 at index 1}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10x10xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with -1 in the shape can't infer the dimension.
func.func @testReshape(%arg0: tensor<10x10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = arith.constant dense<[101, -1]> : tensor<2xi32>
  // expected-error @+1 {{requires 'tensor' number of elements be a multiple of 101, but got 10000}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10x10xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with incorrect output rank.
func.func @testReshape(%arg0: tensor<10x10xf32>) -> tensor<?x?xf32> {
  %shape1 = arith.constant dense<[100]> : tensor<1xi32>
  // expected-error @+1 {{requires 'output' type 'tensor<?x?xf32>' to be cast compatible with expected type 'tensor<100xf32>'}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<?x?xf32>
  func.return %r1 : tensor<?x?xf32>
}

// -----
// tf.Reshape with incorrect output dimension.
func.func @testReshape(%arg0: tensor<1000xf32>) -> tensor<?x8x?xf32> {
  %shape1 = arith.constant dense<[10, 10, 10]> : tensor<3xi32>
  // expected-error @+1 {{requires 'output' type 'tensor<?x8x?xf32>' to be cast compatible with expected type 'tensor<10x10x10xf32>'}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<1000xf32>, tensor<3xi32>) -> tensor<?x8x?xf32>
  func.return %r1 : tensor<?x8x?xf32>
}

// -----
// tf.Reshape with a shape operand that has 0 for one of its elements.
func.func @testReshape(%arg0: tensor<10x10x10xf32>) -> tensor<?x0xf32> {
  %shape1 = arith.constant dense<[-1, 0]> : tensor<2xi32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2xi32>) -> tensor<?x0xf32>
  func.return %r1 : tensor<?x0xf32>
}

// -----
// tf.Reshape with a tensor operand that has 0 for one of its elements.
func.func @testReshape(%arg0: tensor<10x10x0xf32>) -> tensor<?x0xf32> {
  %shape1 = arith.constant dense<[-1, 0]> : tensor<2xi32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x0xf32>, tensor<2xi32>) -> tensor<?x0xf32>
  func.return %r1 : tensor<?x0xf32>
}

// -----
// tf.Reshape with a tensor operand that has non-static shape.
func.func @testReshape(%arg0: tensor<10x10x?xf32>) -> tensor<10x10xf32> {
  %shape1 = arith.constant dense<[10, 10]> : tensor<2xi32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x?xf32>, tensor<2xi32>) -> tensor<10x10xf32>
  func.return %r1 : tensor<10x10xf32>
}

// -----
// tf.Reshape with tensor operand that has non-static shape and shape operand
// with static shape.
func.func @testReshape(%arg0: tensor<10x10x?xf32>, %shape1: tensor<2xi32>) -> tensor<100x100xf32> {
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x?xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with tensor and shape operands with static shape.
func.func @testReshape(%arg0: tensor<10x10x10x10xf32>, %shape1: tensor<2xi32>) -> tensor<100x100xf32> {
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10x10xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  func.return %r1 : tensor<100x100xf32>
}

// -----

// CHECK-LABEL: func @testValidAvgPool
func.func @testValidAvgPool(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: func @testAvgPoolMissingDataFormatOk
func.func @testAvgPoolMissingDataFormatOk(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongDataType(tensor<1x7x7x16xi32>) -> tensor<1x1x1x16xi32> {
^bb0(%arg0: tensor<1x7x7x16xi32>):
  // expected-error @+1 {{must be tensor of floating-point values}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_INT", data_format = "NHWC", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xi32>) -> tensor<1x1x1x16xi32>
  func.return %0 : tensor<1x1x1x16xi32>
}

// -----

func.func @testAvgPoolWrongDataFormat(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'data_format' failed to satisfy constraint: 'NHWC' or 'NCHW' convnet data format}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "HWCN", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolNoKsize(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'ksize'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongKsizeCount(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongKsizeType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7.5, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----
func.func @testAvgPoolWrongKsizeIntType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1 : i32, 7 : i32, 7 : i32, 1 : i32], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolNoPadding(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'padding'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongPadding(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'padding' failed to satisfy constraint: string attribute whose value is SAME, or VALID}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "MAGIC", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolNoStrides(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'strides'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID"} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongStridesCount(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

func.func @testAvgPoolWrongStridesType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = ["1", "1", "1", "1"]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  func.return %0 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: func @testValidConv2D
func.func @testValidConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

// CHECK-LABEL: func @testValidDynamicConv2D
func.func @testValidDynamicConv2D(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<*xf32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @testValidConv3D
func.func @testValidConv3D(%arg0: tensor<256x32x32x32x3xf32>, %arg1: tensor<3x3x3x3x16xf32>) -> tensor<256x32x32x32x16xf32> {
  %0 = "tf.Conv3D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<256x32x32x32x3xf32>, tensor<3x3x3x3x16xf32>) -> tensor<256x32x32x32x16xf32>
  func.return %0 : tensor<256x32x32x32x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {
  // expected-error @+1 {{requires operands to be 4D tensor}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv3D(%arg0: tensor<256x32x32x32x3xf32>, %arg1: tensor<3x3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<256x32x32x32x16xf32>' are incompatible with return type(s) of operation 'tensor<256x32x32x16xf32>'}}
  %0 = "tf.Conv3D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<256x32x32x32x3xf32>, tensor<3x3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x2x16xf32>) -> tensor<256x32x32x16xf32> {
  // expected-error @+1 {{requires the number of input channels to be divisible by the number of filter input channels; found 3 and 2, respectively}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x2x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires attribute 'explicit_paddings'}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "EXPLICIT", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires explicit_paddings attribute length to be 8}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "EXPLICIT", strides = [1, 1, 1, 1], explicit_paddings = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires non negative explicit paddings}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "EXPLICIT", strides = [1, 1, 1, 1], explicit_paddings = [0, 0, 1, -1, 1, -1, 0, 0]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires strides attribute length to be 4}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires positive strides}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [0, 1, 1, 0]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<256x16x11x16xf32>' are incompatible with return type(s) of operation 'tensor<256x30x30x16xf32>'}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 2, 3, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x16x30x16xf32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<256x16x11x16xf32>' are incompatible with return type(s) of operation 'tensor<256x16x30x16xf32>'}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 2, 3, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x16x30x16xf32>
  func.return %0 : tensor<256x16x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<256x6x6x16xf32>' are incompatible with return type(s) of operation 'tensor<256x32x32x16xf32>'}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "EXPLICIT", dilations = [1, 2, 3, 4], explicit_paddings = [1, 2, 3, 4, 5, 6, 7, 8], strides = [5, 6, 7, 8]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32> {
  // expected-error @+1 {{op inferred type(s) 'tensor<256x30x30x16xf32>' are incompatible with return type(s) of operation 'tensor<256x32x32x16xf32>'}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x32x32x16xf32>
  func.return %0 : tensor<256x32x32x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires dilations attribute length to be 4}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1], dilations = [1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testConv2D(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
  // expected-error @+1 {{requires positive dilations}}
  %0 = "tf.Conv2D"(%arg0, %arg1) {padding = "SAME", strides = [1, 1, 1, 1], dilations = [1, 1, 0, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  func.return %0 : tensor<256x30x30x16xf32>
}

// -----

func.func @testMaxPoolGrad(%orig_input: tensor<f32>, %orig_output: tensor<10x12x12x64xf32>, %grad: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  // expected-error @+1 {{requires orig_input to be rank 4}}
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<f32>, tensor<10x12x12x64xf32>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
  func.return %result : tensor<10x24x24x64xf32>
}

// -----

func.func @testMaxPoolGrad(%orig_input: tensor<10x24x24x64xf32>, %orig_output: tensor<12x12x64xf32>, %grad: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  // expected-error @+1 {{requires orig_output to be rank 4}}
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<10x24x24x64xf32>, tensor<12x12x64xf32>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
  func.return %result : tensor<10x24x24x64xf32>
}

// -----

func.func @testMaxPoolGrad(%orig_input: tensor<10x24x24x64xf32>, %orig_output: tensor<10x12x12x64xf32>, %grad: tensor<12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  // expected-error @+1 {{requires grad to be rank 4}}
  %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
     data_format = "NHWC",
     ksize = [1, 2, 2, 1],
     padding = "VALID",
     strides = [1, 2, 2, 1]
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<12x12x64xf32>) -> tensor<10x24x24x64xf32>
  func.return %result : tensor<10x24x24x64xf32>
}

// -----

// CHECK-LABEL: func @testValidDepthwiseConv2dNative
func.func @testValidDepthwiseConv2dNative(tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x4xf32>) :
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {device = "", name = "MobilenetV2/expanded_conv/depthwise/depthwise", T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
  func.return %0 : tensor<256x30x30x12xf32>
}

// -----

// Test valid tf.FakeQuantWithMinMaxArgs
// CHECK-LABEL: func @testValidFakeQuantWithMinMaxArgs
func.func @testValidFakeQuantWithMinMaxArgs(tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 3} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxArgs
func.func @testInvalidFakeQuantWithMinMaxArgsWrongAttr(tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  // expected-error @+1 {{requires num_bits to be between 2 and 16, inclusive}}
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 0} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test valid tf.FakeQuantWithMinMaxVars
// CHECK-LABEL: func @testValidFakeQuantWithMinMaxVars
func.func @testValidFakeQuantWithMinMaxVars(tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func.func @testInvalidFakeQuantWithMinMaxVarsWrongAttr(tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  // expected-error @+1 {{requires num_bits to be between 2 and 16, inclusive}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 0} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func.func @testInvalidFakeQuantWithMinMaxVarsWrongMinRank(tensor<8x8x8x8xf32>, tensor<1xf32>, tensor<2xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>):
  // expected-error @+1 {{requires min to be a 0d float tensor}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<1xf32>, tensor<2xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func.func @testInvalidFakeQuantWithMinMaxVarsWrongMaxRank(tensor<8x8x8x8xf32>, tensor<f32>, tensor<2xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<2xf32>):
  // expected-error @+1 {{requires max to be a 0d float tensor}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<2xf32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func.func @testInvalidFakeQuantWithMinMaxVarsWrongMinType(tensor<8x8x8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
  // expected-error @+1 {{op operand #1 must be tensor of 32-bit float values}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func.func @testInvalidFakeQuantWithMinMaxVarsWrongMaxType(tensor<8x8x8x8xf32>, tensor<f32>, tensor<i32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>):
  // expected-error @+1 {{op operand #2 must be tensor of 32-bit float values}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<i32>) -> tensor<8x8x8x8xf32>
  func.return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test valid tf.FakeQuantWithMinMaxVarsPerChannel
// CHECK-LABEL: func @FakeQuantWithMinMaxVarsPerChannel
func.func @FakeQuantWithMinMaxVarsPerChannel(tensor<1x2x3x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x2x3x8xf32> {
^bb0(%arg0: tensor<1x2x3x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) : (tensor<1x2x3x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<1x2x3x8xf32>
  func.return %0 : tensor<1x2x3x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVarsPerChannel
func.func @FakeQuantWithMinMaxVarsPerChannel_ranked_inputs(tensor<f32>, tensor<8xf32>, tensor<8xf32>) -> tensor<f32> {
^bb0(%arg0: tensor<f32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
  // expected-error @+1 {{requires inputs to be at least 1d float tensor}}
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<8xf32>, tensor<8xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVarsPerChannel
func.func @FakeQuantWithMinMaxVarsPerChannel_mismatch_min_max(tensor<1x2x3x8xf32>, tensor<1xf32>, tensor<8xf32>) -> tensor<1x2x3x8xf32> {
^bb0(%arg0: tensor<1x2x3x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<8xf32>):
  // expected-error @+1 {{requires min and max to have same size as last dimension of inputs}}
  %0 = "tf.FakeQuantWithMinMaxVarsPerChannel"(%arg0, %arg1, %arg2) : (tensor<1x2x3x8xf32>, tensor<1xf32>, tensor<8xf32>) -> tensor<1x2x3x8xf32>
  func.return %0 : tensor<1x2x3x8xf32>
}

// -----

// Test invalid tf.Fill
func.func @testFill(tensor<i32>, tensor<f32>) -> tensor<?x?xf32> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<f32>):
  // expected-error @+1 {{requires dims to be a 1D tensor}}
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<?x?xf32>)
  func.return %0 : tensor<?x?xf32>
}

// -----

// Test invalid tf.Fill
func.func @testFill(tensor<2xi32>, tensor<1xf32>) -> tensor<?x?xf32> {
^bb0(%arg0: tensor<2xi32>, %arg1: tensor<1xf32>):
  // expected-error @+1 {{requires value to be a scalar}}
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<2xi32>, tensor<1xf32>) -> (tensor<?x?xf32>)
  func.return %0 : tensor<?x?xf32>
}

// -----

// Test valid tf.FusedBatchNorm
// CHECK-LABEL: func @testFusedBatchNorm
func.func @testFusedBatchNorm(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func.func @testFusedBatchNormWrongXType(tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires x to be a 4D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func.func @testFusedBatchNormWrongScaleType(tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xi32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{operand #1 must be tensor of 32-bit float values}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func.func @testFusedBatchNormWrongOffsetType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<2x8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires offset to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----
// Test invalid tf.FusedBatchNorm
func.func @testFusedBatchNormWrongMeanType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<?x8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires mean to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----
// Test invalid tf.FusedBatchNorm
func.func @testFusedBatchNormWrongVarianceType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<10x2xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<10x2xf32>):
  // expected-error @+1 {{requires variance to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<10x2xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<10x2xf32>)
  func.return %0#0 : tensor<8x8x8x8xf32>
}

// -----
func.func private @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// Test valid tf.If operation
// CHECK-LABEL: func @testValidIfOp
func.func @testValidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(f32) -> f32
func.func private @testIfElse(f32) -> f32

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, f32) -> f32 {
^bb0(%arg0: tensor<i1>, %arg1: f32):
  // expected-error @+1 {{operand #1 must be tensor of tf.dtype values}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, f32) -> f32

  func.return %1 : f32
}

// -----

func.func private @testIfElse(tensor<2xf32>) -> tensor<2xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{requires attribute 'then_branch'}}
  %1 = "tf.If"(%arg0, %arg1) {
    else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
func.func private @testIfElse(tensor<2xf32>) -> tensor<2xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{'tf.If' op 'then_branch' inputs (size = 2) should have the same number of values as inputs (size = 1)}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
func.func private @testIfElse(tensor<2xf32>) -> tensor<2xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{'tf.If' op 'then_branch' results (size = 2) should have the same number of values as results (size = 1)}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(tensor<*xf16>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{'tf.If' op 'then_branch' input type tensor<*xf16> is incompatible with input type tensor<2xf32> at index 0}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(tensor<2xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<3xf32>) -> tensor<*xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<*xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  // expected-error @+1 {{expects all branch input type(s) (tensor<2xf32>, tensor<3xf32>) at index 0 to be cast compatible}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

func.func private @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<3xf32>

// Test invalid tf.If operation
func.func @testInvalidIfOp(tensor<i1>, tensor<*xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  // expected-error @+1 {{'tf.If' op 'else_branch' result type tensor<3xf32> is incompatible with result type tensor<2xf32> at index 0}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// -----

// Test invalid tf.Yield operation (parent should be IfRegion)
func.func @testInvalidYieldOp(%arg0: f32) -> () {
  // expected-error @+1 {{'tf.Yield' op expects parent op to be one of 'tf.CaseRegion, tf.IfRegion, tf.WhileRegion'}}
  "tf.Yield"(%arg0) : (f32) -> ()
}

// -----

// Test valid tf.IfRegion operation
// CHECK-LABEL: func @testValidIfRegionOp
func.func @testValidIfRegionOp(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %neg = "tf.Neg"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%neg) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// Test valid tf.IfRegion operation with multiple results
// CHECK-LABEL: func @testValidIfRegionOpWithMultipleResults
func.func @testValidIfRegionOpWithMultipleResults(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0, %1, %2 = "tf.IfRegion"(%arg0) ({
     %t0 = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %t1 = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %t2 = "tf.Acosh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%t0, %t1, %t2) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> ()
    }, {
     %e0 = "tf.Neg"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %e1 = "tf.Relu"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %e2 = "tf.Sin"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e0, %e1, %e2) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  %3 = "tf.Add"(%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %4 = "tf.Add"(%2, %3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %4 : tensor<2xf32>
}

// -----

// Test invalid type for operand #0 for tf.IfRegion operation
func.func @testInvalidIfRegionOpType0(%arg0: f32, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{operand #0 must be 0D tensor of 1-bit signless integer values, but got 'f32'}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (f32) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// tf.IfRegion operation should have 2 regions
func.func @testInvalidIfRegionOp1Region(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{op expected 2 regions}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testInvalidIfRegionOpNoRegions(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{op expected 2 regions}}
  %0 = "tf.IfRegion"(%arg0) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testInvalidIfRegionOp3Regions(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{op expected 2 regions}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %te = "tf.Relu"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%te) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// tf.IfRegion regions should be terminated with a tf.Yield
func.func @testIfRegionThenTerminator(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+2 {{block with no terminator}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
   }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testIfRegionElseTerminator(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+5 {{block with no terminator}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// tf.Region yield number of results should match op number of results
func.func @testIfRegionThenResultCount(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.IfRegion' op then results (size = 2) should have the same number of values as results (size = 1)}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t, %t) : (tensor<2xf32>, tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testIfRegionElseResultCount(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.IfRegion' op else results (size = 2) should have the same number of values as results (size = 1)}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e, %e) : (tensor<2xf32>, tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// tf.IfRegion yield types should match op result types
func.func @testIfRegionOpYieldMismatchThen(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.IfRegion' op then result type tensor<i1> is incompatible with result type tensor<2xf32> at index 0}}
  %0 = "tf.IfRegion"(%arg0) ({
     "tf.Yield"(%arg0) : (tensor<i1>) -> ()
    }, {
     %e = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testIfRegionOpYieldMismatchElse(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.IfRegion' op else result type tensor<i1> is incompatible with result type tensor<2xf32> at index 0}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     "tf.Yield"(%arg0) : (tensor<i1>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// value generated in one branch cannot be consumed in the other branch
func.func @testIfRegionElseConsumingThen(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     // expected-error @+1 {{use of undeclared SSA value name}}
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testIfRegionThenConsumingElse(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
   %0 = "tf.IfRegion"(%arg0) ({
     // expected-error @+1 {{does not dominate this use}}
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
      // expected-note @+1 {{operand defined here}}
      %t = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
      "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// The regions for IfRegion themselves cannot have any arguments
func.func @testInvalidIfRegionThenArg(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %neg = "tf.Neg"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  // expected-error @+1 {{'tf.IfRegion' op region #0 should have no arguments}}
  %0 = "tf.IfRegion"(%arg0) ({
     ^bb(%arg_bb: tensor<2xf32>):
     %t = "tf.Abs"(%arg_bb) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     %e = "tf.Acos"(%neg) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

func.func @testInvalidIfRegionElseArg(%arg0: tensor<i1>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  %neg = "tf.Neg"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  // expected-error @+1 {{'tf.IfRegion' op region #1 should have no arguments}}
  %0 = "tf.IfRegion"(%arg0) ({
     %t = "tf.Abs"(%neg) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%t) : (tensor<2xf32>) -> ()
    }, {
     ^bb(%arg_bb: tensor<2xf32>):
     %e = "tf.Acos"(%arg_bb) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e) : (tensor<2xf32>) -> ()
    }) { is_stateless = false} : (tensor<i1>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// -----

// Test valid tf.MatrixBandPart
// CHECK-LABEL: func @testValidMatrixBandPartOp
func.func @testValidMatrixBandPartOp(%arg0: tensor<64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<64x64xbf16> {
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<64x64xbf16>
  func.return %0 : tensor<64x64xbf16>
}

// -----

// Test valid tf.MatrixBandPart
// CHECK-LABEL: func @testValidMatrixBandPartOp3D
func.func @testValidMatrixBandPartOp3D(%arg0: tensor<64x64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<64x64x64xbf16> {
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<64x64x64xbf16>
  func.return %0 : tensor<64x64x64xbf16>
}

// -----

// Test valid tf.MatrixBandPart
// CHECK-LABEL: func @testValidMatrixBandPartOpUnranked
func.func @testValidMatrixBandPartOpUnranked(%arg0: tensor<*xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<*xbf16> {
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<*xbf16>, tensor<i64>, tensor<i64>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

// -----

// Test valid tf.MatrixBandPart
// CHECK-LABEL: func @testValidMatrixBandPartOpUnrankedBand
func.func @testValidMatrixBandPartOpUnrankedBand(%arg0: tensor<64x64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<*xbf16> {
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

// -----

// Test valid tf.MatrixBandPart
// CHECK-LABEL: func @testValidMatrixBandPartOpCompatibleDynamicShapes
func.func @testValidMatrixBandPartOpCompatibleDynamicShapes(%arg0: tensor<?x10x?xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<?x?x8xbf16> {
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<?x10x?xbf16>, tensor<i64>, tensor<i64>) -> tensor<?x?x8xbf16>
  func.return %0 : tensor<?x?x8xbf16>
}

// -----

// Test invalid tf.MatrixBandPart
func.func @testInvalidMatrixBandPartOp(%arg0: tensor<64x64x64xbf16>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<64x64xbf16> {
  // expected-error @+1 {{op failed to verify that all of {input, band} have dynamically equal types}}
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64x64xbf16>, tensor<i64>, tensor<i64>) -> tensor<64x64xbf16>
  func.return %0 : tensor<64x64xbf16>
}

// -----

// Test invalid tf.MatrixBandPart
func.func @testInvalidMatrixBandPartOp(%arg0: tensor<i64>, %arg1: tensor<64x64xi64>, %arg2: tensor<i64>) -> tensor<i64> {
  // expected-error @+1 {{op requires `input` to have rank of at least 2, but found 'tensor<i64>'}}
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<64x64xi64>, tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// Test invalid tf.MatrixBandPart
func.func @testInvalidMatrixBandPartOp(%arg0: tensor<64x64xi64>, %arg1: tensor<32xi64>, %arg2: tensor<i64>) -> tensor<64x64xi64> {
  // expected-error @+1 {{op requires `num_lower` to have 0 dimensions, but found 'tensor<32xi64>'}}
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64xi64>, tensor<32xi64>, tensor<i64>) -> tensor<64x64xi64>
  func.return %0 : tensor<64x64xi64>
}

// -----

// Test invalid tf.MatrixBandPart
func.func @testInvalidMatrixBandPartOp(%arg0: tensor<64x64xi64>, %arg1: tensor<i64>, %arg2: tensor<32xi64>) -> tensor<64x64xi64> {
  // expected-error @+1 {{op requires `num_upper` to have 0 dimensions, but found 'tensor<32xi64>'}}
  %0 = "tf.MatrixBandPart"(%arg0, %arg1, %arg2) : (tensor<64x64xi64>, tensor<i64>, tensor<32xi64>) -> tensor<64x64xi64>
  func.return %0 : tensor<64x64xi64>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.{|Stateful}PartitionedCall
//===--------------------------------------------------------------------===//

// Test valid tf.PartitionedCall
// CHECK-LABEL: func @testValidPartitionedCall
func.func @testValidPartitionedCall(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// Test invalid tf.PartitionedCall
func.func @testUndefinedPartitionedCall(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{'f' attribute refers to an undefined function: @nonexistant_pcall_func}}
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @nonexistant_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// -----

// Test invalid tf.PartitionedCall
func.func @testInvalidPartitionedCall(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{argument count mismatch: 'args' has 1 arguments, but '@pcall_func_2' expects 2}}
  %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func_2} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @pcall_func_2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// Test valid tf.StatefulPartitionedCall
// CHECK-LABEL: func @testValidStatefulPartitionedCall
func.func @testValidStatefulPartitionedCall(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @pcall_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

func.func @testUndefinedCallee(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{'f' attribute refers to an undefined function: @nonexistant_pcall_func}}
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @nonexistant_pcall_func} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// -----

func.func @testArgMismatch(%arg0: tensor<i32>) -> tensor<i32> {
  // expected-error @+1 {{argument count mismatch: 'args' has 1 arguments, but '@pcall_func_2' expects 2}}
  %0 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @pcall_func_2} : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @pcall_func_2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.Select
//===--------------------------------------------------------------------===//

// Test valid tf.Select
// CHECK-LABEL: func @testSelect
func.func @testSelect(%arg0: tensor<3xi1>, %arg1: tensor<3x2xf16>, %arg2: tensor<3x2xf16>) -> tensor<3x2xf16> {
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<3x2xf16>
  func.return %0: tensor<3x2xf16>
}

// -----

func.func @testInvalidSelect(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf16>, %arg2: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error @+1 {{requires that, when pred is a vector, the shape matches the first dimension of t and e}}
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  func.return %0: tensor<2x3xf16>
}

// -----

// Test invalid tf.Select - broadcasting then/else parameters is not supported
func.func @selectBroadcastThen(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // expected-error @+1 {{requires t and e have compatible shapes}}
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// -----

func.func @invalidSelect(%arg0: tensor<2xi1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<2xi32> {
  // expected-error @+1 {{requires that t and e are nonscalar when pred is a vector}}
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// -----

func.func @invalidSelect(%arg0: tensor<1x8xi1>, %arg1: tensor<1x8x8xi32>, %arg2: tensor<1x8x8xi32>) -> tensor<1x8x8xi32> {
  // expected-error @+1 {{requires that pred is a scalar OR has the same rank as t and e OR is a vector}}
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<1x8xi1>, tensor<1x8x8xi32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi32>
  func.return %0: tensor<1x8x8xi32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.SelectV2
//===--------------------------------------------------------------------===//

// Test valid tf.SelectV2
// CHECK-LABEL: func @selectV2BroadcastThen
func.func @selectV2BroadcastThen(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// -----

// Test valid tf.SelectV2
// CHECK-LABEL: func @selectV2BroadcastElse
func.func @selectV2BroadcastElse(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// -----

// Test valid tf.SelectV2
// CHECK-LABEL: func @selectV2BroadcastPred
func.func @selectV2BroadcastPred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// -----

// CHECK-LABEL: func @selectV2BroadcastAll
func.func @selectV2BroadcastAll(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  func.return %0: tensor<8x8x8xi32>
}

// -----

// CHECK-LABEL: func @selectV2DynamicRanked
func.func @selectV2DynamicRanked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  func.return %0: tensor<2x?x8xi32>
}

// -----

// CHECK-LABEL: func @selectV2Unranked
func.func @selectV2Unranked(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

// -----

// Test invalid tf.SelectV2: this is an invalid broadcast for the predicate
func.func @testInvalidSelectV2(%arg0: tensor<3xi1>, %arg1: tensor<3x2xf16>, %arg2: tensor<3x2xf16>) -> tensor<3x2xf16> {
  // expected-error @+1 {{operands don't have broadcast-compatible shapes}}
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<3x2xf16>
  func.return %0: tensor<3x2xf16>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.Softmax
//===--------------------------------------------------------------------===//

// Test valid tf.Softmax
// CHECK-LABEL: func @testSoftmax
func.func @testSoftmax(tensor<8x16xf32>) -> tensor<8x16xf32> {
^bb0(%arg0: tensor<8x16xf32>):
  %0 = "tf.Softmax"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

// Test invalid tf.Softmax
func.func @testSoftmax(%arg0 : tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{requires operand to have rank at least 1}}
  %0 = "tf.Softmax"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Test valid tf.SoftmaxCrossEntropyWithLogits
// CHECK-LABEL: func @testSoftmaxCrossEntropyWithLogits
func.func @testSoftmaxCrossEntropyWithLogits(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<2x3xf32>) {
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<3xf32>, tensor<2x3xf32>
}

// -----

func.func @testSoftmaxCrossEntropyWithLogits(%arg0: tensor<2xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  // expected-error @+1 {{requires features and labels to be broadcast compatible to rank two}}
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<2xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  func.return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
}

// -----

func.func @testSoftmaxCrossEntropyWithLogits(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
  // expected-error @+1 {{requires features and labels to be broadcast compatible to rank two}}
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>)
  func.return %0#0, %0#1 : tensor<3xf32>, tensor<3xf32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.SpaceToBatchND
//===--------------------------------------------------------------------===//

// Test valid tf.SpaceToBatchND
// CHECK-LABEL: func @testSpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x10xf32> {
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf32>
  func.return %0 : tensor<?x?x?x10xf32>
}

// -----

// Test valid tf.SpaceToBatchND
// CHECK-LABEL: func @testSpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>) -> tensor<36x2x3x10xf32> {
  %block_shape = "tf.Const"() {value = dense<[4, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %paddings = "tf.Const"() {value = dense<[[1, 2], [1, 1]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<36x2x3x10xf32>
  func.return %0 : tensor<36x2x3x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2x2xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x10xf32> {
  // expected-error @+1 {{requires rank of block_shape = 1; got 2}}
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf32>
  func.return %0 : tensor<?x?x?x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2xi64>, %paddings: tensor<2xi64>) -> tensor<?x?x?x10xf32> {
  // expected-error @+1 {{requires rank of paddings = 2; got 1}}
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?x?x10xf32>
  func.return %0 : tensor<?x?x?x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2xi64>, %paddings: tensor<2x10xi64>) -> tensor<?x?x?x10xf32> {
  // expected-error @+1 {{requires paddings.shape[1] to be 2; got 10}}
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x10xi64>) -> tensor<?x?x?x10xf32>
  func.return %0 : tensor<?x?x?x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<4xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x?xf32> {
  // expected-error @+1 {{requires block_shape.shape[0] must equal paddings.shape[0]}}
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<4xi64>, tensor<2x2xi64>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5xf32>, %block_shape: tensor<2xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{requires rank of input >= 1 + rank of block}}
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x10xf32> {
  %block_shape = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{requires all values of block_shape to be >= 1; failed for dimension 1}}
  %1 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf32>
  func.return %1 : tensor<?x?x?x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2xi64>) -> tensor<?x?x?x10xf32> {
  %paddings = "tf.Const"() {value = dense<[[1, 0], [-1, 0]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  // expected-error @+1 {{requires all values of paddings to be >= 0; failed for dimension 1}}
  %1 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf32>
  func.return %1 : tensor<?x?x?x10xf32>
}

// -----

// Test invalid tf.SpaceToBatchND
func.func @testSpaceToBatchND(%input: tensor<3x5x7x10xf32>) -> tensor<36x2x3x10xf32> {
  %block_shape = "tf.Const"() {value = dense<[4, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %paddings = "tf.Const"() {value = dense<[[1, 2], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  // expected-error @+1 {{requires block_shape[i] divides input_shape[i + 1] + paddings[i, 0] + paddings[i, 1]; failed for i=1}}
  %1 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<36x2x3x10xf32>
  func.return %1 : tensor<36x2x3x10xf32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.SparseSoftmaxCrossEntropyWithLogits
//===--------------------------------------------------------------------===//

// Test valid tf.SparseSoftmaxCrossEntropyWithLogits
// CHECK-LABEL: func @testSparseSoftmaxCrossEntropyWithLogits
func.func @testSparseSoftmaxCrossEntropyWithLogits(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi32>) -> (tensor<3xf32>, tensor<2x3xf32>) {
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>) -> (tensor<3xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<3xf32>, tensor<2x3xf32>
}

// -----

func.func @testSparseSoftmaxCrossEntropyWithLogits(%arg0: tensor<3xf32>, %arg1: tensor<3xi32>) -> (tensor<3xf32>, tensor<2x3xf32>) {
  // expected-error @+1 {{requires features operand of rank two}}
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xi32>) -> (tensor<3xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<3xf32>, tensor<2x3xf32>
}

// -----

func.func @testSparseSoftmaxCrossEntropyWithLogits(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xi32>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // expected-error @+1 {{requires labels operand of rank one}}
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xi32>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// -----

func.func @testSparseSoftmaxCrossEntropyWithLogits(%arg0: tensor<2x3xf32>, %arg1: tensor<3xi32>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // expected-error @+1 {{requires features and labels with matching first dimension}}
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xi32>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// -----

func.func private @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test valid 'While' operation
// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----
func.func @testWhileUndefinedCond(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{cond refers to an undefined function : undefined_func}}
  %0 = "tf.While"(%arg0, %arg1) {cond = @undefined_func, body = @body, is_stateless = false} : (tensor<i1>, tensor<f32>) -> (tensor<f32>)
  func.return %0 : tensor<f32>
}

func.func private @body(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32>

// -----
func.func @testWhileUndefinedBody(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{body refers to an undefined function : undefined_func}}
  %0 = "tf.While"(%arg0, %arg1) {cond = @cond, body = @undefined_func, is_stateless = false} : (tensor<i1>, tensor<f32>) -> (tensor<f32>)
  func.return %0 : tensor<f32>
}

func.func private @cond(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<i1>

// -----

func.func private @testWhileCond(tensor<*xf32>) -> ()
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{requires cond function to have exactly one result}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----

func.func private @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xi32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{'tf.While' op input type tensor<*xf32> is incompatible with result type tensor<*xi32> at index 0}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xi32>)

  func.return %1 : tensor<*xi32>
}

// -----

func.func private @testWhileCond(tensor<*xi32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{'tf.While' op input type tensor<*xf32> is incompatible with condition input type tensor<*xi32> at index 0}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----

func.func private @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{'tf.While' op inputs (size = 1) should have the same number of values as body inputs (size = 2)}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----

func.func private @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*xf32>) -> (tensor<*xi32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{'tf.While' op result type tensor<*xf32> is incompatible with body result type tensor<*xi32> at index 0}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----

func.func private @testWhileCond(tensor<3xf32>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<4xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func.func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{'tf.While' op condition input type tensor<3xf32> is incompatible with body input type tensor<4xf32> at index 0}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  func.return %1 : tensor<*xf32>
}

// -----

func.func private @testWhileCond(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<16xf32>>>)

// Test invalid 'While' operation verifier that detects incompatible tf.resource
// subtypes.
func.func @testWhileResult(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<16xf32>>>) {
^bb0(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>):
  // expected-error @+1 {{'tf.While' op input type tensor<*x!tf_type.resource<tensor<32xf32>>> is incompatible with result type tensor<!tf_type.resource<tensor<16xf32>>> at index 0}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<16xf32>>>)

  func.return %1 : tensor<!tf_type.resource<tensor<16xf32>>>
}

// -----

func.func private @testWhileCond(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<*xf32>>>)

// Test 'While' operation verifier allows compatible tf.resource subtypes.
// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<*xf32>>>) {
^bb0(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource<tensor<*xf32>>>)

  func.return %1 : tensor<!tf_type.resource<tensor<*xf32>>>
}

// -----

func.func private @testWhileCond(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<i1>)
func.func private @testWhileBody(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource>)

// Test 'While' operation verifier treats tf.resource with subtype and without
// subtype as compatible types.
// CHECK-LABEL: func @testWhileResult
func.func @testWhileResult(tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource>) {
^bb0(%arg0: tensor<*x!tf_type.resource<tensor<32xf32>>>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*x!tf_type.resource<tensor<32xf32>>>) -> (tensor<!tf_type.resource>)

  func.return %1 : tensor<!tf_type.resource>
}

// -----

func.func private @cond(tensor<1x?x3xf32>) -> tensor<i1>
func.func private @body(tensor<1x?x3xf32>) -> tensor<1x?x3xf32>

// Test shape invariant 'While' operation verifier with different operand and
// result shapes.
// CHECK-LABEL: func @testShapeInvariantWhile
func.func @testShapeInvariantWhile(%arg0: tensor<1x2x3xf32>) -> tensor<1x8x3xf32> {
  %0 = "tf.While"(%arg0) {cond = @cond, body = @body, is_stateless = false, shape_invariant} : (tensor<1x2x3xf32>) -> tensor<1x8x3xf32>
  func.return %0 : tensor<1x8x3xf32>
}

// -----
// WhileRegion tests

// Simple While region
// CHECK-LABEL: testValidWhileRegion
func.func @testValidWhileRegion(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%add, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)

  func.return %0#0 : tensor<*xf32>
}

// -----

// While region with no inputs (and hence no outputs) (infinite loop)
// CHECK-LABEL: testValidWhileRegionNoInputs
func.func private @printer(tensor<i32>) -> ()
func.func @testValidWhileRegionNoInputs() -> () {
  "tf.WhileRegion"() (
    {
      %true = arith.constant dense<1> : tensor<i1>
      "tf.Yield"(%true) : (tensor<i1>) -> ()
    },
    {
      %one = arith.constant dense<1> : tensor<i32>
      func.call @printer(%one) : (tensor<i32>) -> ()
      // TODO(b/159753381): tf.IfRegion implicit terminator not working
      "tf.Yield"() : () -> ()
    }
  ) { is_stateless = true } : () -> ()
  func.return
}

// -----
// Invalid while tests. There are 5 sets of type matching that is required
//   I = input, O = output, BI, BO = body input/output, CI = cond input.
//   [I, O], [I, CI], [I, BI], [BO, BI], [BO, O].
// Each check can fail due to number or type mismatch. However, these
// conditions are not all independent. So we just check I->{CI, BI}, O->BO, and
// in addition I->O. BO->BI mismatch cannot be independently created without
// breaking one of these mismatches. That gives us 4x2 tests. In addition
// condition result needs to be tensor<i1>, for which we have 3
// additional validation tests. All these tests are based on the following
// valid while

func.func @testInvalidTestValidBase(%arg0 : tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%barg) : (tensor<i32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

func.func @testInvalidWhileRegion_I_CI_CountMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op inputs (size = 1) should have the same number of values as condition inputs (size = 0)}}
  %0 = "tf.WhileRegion"(%arg0) (
    {
     //^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%barg) : (tensor<i32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_I_CI_TypeMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op input type tensor<i32> is incompatible with condition input type tensor<f32> at index 0}}
  %0 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<f32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%barg) : (tensor<i32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_I_BI_CountMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op inputs (size = 1) should have the same number of values as body inputs (size = 2)}}
  %0 = "tf.WhileRegion"(%arg0) (
     {
       ^bb0(%carg: tensor<i32>):
        %true = arith.constant dense<1> : tensor<i1>
        "tf.Yield"(%true) : (tensor<i1>) -> ()
     },
     {
       ^bb0(%barg0: tensor<i32>, %barg1 : tensor<f32>):
        "tf.Yield"(%barg0) : (tensor<i32>) -> ()
     }
  ) {is_stateless = false} : (tensor<i32>) -> (tensor<i32>)

  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_I_BI_TypeMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op input type tensor<i32> is incompatible with body input type tensor<f32> at index 0}}
  %0 = "tf.WhileRegion"(%arg0) (
     {
       ^bb0(%carg: tensor<i32>):
        %true = arith.constant dense<1> : tensor<i1>
        "tf.Yield"(%true) : (tensor<i1>) -> ()
     },
     {
       ^bb0(%barg: tensor<f32>):
        %c = "tf.Cast"(%barg) : (tensor<f32>) -> tensor<i32>
        "tf.Yield"(%c) : (tensor<i32>) -> ()
     }
  ) {is_stateless = false} : (tensor<i32>) -> (tensor<i32>)

  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_O_BO_CountMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op results (size = 1) should have the same number of values as body results (size = 2)}}
  %0 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%barg, %barg) : (tensor<i32>, tensor<i32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>)
  func.return %0#0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegionMismatch_O_BO_TypeMismatch(%arg0 : tensor<i32>, %arg1: tensor<f32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op result type tensor<i32> is incompatible with body result type tensor<f32> at index 0}}
  %0 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%arg1) : (tensor<f32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>)
  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_I_O_CountMismatch(%arg0 : tensor<i32>) -> (tensor<i32>) {
  // expected-error@+1 {{'tf.WhileRegion' op inputs (size = 1) should have the same number of values as results (size = 2)}}
  %0:2 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%barg, %barg) : (tensor<i32>, tensor<i32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %0#0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegion_I_O_TypeMismatch(%arg0: tensor<i32>, %arg1 : tensor<f32>) -> (tensor<f32>) {
  // expected-error@+1 {{'tf.WhileRegion' op input type tensor<i32> is incompatible with result type tensor<f32> at index 0}}
  %0 = "tf.WhileRegion"(%arg0) (
    {
     ^bb0(%carg: tensor<i32>):
      %false = arith.constant dense<false> : tensor<i1>
      "tf.Yield"(%false) : (tensor<i1>) -> ()
    },
    {
     ^bb0(%barg: tensor<i32>):
      "tf.Yield"(%arg1) : (tensor<f32>) -> ()
    }
  ) { is_stateless = true } : (tensor<i32>) -> (tensor<f32>)
  func.return %0 : tensor<f32>
}
// -----

func.func @testInvalidWhileRegionConditionOutputCount2(%arg : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op condition should have a single tensor<i1> result}}
  %0 = "tf.WhileRegion"(%arg) (
     {
       ^bb0(%carg: tensor<i32>):
        %true = arith.constant dense<1> : tensor<i1>
        "tf.Yield"(%true, %true) : (tensor<i1>, tensor<i1>) -> ()
     },
     {
       ^bb0(%barg: tensor<i32>):
        "tf.Yield"(%barg) : (tensor<i32>) -> ()
     }
  ) {is_stateless = false} : (tensor<i32>) -> (tensor<i32>)

  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegionConditionOutputCount0(%arg : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op condition should have a single tensor<i1> result}}
  %0 = "tf.WhileRegion"(%arg) (
     {
       ^bb0(%carg: tensor<i32>):
        "tf.Yield"() : () -> ()
     },
     {
       ^bb0(%barg: tensor<i32>):
        "tf.Yield"(%barg) : (tensor<i32>) -> ()
     }
  ) {is_stateless = false} : (tensor<i32>) -> (tensor<i32>)

  func.return %0 : tensor<i32>
}

// -----

func.func @testInvalidWhileRegionConditionOutputType(%arg : tensor<i32>) -> (tensor<i32>) {
  // expected-error @+1 {{'tf.WhileRegion' op condition should have a single tensor<i1> result}}
  %0 = "tf.WhileRegion"(%arg) (
     {
       ^bb0(%carg: tensor<i32>):
        "tf.Yield"(%carg) : (tensor<i32>) -> ()
     },
     {
       ^bb0(%barg: tensor<i32>):
        "tf.Yield"(%barg) : (tensor<i32>) -> ()
     }
  ) {is_stateless = false} : (tensor<i32>) -> (tensor<i32>)

  func.return %0 : tensor<i32>
}

// -----

// Test shape invariant 'WhileRegion' operation verifier with different operand
// and result shapes.
// CHECK-LABEL: func @testShapeInvariantWhileRegion
func.func @testShapeInvariantWhileRegion(%arg0: tensor<1x2x3xf32>) -> tensor<1x8x3xf32> {
  %0 = "tf.WhileRegion"(%arg0) ({
  ^cond(%carg0: tensor<1x?x3xf32>):
    %1 = "tf.SomeCondOp"(%carg0) : (tensor<1x?x3xf32>) -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^body(%barg0: tensor<1x?x3xf32>):
    %1 = "tf.SomeBodyOp"(%barg0) : (tensor<1x?x3xf32>) -> tensor<1x?x3xf32>
    "tf.Yield"(%1) : (tensor<1x?x3xf32>) -> ()
  }) {is_stateless = false, shape_invariant} : (tensor<1x2x3xf32>) -> tensor<1x8x3xf32>
  func.return %0 : tensor<1x8x3xf32>
}

// -----

// CHECK-LABEL: func @testValidShape
func.func @testValidShape(tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<4xi32>, tensor<?xi32>) {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<*xf32>):
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<4xi32>
  %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  func.return %0, %1 : tensor<4xi32>, tensor<?xi32>
}

// -----

func.func @testShapeWrongResultElemType(%arg0: tensor<1x32x32x16xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result #0 must be tensor of 32/64-bit signed integer values}}
  %0 = "tf.Shape"(%arg0) : (tensor<1x32x32x16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @testShapeWrongResultDim(tensor<1x32x32x16xf32>) -> tensor<3x2xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires 1D type for result}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @testShapeMismatchDim(tensor<1x32x32x16xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires dimension size of result to match rank of operand}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @testShapeWrongResultDimDynamic(tensor<*xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<*xf32>):
  // expected-warning @+1 {{has static shape result for unranked operand}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @testValidShapeN
func.func @testValidShapeN(%arg0 : tensor<1x32x32x16xf32>, %arg1 : tensor<*xf32>) -> (tensor<4xi32>, tensor<?xi32>) {
  // CHECK-NEXT: "tf.ShapeN"
  %0:2 = "tf.ShapeN"(%arg0, %arg1) : (tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<4xi32>, tensor<?xi32>)
  func.return %0#0, %0#1 : tensor<4xi32>, tensor<?xi32>
}

// -----

func.func @testShapeNWrongResultElemType(%arg0: tensor<1x32x32x16xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result #1 must be tensor of 32/64-bit signed integer values}}
  %0:2 = "tf.ShapeN"(%arg0, %arg0) : (tensor<1x32x32x16xf32>, tensor<1x32x32x16xf32>) -> (tensor<4xi32>, tensor<4xf32>)
  func.return %0#1 : tensor<4xf32>
}

// -----

func.func @testShapeNWrongResultDim(tensor<1x32x32x16xf32>) -> tensor<2x2xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires 1D type for result #1}}
  %0:2 = "tf.ShapeN"(%arg0, %arg0) : (tensor<1x32x32x16xf32>, tensor<1x32x32x16xf32>) -> (tensor<4xi32>, tensor<2x2xi32>)
  func.return %0#1 : tensor<2x2xi32>
}

// -----

func.func @testShapeNMismatchDim(tensor<1x32x32x16xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires dimension size of result #1 to match rank of operand #1}}
  %0:2 = "tf.ShapeN"(%arg0, %arg0) : (tensor<1x32x32x16xf32>, tensor<1x32x32x16xf32>) -> (tensor<4xi32>, tensor<2xi32>)
  func.return %0#1 : tensor<2xi32>
}

// -----

func.func @testShapeNWrongResultDimDynamic(tensor<*xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<*xf32>):
  // expected-warning @+1 {{has static shape result #1 for unranked operand #1}}
  %0:2 = "tf.ShapeN"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<?xi32>, tensor<2xi32>)
  func.return %0#1 : tensor<2xi32>
}

// -----

func.func @testShapeNWrongNumResults(tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{requires 3 result(s), got 2 result(s)}}
  %0:2 = "tf.ShapeN"(%arg0, %arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<?xi32>, tensor<?xi32>)
  func.return
}

// -----

// CHECK-LABEL: func @testValidVariableShape
func.func @testValidVariableShape(%arg0: tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>, %arg1: tensor<*x!tf_type.resource>) -> (tensor<4xi32>, tensor<?xi32>) {
  %0 = "tf.VariableShape"(%arg0) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<4xi32>
  %1 = "tf.VariableShape"(%arg1) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource>) -> tensor<?xi32>
  func.return %0, %1 : tensor<4xi32>, tensor<?xi32>
}

// -----

func.func @testVariableShapeMultipleSubtypes(%arg0: tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>, tensor<1x32x32x16xf32>>>) {
  // expected-error @+1 {{requires resource input type to have at most 1 subtype}}
  %0 = "tf.VariableShape"(%arg0) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>, tensor<1x32x32x16xf32>>>) -> tensor<4xi32>
  func.return
}

// -----

func.func @testVariableShapeWrongResultElemType(%arg0: tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<?xf32> {
  // expected-error @+1 {{result #0 must be tensor of 32/64-bit signed integer values}}
  %0 = "tf.VariableShape"(%arg0) : (tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @testVariableShapeWrongResultDim(%arg0: tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<2x3xi32> {
  // expected-error @+1 {{requires 1D type for result}}
  %0 = "tf.VariableShape"(%arg0) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @testVariableShapeMismatchDim(%arg0: tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<2xi32> {
  // expected-error @+1 {{requires dimension size of result to match rank of operand}}
  %0 = "tf.VariableShape"(%arg0) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @testVariableShapeWrongResultDimDynamic(%arg0: tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<2xi32> {
  // expected-warning @+1 {{has static shape result for unranked operand}}
  %0 = "tf.VariableShape"(%arg0) {output = "tfdtype$DT_INT32"} : (tensor<*x!tf_type.resource<tensor<*xf32>>>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @testVariableShapeWrongNumResources(%arg0: tensor<1x2x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<4xi32> {
  // expected-error @+1 {{requires input to have one resource}}
  %0 = "tf.VariableShape"(%arg0)  : (tensor<1x2x!tf_type.resource<tensor<1x32x32x16xf32>>>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// Test invalid tf.Const
func.func @testConst() -> tensor<f32> {
  // expected-error @+1 {{attribute 'value' failed to satisfy constraint: constant vector/tensor}}
  %0 = "tf.Const"() {T = "tfdtype$DT_FLOAT", value = 1.0 : f32} : () -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Test invalid tf.ToBool
func.func @testInvalidToBool(%arg0: tensor<i32>) -> tensor<1xi1> {
  // expected-error @+1 {{op inferred type(s) 'tensor<i1>' are incompatible with return type(s) of operation 'tensor<1xi1>'}}
  %0 = "tf.ToBool"(%arg0) : (tensor<i32>) -> tensor<1xi1>
  func.return %0 : tensor<1xi1>
}

// -----

// Test invalid tf.TPUPartitionedInputV2 with packing
func.func @testPackedTPUPartitionedInputV2(tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32> {
^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>):
  // expected-error @+1 {{expected 1 inputs, got 2}}
  %0 = "tf.TPUPartitionedInputV2"(%arg0, %arg1) {partition_dims = [2, 1], is_packed = true} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// Test invalid tf.TPUPartitionedInputV2 without packing
func.func @testUnpackedTPUPartitionedInputV2(tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32> {
^bb0(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>):
  // expected-error @+1 {{expected 2 inputs, got 1}}
  %0 = "tf.TPUPartitionedInputV2"(%arg0) {partition_dims = [2, 1], is_packed = false} : (tensor<2x4xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// Test valid tf.Transpose
// CHECK-LABEL: testTranspose
func.func @testTranspose(tensor<2x3xf32>) -> tensor<3x2xf32> {
^bb0(%arg0: tensor<2x3xf32>):
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// Test tf.Transpose with partial unknown shape
// CHECK-LABEL: testTranspose
func.func @testTranspose(tensor<2x?xf32>) -> tensor<?x2xf32> {
^bb0(%arg0: tensor<2x?xf32>):
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x?xf32>, tensor<2xi32>) -> tensor<?x2xf32>
  func.return %0 : tensor<?x2xf32>
}

// -----

// Test tf.Transpose with different partial unknown shape
// CHECK-LABEL: testTranspose
func.func @testTranspose(tensor<2x?x?xf32>) -> tensor<3x?x2xf32> {
^bb0(%arg0: tensor<2x?x?xf32>):
  %cst = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x?x?xf32>, tensor<3xi32>) -> tensor<3x?x2xf32>
  func.return %0 : tensor<3x?x2xf32>
}

// -----

// Test tf.Transpose with invalid rank of perm
func.func @testTranspose(tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<3x2xf32> {
^bb0(%arg0: tensor<2x3xf32>, %arg1: tensor<1x2xi32>):
  // expected-error @+1 {{expected perm to be a 1-D Tensor, got perm of rank 2}}
  %0 = "tf.Transpose"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<1x2xi32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// Test tf.Transpose with invalid size of perm
func.func @testTranspose(tensor<2x3xf32>) -> tensor<3x2xf32> {
^bb0(%arg0: tensor<2x3xf32>):
  %cst = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  // expected-error @+1 {{expected perm to be a 1-D Tensor of size equal to the rank of x, got perm of size 3, and x of rank 2}}
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<3xi32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

// Test tf.Transpose with invalid rank of y
func.func @testTranspose(tensor<2x3xf32>) -> tensor<3x2x1xf32> {
^bb0(%arg0: tensor<2x3xf32>):
  %cst = arith.constant dense<[1, 0]> : tensor<2xi32>
  // expected-error @+1 {{x should be of the same rank with y, got x of rank 2, and y of rank 3}}
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2x1xf32>
  func.return %0 : tensor<3x2x1xf32>
}

// -----

// Test tf.Transpose with invalid shape of y
func.func @testTranspose(tensor<2x3x4xf32>) -> tensor<3x2x4xf32> {
^bb0(%arg0: tensor<2x3x4xf32>):
  %cst = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
  // expected-error @+1 {{requires y.shape[0] (3) to be equal to x.shape[perm[2]] (4)}}
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3x4xf32>, tensor<3xi32>) -> tensor<3x2x4xf32>
  func.return %0 : tensor<3x2x4xf32>
}

// -----

// Test invalid tf.Less
func.func @testLess(tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
  // expected-error @+1 {{op result #0 must be tensor of bool values}}
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// Test valid tf.ConcatV2
func.func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<i32>) -> tensor<?xf32> {
  %0 = "tf.ConcatV2"(%arg, %arg, %axis) : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// tf.ConcatV2 with wrong 'axis' element type
func.func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<f32>) -> tensor<?xf32> {
  // expected-error @+1 {{operand #2 must be tensor of 32/64-bit signed integer values}}
  %0 = "tf.ConcatV2"(%arg, %arg, %axis) : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// tf.ConcatV2 missing required 'axis' operand
func.func @testConcatV2() -> tensor<?xf32> {
  // expected-error @+1 {{expected 1 or more operands}}
  %0 = "tf.ConcatV2"() : () -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: testAll
func.func @testAll(%arg0: tensor<2x2xi1>, %arg1: tensor<i32>) -> tensor<i1> {
  %0 = "tf.All"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi1>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: testAll64
func.func @testAll64(%arg0: tensor<2x2xi1>, %arg1: tensor<i64>) -> tensor<i1> {
  %0 = "tf.All"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi1>, tensor<i64>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @testAllFloat(%arg0: tensor<2x2xi1>, %arg1: tensor<f32>) -> tensor<i1> {
  // expected-error @+1 {{'tf.All' op operand #1 must be tensor of 32/64-bit signed integer values}}
  %0 = "tf.All"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi1>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @testAllI32(%arg0: tensor<2x2xi32>, %arg1: tensor<f32>) -> tensor<i32> {
  // expected-error @+1 {{'tf.All' op operand #0 must be tensor of bool values}}
  %0 = "tf.All"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xi32>, tensor<f32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @testEqualOpIncompatibleShapeTrue(%x: tensor<5xf32>, %y: tensor<4xf32>) -> tensor<5xi1> {
  // expected-error @+1 {{operands don't have broadcast-compatible shapes}}
  %0 = "tf.Equal"(%x, %y) {incompatible_shape_error = true} : (tensor<5xf32>, tensor<4xf32>) -> tensor<5xi1>
  func.return %0 : tensor<5xi1>
}

// -----

// CHECK-LABEL: testEqualOpIncompatibleShapeFalse
func.func @testEqualOpIncompatibleShapeFalse(%x: tensor<5xf32>, %y: tensor<4xf32>) -> tensor<*xi1> {
  %0 = "tf.Equal"(%x, %y) {incompatible_shape_error = false} : (tensor<5xf32>, tensor<4xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

func.func @testNotEqualOpIncompatibleShapeTrue(%x: tensor<5xf32>, %y: tensor<4xf32>) -> tensor<5xi1> {
  // expected-error @+1 {{operands don't have broadcast-compatible shapes}}
  %0 = "tf.NotEqual"(%x, %y) {incompatible_shape_error = true} : (tensor<5xf32>, tensor<4xf32>) -> tensor<5xi1>
  func.return %0 : tensor<5xi1>
}

// -----

// CHECK-LABEL: testNotEqualOpIncompatibleShapeFalse
func.func @testNotEqualOpIncompatibleShapeFalse(%x: tensor<5xf32>, %y: tensor<4xf32>) -> tensor<*xi1> {
  %0 = "tf.NotEqual"(%x, %y) {incompatible_shape_error = false} : (tensor<5xf32>, tensor<4xf32>) -> tensor<*xi1>
  func.return %0 : tensor<*xi1>
}

// -----

func.func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1x1xi32>) -> tensor<*xf32> { // expected-error @+1 {{requires axis to be of scalar type (or vector type for older versions)}}
  %0 = "tf.ConcatV2"(%arg, %arg, %axis) : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<1x1xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1x1xi32>) -> tensor<*xf32> {
  // expected-error @+1 {{requires axis to be of scalar type (or vector type for older versions)}}
  %0 = "tf.Concat"(%axis, %arg, %arg) : (tensor<1x1xi32>, tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testConcatV2(%arg0: tensor<8x16xf32>, %arg1: tensor<8xf32>, %axis: tensor<i32>) -> tensor<*xf32> {
  // expected-error @+1 {{operand type 'tensor<8xf32>' is not compatible with preceding operands; expected rank: 2}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<8x16xf32>, tensor<8xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Valid Concat operation with concat axis 1 or -1.
func.func @testConcatV2(%arg0: tensor<8x16xf32>, %arg1: tensor<8x8xf32>, %axis: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<8x16xf32>, tensor<8x8xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testConcatV2(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>, %axis: tensor<i32>) -> tensor<*xf32> {
  // expected-error @+1 {{operand type 'tensor<16x8xf32>' is not compatible with preceding operands; expected dimension at index 1: 16}}
  %0 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<8x16xf32>, tensor<16x8xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Valid Concat operation with concat axis 1 or -1.
func.func @testConcatV2(%arg0: tensor<8x8xf32>, %arg1: tensor<?x4xf32>, %arg2: tensor<*xf32>, %arg3: tensor<8x?xf32>, %axis: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %arg3, %axis) : (tensor<8x8xf32>, tensor<?x4xf32>, tensor<*xf32>, tensor<8x?xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testInvalidInvertPermutationOp(%arg0: tensor<8x8xi32>) -> tensor<8x8xi32> {
  // expected-error @+1 {{'tf.InvertPermutation' op requires input x to be 1-dimensional}}
  %0 = "tf.InvertPermutation"(%arg0) : (tensor<8x8xi32>) -> tensor<8x8xi32>
  func.return %0 : tensor<8x8xi32>
}

// -----

// Valid Pack operation.
func.func @testPack(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "tf.Pack"(%arg0, %arg1) {axis = 1 : i64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}


// -----

func.func @testPack(%arg0: tensor<4x8xf32>, %arg1: tensor<4x2xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{operand type 'tensor<4x2xf32>' is not compatible with preceding operands; expected dimension at index 1: 8}}
  %0 = "tf.Pack"(%arg0, %arg1) {axis = 1 : i64} : (tensor<4x8xf32>, tensor<4x2xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testPack(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %axis: tensor<i32>) -> tensor<*xf32> {
  // expected-error @+1 {{attribute 'axis' should be within range [-3, 3); actual value: 3}}
  %0 = "tf.Pack"(%arg0, %arg1) {axis = 3 : i64} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// Valid slice operation.
func.func @testSlice(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  %sizes = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  %0 = "tf.Slice"(%arg0, %arg1, %sizes) : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @testSlice_begin_2d(%arg0: tensor<4xi32>, %begins: tensor<2x2xi64>) -> tensor<3xi32> {
  %sizes = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires begin operand to be 1D tensor}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<2x2xi64>, tensor<1xi64>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testSlice_size_two_much_elements(%arg0: tensor<4xi32>) -> tensor<3xi32> {
  %begins = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> (tensor<2xi64>)
  // expected-error @+1 {{requires begin and size operands to have the same number of elements}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<2xi64>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testSlice_begin_negative(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  %begins = "tf.Const"() {value = dense<[-1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires 0 <= begin[i] <= begin[i] + size[i] <= Di}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @testSlice_begin_out_of_bound(%arg0: tensor<4xi32>) -> tensor<2xi32> {
  %begins = "tf.Const"() {value = dense<[4]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires 0 <= begin[i] <= begin[i] + size[i] <= Di}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @testSlice_unknown_begin_out_of_bounds(%arg0: tensor<4xi32>, %begins: tensor<1xi64>) -> tensor<3xi32> {
  %sizes = "tf.Const"() {value = dense<[5]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires size[i] <= Di, even if begin[i] is unknown at compile time}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testSlice_unknown_begin_in_bounds(%arg0: tensor<4xi32>, %begins: tensor<1xi64>) -> tensor<3xi32> {
  %sizes = "tf.Const"() {value = dense<[4]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @testSlice_unequal_output_input_rank(%arg0: tensor<4xi32>, %begins: tensor<1xi64>) -> tensor<i32> {
  %sizes = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires output to have the same rank as input, but got input rank 1 and output rank 0}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @testSlice_wrong_output_size(%arg0: tensor<4xi32>) -> tensor<1xi32> {
  %begins = "tf.Const"() {value = dense<[1]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  %sizes = "tf.Const"() {value = dense<[2]> : tensor<1xi64>} : () -> (tensor<1xi64>)
  // expected-error @+1 {{requires output size to have the same size of slice, got slice size 2 and output size 1}}
  %0 = "tf.Slice"(%arg0, %begins, %sizes) : (tensor<4xi32>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// -----

func.func @testSlice_wrong_type(%arg0: tensor<28x1x100xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<1x1x100xi32> {
  // expected-error @+1 {{failed to verify that input and output must have same element type}}
  %0 = "tf.Slice"(%arg0, %arg1, %arg2) : (tensor<28x1x100xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x1x100xi32>
  func.return %0 : tensor<1x1x100xi32>
}

// -----

// Valid StridedSlice operation.
func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<i64>, %end: tensor<i64>, %strides: tensor<i64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{requires begin, end and strides to be 1D tensors}}
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<32xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{with less than 32 elements}}
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<32xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<?xi64>, %end: tensor<3xi64>, %strides: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{to have the same number of elements}}
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<?xi64>, tensor<3xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>) -> tensor<?x?xf32> {
  %begin = "tf.Const"() { value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<?xi64>
  %end = "tf.Const"() { value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<?xi64>
  %strides = "tf.Const"() { value = dense<[2, 3, 4]> : tensor<3xi64> } : () -> tensor<?xi64>

  // expected-error @+1 {{to have the same number of elements}}
  %1 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<2xi32>, %end: tensor<2xi32>) -> tensor<?x?xf32> {
  %strides = "tf.Const"() { value = dense<[2, 0]> : tensor<2xi32> } : () -> tensor<2xi32>

  // expected-error @+1 {{requires non-zero strides}}
  %1 = "tf.StridedSlice"(%input, %begin, %end, %strides) : (tensor<4x8xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// -----

func.func @testStridedSlice(%input: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{cannot have multiple ellipses}}
  %0 = "tf.StridedSlice"(%input, %begin, %end, %strides) {ellipsis_mask = 3}: (tensor<4x8xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %depth: tensor<i32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  %depth = "tf.Const"() { value = dense<-5> : tensor<i32> } : () -> tensor<i32>
  // expected-error @+1 {{depth must be non-negative}}
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %depth: tensor<2xi32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{requires depth to be a scalar}}
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<2xi32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %depth: tensor<i32>, %on_value: tensor<2xf32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{requires on_value to be a scalar}}
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<2xf32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %depth: tensor<i32>, %on_value: tensor<f32>, %off_value: tensor<2xf32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{requires off_value to be a scalar}}
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -1 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<2xf32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testOneHot(%indices: tensor<3xi32>, %depth: tensor<i32>, %on_value: tensor<f32>, %off_value: tensor<f32>) -> tensor<3x5xf32> {
  // expected-error @+1 {{expected axis (-2) to be -1 or between [0, 1]}}
  %result = "tf.OneHot"(%indices, %depth, %on_value, %off_value) {axis = -2 : i64} : (tensor<3xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<3x5xf32>
  func.return %result : tensor<3x5xf32>
}

// -----

func.func @testSplitNonConstSplitDim(%input: tensor<4x4xf32>, %split_dim: tensor<i32>) {
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<i32>, tensor<4x4xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

func.func @testSplitUnknownRankSplitDim(%input: tensor<4x4xf32>, %split_dim: tensor<*xi32>) {
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<*xi32>, tensor<4x4xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

func.func @testSplitUnknownRankInput(%input: tensor<*xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

func.func @testSplitUnknownDimInput(%input: tensor<4x?x4xf32>) {
  %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x?x4xf32>) -> (tensor<4x?x4xf32>, tensor<4x?x4xf32>)
  func.return
}

// -----

func.func @testSplitNonScalarSplitDim(%input: tensor<4x4xf32>, %split_dim: tensor<1xi32>) {
  // expected-error @+1 {{split dimension should be an integer scalar tensor}}
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<1xi32>, tensor<4x4xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitScalarInput(%input: tensor<f32>, %split_dim: tensor<i32>) {
  // expected-error @+1 {{cannot split scalar input tensor}}
  %0:2 = "tf.Split"(%split_dim, %input) : (tensor<i32>, tensor<f32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitLargeSplitDim(%input: tensor<4x8xf32>) {
  %cst = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{split dimension must be in range [-2, 2)}}
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x8xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitSmallSplitDim(%input: tensor<4x8xf32>) {
  %cst = "tf.Const"() {value = dense<-3> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{split dimension must be in range [-2, 2)}}
  %0:2 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x8xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitSmallSplitDim(%input: tensor<4x8xf32>) {
  %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{dimension #0 not divisible by the number of result tensors}}
  %0:3 = "tf.Split"(%cst, %input) : (tensor<i32>, tensor<4x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSqueezeOutOfBounds(%arg0: tensor<?x?x10xf32>) -> tensor<?x10xf32> {
  // expected-error @+1 {{squeeze dimension -4 not in [-3, 3)}}
  %0 = "tf.Squeeze"(%arg0) { squeeze_dims = [-4] }: (tensor<?x?x10xf32>) -> tensor<?x10xf32>
  func.return %0 : tensor<?x10xf32>
}

// -----

func.func @testTernaryEinsum(%arg0: tensor<2x3xf32>){
  // expected-error @+1 {{supports at most two operands}}
  %0 = "tf.Einsum"(%arg0, %arg0, %arg0) {equation = "ab,cd,ef->"} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<*xf32>)
  func.return
}

// -----

func.func @testTopKV2WrongInputRank(%input: tensor<f32>, %k: tensor<i32>) {
  // expected-error @+1 {{op requires input operand to have at least 1 dimension}}
  %0:2 = "tf.TopKV2"(%input, %k) : (tensor<f32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  func.return
}

// -----

func.func @testTopKV2WrongKRank(%input: tensor<8xf32>, %k: tensor<5xi32>) {
  // expected-error @+1 {{op requires k operand to be 0D tensor}}
  %0:2 = "tf.TopKV2"(%input, %k) : (tensor<8xf32>, tensor<5xi32>) -> (tensor<*xf32>, tensor<*xi32>)
  func.return
}

// -----

func.func @testSplitVScalarInput(%input: tensor<f32>, %split_sizes: tensor<2xi32>, %split_dim: tensor<i32>) {
  // expected-error @+1 {{cannot split scalar input tensor}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<f32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVNonScalarSplitDim(%input: tensor<4x4xf32>, %split_sizes: tensor<2xi32>, %split_dim: tensor<1xi32>) {
  // expected-error @+1 {{split dimension should be an integer scalar tensor}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<1xi32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVSplitDimOutOfRange(%input: tensor<4x4xf32>, %split_sizes: tensor<2xi32>) {
  %split_dim = "tf.Const"() {value = dense<100>: tensor<i32>} : () -> (tensor<i32>)
  // expected-error @+1 {{split dimension must be in range [-2, 2)}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVWrongSplitSizesType(%input: tensor<4x4xf32>, %split_sizes: tensor<2x2xi32>, %split_dim: tensor<i32>) {
  // expected-error @+1 {{op split sizes should be a 1D tensor of 2 elements}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2x2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVMultipleDynamicSizes(%input: tensor<4x4xf32>) {
  %split_dim = "tf.Const"() {value = dense<1>: tensor<i32>} : () -> (tensor<i32>)
  %split_sizes = "tf.Const"() {value = dense<[-1, -1]>: tensor<2xi32>} : () -> (tensor<2xi32>)
  // expected-error @+1 {{cannot have more than one dynamic dimension in split sizes}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVSplitSizeOutOfRange(%input: tensor<4x4xf32>) {
  %split_dim = "tf.Const"() {value = dense<1>: tensor<i32>} : () -> (tensor<i32>)
  %split_sizes = "tf.Const"() {value = dense<[-1, 100]>: tensor<2xi32>} : () -> (tensor<2xi32>)
  // expected-error @+1 {{split sizes must sum up to be less than or equal to the dimension size along split dimension, found 100 vs 4}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVSplitSizeOutOfRange(%input: tensor<4x4xf32>) {
  %split_dim = "tf.Const"() {value = dense<1>: tensor<i32>} : () -> (tensor<i32>)
  %split_sizes = "tf.Const"() {value = dense<[2, 3]>: tensor<2xi32>} : () -> (tensor<2xi32>)
  // expected-error @+1 {{split sizes must sum up to the dimension size along split dimension, found 5 vs 4}}
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitV1(%input: tensor<4x4xf32>) {
  %split_dim = "tf.Const"() {value = dense<1>: tensor<i32>} : () -> (tensor<i32>)
  %split_sizes = "tf.Const"() {value = dense<[-1, 4]>: tensor<2xi32>} : () -> (tensor<2xi32>)
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

func.func @testSplitV2(%input: tensor<4x4xf32>) {
  %split_dim = "tf.Const"() {value = dense<1>: tensor<i32>} : () -> (tensor<i32>)
  %split_sizes = "tf.Const"() {value = dense<[3, 1]>: tensor<2xi32>} : () -> (tensor<2xi32>)
  %0:2 = "tf.SplitV"(%input, %split_sizes, %split_dim) : (tensor<4x4xf32>, tensor<2xi32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return
}

// -----

func.func @testSplitVDynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?xi32>, %arg2: tensor<i32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0:2 = "tf.SplitV"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?xi32>, tensor<i32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)
  func.return %0#0, %0#1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.All
//===--------------------------------------------------------------------===//

func.func @testAllDimWrongRank(%input: tensor<4x6xi1>, %dims: tensor<2x2xi32>) {
  // expected-error @+1 {{dimensions can only be 0D or 1D tensor}}
  %0 = "tf.All"(%input, %dims) : (tensor<4x6xi1>, tensor<2x2xi32>) -> (tensor<*xi1>)
  func.return
}

// -----

func.func @testAllDimOutOfRange(%input: tensor<4x6xi1>) {
  %dims = "tf.Const"() {value = dense<[-1, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  // expected-error @+1 {{1-th dimension should be in the range of [-2, 2)}}
  %0 = "tf.All"(%input, %dims) : (tensor<4x6xi1>, tensor<2xi32>) -> (tensor<*xi1>)
  func.return
}

// -----

//===--------------------------------------------------------------------===//
//  tf.Any
//===--------------------------------------------------------------------===//

func.func @testAnyDimWrongRank(%input: tensor<4x6xi1>, %dims: tensor<2x2xi32>) {
  // expected-error @+1 {{dimensions can only be 0D or 1D tensor}}
  %0 = "tf.Any"(%input, %dims) : (tensor<4x6xi1>, tensor<2x2xi32>) -> (tensor<*xi1>)
  func.return
}

// -----

func.func @testAnyDimOutOfRange(%input: tensor<4x6xi1>) {
  %dims = "tf.Const"() {value = dense<[-1, 5]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  // expected-error @+1 {{1-th dimension should be in the range of [-2, 2)}}
  %0 = "tf.Any"(%input, %dims) : (tensor<4x6xi1>, tensor<2xi32>) -> (tensor<*xi1>)
  func.return
}

// -----

//===--------------------------------------------------------------------===//
//  tf.Unpack
//===--------------------------------------------------------------------===//

func.func @testUnpackAxisOutOfRange(%input: tensor<2x6xf32>) {
  // expected-error @+1 {{axis attribute must be in the range of [-2, 2)}}
  %0:2 = "tf.Unpack"(%input) {axis = 5} : (tensor<2x6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
  func.return
}

// -----

func.func @testAxisUnknownDim(%input: tensor<?x6xf32>) {
  // CHECK: tf.Unpack
  %0:2 = "tf.Unpack"(%input) {axis = 0} : (tensor<?x6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
  func.return
}

// -----

func.func @testAxisDim(%input: tensor<2x6xf32>) {
  // expected-error @+1 {{result count must be equal to 6}}
  %0:2 = "tf.Unpack"(%input) {axis = -1} : (tensor<2x6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
  func.return
}

// -----

//===--------------------------------------------------------------------===//
//  tf.UnsortedSegment{Max|Min|Prod|Sum}
//===--------------------------------------------------------------------===//

// CHECK-LABEL: unsortedSegmentReduction
func.func @unsortedSegmentReduction(%data: tensor<?x10x8xf32>, %segment_ids: tensor<7x?xi32>, %num_segments: tensor<i32>) {
  // CHECK: tf.UnsortedSegmentMin
  %0 = "tf.UnsortedSegmentMin"(%data, %segment_ids, %num_segments) : (tensor<?x10x8xf32>, tensor<7x?xi32>, tensor<i32>) -> (tensor<?x8xf32>)
  func.return
}

// -----

func.func @unsortedSegmentReduction(%data: tensor<7x10x8xf32>, %segment_ids: tensor<7x10xi32>, %num_segments: tensor<2x3xi32>) {
  // expected-error @+1 {{number of segments should be a 0-D tensor}}
  %0 = "tf.UnsortedSegmentMax"(%data, %segment_ids, %num_segments) : (tensor<7x10x8xf32>, tensor<7x10xi32>, tensor<2x3xi32>) -> (tensor<?x8xf32>)
  func.return
}

// -----

func.func @unsortedSegmentReduction(%data: tensor<7x10x8xf32>, %segment_ids: tensor<7x9xi32>, %num_segments: tensor<i32>) {
  // expected-error @+1 {{requires segment ids shape to be a prefix of data shape, but dimension #1 differs: 9 vs. 10}}
  %0 = "tf.UnsortedSegmentProd"(%data, %segment_ids, %num_segments) : (tensor<7x10x8xf32>, tensor<7x9xi32>, tensor<i32>) -> (tensor<?x8xf32>)
  func.return
}

// -----

func.func @unsortedSegmentReduction(%data: tensor<7x10x8xf32>, %segment_ids: tensor<7x10x8x1xi32>, %num_segments: tensor<i32>) {
  // expected-error @+1 {{requires segment ids rank to be less than or equal to data's rank}}
  %0 = "tf.UnsortedSegmentSum"(%data, %segment_ids, %num_segments) : (tensor<7x10x8xf32>, tensor<7x10x8x1xi32>, tensor<i32>) -> (tensor<?x8xf32>)
  func.return
}

// -----

func.func @unsortedSegmentReduction(%data: tensor<7x10x8xf32>, %segment_ids: tensor<7x10xi32>) {
  %num_segments = "tf.Const"() {value = dense<-5> : tensor<i32>} : () -> (tensor<i32>)
  // expected-error @+1 {{num of segments cannot be negative}}
  %0 = "tf.UnsortedSegmentSum"(%data, %segment_ids, %num_segments) : (tensor<7x10x8xf32>, tensor<7x10xi32>, tensor<i32>) -> (tensor<?x8xf32>)
  func.return
}

// -----


//===--------------------------------------------------------------------===//
//  tf.GatherV2
//===--------------------------------------------------------------------===//

func.func @testGatherV2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x3xf32> {
  %0 = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5x3xf32>
  func.return %1 : tensor<16x2x5x3xf32>
}

// -----

// Verify that the batch_dims can be equal to the rank of the indices.
func.func @testGatherV2(%arg0: tensor<16x4xf32>, %arg1: tensor<16xi32>) -> tensor<16xf32> {
  %0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = 1 : i64} : (tensor<16x4xf32>, tensor<16xi32>, tensor<1xi32>) -> tensor<16xf32>
  func.return %1 : tensor<16xf32>
}

// -----

func.func @testGatherV2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x3xf32> {
  %0 = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  // expected-error @+1 {{batch_dims (-3) must be in range [-2, 3)}}
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -3 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5x3xf32>
  func.return %1 : tensor<16x2x5x3xf32>
}

// -----

func.func @testGatherV2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x3xf32> {
  %0 = "tf.Const"() { value = dense<[[-4]]> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
  // expected-error @+1 {{requires axis to have rank at most 1}}
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1x1xi32>) -> tensor<16x2x5x3xf32>
  func.return %1 : tensor<16x2x5x3xf32>
}

// -----

func.func @testGatherV2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x3xf32> {
  %0 = "tf.Const"() { value = dense<[-4]> : tensor<1xi32> } : () -> tensor<1xi32>
  // expected-error @+1 {{axis (-4) must be in range [-3, 3)}}
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5x3xf32>
  func.return %1 : tensor<16x2x5x3xf32>
}

// -----

func.func @testGatherV2(%arg0: tensor<16x2x3xf32>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x3xf32> {
  %0 = "tf.Const"() { value = dense<[0]> : tensor<1xi32> } : () -> tensor<1xi32>
  // expected-error @+1 {{requires axis (0) to be greater than or equal to batch_dims (1)}}
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) {batch_dims = -1 : i64} : (tensor<16x2x3xf32>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5x3xf32>
  func.return %1 : tensor<16x2x5x3xf32>
}

// -----

//===--------------------------------------------------------------------===//
//  tf.StridedSliceGrad
//===--------------------------------------------------------------------===//

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  // CHECK: tf.StridedSliceGrad
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<i64>, %end: tensor<2xi64>, %strides: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{requires begin, end and strides to be 1D tensors}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<32xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{with less than 32 elements}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<32xi64>, tensor<2xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<?xi64>, %end: tensor<3xi64>, %strides: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{have the same number of elements}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<?xi64>, tensor<3xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  %begin = "tf.Const"() { value = dense<[0, 0]> : tensor<2xi64> } : () -> tensor<?xi64>
  %end = "tf.Const"() { value = dense<[5, 10]> : tensor<2xi64> } : () -> tensor<?xi64>
  %strides = "tf.Const"() { value = dense<[2, 3, 4]> : tensor<3xi64> } : () -> tensor<?xi64>

  // expected-error @+1 {{have the same number of elements}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  %strides = "tf.Const"() { value = dense<[2, 0]> : tensor<2xi32> } : () -> tensor<2xi32>

  // expected-error @+1 {{requires non-zero strides}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi32>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>, %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{cannot have multiple ellipses}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) {ellipsis_mask = 3} : (tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @stridedSliceGrad(%dy: tensor<4x8xf32>, %begin: tensor<2xi64>, %end: tensor<2xi64>, %strides: tensor<2xi64>, %shape: tensor<1x2xi64>) -> tensor<?x?xf32> {
  // expected-error @+1 {{'shape' operand must be 1D tensor, but got 2D tensor}}
  %0 = "tf.StridedSliceGrad"(%shape, %begin, %end, %strides, %dy) : (tensor<1x2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<4x8xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @testDynamicStitch() -> tensor<2x2xf32> {
  // expected-error @+1 {{requires attribute N with value >= 1}}
  %0 = "tf.DynamicStitch"() : () -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<f32> {
  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires non scalar output}}
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %indices = "tf.Const"() {value = dense<[-1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires non-negative index values; found -1}}
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{requires shape of data with type 'tensor<3x2xf32>' to have prefix matching with shape of the corresponding index type 'tensor<2xi32>'}}
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2xf32>, %arg1: tensor<2x2x3xf32>) -> (tensor<5x2xf32>) {
  %indices0 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %indices1 = "tf.Const"() {value = dense<[[3, 2], [1, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>

  // expected-error @+1 {{inconsistent shaped data and index pairs; inferred item shapes [2] and [3] don't match}}
  %0 = "tf.DynamicStitch"(%indices0, %indices1, %arg0, %arg1) : (tensor<i32>, tensor<2x2xi32>, tensor<2xf32>, tensor<2x2x3xf32>) -> tensor<5x2xf32>
  func.return %0 : tensor<5x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %indices = "tf.Const"() {value = dense<[2, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{missing index 1}}
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<3x2xf32> {
  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error @+1 {{has invalid output type; should be compatible with inferred type 'tensor<2x2xf32>'}}
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  func.return %0 : tensor<3x2xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<?x2xi32>, %arg1: tensor<?x3x3xf32>) -> (tensor<*xf32>) {
  // expected-error @+1 {{requires shape of data with type 'tensor<?x3x3xf32>' to have prefix matching with shape of the corresponding index type 'tensor<?x2xi32>'}}
  %0 = "tf.DynamicStitch"(%arg0, %arg1) : (tensor<?x2xi32>, tensor<?x3x3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testDynamicStitch(%arg0: tensor<?x3xf32>, %arg1: tensor<2x?xf32>) -> (tensor<2x3x2xf32>) {
  %indices0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %indices1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // expected-error @+1 {{has invalid output type; should be compatible with inferred type 'tensor<2x2x3xf32>'}}
  %0 = "tf.DynamicStitch"(%indices0, %indices1, %arg0, %arg1) : (tensor<i32>, tensor<i32>, tensor<?x3xf32>, tensor<2x?xf32>) -> tensor<2x3x2xf32>
  func.return %0 : tensor<2x3x2xf32>
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<i32>, %shape0: tensor<3xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires N to be at least 2, got 1}}
  %0 = "tf.ConcatOffset"(%concat_dim, %shape0) : (tensor<i32>, tensor<3xi32>) -> tensor<3xi32>
  func.return
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<i32>, %shape0: tensor<3xi32>, %shape1: tensor<3xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires sizes of shapes and offsets to be the same, got sizes 2 and 3}}
  %0:3 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>, tensor<3xi32>)
  func.return
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<1xi32>, %shape0: tensor<3xi32>, %shape1: tensor<3xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires concat_dim to be a scalar, got tensor of rank 1}}
  %0:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<1xi32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<i32>, %shape0: tensor<3xi32>, %shape1: tensor<3xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires operand and result 1 to have compatible shapes}}
  %0:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3xi32>) -> (tensor<3xi32>, tensor<8xi32>)
  func.return
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<i32>, %shape0: tensor<3xi32>, %shape1: tensor<3x3xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires shape tensor operand 1 to be of rank 1, got tensor of rank 2}}
  %0:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<3x3xi32>) -> (tensor<3xi32>, tensor<3x3xi32>)
  func.return
}

// -----

func.func @testConcatOffest(%concat_dim: tensor<i32>, %shape0: tensor<3xi32>, %shape1: tensor<8xi32>) {
  // expected-error @+1 {{'tf.ConcatOffset' op requires shape tensor (rank 1) operand 1 to be of length 3, got tensor (rank 1) of length 8}}
  %0:2 = "tf.ConcatOffset"(%concat_dim, %shape0, %shape1) : (tensor<i32>, tensor<3xi32>, tensor<8xi32>) -> (tensor<3xi32>, tensor<8xi32>)
  func.return
}

// -----

func.func @tensor_scatter_update(%tensor: tensor<f32>, %indices: tensor<4x2xi32>, %updates: tensor<4x4xf32>) -> tensor<f32> {
  // expected-error @+1 {{op requires tensor operand to have at least 1 dimension}}
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<f32>, tensor<4x2xi32>, tensor<4x4xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @tensor_scatter_update(%tensor: tensor<4x4x4xf32>, %indices: tensor<i32>, %updates: tensor<4x4xf32>) -> tensor<4x4x4xf32> {
  // expected-error @+1 {{op requires indices operand to have at least 1 dimension}}
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<4x4x4xf32>, tensor<i32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %0 : tensor<4x4x4xf32>
}

// -----

func.func @tensor_scatter_update(%tensor: tensor<4x4x4xf32>, %indices: tensor<4x2xi32>, %updates: tensor<f32>) -> tensor<4x4x4xf32> {
  // CHECK: TensorScatterUpdate
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<4x4x4xf32>, tensor<4x2xi32>, tensor<f32>) -> tensor<4x4x4xf32>
  func.return %0 : tensor<4x4x4xf32>
}

// -----

func.func @tensor_scatter_update(%tensor: tensor<4xf32>, %indices: tensor<4x2xi32>, %updates: tensor<4x4xf32>) -> tensor<4x4x4xf32> {
  // expected-error @+1 {{op requires tensor operand with rank greater than or equal to the indices operand's last dimensions}}
  %0 = "tf.TensorScatterUpdate"(%tensor, %indices, %updates) : (tensor<4xf32>, tensor<4x2xi32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  func.return %0 : tensor<4x4x4xf32>
}

// -----

// CHECK-LABEL: func @testParseExampleV2DenseOnlyValid
func.func @testParseExampleV2DenseOnlyValid(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %dense_keys : tensor<2x!tf_type.string>, %dense_default_0 : tensor<?xf32>, %dense_default_1 : tensor<?xf32>) -> (tensor<32xf32>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  %result:2 = "tf.ParseExampleV2"(%serialized, %names, %empty_str_vector, %dense_keys, %empty_str_vector, %dense_default_0, %dense_default_1) {dense_shapes = [#tf_type.shape<>, #tf_type.shape<>], num_sparse = 0 : i64, result_segment_sizes = array<i32: 0, 0, 0, 2, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<?xf32>, tensor<?xf32>) -> (tensor<32xf32>, tensor<32xf32>)
  func.return %result#0 : tensor<32xf32>
}

// -----

func.func @testParseExampleV2DenseMismatchedInputOutput(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %dense_keys : tensor<2x!tf_type.string>, %dense_default_0 : tensor<?xf32>, %dense_default_1 : tensor<?xf32>) -> (tensor<32xf32>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // expected-error @+1 {{output 'dense_values' should have same length as attribute 'Tdense'}}
  %result:3 = "tf.ParseExampleV2"(%serialized, %names, %empty_str_vector, %dense_keys, %empty_str_vector, %dense_default_0, %dense_default_1) {dense_shapes = [#tf_type.shape<>, #tf_type.shape<>], num_sparse = 0 : i64, result_segment_sizes = array<i32: 0, 0, 0, 3, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<?xf32>, tensor<?xf32>) -> (tensor<32xf32>, tensor<32xf32>, tensor<32xi64>)
  func.return %result#0 : tensor<32xf32>
}

// -----

// CHECK-LABEL: func @testParseExampleV2SparseOnlyValid
func.func @testParseExampleV2SparseOnlyValid(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %sparse_keys : tensor<2x!tf_type.string>) -> (tensor<?x2xi64>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  %result:6 = "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %empty_str_vector, %empty_str_vector) {dense_shapes = [], num_sparse = 2 : i64, result_segment_sizes = array<i32: 2, 2, 2, 0, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>) -> (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>, tensor<2xi64>)
  func.return %result#0 : tensor<?x2xi64>
}

// -----

func.func @testParseExampleV2SparseInvalidNumSparse(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %sparse_keys : tensor<2x!tf_type.string>) -> (tensor<?x2xi64>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // expected-error @+1 {{attribute 'num_sparse' should be the same as the length of attribute 'sparse_types'}}
  %result:6 = "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %empty_str_vector, %empty_str_vector) {dense_shapes = [], num_sparse = 3 : i64, result_segment_sizes = array<i32: 2, 2, 2, 0, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>) -> (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>, tensor<2xi64>)
  func.return %result#0 : tensor<?x2xi64>
}

// -----

func.func @testParseExampleV2SparseInvalidSparseIndicesOutput(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %sparse_keys : tensor<2x!tf_type.string>) -> (tensor<?x2xi64>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // expected-error @+1 {{output 'sparse_indices' should have same length as attribute 'sparse_types'}}
  %result:5 = "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %empty_str_vector, %empty_str_vector) {dense_shapes = [], num_sparse = 2 : i64, result_segment_sizes = array<i32: 1, 2, 2, 0, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>) -> (tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>, tensor<2xi64>)
  func.return %result#0 : tensor<?x2xi64>
}

// -----

func.func @testParseExampleV2SparseOnlyValid(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %sparse_keys : tensor<2x!tf_type.string>) -> (tensor<?x2xi64>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // expected-error @+1 {{output 'sparse_shapes' should have same length as attribute 'sparse_types'}}
  %result:5 = "tf.ParseExampleV2"(%serialized, %names, %sparse_keys, %empty_str_vector, %empty_str_vector) {dense_shapes = [], num_sparse = 2 : i64, result_segment_sizes = array<i32: 2, 2, 1, 0, 0, 0>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>) -> (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>)
  func.return %result#0 : tensor<?x2xi64>
}

// -----

// CHECK-LABEL: func @testParseExampleV2RaggedOnlyValid
func.func @testParseExampleV2RaggedOnlyValid(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %ragged_keys : tensor<2x!tf_type.string>) -> (tensor<?xf32>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  %result:4 = "tf.ParseExampleV2"(%serialized, %names, %empty_str_vector, %empty_str_vector, %ragged_keys) {dense_shapes = [], num_sparse = 0 : i64, result_segment_sizes = array<i32: 0, 0, 0, 0, 2, 2>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<?xf32>, tensor<?x!tf_type.string>, tensor<?xi32>, tensor<?xi64>)
  func.return %result#0 : tensor<?xf32>
}

// -----

func.func @testParseExampleV2RaggedMismatchedOutputLengths(%serialized: tensor<32x!tf_type.string>, %names : tensor<32x!tf_type.string>, %ragged_keys : tensor<2x!tf_type.string>) -> (tensor<?xf32>) {
  %empty_str_vector = "tf.Const"() {dtype = !tf_type.string, value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F535452494E472074656E736F725F7368617065207B2064696D207B207D207D"> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
  // expected-error @+1 {{attribute 'ragged_value_types' should have same length as attribute 'ragged_split_types'}}
  %result:3 = "tf.ParseExampleV2"(%serialized, %names, %empty_str_vector, %empty_str_vector, %ragged_keys) {dense_shapes = [], num_sparse = 0 : i64, result_segment_sizes = array<i32: 0, 0, 0, 0, 2, 1>} : (tensor<32x!tf_type.string>, tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<?xf32>, tensor<?x!tf_type.string>, tensor<?xi32>)
  func.return %result#0 : tensor<?xf32>
}

// -----

// Legal BatchMatMul op.
func.func @testBatchMatMul(%lhs: tensor<2x?x2x?x3x5xf32>, %rhs: tensor<2x2x?x?x5x7xf32>) {
  %0 = "tf.BatchMatMul"(%lhs, %rhs) : (tensor<2x?x2x?x3x5xf32>, tensor<2x2x?x?x5x7xf32>) -> tensor<2x?x?x?x3x7xf32>
  func.return
}

// -----

// Mismatching batch dimensions.
func.func @testBatchMatMul(%lhs: tensor<1x3x5xf32>, %rhs: tensor<2x5x7xf32>) {
  // expected-error @+1 {{found mismatching batch dimensions for lhs shape 'tensor<1x3x5xf32>' and rhs shape 'tensor<2x5x7xf32>'}}
  %0 = "tf.BatchMatMul"(%lhs, %rhs) : (tensor<1x3x5xf32>, tensor<2x5x7xf32>) -> tensor<2x3x7xf32>
}

// -----

func.func @testBatchMatMulV2(%lhs: tensor<f32>, %rhs: tensor<10x10xf32>) {
  // expected-error @+1 {{requires lhs operand to have rank at least two}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<f32>, tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @testBatchMatMulV2(%lhs: tensor<10x10xf32>, %rhs: tensor<f32>) {
  // expected-error @+1 {{requires rhs operand to have rank at least two}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x10xf32>, tensor<f32>) -> tensor<10x10xf32>
}

// -----

// CHECK-LABEL: func @testBatchMatMulV2NoBatchDimension
func.func @testBatchMatMulV2NoBatchDimension(%lhs: tensor<5x10xf32>, %rhs: tensor<10x10xf32>) -> (tensor<5x10xf32>) {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<5x10xf32>, tensor<10x10xf32>) -> tensor<5x10xf32>
  func.return %0 : tensor<5x10xf32>
}

// -----

// CHECK-LABEL: func @testBatchMatMulV2ValidBroadcastingBatchDimension
func.func @testBatchMatMulV2ValidBroadcastingBatchDimension(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x10xf32>) -> (tensor<10x2x5x10xf32>) {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x10xf32>) -> tensor<10x2x5x10xf32>
  func.return %0 : tensor<10x2x5x10xf32>
}

// -----

// CHECK-LABEL: func @testBatchMatMulV2ValidMultiBatchDimension
func.func @testBatchMatMulV2ValidMultiBatchDimension(%lhs: tensor<4x5x1x3x2xf32>, %rhs: tensor<1x1x3x5xf32>) -> (tensor<4x5x1x2x5xf32>) {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) { adj_x = true } : (tensor<4x5x1x3x2xf32>, tensor<1x1x3x5xf32>) -> tensor<4x5x1x2x5xf32>
  func.return %0 : tensor<4x5x1x2x5xf32>
}

// -----

func.func @testBatchMatMulV2InvalidBroadcastingBatchDimensionWithHigherXRank(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x10x10xf32>) {
  // expected-error @+1 {{found incompatible broadcast batch dimensions for lhs shape 'tensor<10x2x5x10xf32>' and rhs shape 'tensor<10x10x10xf32>'}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidBroadcastingBatchDimensionWithSameRank(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x10x10x10xf32>) {
  // expected-error @+1 {{found incompatible broadcast batch dimensions for lhs shape 'tensor<10x2x5x10xf32>' and rhs shape 'tensor<10x10x10x10xf32>'}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x10x10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidBroadcastingBatchDimensionWithHigherYRank(%lhs: tensor<2x5x10xf32>, %rhs: tensor<10x10x10x10xf32>) {
  // expected-error @+1 {{found incompatible broadcast batch dimensions for lhs shape 'tensor<2x5x10xf32>' and rhs shape 'tensor<10x10x10x10xf32>'}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<2x5x10xf32>, tensor<10x10x10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidOutputBatchDimension(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<2x10x10xf32>) {
  // expected-error @+1 {{has mismatching input batch dimension 2 and output batch dimension 3}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<2x10x10xf32>) -> tensor<10x3x10x10xf32>
}

// -----

func.func @testBatchMatMulV2DynamicInputBatchDimension(%lhs: tensor<?x2x5x10xf32>, %rhs: tensor<?x10xf32>) -> tensor<5x2x5x10xf32> {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<?x2x5x10xf32>, tensor<?x10xf32>) -> tensor<5x2x5x10xf32>
  func.return %0 : tensor<5x2x5x10xf32>
}

// -----

func.func @testBatchMatMulV2DynamicOutputBatchDimension(%lhs: tensor<1x2x5x10xf32>, %rhs: tensor<1x10xf32>) -> tensor<?x2x5x10xf32> {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<1x2x5x10xf32>, tensor<1x10xf32>) -> tensor<?x2x5x10xf32>
  func.return %0 : tensor<?x2x5x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidOutputRank(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x1x10x10xf32>) {
  // expected-error @+1 {{found invalid output rank, expected 4 but got 3}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x1x10x10xf32>) -> tensor<10x5x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidOutputRowDim(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x10xf32>) {
  // expected-error @+1 {{found invalid output dimension on row, expected 5 but got 10}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x10xf32>) -> tensor<10x2x10x10xf32>
}

// -----

func.func @testBatchMatMulV2AdjXInvalidOutputRowDim(%lhs: tensor<10x2x10x5xf32>, %rhs: tensor<10x10xf32>) {
  // expected-error @+1 {{found invalid output dimension on row, expected 5 but got 10}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) { adj_x = true } : (tensor<10x2x10x5xf32>, tensor<10x10xf32>) -> tensor<10x2x10x10xf32>
}

// -----

func.func @testBatchMatMulV2InvalidOutputColDim(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<10x10xf32>) {
  // expected-error @+1 {{found invalid output dimension on col, expected 10 but got 5}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<10x2x5x10xf32>, tensor<10x10xf32>) -> tensor<10x2x5x5xf32>
}

// -----

func.func @testBatchMatMulV2AdjYInvalidOutputColDim(%lhs: tensor<10x2x5x10xf32>, %rhs: tensor<4x10xf32>) {
  // expected-error @+1 {{found invalid output dimension on col, expected 4 but got 10}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) { adj_y = true } : (tensor<10x2x5x10xf32>, tensor<4x10xf32>) -> tensor<10x2x5x10xf32>
}

// -----

// CHECK-LABEL: func @testBatchMatMulV2PartiallyKnownInputBatchDim
func.func @testBatchMatMulV2PartiallyKnownInputBatchDim(%lhs: tensor<4x5x?x3x2xf32>, %rhs: tensor<1x1x3x5xf32>) -> (tensor<4x5x?x2x5xf32>) {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) { adj_x = true } : (tensor<4x5x?x3x2xf32>, tensor<1x1x3x5xf32>) -> tensor<4x5x?x2x5xf32>
  func.return %0 : tensor<4x5x?x2x5xf32>
}

// -----

// CHECK-LABEL: func @testBatchMatMulV2PartiallyKnownMatmulDim
func.func @testBatchMatMulV2PartiallyKnownMatmulDim(%lhs: tensor<4x5x1x?x3xf32>, %rhs: tensor<1x1x3x5xf32>) -> (tensor<4x5x1x?x5xf32>) {
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<4x5x1x?x3xf32>, tensor<1x1x3x5xf32>) -> tensor<4x5x1x?x5xf32>
  func.return %0 : tensor<4x5x1x?x5xf32>
}

// -----

func.func @testBatchMatMulV2InvalidPartiallyKnownMatmulDim(%lhs: tensor<4x5x1x?x3xf32>, %rhs: tensor<1x1x3x5xf32>) -> (tensor<4x5x1x?x3xf32>) {
  // expected-error @+1 {{found invalid output dimension on col, expected 5 but got 3}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) : (tensor<4x5x1x?x3xf32>, tensor<1x1x3x5xf32>) -> tensor<4x5x1x?x3xf32>
  func.return %0 : tensor<4x5x1x?x3xf32>
}

// -----

func.func @testBatchMatMulV2AdjXInvalidPartiallyKnownMatmulDim(%lhs: tensor<4x5x1x3x?xf32>, %rhs: tensor<1x1x3x5xf32>) -> (tensor<4x5x1x?x3xf32>) {
  // expected-error @+1 {{found invalid output dimension on col, expected 5 but got 3}}
  %0 = "tf.BatchMatMulV2"(%lhs, %rhs) { adj_x = true } : (tensor<4x5x1x3x?xf32>, tensor<1x1x3x5xf32>) -> tensor<4x5x1x?x3xf32>
  func.return %0 : tensor<4x5x1x?x3xf32>
}

// -----

func.func @testDataFormatVecPermuteInvalid1dInput(%x: tensor<5xi32>) {
  // expected-error @+1 {{requires 1D input of size 4}}
  %0 = "tf.DataFormatVecPermute"(%x): (tensor<5xi32>) -> tensor<5xi32>
  func.return
}

// -----

func.func @testDataFormatVecPermuteInvalid2dDim0Input(%x: tensor<5x2xi32>) {
  // expected-error @+1 {{requires first dimensions of 2D input to be of size 4}}
  %0 = "tf.DataFormatVecPermute"(%x): (tensor<5x2xi32>) -> tensor<5x2xi32>
  func.return
}

// -----

func.func @testDataFormatVecPermuteInvalid2dDim1Input(%x: tensor<4x3xi32>) {
  // expected-error @+1 {{requires second dimensions of 2D input to be of size 2}}
  %0 = "tf.DataFormatVecPermute"(%x): (tensor<4x3xi32>) -> tensor<4x3xi32>
  func.return
}

// -----

func.func @testDataFormatVecPermuteInvalid3dInput(%x: tensor<4x2x2xi32>) {
  // expected-error @+1 {{requires input of rank 1 or 2}}
  %0 = "tf.DataFormatVecPermute"(%x): (tensor<4x2x2xi32>) -> tensor<4x2x2xi32>
  func.return
}

// -----

func.func @testSendTPUEmbeddingGradients(%x: tensor<512x256xf32>) {
  "tf.SendTPUEmbeddingGradients"(%x) {N = 1 : i64, NN = 0 : i64, config = "", operand_segment_sizes = array<i32: 1, 0>} : (tensor<512x256xf32>) -> ()
  func.return
}

// -----

//===--------------------------------------------------------------------===//
//  tf.BatchToSpace
//===--------------------------------------------------------------------===//

func.func @testBatchToSpaceDynamic(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return
}

func.func @testBatchToSpaceRankedInput(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<*xi32>) {
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<?x?x?x?xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return
}

func.func @testBatchToSpaceRankedCrops(%arg0: tensor<*xf32>, %arg1: tensor<?x?xi32>) {
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<?x?xi32>) -> tensor<*xf32>
  func.return
}

func.func @testBatchToSpaceRankedOutput(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<?x?x?x?xf32>
  func.return
}

func.func @testBatchToSpaceStatic(%arg0: tensor<36x8x8x8xf32>) {
  %crops = "tf.Const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %0 = "tf.BatchToSpace"(%arg0, %crops) {block_size = 3 : i64} : (tensor<36x8x8x8xf32>, tensor<2x2xi32>) -> tensor<4x21x17x8xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidInputRank(%arg0: tensor<8xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires input to be a 4D tensor, but got 'tensor<8xf32>'}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<8xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidInputBatch(%arg0: tensor<2x4x6x8xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires input batch (dimension 0) to be evenly divisible by (block_size * block_size), but got input batch 2 and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<2x4x6x8xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidCropsRank(%arg0: tensor<*xf32>, %arg1: tensor<?x?x?xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires crops to be a 2D tensor, but got 'tensor<?x?x?xi32>'}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<?x?x?xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidCropsFirstDim(%arg0: tensor<*xf32>, %arg1: tensor<3x?xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires crops to be a tensor<2x2>, but got 'tensor<3x?xi32>'}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<3x?xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidCropsSecondDim(%arg0: tensor<*xf32>, %arg1: tensor<?x3xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires crops to be a tensor<2x2>, but got 'tensor<?x3xi32>'}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<?x3xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceBadCropValues(%arg0: tensor<*xf32>) {
  %crops = "tf.Const"() {value = dense<[[-1, -2], [-3, -4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // expected-error @+1 {{'tf.BatchToSpace' op requires all crop values to be nonnegative, but got dense<[[-1, -2], [-3, -4]]> : tensor<2x2xi32>}}
  %0 = "tf.BatchToSpace"(%arg0, %crops) {block_size = 2 : i64} : (tensor<*xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputRank(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires output to be a 4D tensor, but got 'tensor<8xf32>'}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<*xf32>, tensor<*xi32>) -> tensor<8xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputBatch(%arg0: tensor<16x8x8x3xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires output batch (dimension 0) to be equal to input batch (dimension 0) / (block_size * block_size), but got output batch 8, input batch 16, and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<16x8x8x3xf32>, tensor<*xi32>) -> tensor<8x8x8x3xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputHeight(%arg0: tensor<16x8x8x3xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires output height (dimension 1) to be less than or equal to input height (dimension 1) * block_size, but got output height 17, input height 8, and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<16x8x8x3xf32>, tensor<*xi32>) -> tensor<4x17x8x3xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputHeightCrops(%arg0: tensor<16x8x8x3xf32>) {
  %crops = "tf.Const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // expected-error @+1 {{'tf.BatchToSpace' op requires output height (dimension 1) to be equal to input height (dimension 1) * block_size - crop_top - crop_bottom, but got output height 8, input height 8, crop_top 1, crop_bottom 2, and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %crops) {block_size = 2 : i64} : (tensor<16x8x8x3xf32>, tensor<2x2xi32>) -> tensor<4x8x9x3xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputWidth(%arg0: tensor<16x4x4x3xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires output width (dimension 2) to be less than or equal to input width (dimension 2) * block_size, but got output width 9, input width 4, and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<16x4x4x3xf32>, tensor<*xi32>) -> tensor<4x4x9x3xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputWidthCrops(%arg0: tensor<16x8x8x3xf32>) {
  %crops = "tf.Const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // expected-error @+1 {{'tf.BatchToSpace' op requires output width (dimension 2) to be equal to input width (dimension 2) * block_size - crop_left - crop_right, but got output width 8, input width 8, crop_left 3, crop_right 4, and block_size 2}}
  %0 = "tf.BatchToSpace"(%arg0, %crops) {block_size = 2 : i64} : (tensor<16x8x8x3xf32>, tensor<2x2xi32>) -> tensor<4x13x8x3xf32>
  func.return
}

// -----

func.func @testBatchToSpaceInvalidOutputDepth(%arg0: tensor<16x8x8x3xf32>, %arg1: tensor<*xi32>) {
  // expected-error @+1 {{'tf.BatchToSpace' op requires output depth (dimension 3) to be equal to input depth (dimension 3), but got output depth 8 and input depth 3}}
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 2 : i64} : (tensor<16x8x8x3xf32>, tensor<*xi32>) -> tensor<4x8x8x8xf32>
  func.return
}

// -----

func.func private @branch()

func.func @testCaseBadBranchIndicesShape(%arg0: tensor<8xi32>) {
  // expected-error @+1 {{expects 'branch_index' to be a scalar, but got 'tensor<8xi32>'}}
  "tf.Case"(%arg0) {branches = [@branch], is_stateless = false} : (tensor<8xi32>) -> ()
  func.return
}

// -----

func.func private @branch0(tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
func.func private @branch1(tensor<2xf32>) -> tensor<2xf32>

func.func @testCaseMismatchedNumOperands(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.Case' op branch #0 inputs (size = 2) should have the same number of values as inputs (size = 1)}}
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func private @branch0(tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
func.func private @branch1(tensor<2xf32>) -> tensor<2xf32>

func.func @testCaseMismatchedNumResults(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.Case' op branch #0 results (size = 2) should have the same number of values as results (size = 1)}}
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func private @branch0(tensor<*xf16>) -> tensor<*xf32>
func.func private @branch1(tensor<*xf32>) -> tensor<*xf32>

func.func @testCaseOperandNotCastCompatible(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.Case' op branch #0 input type tensor<*xf16> is incompatible with input type tensor<2xf32> at index 0}}
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func private @branch0(tensor<2xf32>) -> tensor<*xf32>
func.func private @branch1(tensor<3xf32>) -> tensor<*xf32>

func.func @testCaseBranchArgumentsNotCastCompatible(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{expects all branch input type(s) (tensor<2xf32>, tensor<3xf32>) at index 0 to be cast compatible}}
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<*xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func private @branch0(tensor<*xf32>) -> tensor<*xf32>
func.func private @branch1(tensor<*xf32>) -> tensor<3xf32>

func.func @testCaseResultNotCastCompatible(%arg0: tensor<i32>, %arg1: tensor<*xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{'tf.Case' op branch #1 result type tensor<3xf32> is incompatible with result type tensor<2xf32> at index 0}}
  %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<*xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func @testCaseRegionNoRegions(%arg0: tensor<i32>) {
  // expected-error @+1 {{expects to have at least 1 region}}
  "tf.CaseRegion"(%arg0) {is_stateless = false} : (tensor<i32>) -> ()
  func.return
}

// -----

func.func @testCaseRegionBadBranchIndicesShape(%arg0: tensor<8xi32>) {
  // expected-error @+1 {{expects 'branch_index' to be a scalar, but got 'tensor<8xi32>'}}
  "tf.CaseRegion"(%arg0) ({
    "tf.Yield"() : () -> ()
  }) {is_stateless = false} : (tensor<8xi32>) -> ()
  func.return
}

// -----

func.func @testCaseRegionMismatchedNumResults(%arg0: tensor<i32>) {
  // expected-error @+1 {{'tf.CaseRegion' op branch #0 results (size = 0) should have the same number of values as results (size = 1)}}
  %1 = "tf.CaseRegion"(%arg0) ({
    "tf.Yield"() : () -> ()
  }) {is_stateless = false} : (tensor<i32>) -> tensor<i1>
  func.return
}

// -----

func.func @testCaseRegionMismatchedResultTypes(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  // expected-error @+1 {{'tf.CaseRegion' op branch #0 result type tensor<f32> is incompatible with result type tensor<i1> at index 0}}
  %1 = "tf.CaseRegion"(%arg0) ({
    "tf.Yield"(%arg1) : (tensor<f32>) -> ()
  }) {is_stateless = false} : (tensor<i32>) -> tensor<i1>
  func.return
}

// -----

// Test valid tf.Cumsum
func.func @testCumsum(%arg: tensor<8x16xf32>, %axis: tensor<i32>) -> tensor<8x16xf32> {
  %0 = "tf.Cumsum"(%arg, %axis) : (tensor<8x16xf32>, tensor<i32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

func.func @testCumprod(%arg: tensor<8x16xf32>, %axis: tensor<2xi32>) -> tensor<8x16xf32> {
  // expected-error @+1 {{requires scalar axis operand}}
  %0 = "tf.Cumprod"(%arg, %axis) : (tensor<8x16xf32>, tensor<2xi32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

func.func @testCumprod(%arg: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %axis = arith.constant dense<-3> : tensor<i32>
  // expected-error @+1 {{axis operand should be within range [-2, 2)}}
  %0 = "tf.Cumprod"(%arg, %axis) : (tensor<8x16xf32>, tensor<i32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

func.func @testTile(%arg0: tensor<2x3x?xf32>) {
  %cst = arith.constant dense <[2, 3, 4]> : tensor<3xi32>
  %0 = "tf.Tile"(%arg0, %cst) : (tensor<2x3x?xf32>, tensor<3xi32>) -> tensor<4x9x?xf32>
  func.return
}

// -----

func.func @testTileMultipleNotRank1(%arg0: tensor<2x3xf32>, %arg1: tensor<1x1xi32>) {
  // expected-error @+1 {{expected multiples to be rank 1, got rank = 2}}
  %0 = "tf.Tile"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<1x1xi32>) -> tensor<2x3xf32>
  func.return
}

// -----

func.func @testTileInputRankNotEqualToMultiplesSize(%arg0: tensor<2x3xf32>, %arg1: tensor<3xi32>) {
  // expected-error @+1 {{expected size of multiples equal to rank of input, got multiples of size 3, and input of rank 2}}
  %0 = "tf.Tile"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xi32>) -> tensor<2x3xf32>
  func.return
}

// -----

func.func @testTileInputRankNotEqualToOutputRank(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi32>) {
  // expected-error @+1 {{expected rank of input to equal to rank of output, got input of rank 2, and output of rank 3}}
  %0 = "tf.Tile"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<2x3x1xf32>
  func.return
}

// -----

func.func @testTileNegativeMultiples(%arg0: tensor<2x3xf32>) {
  %cst = arith.constant dense <[-1, 1]> : tensor<2xi32>
  // expected-error @+1 {{expected multiples to be non-negative, got multiples[0] = -1}}
  %0 = "tf.Tile"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  func.return
}

// -----

func.func @testTileInvalidOutputShape(%arg0: tensor<2x3xf32>) {
  %cst = arith.constant dense <[2, 3]> : tensor<2xi32>
  // expected-error @+1 {{requires input.shape[1] (3) * 3 to be equal to output.shape[1] (6)}}
  %0 = "tf.Tile"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<4x6xf32>
  func.return
}

// -----

// Test reference variable support for some ops (no errors expected)

// CHECK-LABEL: @testMaximumWithRef
func.func @testMaximumWithRef(%arg0: tensor<!tf_type.f32ref>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: tf.Maximum
  %0 = "tf.Maximum"(%arg0, %arg1) : (tensor<!tf_type.f32ref>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @testAddV2WithRef
func.func @testAddV2WithRef(%arg0: tensor<!tf_type.int16ref>, %arg1: tensor<i16>) -> tensor<i16> {
  // CHECK: tf.AddV2
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<!tf_type.int16ref>, tensor<i16>) -> tensor<i16>
  func.return %0 : tensor<i16>
}

// CHECK-LABEL: @testRealDivWithRef
func.func @testRealDivWithRef(%arg0: tensor<f64>, %arg1: tensor<!tf_type.f64ref>) -> tensor<f64> {
  // CHECK: tf.RealDivOp
  %0 = "tf.RealDivOp"(%arg0, %arg1) : (tensor<f64>, tensor<!tf_type.f64ref>) -> tensor<f64>
  func.return %0 : tensor<f64>
}

// CHECK-LABEL: @testDivNoNanWithRef
func.func @testDivNoNanWithRef(%arg0: tensor<f32>, %arg1: tensor<!tf_type.f32ref>) -> tensor<f32> {
  // CHECK: tf.DivNoNanOp
  %0 = "tf.DivNoNanOp"(%arg0, %arg1) : (tensor<f32>, tensor<!tf_type.f32ref>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @testAddWithRef
func.func @testAddWithRef(%arg0: tensor<!tf_type.f64ref>, %arg1: tensor<f64>) -> tensor<f64> {
  // CHECK: tf.Add
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<!tf_type.f64ref>, tensor<f64>) -> tensor<f64>
  func.return %0 : tensor<f64>
}

// -----

func.func @testInvalidTPUExecuteAndUpdateVariables(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<3x!tf_type.string>) {
  // expected-error@below {{requires 'device_var_reads_indices' to be the same size as number of resource handles in 'args' (1), but got 2}}
  "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1) {device_var_reads_indices = [0, 1], device_var_updates_indices = [0]} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<3x!tf_type.string>) -> ()
  func.return
}

// -----

func.func @testInvalidTPUExecuteAndUpdateVariables(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<3x!tf_type.string>) {
  // expected-error@below {{requires 'device_var_updates_indices' to be the same size as number of resource handles in 'args' (1), but got 2}}
  "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1) {device_var_reads_indices = [0], device_var_updates_indices = [0, 1]} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<3x!tf_type.string>) -> ()
  func.return
}

// -----

func.func @testInvalidTPUExecuteAndUpdateVariables(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<3x!tf_type.string>) {
  // expected-error@below {{requires 'device_var_reads_indices' to contain values of at least 0, but got -1 at index 0}}
  "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1) {device_var_reads_indices = [-1], device_var_updates_indices = [0]} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<3x!tf_type.string>) -> ()
  func.return
}

// -----

func.func @testInvalidTPUExecuteAndUpdateVariables(%arg0: tensor<!tf_type.resource<tensor<i32>>>, %arg1: tensor<3x!tf_type.string>) {
  // expected-error@below {{requires 'device_var_updates_indices' to contain values of at least -1, but got -2 at index 0}}
  "tf.TPUExecuteAndUpdateVariables"(%arg0, %arg1) {device_var_reads_indices = [0], device_var_updates_indices = [-2]} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<3x!tf_type.string>) -> ()
  func.return
}

// -----

// Valid VarHandleOp operation.
// CHECK-LABEL: func @testVarHandleOp
func.func @testVarHandleOp() -> tensor<!tf_type.resource<tensor<*xf32>>> {
  %0 = "tf.VarHandleOp"() {
    container = "",
    shared_name = "cd2c89b7-88b7-44c8-ad83-06c2a9158347"
  } : () -> tensor<!tf_type.resource<tensor<*xf32>>>
  func.return %0 : tensor<!tf_type.resource<tensor<*xf32>>>
}

// -----

// VarHandleOp operation missing the required resource subtype.
func.func @testVarHandleOp() -> tensor<*x!tf_type.resource> {
  // expected-error @+1 {{must have exactly one subtype in the result resource type}}
  %0 = "tf.VarHandleOp"() {
    container = "",
    shared_name = "cd2c89b7-88b7-44c8-ad83-06c2a9158347"
  } : () -> tensor<*x!tf_type.resource>
  func.return %0 : tensor<*x!tf_type.resource>
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<2x3x5xi32>, %arg1: tensor<5x2xi32>) -> () {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{broadcast_dims must have size equal to the smaller argument rank}}
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<2x3x5xi32>, tensor<5x2xi32>, tensor<1xi64>) -> (tensor<2x3x5xi32>, tensor<2x1x5xi32>)
  func.return
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<2x3x5xi32>, %arg1: tensor<5x2xi32>) -> () {
  %0 = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
  // expected-error @+1 {{if broadcast_dims is empty, both arguments must have equal rank or at least one argument must be a scalar}}
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<2x3x5xi32>, tensor<5x2xi32>, tensor<0xi64>) -> (tensor<2x3x5xi32>, tensor<2x1x5xi32>)
  func.return
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<5x2xi32>, %arg1: tensor<2x3x5xi32>) -> () {
  %0 = "tf.Const"() {value = dense<0> : tensor<2xi64>} : () -> tensor<2xi64>
  // expected-error @+1 {{broadcast_dims has duplicates}}
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<5x2xi32>, tensor<2x3x5xi32>, tensor<2xi64>) -> (tensor<2x1x5xi32>, tensor<2x3x5xi32>)
  func.return
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<2xi32>, %arg1: tensor<i32>) -> () {
  %0 = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<2xi32>, tensor<i32>, tensor<0xi64>) -> (tensor<2xi32>, tensor<i32>)
  func.return
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<5x2xi32>, %arg1: tensor<2x3x5xi32>) -> () {
  %0 = "tf.Const"() {value = dense<[2, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<5x2xi32>, tensor<2x3x5xi32>, tensor<2xi64>) -> (tensor<2x1x5xi32>, tensor<2x3x5xi32>)
  func.return
}

// -----

func.func @testXlaBroadcastHelper(%arg0: tensor<2x3x5xi32>, %arg1: tensor<5x2xi32>) -> () {
  %0 = "tf.Const"() {value = dense<[2, 0]> : tensor<2xi64>} : () -> tensor<2xi64>
  %lhs_output, %rhs_output = "tf.XlaBroadcastHelper"(%arg0, %arg1, %0) : (tensor<2x3x5xi32>, tensor<5x2xi32>, tensor<2xi64>) -> (tensor<2x3x5xi32>, tensor<2x1x5xi32>)
  func.return
}

// -----

func.func @testXlaConvV2InvalidFeatureGroupCount(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %lhs_dilation = "tf.Const"() {value = dense<[4, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // / expected-error@+1 {{'tf.XlaConvV2' op expects feature_group_count to be a scalar}}
  %0 = "tf.XlaConvV2"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<2xi32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}

// -----

func.func @testXlaConvV2InvalidLhsDilation(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %lhs_dilation = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+1 {{'tf.XlaConvV2' op expects lhs_dilation to be a vecotr}}
  %0 = "tf.XlaConvV2"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x2xi32>, tensor<i32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}

// -----

func.func @testXlaConvV2InvalidRhsDilation(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %lhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+1 {{'tf.XlaConvV2' op expects rhs_dilation to be a vecotr}}
  %0 = "tf.XlaConvV2"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<i32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}

// -----

func.func @testXlaConvV2InvalidWindowStrides(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %lhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %strides = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tf.XlaConvV2' op expects window_stride to be a vector}}
  %0 = "tf.XlaConvV2"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<i32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}

// -----

func.func @testXlaConvV2InvalidPadding(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %lhs_dilation = "tf.Const"() {value = dense<[4, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x3xi32>} : () -> tensor<3x3xi32>
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+1 {{'tf.XlaConvV2' op expects padding to be a matrix with minor dimension 2}}
  %0 = "tf.XlaConvV2"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  "tf._XlaHostComputeMlir"(%arg0) {send_key="", recv_key="", host_mlir_module=""} : (tensor<2xf32>) -> ()
  func.return
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  "tf._XlaHostComputeMlir"(%arg0) {send_key="", recv_key="", host_mlir_module="module  {\0A  func.func @host_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {\0A    %0 = \22tf.Identity\22(%arg0) {_xla_outside_compilation = \22cluster1\22} : (tensor<*xf32>) -> tensor<*xf32> \0A    func.return %0 : tensor<*xf32> \0A  } \0A} \0A"} : (tensor<2xf32>) -> (tensor<2xf32>)
  func.return
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  // expected-error @+1 {{can not be deserialized}}
  "tf._XlaHostComputeMlir"(%arg0) {send_key="", recv_key="", host_mlir_module="bad_module"} : (tensor<2xf32>) -> ()
  func.return
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  // expected-error @+1 {{'host_mlir_module' does not contain 'host_func' function}}
  "tf._XlaHostComputeMlir"(%arg0) {send_key="", recv_key="", host_mlir_module="module  {\0A  func.func @bad_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {\0A    %0 = \22tf.Identity\22(%arg0) {_xla_outside_compilation = \22cluster1\22} : (tensor<*xf32>) -> tensor<*xf32> \0A    func.return %0 : tensor<*xf32> \0A  } \0A} \0A"} : (tensor<2xf32>) -> ()
  func.return
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  // expected-error @+1 {{Number of operands/inputs should be the same}}
  "tf._XlaHostComputeMlir"() {send_key="", recv_key="", host_mlir_module="module  {\0A  func.func @host_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {\0A    %0 = \22tf.Identity\22(%arg0) {_xla_outside_compilation = \22cluster1\22} : (tensor<*xf32>) -> tensor<*xf32> \0A    func.return %0 : tensor<*xf32> \0A  } \0A} \0A"} : () -> ()
  func.return
}

// -----

func.func @testXlaHostComputeMlir(%arg0: tensor<2xf32>) -> () {
  // expected-error @+1 {{Number of results should be the same}}
  "tf._XlaHostComputeMlir"(%arg0) {send_key="", recv_key="", host_mlir_module="module  {\0A  func.func @host_func(%arg0: tensor<*xf32>) -> tensor<*xf32> {\0A    %0 = \22tf.Identity\22(%arg0) {_xla_outside_compilation = \22cluster1\22} : (tensor<*xf32>) -> tensor<*xf32> \0A    func.return %0 : tensor<*xf32> \0A  } \0A} \0A"} : (tensor<2xf32>) -> ()
  func.return
}

// -----
func.func @testXlaSelectAndScatterAttrs(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op expects the size of window_dimensionsto be equal to the input rank (3 vs. 4)}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @no_select} : (tensor<4x5x1x1xbf16>, tensor<3xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

// -----

func.func @testXlaSelectAndScatterPadding(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op expects padding to be a matrix with minor dimension 2, got 4, 2, 3}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @no_select} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2x3xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

// -----

func.func @testXlaSelectAndScatterSelect(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op has no select function specified}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @no_select} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

// -----

func.func @testXlaSelectAndScatterSelectNumArgs(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op expects select function to take 2 parameters, but has 3 parameter(s)}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @xla_select_and_scatter_select_3_args} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

func.func private @xla_select_and_scatter_select_3_args(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<bf16>) -> tensor<i1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @testXlaSelectAndScatterSelectReturnType(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op expects select function to return a single boolean result but got 'tensor<4xi32>'}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @xla_select_and_scatter_select_return_int32_vector} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

func.func private @xla_select_and_scatter_select_return_int32_vector(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<4xi32> {
  %0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @testXlaSelectAndScatterScatter(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op has no scatter function specified}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @no_scatter, select = @xla_select_and_scatter_select1} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

func.func private @xla_select_and_scatter_select1(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<i1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

func.func @testXlaSelectAndScatterSelectNumArgs(%arg0: tensor<4x5x1x1xbf16>, %arg1: tensor<2x2x1x1xbf16>, %arg2: tensor<bf16>) -> tensor<?x?x?x?xbf16> {
  %cst = "tf.Const"() {value = dense<0> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  %cst_0 = "tf.Const"() {value = dense<[2, 2, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %cst_1 = "tf.Const"() {value = dense<[2, 3, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error @+1 {{'tf.XlaSelectAndScatter' op expects scatter function to take 2 parameters, but has 3 parameter(s)}}
  %0 = "tf.XlaSelectAndScatter"(%arg0, %cst_1, %cst_0, %cst, %arg1, %arg2) {scatter = @xla_select_and_scatter_scatter, select = @xla_select_and_scatter_select2} : (tensor<4x5x1x1xbf16>, tensor<4xi32>, tensor<4xi32>, tensor<4x2xi32>, tensor<2x2x1x1xbf16>, tensor<bf16>) -> tensor<?x?x?x?xbf16>
  func.return %0 : tensor<?x?x?x?xbf16>
}

func.func private @xla_select_and_scatter_select2(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<i1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func private @xla_select_and_scatter_scatter(%arg0: tensor<*xbf16>, %arg1: tensor<*xbf16>, %arg2: tensor<*xbf16>) -> tensor<*xbf16> {
  %0 = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

// -----

func.func @testXlaReduceWindowAttrs(%arg0: tensor<7xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<3> : tensor<2xi32>} : () -> tensor<2xi32>
  %cst_3 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error @+1 {{tf.XlaReduceWindow' op expects the size of base_dilations to be equal to the input rank (2 vs. 1)}}
  %0 = "tf.XlaReduceWindow"(%arg0, %arg1, %cst_0, %cst_1, %cst_2, %cst_3, %cst) {computation = @no_reducer} : (tensor<7xf32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<2xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testXlaReduceWindowPadding(%arg0: tensor<7xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x3xi32>} : () -> tensor<1x3xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_3 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error @+1 {{'tf.XlaReduceWindow' op expects padding to be a matrix with minor dimension 2, got 1, 3}}
  %0 = "tf.XlaReduceWindow"(%arg0, %arg1, %cst_0, %cst_1, %cst_2, %cst_3, %cst) {computation = @no_reducer} : (tensor<7xf32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x3xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testXlaReduceWindowComputation(%arg0: tensor<7xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_3 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error @+1 {{'tf.XlaReduceWindow' op has no reduction function specified}}
  %0 = "tf.XlaReduceWindow"(%arg0, %arg1, %cst_0, %cst_1, %cst_2, %cst_3, %cst) {computation = @no_reducer} : (tensor<7xf32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @testXlaReduceWindowComputationNumArgs(%arg0: tensor<7xf32>, %arg1: tensor<f32>) -> tensor<?xf32> {
  %cst = "tf.Const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_2 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst_3 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  // expected-error @+1 {{'tf.XlaReduceWindow' op expects reduction function to take 2 parameters, but has 3 parameter(s)}}
  %0 = "tf.XlaReduceWindow"(%arg0, %arg1, %cst_0, %cst_1, %cst_2, %cst_3, %cst) {computation = @xla_reduce_window_op_reducer} : (tensor<7xf32>, tensor<f32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

func.func private @xla_reduce_window_op_reducer(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testLogStaticShapeInputAndDynamicShapeOutput(%arg0: tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "tf.Log"(%arg0) : (tensor<8x16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testReluStaticShapeInputAndDynamicShapeOutput(%arg0: tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "tf.Relu"(%arg0) : (tensor<8x16xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// -----

func.func @set_dynamic_dimension_size(%input: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf16> {
  %dimension = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  // expected-error @+1 {{dim_index (1) is out of range [0, 1)}}
  %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4xf32>, tensor<i32>, tensor<i32>) -> tensor<?xf16>
  func.return %0 : tensor<?xf16>
}

// -----

func.func @testSetStaticDimensionBounds(%arg0: f32, %arg1: tensor<?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @below {{'tf.SetStaticDimensionBounds' op operand #0 must be tensor of tf.dtype values, but got 'f32'}}
  %dyn_arg0 = "tf.SetStaticDimensionBounds" (%arg0, %arg1) :(f32, tensor<?xi32>) -> tensor<?x?x?xi32>
  func.return %dyn_arg0 : tensor<?x?x?xi32>
}

// -----

func.func @testSetStaticDimensionBounds(%arg0: tensor<?x?x?xi32>, %arg1: tensor<?xi32>) -> tensor<?x?x?xi32> {
  // expected-error @below {{'tf.SetStaticDimensionBounds' op was used with an input tensor with rank > 2, only tensors of rank 1,2 are supported}}
  %dyn_arg0 = "tf.SetStaticDimensionBounds" (%arg0, %arg1) :(tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?xi32>
  func.return %dyn_arg0 : tensor<?x?x?xi32>
}

// -----

func.func @testSetStaticDimensionBounds(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // expected-error @below {{'tf.SetStaticDimensionBounds' op static shape must be of rank 1 (vector)}}
  %dyn_arg0 = "tf.SetStaticDimensionBounds" (%arg0, %arg1) :(tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %dyn_arg0 : tensor<?x?xi32>
}

// -----

func.func @testSetStaticDimensionBounds(%arg0: tensor<?x?xi32>, %arg1: tensor<4xi32>) -> tensor<?x?xi32> {
  // expected-error @below {{'tf.SetStaticDimensionBounds' op static shape must have num_elements == rank of input tensor}}
  %dyn_arg0 = "tf.SetStaticDimensionBounds" (%arg0, %arg1) :(tensor<?x?xi32>, tensor<4xi32>) -> tensor<?x?xi32>
  func.return %dyn_arg0 : tensor<?x?xi32>
}

// -----

func.func @testUniformQuantizedDotHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<i32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedDotHybrid' op quantization_axis is -1, scales must have 0 rank.}}
  %0 = "tf.UniformQuantizedDotHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedDotHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<f32>, %rhs_zps: tensor<2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedDotHybrid' op quantization_axis is -1, zero_points must have 0 rank.}}
  %0 = "tf.UniformQuantizedDotHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<f32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedDotHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2x2xf32>, %rhs_zps: tensor<2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedDotHybrid' op quantization_axis is not -1, scales must have 1 rank.}}
  %0 = "tf.UniformQuantizedDotHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2x2xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedDotHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<2x2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedDotHybrid' op quantization_axis is not -1, zero_points must have 1 rank.}}
  %0 = "tf.UniformQuantizedDotHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedDotHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<3xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedDotHybrid' op scales and zero points must have same number of elements.}}
  %0 = "tf.UniformQuantizedDotHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedConvolutionHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<i32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedConvolutionHybrid' op quantization_axis is -1, scales must have 0 rank.}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedConvolutionHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<f32>, %rhs_zps: tensor<2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedConvolutionHybrid' op quantization_axis is -1, zero_points must have 0 rank.}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = -1 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<f32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedConvolutionHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2x2xf32>, %rhs_zps: tensor<2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedConvolutionHybrid' op quantization_axis is not -1, scales must have 1 rank.}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2x2xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedConvolutionHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<2x2xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedConvolutionHybrid' op quantization_axis is not -1, zero_points must have 1 rank.}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<2x2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedConvolutionHybrid(%lhs: tensor<*xf32>, %rhs: tensor<2x2x!tf_type.qint8>, %rhs_scales: tensor<2xf32>, %rhs_zps: tensor<3xi32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformQuantizedConvolutionHybrid' op scales and zero points must have same number of elements.}}
  %0 = "tf.UniformQuantizedConvolutionHybrid"(%lhs, %rhs, %rhs_scales, %rhs_zps) {
    window_strides = [1, 2],
    padding = "VALID",
    explicit_padding = [],
    lhs_dilation = [1, 1],
    rhs_dilation = [2, 2],
    batch_group_count = 1 : i64,
    feature_group_count = 1 : i64,
    dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
    rhs_quantization_axis = 0 : i64, rhs_quantization_min_val = -128 : i64, rhs_quantization_max_val = 127 : i64
    } : (tensor<*xf32>, tensor<2x2x!tf_type.qint8>, tensor<2xf32>, tensor<3xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantize(%arg0: tensor<*xf32>, %scales: tensor<2xf32>, %zps: tensor<i32>) -> tensor<*x!tf_type.qint8> {
  // expected-error @below {{'tf.UniformQuantize' op quantization_axis is -1, scales must have 0 rank.}}
   %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<*xf32>, tensor<2xf32>, tensor<i32>) -> tensor<*x!tf_type.qint8>
  func.return %0 : tensor<*x!tf_type.qint8>
}

// -----

func.func @testUniformRequantize(
  %arg0: tensor<*x!tf_type.qint8>,
  %scales_0: tensor<2xf32>, %zps_0: tensor<i32>,
  %scales_1: tensor<f32>, %zps_1: tensor<i32>) -> tensor<*x!tf_type.qint8> {
  // expected-error @below {{'tf.UniformRequantize' op quantization_axis is -1, scales must have 0 rank.}}
  %0 = "tf.UniformRequantize"(%arg0, %scales_0, %zps_0, %scales_1, %zps_1) {
    input_quantization_axis = -1 : i64, input_quantization_min_val = -2147483648 : i64, input_quantization_max_val = 2147483647 : i64,
    output_quantization_axis = -1 : i64, output_quantization_min_val = -128 : i64, output_quantization_max_val = 127 : i64
  } : (tensor<*x!tf_type.qint8>, tensor<2xf32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<*x!tf_type.qint8>
  func.return %0 : tensor<*x!tf_type.qint8>
}

// -----

func.func @testUniformRequantize(
  %arg0: tensor<*x!tf_type.qint8>,
  %scales_0: tensor<f32>, %zps_0: tensor<i32>,
  %scales_1: tensor<2xf32>, %zps_1: tensor<i32>) -> tensor<*x!tf_type.qint8> {
  // expected-error @below {{'tf.UniformRequantize' op quantization_axis is -1, scales must have 0 rank.}}
  %0 = "tf.UniformRequantize"(%arg0, %scales_0, %zps_0, %scales_1, %zps_1) {
    input_quantization_axis = -1 : i64, input_quantization_min_val = -2147483648 : i64, input_quantization_max_val = 2147483647 : i64,
    output_quantization_axis = -1 : i64, output_quantization_min_val = -128 : i64, output_quantization_max_val = 127 : i64
  } : (tensor<*x!tf_type.qint8>, tensor<f32>, tensor<i32>, tensor<2xf32>, tensor<i32>) -> tensor<*x!tf_type.qint8>
  func.return %0 : tensor<*x!tf_type.qint8>
}

// -----

func.func @testUniformDequantize(%arg0: tensor<*x!tf_type.qint8>, %scales: tensor<2xf32>, %zps: tensor<i32>) -> tensor<*xf32> {
  // expected-error @below {{'tf.UniformDequantize' op quantization_axis is -1, scales must have 0 rank.}}
   %0 = "tf.UniformDequantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<*x!tf_type.qint8>, tensor<2xf32>, tensor<i32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @testUniformQuantizedDot(
  %input: tensor<*x!tf_type.qint8>, %weight: tensor<2x2x!tf_type.qint8>,
  %input_scales: tensor<2xf32>, %input_zps: tensor<i32>,
  %weight_scales: tensor<f32>, %weight_zps: tensor<i32>,
  %output_scales: tensor<f32>, %output_zps: tensor<i32>) -> () {
  // expected-error @below {{'tf.UniformQuantizedDot' op quantization_axis is -1, scales must have 0 rank.}}
  %1 = "tf.UniformQuantizedDot"(
    %input, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<*x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>,
        tensor<2xf32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<*x!tf_type.qint32>
  func.return
}

// -----

func.func @testUniformQuantizedDot(
  %input: tensor<*x!tf_type.qint8>, %weight: tensor<2x2x!tf_type.qint8>,
  %input_scales: tensor<f32>, %input_zps: tensor<i32>,
  %weight_scales: tensor<2xf32>, %weight_zps: tensor<i32>,
  %output_scales: tensor<f32>, %output_zps: tensor<i32>) -> () {
  // expected-error @below {{'tf.UniformQuantizedDot' op quantization_axis is -1, scales must have 0 rank.}}
  %1 = "tf.UniformQuantizedDot"(
    %input, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<*x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>,
        tensor<f32>, tensor<i32>,
        tensor<2xf32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<*x!tf_type.qint32>
  func.return
}

// -----

func.func @testUniformQuantizedDot(
  %input: tensor<*x!tf_type.qint8>, %weight: tensor<2x2x!tf_type.qint8>,
  %input_scales: tensor<f32>, %input_zps: tensor<i32>,
  %weight_scales: tensor<f32>, %weight_zps: tensor<i32>,
  %output_scales: tensor<2xf32>, %output_zps: tensor<i32>) -> () {
  // expected-error @below {{'tf.UniformQuantizedDot' op quantization_axis is -1, scales must have 0 rank.}}
  %1 = "tf.UniformQuantizedDot"(
    %input, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<*x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<2xf32>, tensor<i32>) -> tensor<*x!tf_type.qint32>
  func.return
}

// -----

func.func @testUniformQuantizedConvolution(
  %input: tensor<*x!tf_type.qint8>, %weight: tensor<2x2x!tf_type.qint8>,
  %input_scales: tensor<f32>, %input_zps: tensor<i32>,
  %weight_scales: tensor<2xf32>, %weight_zps: tensor<i32>,
  %output_scales: tensor<f32>, %output_zps: tensor<i32>) -> () {
  // expected-error @below {{'tf.UniformQuantizedConvolution' op quantization_axis is -1, scales must have 0 rank.}}
  %1 = "tf.UniformQuantizedConvolution"(
    %input, %weight,
    %input_scales, %input_zps,
    %weight_scales, %weight_zps,
    %output_scales, %output_zps) {
      window_strides = [1, 2],
      padding = "VALID",
      explicit_padding = [],
      lhs_dilation = [1, 1],
      rhs_dilation = [2, 2],
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      dimension_numbers = "\10\03\1A\02\01\02 \02(\032\02\00\01@\03J\02\01\02",
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -128 : i64,
      lhs_quantization_max_val = 127 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -128 : i64,
      rhs_quantization_max_val = 127 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<*x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>,
        tensor<f32>, tensor<i32>,
        tensor<2xf32>, tensor<i32>,
        tensor<f32>, tensor<i32>) -> tensor<*x!tf_type.qint32>
  func.return
}

// -----

func.func @testUniformQuantizedAdd(
  %input: tensor<2x2x!tf_type.qint32>, %bias: tensor<2x!tf_type.qint32>,
  %input_scales: tensor<f32>, %input_zps: tensor<i32>,
  %bias_scales: tensor<f32>, %bias_zps: tensor<i32>,
  %output_scales: tensor<2xf32>, %output_zps: tensor<i32>) -> () {
  // expected-error @below {{'tf.UniformQuantizedAdd' op quantization_axis is -1, scales must have 0 rank.}}
  %1 = "tf.UniformQuantizedAdd"(
    %input, %bias,
    %input_scales, %input_zps,
    %bias_scales, %bias_zps,
    %output_scales, %output_zps) {
      lhs_quantization_axis = -1 : i64,
      lhs_quantization_min_val = -2147483648 : i64,
      lhs_quantization_max_val = 2147483647 : i64,
      rhs_quantization_axis = -1 : i64,
      rhs_quantization_min_val = -2147483648 : i64,
      rhs_quantization_max_val = 2147483647 : i64,
      output_quantization_axis = -1 : i64,
      output_quantization_min_val = -2147483648 : i64,
      output_quantization_max_val = 2147483647 : i64} : (
        tensor<2x2x!tf_type.qint32>, tensor<2x!tf_type.qint32>,
        tensor<f32>, tensor<i32>,
        tensor<f32>, tensor<i32>,
        tensor<2xf32>, tensor<i32>) -> tensor<2x2x!tf_type.qint32>
  func.return
}

// Following tests are for LegacyCall symbol use verifier.

// -----

// Tests that valid symbol use does not produce any error.
func.func @valid_symbol_use(%arg0: tensor<i32>) -> () {
  "tf.LegacyCall"(%arg0) {f = @call_func} : (tensor<i32>) -> (tensor<i32>)
  func.return
}

func.func @call_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}

// -----

// Tests that undefined call function produces error.
func.func @test_undefined_function() -> () {
  // expected-error @below {{'f' attribute refers to an undefined function: undefined_func}}
  "tf.LegacyCall"() {f = @undefined_func} : () -> ()
  func.return
}

// -----

// Tests that argument count mismatch produces error.
func.func @test_arg_count_mismatch(%arg0: tensor<i32>) -> () {
  // expected-error @below {{argument count mismatch: 'args' has 1 argument(s), but 'call_func' expects 2}}
  "tf.LegacyCall"(%arg0) {f = @call_func} : (tensor<i32>) -> tensor<i32>
  func.return
}

func.func @call_func(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  func.return %arg0 : tensor<f32>
}

// -----

func.func @test_batch_function_with_valid_symbol(%arg0: tensor<1x3xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> () {
  "tf.BatchFunction"(%arg0, %arg1) {batch_timeout_micros = 100000 : i64, f = @batched_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return
}

func.func private @batched_function(%arg0: tensor<1x3xf32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<1x3xf32> {
  %0 = "tf.Identity"(%arg0) : (tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %0 : tensor<1x3xf32>
}

// -----

func.func @test_batch_function_with_invalid_symbol(%arg0: tensor<1x3xf32>, %arg1: tensor<!tf_type.resource<tensor<1x3xf32>>>) -> () {
  // expected-error @below {{'f' attribute refers to an undefined function: undefined_function}}
  "tf.BatchFunction"(%arg0, %arg1) {batch_timeout_micros = 100000 : i64, f = @undefined_function, max_batch_size = 6 : i64, max_enqueued_batches = 10 : i64, num_batch_threads = 1 : i64, operand_segment_sizes = array<i32: 1, 1>} : (tensor<1x3xf32>, tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<*xf32>
  func.return
}
