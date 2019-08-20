// RUN: tf-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===--------------------------------------------------------------------===//
//  Test TF opaque attributes
//===--------------------------------------------------------------------===//

// CHECK-LABEL: func @opaquetensorattr
func @opaquetensorattr() -> () {
^bb0:
// CHECK: "opaqueIntTensor"() {bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4xi32>} : () -> ()
  "opaqueIntTensor"(){bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4xi32>} : () -> ()
// CHECK: "opaqueFloatTensor"() {bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4xf32>} : () -> ()
  "opaqueFloatTensor"(){bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4xf32>} : () -> ()
// CHECK: "opaqueStringTensor"() {bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4x!tf.string>} : () -> ()
  "opaqueStringTensor"(){bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4x!tf.string>} : () -> ()
// CHECK: "opaqueResourceTensor"() {bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4x!tf.resource>} : () -> ()
  "opaqueResourceTensor"(){bar = opaque<"tf", "0x68656C6C6F"> : tensor<2x1x4x!tf.resource>} : () -> ()
  return
}

//===--------------------------------------------------------------------===//
//  Test raw TF operations (_tf.*)
//===--------------------------------------------------------------------===//

// Test of parsing !tf.resource type
// CHECK-LABEL: func @testTFResource(%arg0: !tf.resource)
func @testTFResource(!tf.resource) -> () {
^bb0(%arg0: !tf.resource):
  // CHECK: %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_Resource", name = "Const"} : () -> (tensor<*x!tf.resource>, !_tf.control)
  %0:2 = "_tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_Resource"} : () -> (tensor<*x!tf.resource>, !_tf.control)
  // CHECK: %1 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (!tf.resource, tensor<*x!tf.resource>) -> !_tf.control
  %1 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (!tf.resource, tensor<*x!tf.resource>) -> !_tf.control
  return
}

// Test of parsing !tf.variant type
// CHECK-LABEL: func @testTFVariant(%arg0: tensor<*x!tf.variant>)
func @testTFVariant(tensor<*x!tf.variant>) -> () {
^bb0(%arg0: tensor<*x!tf.variant>):
  // CHECK: %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_VARIANT", name = "Const"} : () -> (!tf.variant, !_tf.control)
  %0:2 = "_tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_VARIANT"} : () -> (!tf.variant, !_tf.control)
  // CHECK: %1 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.variant>, !tf.variant) -> !_tf.control
  %1 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.variant>, !tf.variant) -> !_tf.control
  return
}

// Test of parsing tf_complex type
// CHECK-LABEL: func @testTFComplex(%arg0: tensor<*x!tf.complex64>, %arg1: tensor<*x!tf.complex128>)
func @testTFComplex(tensor<*x!tf.complex64>, tensor<*x!tf.complex128>) -> (!tf.complex64, !tf.complex128) {
^bb0(%arg0: tensor<*x!tf.complex64>, %arg1: tensor<*x!tf.complex128>):
  // CHECK: %0:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_COMPLEX64", name = "Const"} : () -> (!tf.complex64, !_tf.control)
  %0:2 = "_tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_COMPLEX64"} : () -> (!tf.complex64, !_tf.control)
  // CHECK: %1:2 = "_tf.Const"() {device = "", dtype = "tfdtype$DT_COMPLEX128", name = "Const"} : () -> (!tf.complex128, !_tf.control)
  %1:2 = "_tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_COMPLEX128"} : () -> (!tf.complex128, !_tf.control)
  // CHECK: %2:2 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.complex64>, !tf.complex64) -> (!tf.complex64, !_tf.control)
  %2:2 = "_tf.AssignAddVariableOp"(%arg0, %0#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.complex64>, !tf.complex64) -> (!tf.complex64, !_tf.control)
  // CHECK: %3:2 = "_tf.AssignAddVariableOp"(%arg1, %1#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.complex128>, !tf.complex128) -> (!tf.complex128, !_tf.control)
  %3:2 = "_tf.AssignAddVariableOp"(%arg1, %1#0) {device = "", name = "AssignAddVariableOp"} : (tensor<*x!tf.complex128>, !tf.complex128) -> (!tf.complex128, !_tf.control)
  return %2#0, %3#0 : !tf.complex64, !tf.complex128
}

//===--------------------------------------------------------------------===//
//  Test TF operations (tf.*)
//===--------------------------------------------------------------------===//

// -----

// CHECK-LABEL: func @testIdentity
func @testIdentity(%arg0: tensor<4x2x!tf.stringref>) -> tensor<4x2x!tf.string> {
  // CHECK: tf.Identity
  %0 = "tf.Identity"(%arg0) : (tensor<4x2x!tf.stringref>) -> tensor<4x2x!tf.string>
  return %0 : tensor<4x2x!tf.string>
}

// -----

func @testIdentityWrongType(%arg0: tensor<4x2x!tf.string>) -> tensor<4x2x!tf.stringref> {
  // expected-error @+1 {{requires all operands to be either same as or ref type of results}}
  %0 = "tf.Identity"(%arg0) : (tensor<4x2x!tf.string>) -> tensor<4x2x!tf.stringref>
  return %0 : tensor<4x2x!tf.stringref>
}

// -----

// TODO(hinsu): Move this to MLIR core once the test dialect have a custom type.

// Check that broadcastable trait accepts TF specific element type
// CHECK-LABEL: func @testAdd
func @testAdd(%arg0: tensor<4x2x!tf.string>, %arg1: tensor<2x!tf.string>) -> tensor<4x2x!tf.string> {
  // CHECK: %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x2x!tf.string>, tensor<2x!tf.string>) -> tensor<4x2x!tf.string>
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<4x2x!tf.string>, tensor<2x!tf.string>) -> tensor<4x2x!tf.string>
  return %0 : tensor<4x2x!tf.string>
}

// -----

func @testValidBiasAdd(tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32> {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<16xf32>):
  // CHECK: %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func @testBiasAddNoDataFormatOk(tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32> {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<16xf32>):
  // CHECK: %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT"} : (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT"}: (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

func @testBiasAddWrongDataFormat(tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32> {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<16xf32>):
  // expected-error @+1 {{attribute 'data_format' failed to satisfy constraint: 'NHWC' or 'NCHW' convnet data format}}
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "HWCN"} : (tensor<1x32x32x16xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----

// Test valid tf.BroadcastTo
// CHECK-LABEL: func @testBroadcastTo(%arg0: tensor<16xf32>)
func @testBroadcastTo(%arg0: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %cst = constant dense<16> : tensor<4xi32>
  // CHECK: %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  %0 = "tf.BroadcastTo"(%arg0, %cst) : (tensor<16xf32>, tensor<4xi32>) -> tensor<16x16x16x16xf32>
  return %0 : tensor<16x16x16x16xf32>
}

// -----

// Test valid tf.LeakyRelu
// CHECK-LABEL: func @testLeakyRelu(%arg0: tensor<16xf32>)
func @testLeakyRelu(tensor<16xf32>) -> tensor<16xf32> {
^bb0(%arg0: tensor<16xf32>):
  // CHECK: %0 = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "tf.LeakyRelu"(%arg0) {alpha = 0.2 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----
func @testLeakyWrongAlphaType(tensor<16xf32>) -> tensor<16xf32> {
^bb0(%arg0: tensor<16xf32>):
  // expected-error @+1 {{attribute 'alpha' failed to satisfy constraint: 32-bit float}}
  %0 = "tf.LeakyRelu"(%arg0) {alpha = 1: i32}: (tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: func @testMul
func @testMul(%arg0: tensor<2x!tf.uint16>) -> (tensor<2x!tf.uint16>) {
  // CHECK: tf.Mul
  %0 = "tf.Mul"(%arg0, %arg0) {T = "tfdtype$DT_UINT16", device = "/device:CPU:0", name = "Mul"} : (tensor<2x!tf.uint16>, tensor<2x!tf.uint16>) -> tensor<2x!tf.uint16>
  return %0 : tensor<2x!tf.uint16>
}

// -----

// CHECK-LABEL: func @testReshape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<10000xf32>, %arg3: tensor<*xi32>)
func @testReshape(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<10000xf32>, %arg3: tensor<*xi32>) -> (tensor<100x100xf32>, tensor<*xf32>, tensor<10000xf32>, tensor<100x100xf32>, tensor<*xf32>, tensor<*xf32>) {
  // CHECK: %cst = constant dense<100> : tensor<2xi32>
  %shape1 = constant dense<100> : tensor<2xi32>
  // CHECK: %0 = "tf.Reshape"(%arg0, %cst) : (tensor<*xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<*xf32>, tensor<2xi32>) -> (tensor<100x100xf32>)
  // CHECK: %1 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", device = "", name = "Shape", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  %shape2 = "tf.Shape"(%arg0) {device = "", name = "Shape", T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> (tensor<?xi32>)
  // CHECK: %2 = "tf.Reshape"(%arg1, %1) {T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32", device = "", name = "Reshape_1"} : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
  %r2 = "tf.Reshape"(%arg1, %shape2) {device = "", name = "Reshape_1", T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32"} : (tensor<*xf32>, tensor<?xi32>) -> (tensor<*xf32>)
  // CHECK: %3 = "tf.Reshape"(%arg2, %cst) {T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32", device = "", name = "Reshape_1"} : (tensor<10000xf32>, tensor<2xi32>) -> tensor<10000xf32>
  %r3 = "tf.Reshape"(%arg2, %shape1) {device = "", name = "Reshape_1", T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32"} : (tensor<10000xf32>, tensor<2xi32>) -> (tensor<10000xf32>)
  // CHECK: %cst_0 = constant dense<[-1, 100]> : tensor<2xi32>
  %shape3 = constant dense<[-1, 100]> : tensor<2xi32>
  // CHECK: %4 = "tf.Reshape"(%arg2, %cst_0) {T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32", device = "", name = "Reshape_1"} : (tensor<10000xf32>, tensor<2xi32>) -> tensor<100x100xf32>
  %r4 = "tf.Reshape"(%arg2, %shape3) {device = "", name = "Reshape_1", T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32"} : (tensor<10000xf32>, tensor<2xi32>) -> (tensor<100x100xf32>)
  // CHECK: "tf.Reshape"(%arg0, %arg3)
  %r5 = "tf.Reshape"(%arg0, %arg3) {T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32"} : (tensor<*xf32>, tensor<*xi32>) -> (tensor<*xf32>)
  // CHECK: "tf.Reshape"(%arg2, %arg3)
  %r6 = "tf.Reshape"(%arg2, %arg3) {T = "tfdtype$DT_FLOAT", Tshape = "tfdtype$DT_INT32"} : (tensor<10000xf32>, tensor<*xi32>) -> (tensor<*xf32>)
  return %r1, %r2, %r3, %r4, %r5, %r6: tensor<100x100xf32>, tensor<*xf32>, tensor<10000xf32>, tensor<100x100xf32>, tensor<*xf32>, tensor<*xf32>
}

// -----
// tf.Reshape with incorrect type.
func @testReshape(tensor<*xf32>, tensor<*xf32>) -> (tensor<100x100xf32>) {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>):
  %shape1 = constant dense<100.> : tensor<2xf32>
  // expected-error @+1 {{must be tensor of 32/64-bit integer values}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<*xf32>, tensor<2xf32>) -> (tensor<100x100xf32>)
  return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with incorrect element number.
func @testReshape(%arg0: tensor<10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = constant dense<100> : tensor<2xi32>
  // expected-error @+1 {{mismatch in tensor elements and shape implied elements}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2xi32>) -> (tensor<100x100xf32>)
  return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with more than one -1 in the shape.
func @testReshape(%arg0: tensor<10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = constant dense<-1> : tensor<2xi32>
  // expected-error @+1 {{more than one component of shape are -1}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2xi32>) -> (tensor<100x100xf32>)
  return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with -1 in the shape can't infer the dimension.
func @testReshape(%arg0: tensor<10x10x10xf32>) -> tensor<100x100xf32> {
  %shape1 = constant dense<[101, -1]> : tensor<2xi32>
  // expected-error @+1 {{one component of shape is -1 but couldn't infer the dimension}}
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x10xf32>, tensor<2xi32>) -> (tensor<100x100xf32>)
  return %r1 : tensor<100x100xf32>
}

// -----
// tf.Reshape with a first operand that has non-static shape.
func @testReshape(%arg0: tensor<10x10x?xf32>) -> tensor<10x10xf32> {
  %shape1 = constant dense<[10, 10]> : tensor<2xi32>
  %r1 = "tf.Reshape" (%arg0, %shape1) : (tensor<10x10x?xf32>, tensor<2xi32>) -> (tensor<10x10xf32>)
  return %r1 : tensor<10x10xf32>
}

// -----

// CHECK-LABEL: func @testValidAvgPool
func @testValidAvgPool(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // CHECK: %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: func @testAvgPoolMissingDataFormatOk
func @testAvgPoolMissingDataFormatOk(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // CHECK: %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongDataType(tensor<1x7x7x16xi32>) -> tensor<1x1x1x16xi32> {
^bb0(%arg0: tensor<1x7x7x16xi32>):
  // expected-error @+1 {{must be tensor of floating-point values}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_INT", data_format = "NHWC", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xi32>) -> tensor<1x1x1x16xi32>
  return %0 : tensor<1x1x1x16xi32>
}

// -----

func @testAvgPoolWrongDataFormat(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'data_format' failed to satisfy constraint: 'NHWC' or 'NCHW' convnet data format}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "HWCN", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolNoKsize(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'ksize'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongKsizeCount(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [7, 7, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongKsizeType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7.5, 1], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----
func @testAvgPoolWrongKsizeIntType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{'ksize' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1 : i32, 7 : i32, 7 : i32, 1 : i32], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolNoPadding(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'padding'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongPadding(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'padding' failed to satisfy constraint: string attribute whose value is SAME, or VALID}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "MAGIC", strides = [1, 1, 1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolNoStrides(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{requires attribute 'strides'}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID"} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongStridesCount(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 1]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

func @testAvgPoolWrongStridesType(tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32> {
^bb0(%arg0: tensor<1x7x7x16xf32>):
  // expected-error @+1 {{attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute with at least 4 elements}}
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", ksize = [1, 7, 7, 1], padding = "VALID", strides = ["1", "1", "1", "1"]} : (tensor<1x7x7x16xf32>) -> tensor<1x1x1x16xf32>
  return %0 : tensor<1x1x1x16xf32>
}

// -----

// CHECK-LABEL: func @testValidConv2D
func @testValidConv2D(tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x16xf32>) :
  // CHECK: %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", device = "", name = "MobilenetV2/expanded_conv/depthwise/depthwise", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  %0 = "tf.Conv2D"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", device = "", name = "MobilenetV2/expanded_conv/depthwise/depthwise", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x16xf32>) -> tensor<256x30x30x16xf32>
  return %0 : tensor<256x30x30x16xf32>
}

// -----

// CHECK-LABEL: func @testValidDepthwiseConv2dNative
func @testValidDepthwiseConv2dNative(tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32> {
^bb0(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<3x3x3x4xf32>) :
  // CHECK: %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], name = "MobilenetV2/expanded_conv/depthwise/depthwise", padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
  %0 = "tf.DepthwiseConv2dNative"(%arg0, %arg1) {device = "", name = "MobilenetV2/expanded_conv/depthwise/depthwise", T = "tfdtype$DT_FLOAT", data_format = "NHWC", dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<256x32x32x3xf32>, tensor<3x3x3x4xf32>) -> tensor<256x30x30x12xf32>
  return %0 : tensor<256x30x30x12xf32>
}

// -----

// Test valid tf.FakeQuantWithMinMaxArgs
// CHECK-LABEL: func @testValidFakeQuantWithMinMaxArgs
func @testValidFakeQuantWithMinMaxArgs(tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  // CHECK: %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {max = 1.000000e+00 : f32, min = -1.000000e+00 : f32, num_bits = 3 : i64} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 3} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxArgs
func @testInvalidFakeQuantWithMinMaxArgsWrongAttr(tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>):
  // expected-error @+1 {{requires num_bits to be between 2 and 16, inclusive}}
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 0} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test valid tf.FakeQuantWithMinMaxVars
// CHECK-LABEL: func @testValidFakeQuantWithMinMaxVars
func @testValidFakeQuantWithMinMaxVars(tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK: %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func @testInvalidFakeQuantWithMinMaxVarsWrongAttr(tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
  // expected-error @+1 {{requires num_bits to be between 2 and 16, inclusive}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {min = -1.0 : f32, max = 1.0 : f32, num_bits = 0} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func @testInvalidFakeQuantWithMinMaxVarsWrongMinRank(tensor<8x8x8x8xf32>, tensor<1xf32>, tensor<2xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<1xf32>, %arg2: tensor<2xf32>):
  // expected-error @+1 {{requires min to be a 0d float tensor}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<1xf32>, tensor<2xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func @testInvalidFakeQuantWithMinMaxVarsWrongMaxRank(tensor<8x8x8x8xf32>, tensor<f32>, tensor<2xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<2xf32>):
  // expected-error @+1 {{requires max to be a 0d float tensor}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<2xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func @testInvalidFakeQuantWithMinMaxVarsWrongMinType(tensor<8x8x8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>):
  // expected-error @+1 {{op operand #1 must be tensor of 32-bit float values}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FakeQuantWithMinMaxVars
func @testInvalidFakeQuantWithMinMaxVarsWrongMaxType(tensor<8x8x8x8xf32>, tensor<f32>, tensor<i32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>):
  // expected-error @+1 {{op operand #2 must be tensor of 32-bit float values}}
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<i32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>
}

// -----

// Test valid tf.FusedBatchNorm
// CHECK-LABEL: func @testFusedBatchNorm
func @testFusedBatchNorm(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // CHECK: %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 1.000000e-03 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func @testFusedBatchNormWrongXType(tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires x to be a 4D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func @testFusedBatchNormWrongScaleType(tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xi32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{operand #1 must be tensor of 32-bit float values}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// -----

// Test invalid tf.FusedBatchNorm
func @testFusedBatchNormWrongOffsetType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<2x8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires offset to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// -----
// Test invalid tf.FusedBatchNorm
func @testFusedBatchNormWrongMeanType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<?x8xf32>, %arg4: tensor<8xf32>):
  // expected-error @+1 {{requires mean to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<?x8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// -----
// Test invalid tf.FusedBatchNorm
func @testFusedBatchNormWrongVarianceType(tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) -> tensor<8x8x8x8xf32> {
^bb0(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<*xf32>):
  // expected-error @+1 {{requires variance to be a 1D float tensor}}
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<*xf32>)
  return %0#0 : tensor<8x8x8x8xf32>
}

// -----
func @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// Test valid tf.If operation
// CHECK-LABEL: func @testValidIfOp
func @testValidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

func @testIfThen(f32) -> f32
func @testIfElse(f32) -> f32

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, f32) -> f32 {
^bb0(%arg0: tensor<i1>, %arg1: f32):
  // expected-error @+1 {{operand #1 must be tensor of tf.dtype values}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, f32) -> f32

  return %1 : f32
}

// -----

func @testIfElse(tensor<2xf32>) -> tensor<2xf32>

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{requires attribute 'then_branch'}}
  %1 = "tf.If"(%arg0, %arg1) {
    else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

func @testIfThen(tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
func @testIfElse(tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{branches should have 1 inputs}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

func @testIfThen(tensor<*xf16>) -> tensor<*xf32>
func @testIfElse(tensor<*xf32>) -> tensor<*xf32>

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // expected-error @+1 {{then branch input type tensor<*xf16> is incompatible with operand type tensor<2xf32>}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

func @testIfThen(tensor<2xf32>) -> tensor<*xf32>
func @testIfElse(tensor<3xf32>) -> tensor<*xf32>

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, tensor<*xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  // expected-error @+1 {{branches inputs have incompatible types tensor<2xf32> and tensor<3xf32>}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

func @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func @testIfElse(tensor<*xf32>) -> tensor<3xf32>

// Test invalid tf.If operation
func @testInvalidIfOp(tensor<i1>, tensor<*xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  // expected-error @+1 {{else branch result type tensor<3xf32> is incompatible with op result type tensor<2xf32>}}
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen,
    else_branch = @testIfElse,
    is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> tensor<2xf32>

  return %1 : tensor<2xf32>
}

// -----

// Test valid tf.Softmax
// CHECK-LABEL: func @testSoftmax
func @testSoftmax(tensor<8x16xf32>) -> tensor<8x16xf32> {
^bb0(%arg0: tensor<8x16xf32>):
  // CHECK: %0 = "tf.Softmax"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = "tf.Softmax"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

// Test invalid tf.Softmax
func @testSoftmax(%arg0 : tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{requires operand to have rank at least 1}}
  %0 = "tf.Softmax"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test valid 'While' operation
// CHECK-LABEL: func @testWhileResult
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----
func @testWhileUndefinedCond(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{cond refers to an undefined function : undefined_func}}
  %0 = "tf.While"(%arg0, %arg1) {cond = @undefined_func, body = @body, is_stateless = false} : (tensor<i1>, tensor<f32>) -> (tensor<f32>)
  return %0 : tensor<f32>
}

func @body(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32>

// -----
func @testWhileUndefinedBody(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<f32> {
  // expected-error @+1 {{body refers to an undefined function : undefined_func}}
  %0 = "tf.While"(%arg0, %arg1) {cond = @cond, body = @undefined_func, is_stateless = false} : (tensor<i1>, tensor<f32>) -> (tensor<f32>)
  return %0 : tensor<f32>
}

func @cond(%arg0: tensor<i1>, %arg1: tensor<f32>) -> tensor<i1>

// -----

func @testWhileCond(tensor<*xf32>) -> ()
func @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{requires cond function to have exactly one result}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----

func @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xi32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{operand type tensor<*xf32> is incompatible with result type}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xi32>)

  return %1 : tensor<*xi32>
}

// -----

func @testWhileCond(tensor<*xi32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{operand type tensor<*xf32> is incompatible with cond function input type}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----

func @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{requires the number of operands to be equal to the number of body function inputs. Found 1 and 2, respectively}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----

func @testWhileCond(tensor<*xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>) -> (tensor<*xi32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{body function result type tensor<*xi32> is incompatible with result type}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----

func @testWhileCond(tensor<3xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<4xf32>) -> (tensor<*xf32>)

// Test invalid 'While' operation
func @testWhileResult(tensor<*xf32>) -> (tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{cond function input type tensor<3xf32> is incompatible with body function input type}}
  %1 = "tf.While"(%arg0) {
    cond = @testWhileCond,
    body = @testWhileBody,
    is_stateless = false
  } : (tensor<*xf32>) -> (tensor<*xf32>)

  return %1 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @testValidShape
func @testValidShape(tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<4xi32>, tensor<?xi32>) {
^bb0(%arg0: tensor<1x32x32x16xf32>, %arg1: tensor<*xf32>):
  // CHECK: %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<4xi32>
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<4xi32>
  // CHECK: %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  return %0, %1 : tensor<4xi32>, tensor<?xi32>
}

// -----

func @testShapeWrongResultElemType(%arg0: tensor<1x32x32x16xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result #0 must be tensor of 32/64-bit integer values}}
  %0 = "tf.Shape"(%arg0) : (tensor<1x32x32x16xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func @testShapeWrongResultDim(tensor<1x32x32x16xf32>) -> tensor<*xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires 1D result type}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

func @testShapeMismatchDim(tensor<1x32x32x16xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<1x32x32x16xf32>):
  // expected-error @+1 {{requires dimension size of result to match rank of operand}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

func @testShapeWrongResultDim(tensor<*xf32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<*xf32>):
  // expected-error @+1 {{requires dynamic shape result for unranked input}}
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// Test invalid tf.Const
func @testConst() -> tensor<f32> {
  // expected-error @+1 {{attribute 'value' failed to satisfy constraint: constant vector/tensor}}
  %0 = "tf.Const"() {T = "tfdtype$DT_FLOAT", value = 1.0 : f32} : () -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Test valid tf.Transpose
func @testTranspose(tensor<2x3xf32>) -> tensor<3x2xf32> {
^bb0(%arg0: tensor<2x3xf32>):
  %cst = constant dense<[1, 0]> : tensor<2xi32>
  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>

// CHECK-LABEL: testTranspose
// CHECK:  %0 = "tf.Transpose"(%arg0, %cst) {T = "tfdtype$DT_FLOAT", Tperm = "tfdtype$DT_INT32"} : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
}

// -----

// Test invalid tf.Less
func @testLess(tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
  // expected-error @+1 {{op result #0 must be tensor of 1-bit integer values}}
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Test valid tf.ConcatV2
func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1xi32>) -> tensor<?xf32> {
  // CHECK: %0 = "tf.ConcatV2"(%arg0, %arg0, %arg1) {N = 2 : i64} : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<1xi32>) -> tensor<?xf32>
  %0 = "tf.ConcatV2"(%arg, %arg, %axis) {N = 2: i64} : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// tf.ConcatV2 with wrong 'axis' element type
func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{operand #2 must be tensor of 32/64-bit integer values}}
  %0 = "tf.ConcatV2"(%arg, %arg, %axis) {N = 2: i64} : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<1xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// tf.ConcatV2 missing required 'axis' operand
func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected 1 or more operands}}
  %0 = "tf.ConcatV2"() {N = 0: i64} : () -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// tf.ConcatV2 with less than required number of values for the variadic operand
func @testConcatV2(%arg: tensor<8x16xf32>, %axis: tensor<1xi32>) -> tensor<?xf32> {
  // expected-error @+1 {{attribute 'N' failed to satisfy constraint: 64-bit integer attribute whose minimal value is 2}}
  %0 = "tf.ConcatV2"(%arg, %axis) {N = 1: i64} : (tensor<8x16xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

