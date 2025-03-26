// RUN: stablehlo-quant-opt %s -convert-tf-quant-types -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @relu_qint8
func.func @relu_qint8(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8> {
  // CHECK: %[[X:.*]] = "tf.Relu"(%arg0) : (tensor<1xi8>) -> tensor<1xi8>
  %0 = "tf.Relu"(%arg0) : (tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8>
  func.return %0: tensor<1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @if_qint8(%arg0: tensor<i1>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1xi8>
func.func @if_qint8(%arg0: tensor<i1>, %arg1: tensor<1x!tf_type.qint8>, %arg2: tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8> {
  // CHECK-NEXT: %0 = "tf.IfRegion"(%arg0) <{is_stateless = false}> ({
  // CHECK-NEXT:   "tf.Yield"(%arg1) : (tensor<1xi8>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   "tf.Yield"(%arg2) : (tensor<1xi8>) -> ()
  // CHECK-NEXT:  }) : (tensor<i1>) -> tensor<1xi8>
  // CHECK-NEXT: return %0 : tensor<1xi8>
  %0 = "tf.IfRegion"(%arg0) <{is_stateless = false}> ({
    "tf.Yield"(%arg1) : (tensor<1x!tf_type.qint8>) -> ()
    }, {
    "tf.Yield"(%arg2) : (tensor<1x!tf_type.qint8>) -> ()
   }) : (tensor<i1>) -> tensor<1x!tf_type.qint8>
  func.return %0 : tensor<1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @id_qint8(%arg0: tensor<1xi8>) -> tensor<1xi8> {
func.func @id_qint8(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8> {
  // CHECK-NEXT: return %arg0 : tensor<1xi8>
  func.return %arg0: tensor<1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @id_qint16(%arg0: tensor<1xi16>) -> tensor<1xi16> {
func.func @id_qint16(%arg0: tensor<1x!tf_type.qint16>) -> tensor<1x!tf_type.qint16> {
  // CHECK-NEXT: return %arg0 : tensor<1xi16>
  func.return %arg0: tensor<1x!tf_type.qint16>
}

// -----

// CHECK-LABEL: func @id_qint32(%arg0: tensor<1xi32>) -> tensor<1xi32> {
func.func @id_qint32(%arg0: tensor<1x!tf_type.qint32>) -> tensor<1x!tf_type.qint32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  func.return %arg0: tensor<1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @id_quint8(%arg0: tensor<1xui8>) -> tensor<1xui8> {
func.func @id_quint8(%arg0: tensor<1x!tf_type.quint8>) -> tensor<1x!tf_type.quint8> {
  // CHECK-NEXT: return %arg0 : tensor<1xui8>
  func.return %arg0: tensor<1x!tf_type.quint8>
}

// -----

// CHECK-LABEL: func @id_quint16(%arg0: tensor<1xui16>) -> tensor<1xui16> {
func.func @id_quint16(%arg0: tensor<1x!tf_type.quint16>) -> tensor<1x!tf_type.quint16> {
  // CHECK-NEXT: return %arg0 : tensor<1xui16>
  func.return %arg0: tensor<1x!tf_type.quint16>
}

// -----

// CHECK-LABEL: func @uniform_quantize
func.func @uniform_quantize(%arg0: tensor<1xf32>) -> tensor<1x!tf_type.qint8>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[qint:.*]] = "tf.UniformQuantize"
  // CHECK: %[[int:.*]] = "tf.Cast"(%[[qint]]) <{Truncate = false}> : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>

  // CHECK: return %[[int]] : tensor<1xi8>
  func.return %0 : tensor<1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @uniform_quantize_no_return
func.func @uniform_quantize_no_return(%arg0: tensor<1xf32>) -> ()
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[qint:.*]] = "tf.UniformQuantize"
  // CHECK: %[[int:.*]] = "tf.Cast"(%[[qint]]) <{Truncate = false}> : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>

  // CHECK: return
  func.return
}

// -----

// CHECK-LABEL: func @uniform_dequantize
func.func @uniform_dequantize(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xf32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[x:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1x!tf_type.qint8>
  // CHECK: %[[y:.*]] = "tf.UniformDequantize"(%[[x]]
  %0 = "tf.UniformDequantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: return %[[y]] : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_dequantize
func.func @uniform_quantize_dequantize(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[qint0:.*]] = "tf.UniformQuantize"
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>

  // CHECK: %[[int:.*]] = "tf.Cast"(%[[qint0]]) <{Truncate = false}> : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  // CHECK: %[[qint1:.*]] = "tf.Cast"(%[[int]]) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1x!tf_type.qint8>
  // CHECK: %[[res:.*]] = "tf.UniformDequantize"(%[[qint1]]
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: return %[[res]] : tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantized_add
func.func @uniform_quantized_add(%arg0: tensor<2x!tf_type.qint32>, %arg1: tensor<2x!tf_type.qint32>) -> tensor<2x!tf_type.qint32>
{
    %input_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
    %input_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
    %bias_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
    %bias_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
    %output_scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
    %output_zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

    // CHECK: %[[lhs:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<2xi32>) -> tensor<2x!tf_type.qint32>
    // CHECK: %[[rhs:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<2xi32>) -> tensor<2x!tf_type.qint32>
    // CHECK: %[[res_qint:.*]] = "tf.UniformQuantizedAdd"(%[[lhs]], %[[rhs]]
    // CHECK: %[[res_int:.*]] = "tf.Cast"(%[[res_qint]]) <{Truncate = false}> : (tensor<2x!tf_type.qint32>) -> tensor<2xi32>
    // CHECK: return %[[res_int]] : tensor<2xi32>
    %1 = "tf.UniformQuantizedAdd"(
      %arg0, %arg1,
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
          tensor<2x!tf_type.qint32>, tensor<2x!tf_type.qint32>,
          tensor<f32>, tensor<i32>,
          tensor<f32>, tensor<i32>,
          tensor<f32>, tensor<i32>) -> tensor<2x!tf_type.qint32>
    func.return %1 : tensor<2x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @while_region_qint
func.func @while_region_qint(%arg0: tensor<2x2xf32>) -> (tensor<2x?xf32>, tensor<?x2xf32>) {
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps2 = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>
  %zps4 = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[qint_0:.*]] = "tf.UniformQuantize"
  // CHECK: %[[int_0:.*]] = "tf.Cast"(%[[qint_0]]) <{Truncate = false}> : (tensor<2x2x!tf_type.qint8>) -> tensor<2x2xi8>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps2) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint8>

  // CHECK: %[[qint_1:.*]] = "tf.UniformQuantize"
  // CHECK: %[[int_1:.*]] = "tf.Cast"(%[[qint_1]]) <{Truncate = false}> : (tensor<2x2x!tf_type.qint8>) -> tensor<2x2xi8>
  %1 = "tf.UniformQuantize"(%arg0, %scales, %zps4) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<2x2xf32>, tensor<f32>, tensor<i32>) -> tensor<2x2x!tf_type.qint8>

  // CHECK: %[[while_result:.*]]:2 = "tf.WhileRegion"(%[[int_0]], %[[int_1]])
  %2:2 = "tf.WhileRegion"(%0, %1) ({
  ^bb0(%carg0: tensor<2x?x!tf_type.qint8>, %carg1: tensor<?x2x!tf_type.qint8>):
    %cst = "tf.Const"()  {value = dense<1> : tensor<i1>}  : () -> tensor<i1>
    "tf.Yield"(%cst) : (tensor<i1>) -> ()
    }, {
  ^bb0(%barg0: tensor<2x?x!tf_type.qint8>, %barg1: tensor<?x2x!tf_type.qint8>):
    %id = "tf.Identity"(%barg0) : (tensor<2x?x!tf_type.qint8>) -> tensor<2x?x!tf_type.qint8>
    "tf.Yield"(%id, %barg1) : (tensor<2x?x!tf_type.qint8>, tensor<?x2x!tf_type.qint8>) -> ()
  }) {is_stateless = false} : (tensor<2x2x!tf_type.qint8>, tensor<2x2x!tf_type.qint8>) -> (tensor<2x?x!tf_type.qint8>, tensor<?x2x!tf_type.qint8>)

  // CHECK: %[[out_qint_0:.*]] = "tf.Cast"(%[[while_result]]#0) <{Truncate = false}> : (tensor<2x?xi8>) -> tensor<2x?x!tf_type.qint8>
  // CHECK: %[[out_f_0:.*]] = "tf.UniformDequantize"(%[[out_qint_0]]
  %3 = "tf.UniformDequantize"(%2#0, %scales, %zps2) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<2x?x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<2x?xf32>

  // CHECK: %[[out_qint_1:.*]] = "tf.Cast"(%[[while_result]]#1) <{Truncate = false}> : (tensor<?x2xi8>) -> tensor<?x2x!tf_type.qint8>
  // CHECK: %[[out_f_1:.*]] = "tf.UniformDequantize"(%[[out_qint_1]]
  %4 = "tf.UniformDequantize"(%2#1, %scales, %zps4) {quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64} : (tensor<?x2x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<?x2xf32>

  // CHECK: return %[[out_f_0]], %[[out_f_1]]
  func.return %3, %4 : tensor<2x?xf32>, tensor<?x2xf32>
}

// -----

// CHECK-LABEL: func @concat_uniform_quantize
func.func @concat_uniform_quantize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3x!tf_type.qint8> {
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[input:.*]] = "tf.ConcatV2"(%arg0, %arg1
  // CHECK: %[[output_qint:.*]] = "tf.UniformQuantize"(%[[input]]
  // CHECK: %[[output:.*]] = "tf.Cast"(%[[output_qint]]) <{Truncate = false}> : (tensor<6x3x!tf_type.qint8>) -> tensor<6x3xi8>
  // CHECK: return %[[output]] : tensor<6x3xi8>
  %0 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
  %1 = "tf.UniformQuantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<6x3xf32>, tensor<f32>, tensor<i32>) -> tensor<6x3x!tf_type.qint8>
  func.return %1 : tensor<6x3x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @concat_uniform_dequantize
func.func @concat_uniform_dequantize(%arg0: tensor<3x3x!tf_type.qint8>, %arg1: tensor<3x3x!tf_type.qint8>) -> tensor<6x3xf32> {
  %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[input:.*]] = "tf.ConcatV2"(%arg0, %arg1, %[[VAL:.*]]) : (tensor<3x3xi8>, tensor<3x3xi8>, tensor<i64>) -> tensor<6x3xi8>
  // CHECK: %[[input_qint:.*]] = "tf.Cast"(%[[input]]) <{Truncate = false}> : (tensor<6x3xi8>) -> tensor<6x3x!tf_type.qint8>
  // CHECK: %[[output:.*]] = "tf.UniformDequantize"(%[[input_qint]]
  // CHECK: return %[[output]] : tensor<6x3xf32>
  %0 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3x!tf_type.qint8>, tensor<3x3x!tf_type.qint8>, tensor<i64>) -> tensor<6x3x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<6x3x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<6x3xf32>
  func.return %1 : tensor<6x3xf32>
}

// -----

// CHECK-LABEL: func @tf_const_qint32
func.func @tf_const_qint32() -> tensor<1x!tf_type.qint32> {
  // CHECK: %[[result:.*]] = "tf.Const"() <{value = dense<127> : tensor<1xi32>}> : () -> tensor<1xi32>
  %0 = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F51494E5433322074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20225C3137375C3030305C3030305C30303022"> : tensor<1x!tf_type.qint32> } : () -> tensor<1x!tf_type.qint32>
  // CHECK: return %[[result]] : tensor<1xi32>
  func.return %0 :  tensor<1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @tf_const_qint8
func.func @tf_const_qint8() -> tensor<2x!tf_type.qint8> {
  // CHECK: %[[result:.*]] = "tf.Const"() <{value = dense<[127, 18]> : tensor<2xi8>}> : () -> tensor<2xi8>
  %0 = "tf.Const"() { value = #tf_type<tensor_proto : "0x746674656e736f722464747970653a2044545f51494e54382074656e736f725f7368617065207b2064696d207b2073697a653a2032207d207d2074656e736f725f636f6e74656e743a20225c3137375c30323222"> : tensor<2x!tf_type.qint8> } : () -> tensor<2x!tf_type.qint8>
  // CHECK: return %[[result]] : tensor<2xi8>
  func.return %0 :  tensor<2x!tf_type.qint8>
}

// -----

func.func @tf_const_invalid_proto() -> tensor<2x!tf_type.qint32> {
  // expected-error@+2 {{failed to get DenseElementAttr}}
  // expected-error@+1 {{failed to legalize operation 'tf.Const'}}
  %0 = "tf.Const"() { value = #tf_type<tensor_proto : "0x2532"> : tensor<2x!tf_type.qint32> } : () -> tensor<2x!tf_type.qint32>
  func.return %0 :  tensor<2x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @cast_op_qint32_int32
func.func @cast_op_qint32_int32(%arg0: tensor<1x!tf_type.qint32>) -> tensor<1xi32> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x!tf_type.qint32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @cast_op_int32_qint32
func.func @cast_op_int32_qint32(%arg0: tensor<1xi32>) -> tensor<1x!tf_type.qint32> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi32>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1xi32>) -> tensor<1x!tf_type.qint32>
  func.return %0: tensor<1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @cast_op_qint8_int8
func.func @cast_op_qint8_int8(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1xi8> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1xi8>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  func.return %0: tensor<1xi8>
}

// -----

// CHECK-LABEL: func @cast_op_int8_qint8
func.func @cast_op_int8_qint8(%arg0: tensor<1xi8>) -> tensor<1x!tf_type.qint8> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1xi8>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1xi8>) -> tensor<1x!tf_type.qint8>
  func.return %0: tensor<1x!tf_type.qint8>
}

// -----

// CHECK-LABEL: func @cast_op_qint32_int8
func.func @cast_op_qint32_int8(%arg0: tensor<1x!tf_type.qint32>) -> tensor<1xi8> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi8>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x!tf_type.qint32>) -> tensor<1xi8>
  func.return %0: tensor<1xi8>
}

// -----

// CHECK-LABEL: func @cast_op_int8_qint32
func.func @cast_op_int8_qint32(%arg0: tensor<1xi8>) -> tensor<1x!tf_type.qint32> {
  // CHECK: "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1xi32>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1xi8>) -> tensor<1x!tf_type.qint32>
  func.return %0: tensor<1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @cast_uniform_dequantize
func.func @cast_uniform_dequantize(%arg0: tensor<1x!tf_type.qint32>) -> tensor<1xf32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[x:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1xi8>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x!tf_type.qint32>) -> tensor<1x!tf_type.qint8>

  // CHECK: %[[y:.*]] = "tf.Cast"(%[[x]]) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1x!tf_type.qint8>
  // CHECK: %[[z:.*]] = "tf.UniformDequantize"(%[[y]]
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: return %[[z]] : tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_cast
func.func @uniform_quantize_cast(%arg0: tensor<1xf32>) -> tensor<1x!tf_type.qint32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK: tf.UniformQuantize
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>

  // CHECK: %1 = "tf.Cast"(%0) <{Truncate = false}> : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  // CHECK: %2 = "tf.Cast"(%1) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1xi32>
  %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint32>

  // CHECK: return %2 : tensor<1xi32>
  func.return %1 : tensor<1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_cast_dequantize
func.func @uniform_quantize_cast_dequantize(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
  %scales = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>
  %scales1 = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %zps1 = "tf.Const"() { value = dense<2> : tensor<i32> } : () -> tensor<i32>

  // CHECK: %[[qint_1:.*]] = "tf.UniformQuantize"
  // CHECK: %[[int_1:.*]] = "tf.Cast"(%[[qint_1]]) <{Truncate = false}> : (tensor<1x!tf_type.qint8>) -> tensor<1xi8>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>

  // CHECK: %[[int_2:.*]] = "tf.Cast"(%[[int_1]]) <{Truncate = false}> : (tensor<1xi8>) -> tensor<1xi32>
  %1 = "tf.Cast"(%0) {Truncate = false} : (tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint32>

  // CHECK: %[[qint_2:.*]] = "tf.Cast"(%[[int_2]]) <{Truncate = false}> : (tensor<1xi32>) -> tensor<1x!tf_type.qint32>
  // CHECK: %[[int_3:.*]] = "tf.UniformDequantize"(%[[qint_2]]
  %2 = "tf.UniformDequantize"(%1, %scales1, %zps1) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1xf32>

  // CHECK: return %[[int_3]] : tensor<1xf32>
  func.return %2 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_clip_min_cast
func.func @uniform_quantize_clip_min_cast(%arg0: tensor<1x2x2x1x!tf_type.qint32>, %arg1: tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32> {
  %scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[MIN_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  %q_min = "tf.Cast"(%arg1) {Truncate = false} : (tensor<i32>) -> tensor<!tf_type.qint32>

  // CHECK-DAG: %[[INPUT_QINT:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>
  // CHECK: "tf.UniformQuantizedClipByValue"(%[[INPUT_QINT]], %[[MIN_QINT]], %[[MIN_QINT]]
  %output = "tf.UniformQuantizedClipByValue"(%arg0, %q_min, %q_min, %scale, %zp)
    {quantization_axis = -1 : i64, quantization_max_val = 2147483647 : i64, quantization_min_val = -2147483648 : i64} :
    (tensor<1x2x2x1x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32>
  return %output : tensor<1x2x2x1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_clip_input_cast
func.func @uniform_quantize_clip_input_cast(%arg0: tensor<1x2x2x1xi32>, %arg1: tensor<!tf_type.qint32>) -> tensor<1x2x2x1x!tf_type.qint32> {
  %scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[INPUT_QINT:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>
  %q_input = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>

  // CHECK-DAG: %[[MIN_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK-DAG: %[[MAX_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK: "tf.UniformQuantizedClipByValue"(%[[INPUT_QINT]], %[[MIN_QINT]], %[[MAX_QINT]]
  %output = "tf.UniformQuantizedClipByValue"(%q_input, %arg1, %arg1, %scale, %zp)
    {quantization_axis = -1 : i64, quantization_max_val = 2147483647 : i64, quantization_min_val = -2147483648 : i64} :
    (tensor<1x2x2x1x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32>
  return %output : tensor<1x2x2x1x!tf_type.qint32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_clip_output_cast
func.func @uniform_quantize_clip_output_cast(%arg0: tensor<1x2x2x1x!tf_type.qint32>, %arg1: tensor<!tf_type.qint32>) -> tensor<1x2x2x1xi32> {
  %scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[INPUT_QINT:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>
  // CHECK-DAG: %[[MIN_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK-DAG: %[[MAX_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK: %[[OUTPUT_QINT:.*]] = "tf.UniformQuantizedClipByValue"(%[[INPUT_QINT]], %[[MIN_QINT]], %[[MAX_QINT]]
  %q_output = "tf.UniformQuantizedClipByValue"(%arg0, %arg1, %arg1, %scale, %zp)
    {quantization_axis = -1 : i64, quantization_max_val = 2147483647 : i64, quantization_min_val = -2147483648 : i64} :
    (tensor<1x2x2x1x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32>

  // CHECK: %[[OUTPUT:.*]] = "tf.Cast"(%[[OUTPUT_QINT]]) <{Truncate = false}> : (tensor<1x2x2x1x!tf_type.qint32>) -> tensor<1x2x2x1xi32>
  %output = "tf.Cast"(%q_output) {Truncate = false} : (tensor<1x2x2x1x!tf_type.qint32>) -> tensor<1x2x2x1xi32>

  return %output : tensor<1x2x2x1xi32>
}

// -----

// CHECK-LABEL: func @uniform_quantize_clip_output_cast_multiple_uses
func.func @uniform_quantize_clip_output_cast_multiple_uses(%arg0: tensor<1x2x2x1x!tf_type.qint32>, %arg1: tensor<!tf_type.qint32>) -> tensor<1x2x2x1xi32> {
  %scale = "tf.Const"() { value = dense<1.0> : tensor<f32> } : () -> tensor<f32>
  %zp = "tf.Const"() { value = dense<3> : tensor<i32> } : () -> tensor<i32>

  // CHECK-DAG: %[[INPUT_QINT:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>
  // CHECK-DAG: %[[MIN_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK-DAG: %[[MAX_QINT:.*]] = "tf.Cast"(%arg1) <{Truncate = false}> : (tensor<i32>) -> tensor<!tf_type.qint32>
  // CHECK: %[[OUTPUT_QINT:.*]] = "tf.UniformQuantizedClipByValue"(%[[INPUT_QINT]], %[[MIN_QINT]], %[[MAX_QINT]]
  %q_output = "tf.UniformQuantizedClipByValue"(%arg0, %arg1, %arg1, %scale, %zp)
    {quantization_axis = -1 : i64, quantization_max_val = 2147483647 : i64, quantization_min_val = -2147483648 : i64} :
    (tensor<1x2x2x1x!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1x!tf_type.qint32>

  // CHECK-DAG: %[[OUTPUT_1:.*]] = "tf.Cast"(%[[OUTPUT_QINT]]) <{Truncate = false}> : (tensor<1x2x2x1x!tf_type.qint32>) -> tensor<1x2x2x1xi32>
  %output = "tf.Cast"(%q_output) {Truncate = false} : (tensor<1x2x2x1x!tf_type.qint32>) -> tensor<1x2x2x1xi32>

  // CHECK-DAG: %[[OUTPUT_2:.*]] = "tf.Cast"(%[[OUTPUT_QINT]]) <{Truncate = false}> : (tensor<1x2x2x1x!tf_type.qint32>) -> tensor<1x2x2x1xi32>
  // CHECK-DAG: %[[OUTPUT_QINT_1:.*]] = "tf.Cast"(%[[OUTPUT_1]]) <{Truncate = false}> : (tensor<1x2x2x1xi32>) -> tensor<1x2x2x1x!tf_type.qint32>
  // CHECK: "tf.UniformDequantize"(%[[OUTPUT_QINT_1:.*]]
  %dq = "tf.UniformDequantize"(%q_output, %scale, %zp) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x2x2x1x!tf_type.qint32>, tensor<f32>, tensor<i32>) -> tensor<1x2x2x1xf32>

  return %output : tensor<1x2x2x1xi32>
}
