// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// Ensure constants roundtrip exactly

func.func @bool() -> tensor<4xi1> {
  // CHECK-LABEL: @bool
  // CHECK: value = dense<[false, true, true, false]> : tensor<4xi1>
  %0 = "tfl.pseudo_const"() { value = dense<[false, true, true, false]> : tensor<4xi1> } : () -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

func.func @complex64() -> tensor<4xcomplex<f32>> {
  // CHECK-LABEL: @complex64
  // CHECK: value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F434F4D504C455836342074656E736F725F7368617065207B2064696D207B2073697A653A2034207D207D2074656E736F725F636F6E74656E743A20225C3030305C3030305C3230303F5C3030305C3030305C3230303F5C3030305C3030305C303030405C3030305C3030305C303030405C3030305C30303040405C3030305C30303040405C3030305C3030305C323030405C3030305C3030305C3230304022"> : tensor<4xcomplex<f32>>
  %0 = "tfl.pseudo_const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F434F4D504C455836342074656E736F725F7368617065207B2064696D207B2073697A653A2034207D207D2074656E736F725F636F6E74656E743A20225C3030305C3030305C3230303F5C3030305C3030305C3230303F5C3030305C3030305C303030405C3030305C3030305C303030405C3030305C30303040405C3030305C30303040405C3030305C3030305C323030405C3030305C3030305C3230304022"> : tensor<4xcomplex<f32>> } : () -> tensor<4xcomplex<f32>>
  func.return %0 : tensor<4xcomplex<f32>>
}

func.func @complex128() -> tensor<4xcomplex<f64>> {
  // CHECK-LABEL: @complex128
  // CHECK: value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F434F4D504C45583132382074656E736F725F7368617065207B2064696D207B2073697A653A2034207D207D2074656E736F725F636F6E74656E743A20225C3030305C3030305C3030305C3030305C3030305C3030305C3336303F5C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303130405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303230405C3030305C3030305C3030305C3030305C3030305C3030305C3030304022"> : tensor<4xcomplex<f64>>
  %0 = "tfl.pseudo_const"() { value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F434F4D504C45583132382074656E736F725F7368617065207B2064696D207B2073697A653A2034207D207D2074656E736F725F636F6E74656E743A20225C3030305C3030305C3030305C3030305C3030305C3030305C3336303F5C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303130405C3030305C3030305C3030305C3030305C3030305C3030305C303030405C3030305C3030305C3030305C3030305C3030305C3030305C303230405C3030305C3030305C3030305C3030305C3030305C3030305C3030304022"> : tensor<4xcomplex<f64>> } : () -> tensor<4xcomplex<f64>>
  func.return %0 : tensor<4xcomplex<f64>>
}

func.func @f16() -> tensor<4xf16> {
  // CHECK-LABEL: @f16
  // CHECK: value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf16>
  %0 = "tfl.pseudo_const"() { value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf16> } : () -> tensor<4xf16>
  func.return %0 : tensor<4xf16>
}

func.func @f32() -> tensor<4xf32> {
  // CHECK-LABEL: @f32
  // CHECK: value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = "tfl.pseudo_const"() { value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32> } : () -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

func.func @f64() -> tensor<4xf64> {
  // CHECK-LABEL: @f64
  // CHECK: value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf64>
  %0 = "tfl.pseudo_const"() { value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64> } : () -> tensor<4xf64>
  func.return %0 : tensor<4xf64>
}

func.func @i8() -> tensor<4xi8> {
  // CHECK-LABEL: @i8
  // CHECK: value = dense<[1, 2, 3, 4]> : tensor<4xi8>
  %0 = "tfl.pseudo_const" () { value = dense<[1, 2, 3, 4]> : tensor<4xi8> } : () -> tensor<4xi8>
  func.return %0 : tensor<4xi8>
}

func.func @i16() -> tensor<4xi16> {
  // CHECK-LABEL: @i16
  // CHECK: value = dense<[1, 2, 3, 258]> : tensor<4xi16>
  %0 = "tfl.pseudo_const" () { value = dense<[1, 2, 3, 258]> : tensor<4xi16> } : () -> tensor<4xi16>
  func.return %0 : tensor<4xi16>
}

func.func @i32() -> tensor<4xi32> {
  // CHECK-LABEL: @i32
  // CHECK: value = dense<[1, 2, 3, 16909060]> : tensor<4xi32>
  // Check bytes come back in the right order
  %0 = "tfl.pseudo_const" () { value = dense<[1, 2, 3, 16909060]> : tensor<4xi32> } : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

func.func @i64() -> tensor<4xi64> {
  // CHECK-LABEL: @i64
  // CHECK: value = dense<[1, 2, 3, 72623859790382856]> : tensor<4xi64>
  %0 = "tfl.pseudo_const" () { value = dense<[1, 2, 3, 72623859790382856]> : tensor<4xi64> } : () -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

func.func @string() -> tensor<2x2x!tf_type.string> {
  // CHECK-LABEL: @string
  // CHECK: value = dense<{{\[\["1", "12"\], \["123", "1234"\]\]}}> : tensor<2x2x!tf_type.string>
  %0 = "tfl.pseudo_const"() { value = dense<[["1", "12"], ["123", "1234"]]> : tensor<2x2x!tf_type.string> } : () -> tensor<2x2x!tf_type.string>
  func.return %0 : tensor<2x2x!tf_type.string>
}

func.func @string_norank() -> tensor<!tf_type.string> {
  // CHECK-LABEL: @string_norank
  // CHECK: value = dense<"test"> : tensor<!tf_type.string>
  %0 = "tfl.pseudo_const"() { value = dense<"test"> : tensor<!tf_type.string> } : () -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}

func.func @uint8() -> tensor<4xui8> {
  // CHECK-LABEL: @uint8
  // CHECK: value = dense<[222, 173, 190, 239]> : tensor<4xui8>
  %0 = "tfl.pseudo_const"() {value = dense<[222, 173, 190, 239]> : tensor<4xui8>} : () -> tensor<4xui8>
  func.return %0 : tensor<4xui8>
}

func.func @int4() -> tensor<5xi4> {
  // CHECK-LABEL: @int4
  // CHECK: value = dense<[6, 3, -2, -7, -8]> : tensor<5xi4>
  %0 = "tfl.pseudo_const"() {value = dense<[6, 3, -2, -7, -8]> : tensor<5xi4>} : () -> tensor<5xi4>
  func.return %0 : tensor<5xi4>
}

func.func @qi32_per_axis() -> tensor<3x3x!quant.uniform<i32:f32:1, {1.0, 0.5:1, 0.25:1}>> {
  // CHECK-LABEL: @qi32_per_axis
  // CHECK: {qtype = tensor<3x3x!quant.uniform<i32:f32:1, {1.000000e+00,5.000000e-01:1,2.500000e-01:1}>>, value = dense<1> : tensor<3x3xi32>} : () -> tensor<3x3x!quant.uniform<i32:f32:1, {1.000000e+00,5.000000e-01:1,2.500000e-01:1}>>
  %0 = "tfl.pseudo_qconst"() { qtype = tensor<3x3x!quant.uniform<i32:f32:1, {1.0, 0.5:1, 0.25:1}>>, value = dense<1> : tensor<3x3xi32>} : () -> tensor<3x3x!quant.uniform<i32:f32:1, {1.0, 0.5:1, 0.25:1}>>
  func.return %0 : tensor<3x3x!quant.uniform<i32:f32:1, {1.0, 0.5:1, 0.25:1}>>
}

func.func @qi32_per_axis_zero() -> tensor<3x3x!quant.uniform<i32:f32:0, {1.0, 0.5:1, 0.25:1}>> {
  // CHECK-LABEL: @qi32_per_axis_zero
  // CHECK: {qtype = tensor<3x3x!quant.uniform<i32:f32:0, {1.000000e+00,5.000000e-01:1,2.500000e-01:1}>>, value = dense<1> : tensor<3x3xi32>} : () -> tensor<3x3x!quant.uniform<i32:f32:0, {1.000000e+00,5.000000e-01:1,2.500000e-01:1}>>
  %0 = "tfl.pseudo_qconst"() { qtype = tensor<3x3x!quant.uniform<i32:f32:0, {1.0, 0.5:1, 0.25:1}>>, value = dense<1> : tensor<3x3xi32>} : () -> tensor<3x3x!quant.uniform<i32:f32:0, {1.0, 0.5:1, 0.25:1}>>
  func.return %0 : tensor<3x3x!quant.uniform<i32:f32:0, {1.0, 0.5:1, 0.25:1}>>
}

func.func @qu8() -> tensor<3x!quant.uniform<u8<1:255>:f32, 1.0>> {
  // CHECK-LABEL: @qu8
  // CHECK: {qtype = tensor<3x!quant.uniform<u8<1:255>:f32, 1.000000e+00>>, value = dense<1> : tensor<3xi8>} : () -> tensor<3x!quant.uniform<u8<1:255>:f32, 1.000000e+00>>
  %0 = "tfl.pseudo_qconst"() { qtype = tensor<3x!quant.uniform<u8<1:255>:f32, 1.0>>, value = dense<1> : tensor<3xi8>} : () -> tensor<3x!quant.uniform<u8<1:255>:f32, 1.0>>
  func.return %0 : tensor<3x!quant.uniform<u8<1:255>:f32, 1.0>>
}

func.func @sparse_f32() -> tensor<3x2xf32> {
  // CHECK-LABEL: @sparse_f32
  // CHECK: {compressed_data = dense<[1.000000e+00, 2.000000e+00, 5.000000e-01, 2.500000e-01, -1.000000e+00, -2.000000e+00, -5.000000e-01, -2.500000e-01]> : tensor<8xf32>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = <format = DENSE, dense_size = 16, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<0.000000e+00> : tensor<3x2xf32>}
  %0 = "tfl.pseudo_sparse_const"() {compressed_data = dense<[1.0, 2.0, 0.5, 0.25, -1.0, -2.0, -0.5, -0.25]> : tensor<8xf32>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = #tfl.dimension_metadata<format = DENSE, dense_size = 16, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<0.000000e+00> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  func.return %0: tensor<3x2xf32>
}

func.func @sparse_f16() -> tensor<3x2xf16> {
  // CHECK-LABEL: @sparse_f16
  // CHECK: {compressed_data = dense<[1.000000e+00, 2.000000e+00, 5.000000e-01, 2.500000e-01, -1.000000e+00, -2.000000e+00, -5.000000e-01, -2.500000e-01]> : tensor<8xf16>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = <format = DENSE, dense_size = 16, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<0.000000e+00> : tensor<3x2xf16>}
  %0 = "tfl.pseudo_sparse_const"() {compressed_data = dense<[1.0, 2.0, 0.5, 0.25, -1.0, -2.0, -0.5, -0.25]> : tensor<8xf16>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = #tfl.dimension_metadata<format = DENSE, dense_size = 16, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<0.000000e+00> : tensor<3x2xf16>} : () -> tensor<3x2xf16>
  func.return %0: tensor<3x2xf16>
}

func.func @sparse_qu8() -> tensor<3x2x!quant.uniform<u8<1:255>:f32, 1.0>> {
  // CHECK-LABEL: @sparse_qu8
  // CHECK: {compressed_data = dense<[1, 2, 3, 4, -1, -2, -3, -4]> : tensor<8xi8>, qtype = tensor<3x2x!quant.uniform<u8<1:255>:f32, 1.000000e+00>>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = <format = DENSE, dense_size = 16, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = DENSE, dense_size = 1, segments = [], indices = []>, <format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<0> : tensor<3x2xi8>}
  %0 = "tfl.pseudo_sparse_qconst"() {compressed_data = dense<[1, 2, 3, 4, -1, -2, -3, -4]> : tensor<8xi8>, qtype = tensor<3x2x!quant.uniform<u8<1:255>:f32, 1.0>>, s_param = #tfl.sparsity_parameter<traversal_order = [0, 1, 2, 3], block_map = [3, 1], dim_metadata = #tfl.dimension_metadata<format = DENSE, dense_size = 16, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = DENSE, dense_size = 1, segments = [], indices = []>, #tfl.dimension_metadata<format = SPARSE_CSR, dense_size = 0, segments = [0, 5, 11], indices = [1, 4, 9]>>, value = dense<42> : tensor<3x2xi8>} : () -> tensor<3x2x!quant.uniform<u8<1:255>:f32, 1.0>>
  func.return %0: tensor<3x2x!quant.uniform<u8<1:255>:f32, 1.0>>
}

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}
